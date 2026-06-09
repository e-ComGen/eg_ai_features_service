"""
app/services/providers/domain_health.py

Persistent dead-domain memory for the Scrappey fallback.

Tracks per-domain Scrappey failure outcomes and marks domains "dead"
when enough distinct-URL block failures accumulate.  Dead domains are
skipped on subsequent runs to avoid wasting paid Scrappey credits on
sites that consistently return anti-bot shells.

Death validation rules
----------------------
- Three (DEAD_DOMAIN_STRIKES) confirmed BLOCKED outcomes on DISTINCT URLs
  are required before declaring a domain dead.  A repeat failure on the
  same URL does not increment the strike counter (one bad page ≠ dead
  domain).
- TRANSIENT errors (timeouts, connect errors, Scrappey 5xx) never count
  toward death — they reflect network hiccups, not structural blocks.
- Dead status expires after DEAD_DOMAIN_TTL_DAYS (default 14) so that
  sites that lift their blocks are re-probed automatically.
- A USABLE result resets all strikes and clears dead_until immediately
  (domain proven alive).

Exceptions (DOMAIN_HEALTH_NEVER_DEAD)
--------------------------------------
ozon.ru and wildberries.ru (and their subdomains) are always exempt —
should_skip_scrappey returns False for them even if block_strikes is
maxed out.  Configurable via the DOMAIN_HEALTH_NEVER_DEAD env var
(comma-separated registrable domains).

Persistence
-----------
State is stored in a JSON file whose path is resolved the same way as
other project caches: relative to this file's package root, in
.domain_health.json.  Override with DOMAIN_HEALTH_STORE env var.
The file is loaded once at module import, saved on every meaningful
update, and registered with atexit for a final flush on process exit.

All I/O errors are caught and silently degraded — a corrupt store
must never crash a fetch.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------

def _dead_domain_strikes() -> int:
    try:
        return int(os.environ.get("DEAD_DOMAIN_STRIKES", "3"))
    except ValueError:
        return 3


def _dead_domain_ttl_days() -> float:
    try:
        return float(os.environ.get("DEAD_DOMAIN_TTL_DAYS", "14"))
    except ValueError:
        return 14.0


def _never_dead_domains() -> frozenset[str]:
    """Return the set of registrable domains that are always Scrappey-eligible."""
    raw = os.environ.get("DOMAIN_HEALTH_NEVER_DEAD", "ozon.ru,wildberries.ru")
    return frozenset(d.strip().lower() for d in raw.split(",") if d.strip())


# ---------------------------------------------------------------------------
# Store path
# ---------------------------------------------------------------------------

_DEFAULT_STORE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",  # project root
    ".domain_health.json",
)


def _store_path() -> str:
    return os.environ.get("DOMAIN_HEALTH_STORE", _DEFAULT_STORE_PATH)


# ---------------------------------------------------------------------------
# Domain record
# ---------------------------------------------------------------------------

@dataclass
class DomainRecord:
    block_strikes: int = 0
    transient_fails: int = 0
    last_failed_url: str = ""
    dead_until: Optional[float] = None
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DomainRecord":
        return cls(
            block_strikes=int(d.get("block_strikes", 0)),
            transient_fails=int(d.get("transient_fails", 0)),
            last_failed_url=str(d.get("last_failed_url", "")),
            dead_until=d.get("dead_until"),  # float or None
            last_seen=float(d.get("last_seen", time.time())),
        )


# ---------------------------------------------------------------------------
# In-memory store (loaded once at import)
# ---------------------------------------------------------------------------

# Keyed by registrable domain (e.g. "citilink.ru")
_store: dict[str, DomainRecord] = {}
_store_loaded: bool = False


def _load_store() -> None:
    global _store, _store_loaded
    if _store_loaded:
        return
    _store_loaded = True
    path = _store_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict = json.load(fh)
        _store = {k: DomainRecord.from_dict(v) for k, v in raw.items()}
        logger.debug("domain_health: loaded %d records from %s", len(_store), path)
    except FileNotFoundError:
        _store = {}
    except Exception as exc:
        logger.warning("domain_health: could not load store from %r: %s — starting fresh", path, exc)
        _store = {}


def _save_store() -> None:
    path = _store_path()
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump({k: v.to_dict() for k, v in _store.items()}, fh, indent=2)
        os.replace(tmp_path, path)
        logger.debug("domain_health: saved %d records to %s", len(_store), path)
    except Exception as exc:
        logger.warning("domain_health: could not save store to %r: %s", path, exc)


atexit.register(_save_store)


# ---------------------------------------------------------------------------
# Registrable-domain extraction
# ---------------------------------------------------------------------------

# Two-part TLDs that are common in RU e-commerce space
_TWO_PART_TLDS = frozenset({
    "co.uk", "co.jp", "com.au", "com.br", "com.mx",
    "com.tr", "com.ar", "co.nz", "co.za",
})

_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _registrable_domain(host: str) -> str:
    """Fold host to its registrable domain (strip subdomains).

    Examples:
        www.citilink.ru   → citilink.ru
        dns-shop.ru       → dns-shop.ru
        m.wildberries.ru  → wildberries.ru
        example.co.uk     → example.co.uk
        1.2.3.4           → 1.2.3.4  (IP kept as-is)
    """
    host = host.lower().strip()
    if _IP_RE.match(host):
        return host
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    # Check for two-part TLD
    two_part = ".".join(parts[-2:])
    if two_part in _TWO_PART_TLDS and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _is_never_dead(domain: str) -> bool:
    """Return True if ``domain`` matches any never-dead exception."""
    never = _never_dead_domains()
    return domain in never or any(
        domain == nd or domain.endswith("." + nd) for nd in never
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ScrappeyOutcome = Literal["USABLE", "BLOCKED", "TRANSIENT"]

# How long after a "same URL" block before we allow it to count again.
# 1 hour: if the same page is re-attempted after cooling down, the domain
# is probably being re-probed intentionally, so count it.
_SAME_URL_COOLDOWN_SECS = 3600.0


def should_skip_scrappey(host: str) -> bool:
    """Return True if Scrappey should be skipped for this host.

    True only when:
    - The registrable domain is currently marked dead (dead_until in the future), AND
    - The domain is NOT in the never-dead exception list.

    Thread safety: asyncio is single-threaded so no lock is needed.
    """
    _load_store()
    try:
        domain = _registrable_domain(host)
        if _is_never_dead(domain):
            return False
        rec = _store.get(domain)
        if rec is None:
            return False
        if rec.dead_until is None:
            return False
        return time.time() < rec.dead_until
    except Exception as exc:
        logger.debug("domain_health.should_skip_scrappey error for %r: %s", host, exc)
        return False  # degrade gracefully — never block a fetch on store error


def record_scrappey_outcome(host: str, url: str, outcome: ScrappeyOutcome) -> None:
    """Update the domain health record based on a Scrappey call outcome.

    USABLE  → reset strikes, clear dead_until (domain alive).
    BLOCKED → increment block_strikes only on distinct URLs (or after cooldown);
              set dead_until when threshold reached.
    TRANSIENT → increment transient_fails only (network hiccup, NOT death).
    """
    _load_store()
    try:
        domain = _registrable_domain(host)
        rec = _store.get(domain)
        if rec is None:
            rec = DomainRecord()
            _store[domain] = rec

        rec.last_seen = time.time()

        if outcome == "USABLE":
            rec.block_strikes = 0
            rec.dead_until = None
            logger.debug("domain_health: %s — USABLE, strikes reset", domain)

        elif outcome == "BLOCKED":
            # Distinct-URL guard: only count if the URL differs from the last
            # failure, OR enough time has passed (re-probe of the same URL).
            same_url = (url == rec.last_failed_url)
            enough_time = (
                rec.last_seen > 0
                and (time.time() - rec.last_seen) >= _SAME_URL_COOLDOWN_SECS
            )
            if not same_url or enough_time:
                rec.block_strikes += 1
                rec.last_failed_url = url
                logger.debug(
                    "domain_health: %s — BLOCKED on %r (strikes=%d)", domain, url, rec.block_strikes
                )
            else:
                logger.debug(
                    "domain_health: %s — BLOCKED on same URL %r, not counting again", domain, url
                )

            threshold = _dead_domain_strikes()
            if rec.block_strikes >= threshold and rec.dead_until is None:
                ttl_secs = _dead_domain_ttl_days() * 86400.0
                rec.dead_until = time.time() + ttl_secs
                logger.warning(
                    "domain_health: %s declared DEAD (strikes=%d) — dead until %.0f",
                    domain, rec.block_strikes, rec.dead_until,
                )

        elif outcome == "TRANSIENT":
            rec.transient_fails += 1
            logger.debug(
                "domain_health: %s — TRANSIENT fail (total=%d), not counting toward death",
                domain, rec.transient_fails,
            )

        _save_store()

    except Exception as exc:
        logger.warning("domain_health.record_scrappey_outcome error for %r: %s", host, exc)
