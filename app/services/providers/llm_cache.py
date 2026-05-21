"""Disk-backed LLM response cache using SQLite.

Activated only when env LLM_CACHE_ENABLED=1.
DB path: env LLM_CACHE_DB (default: .llm_cache.sqlite in project root).

Usage is transparent — StructuredLlmManager checks the cache before every
structured_request() call and stores successful results for future reuse.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _db_path() -> Path:
    env_path = os.environ.get("LLM_CACHE_DB", "")
    if env_path:
        return Path(env_path)
    # Default: project root (two levels up from this file: providers → services → app → root)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    return project_root / ".llm_cache.sqlite"


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def make_cache_key(system_prompt: str, user_text: str, response_model: Type[BaseModel]) -> str:
    """SHA-256 of system + user + model name + model schema."""
    try:
        schema_str = json.dumps(response_model.model_json_schema(), sort_keys=True)
    except Exception:
        schema_str = ""
    raw = "\x00".join([
        system_prompt,
        user_text,
        response_model.__name__,
        schema_str,
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# LlmCache
# ---------------------------------------------------------------------------

class LlmCache:
    """SQLite-backed cache for structured LLM responses.

    Thread-safe for read-heavy workloads (each call opens a short connection).
    No external dependencies — stdlib sqlite3 only.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS llm_cache (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    def __init__(self) -> None:
        self._path = _db_path()
        self._init_db()

    def _connect(self) -> Optional[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(str(self._path), timeout=5)
            conn.execute("PRAGMA journal_mode=WAL")
            return conn
        except Exception as exc:
            logger.warning("LlmCache: cannot connect to %s: %s", self._path, exc)
            return None

    def _init_db(self) -> None:
        conn = self._connect()
        if conn is None:
            return
        try:
            conn.execute(self._SCHEMA)
            conn.commit()
        except Exception as exc:
            logger.warning("LlmCache: schema init failed: %s", exc)
        finally:
            conn.close()

    def get(self, key: str) -> Optional[str]:
        """Return cached JSON string or None (never raises)."""
        conn = self._connect()
        if conn is None:
            return None
        try:
            cur = conn.execute("SELECT value FROM llm_cache WHERE key = ?", (key,))
            row = cur.fetchone()
            return row[0] if row else None
        except Exception as exc:
            logger.warning("LlmCache.get failed: %s", exc)
            return None
        finally:
            conn.close()

    def set(self, key: str, value: str) -> None:
        """Store JSON string (never raises)."""
        conn = self._connect()
        if conn is None:
            return
        try:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()
        except Exception as exc:
            logger.warning("LlmCache.set failed: %s", exc)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Module-level singleton (created lazily)
# ---------------------------------------------------------------------------

_cache: Optional[LlmCache] = None


def get_cache() -> Optional[LlmCache]:
    """Return the singleton LlmCache if LLM_CACHE_ENABLED=1, else None."""
    enabled = os.environ.get("LLM_CACHE_ENABLED", "0").strip() in ("1", "true", "yes")
    if not enabled:
        return None
    global _cache
    if _cache is None:
        _cache = LlmCache()
    return _cache
