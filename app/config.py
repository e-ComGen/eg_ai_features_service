import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
INTERNAL_SERVICE_SECRET = os.getenv("INTERNAL_SERVICE_SECRET", "")

# --- Cheap LLM providers (Tier 1 cost reduction) ---
# DeepSeek: env var is DEEP_SEEK_API_KEY (user convention)
DEEPSEEK_API_KEY = os.getenv("DEEP_SEEK_API_KEY", "")
DEEPSEEK_DEFAULT_MODEL = os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat")
DEEPSEEK_PRO_MODEL = os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro")

# OpenRouter: env var is OPEN_ROUTER_API_KEY (user convention)
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY", "")

# Serper web search API
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

WEB_SEARCH_MODEL = os.getenv("WEB_SEARCH_MODEL", "gpt-4o")
WEB_SEARCH_MAX_CONCURRENT = int(os.getenv("WEB_SEARCH_MAX_CONCURRENT", "10"))

# ---------------------------------------------------------------------------
# Provider selection — config-driven, override via env vars
# ---------------------------------------------------------------------------
# Who handles what type of LLM call:
#   "openai"      → existing OpenAIManager (gpt-4o-mini, structured parse)
#   "deepseek"    → DeepSeekProvider (deepseek-v4-flash / pro, JSON mode)
#   "openrouter"  → OpenRouterProvider (any model via OpenRouter gateway)
PROVIDER_MAIN: str = os.getenv("PROVIDER_MAIN", "deepseek")       # parser, deduction, judge, extraction
PROVIDER_VISION: str = os.getenv("PROVIDER_VISION", "openrouter") # vision producer (Gemini)
PROVIDER_WEB_SEARCH: str = os.getenv("PROVIDER_WEB_SEARCH", "serper")  # web search: "openai" | "serper"

# Model identifiers per stage (override via env)
MAIN_MODEL: str = os.getenv("MAIN_MODEL", "deepseek-chat")
PREMIUM_MODEL: str = os.getenv("PREMIUM_MODEL", "deepseek-reasoner")
VISION_MODEL: str = os.getenv("VISION_MODEL", "google/gemini-2.5-flash")
EXTRACTION_FROM_TEXT_MODEL: str = os.getenv("EXTRACTION_FROM_TEXT_MODEL", "deepseek-chat")

# Tree routing — defaults to the same provider/model as PROVIDER_MAIN
PROVIDER_ROUTER: str = os.getenv("PROVIDER_ROUTER", os.getenv("PROVIDER_MAIN", "deepseek"))
ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", os.getenv("MAIN_MODEL", "deepseek-chat"))

# ---------------------------------------------------------------------------
# Fallback to OpenAI on transient provider errors
# ---------------------------------------------------------------------------
ENABLE_OPENAI_FALLBACK: bool = os.getenv("ENABLE_OPENAI_FALLBACK", "true").lower() in ("1", "true", "yes")
OPENAI_FALLBACK_MODEL: str = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")
OPENAI_FALLBACK_MODEL_VISION: str = os.getenv("OPENAI_FALLBACK_MODEL_VISION", "gpt-4o")

# ---------------------------------------------------------------------------
# OpenAI strict json_schema — для enum-heavy structured extraction
# ---------------------------------------------------------------------------
# gpt-4.1-mini поддерживает token-level enum enforcement через llguidance.
# Используется ТОЛЬКО когда response_model.__has_enum_constraints__ == True.
# $0.40 input / $1.60 output per 1M tokens — дороже DeepSeek, но без retries.
OPENAI_STRUCTURED_MODEL: str = os.getenv("OPENAI_STRUCTURED_MODEL", "gpt-4.1-mini")
# Toggle for routing enum-heavy extraction through OpenAI strict mode.
# When false, get_openai_strict_manager() returns None and enum extraction falls
# back to the main provider (DeepSeek). Set false while the OpenAI quota is dead
# (a present-but-429 key otherwise routes strict calls into a dead provider with
# no fallback → sources return []). Restore to true when OpenAI quota returns.
USE_OPENAI_STRICT: bool = os.getenv("USE_OPENAI_STRICT", "true").lower() in ("1", "true", "yes")

# Vague / placeholder feature names that must NEVER trigger an LLM call.
# These are operator-defined placeholders (e.g. "NewFeature" from CS-Cart's
# default schema templates) — extracting any value for them would be a
# hallucination by definition. Match is case-insensitive, regex, full-string
# (re.fullmatch against the trimmed feature name).
VAGUE_FEATURE_PATTERNS = [
    r"new[\s_-]*feature\s*\d*",       # NewFeature, new_feature, NewFeature2
    r"feature[\s_-]*\d+",             # Feature 1, Feature_2, feature-3
    r"field[\s_-]*\d*",               # Field, Field 1
    r"attr(?:ibute)?[\s_-]*\d*",      # Attr, Attribute, Attr1
    r"[a-z]",                          # single-char placeholder X, Y, Z
    r"\d+",                            # purely numeric names
    r"none",                           # literal "None"
    r"null",                           # literal "Null"
    r"undefined",
    r"placeholder",
    r"todo",
    r"tbd",
]

# ---------------------------------------------------------------------------
# Feature flag: new cost-aware enrichment pipeline (PipelineOrchestrator)
# ---------------------------------------------------------------------------
# When True, job_processor routes attribute filling through PipelineOrchestrator
# via PipelineAdapter instead of the legacy per-feature LLM flow.
# Default OFF so existing behaviour is completely unchanged.
# Enable via env: USE_NEW_PIPELINE=true (or "1" / "yes").
USE_NEW_PIPELINE: bool = os.getenv("USE_NEW_PIPELINE", "false").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Feature flag: bypass the prompt-tree (TreeRouter) verification — EXPLICIT,
# decoupled from USE_NEW_PIPELINE so "do we route through the owner's prompt-
# tree" is its own reversible switch.
# ---------------------------------------------------------------------------
# True  (current default, owner decision 2026-06-16) = BYPASS the prompt-tree.
#   The cost-aware source-merge pipeline fills attributes and the deterministic
#   UNIVERSAL_VERIFY gate (+ LLM judge once re-enabled) acts as an after-the-
#   fact guard. This is a STOPGAP — it CURES hallucinations after generation
#   (unit-sanity / country-rule / evidence-self-admission = band-aids).
# False (the PROPER future investment) = route values through the prompt-tree's
#   crafted per-leaf prompts so hallucinations are PREVENTED at generation and
#   never arise. NOT yet wired into the source-merge pipeline; flipping False
#   today is a no-op fallback to bypass + a logged TODO until that lands.
# Kept ON now: defer the tree investment, keep the stopgap, but the bypass is
# now an explicit conscious toggle instead of zashitое behaviour.
BYPASS_PROMPT_TREE: bool = os.getenv("BYPASS_PROMPT_TREE", "true").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Scrapfly — last-resort Ozon card gap-filler
# ---------------------------------------------------------------------------
# API key: read SCRAPFLY_API_KEY (canonical) or the legacy SCRAPFLY_KEY env var.
SCRAPFLY_API_KEY: str = os.getenv("SCRAPFLY_API_KEY", "") or os.getenv("SCRAPFLY_KEY", "")

# Feature flag: Scrapfly gap-fill fires ONLY when this env var is set to "true".
# Default OFF — prevents unexpected credits spend unless explicitly enabled.
# Enable via env: SCRAPFLY_OZON_FALLBACK_ENABLED=true (or "1" / "yes").
SCRAPFLY_OZON_FALLBACK_ENABLED: bool = os.getenv(
    "SCRAPFLY_OZON_FALLBACK_ENABLED", "false"
).lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Lamoda Scrapfly — last-resort clothing attribute gap-filler
# ---------------------------------------------------------------------------
# Fetches attributes from Lamoda product cards via Scrapfly (residential proxy,
# DataDome bypass). Pre-gated by LLM "is this clothing?" classifier — non-clothing
# products pay zero Scrapfly credits. Cost: 30 credits/product + 1 Serper call.
# Default OFF — set LAMODA_SCRAPFLY_ENABLED=true to enable.
LAMODA_SCRAPFLY_ENABLED: bool = os.getenv(
    "LAMODA_SCRAPFLY_ENABLED", "false"
).lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Safe LLM enum fill — gated fill for short optional enum attrs
# ---------------------------------------------------------------------------
# When True, PipelineOrchestrator runs SafeEnumFillSource as a late stage
# (after all card/web sources) for still-empty optional short-enum targets.
# Every proposed fill is gated: verbatim-evidence in fetched text (Gate A)
# OR adversarial LLM verifier (Gate B). Gate B is a second focused LLM call
# that defaults to RETRACT, batched per product (1 call covers all fills).
# Cost per product: 1 proposal call + 1 adversarial call (when Gate A misses).
# Default OFF — set SAFE_LLM_ENUM_FILL_ENABLED=true to enable.
SAFE_LLM_ENUM_FILL_ENABLED: bool = os.getenv(
    "SAFE_LLM_ENUM_FILL_ENABLED", "false"
).lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# TnvedSource authoritative fix — drop LLM/web garbage codes before merge
# ---------------------------------------------------------------------------
# When True (default), values for ТН ВЭД attributes from LLM_KNOWLEDGE or
# WEB_SEARCH are dropped before _finalize unless they came from TnvedSource
# (identified by evidence prefix "tnved_resolver:").  TnvedSource itself uses
# LLM_KNOWLEDGE but produces a validated 10-digit code via per-category prompt
# — its fills are deliberately preserved.  Set TNVED_SOURCE_FIX_ENABLED=false
# to revert to the old behaviour (web_search wins with garbage codes).
TNVED_SOURCE_FIX_ENABLED: bool = os.getenv(
    "TNVED_SOURCE_FIX_ENABLED", "true"
).lower() not in ("0", "false", "no")

# ---------------------------------------------------------------------------
# Unit normalization for card-sourced values — deterministic, data-driven
# ---------------------------------------------------------------------------
# WB/Ozon card extraction maps a spec line to an attribute by fuzzy NAME match
# without checking unit compatibility, so a value with its own unit ("5.4 см")
# can land in a field that declares a different one ("Ширина, мм"). When True
# (default), WbCardSource losslessly converts such EXPLICIT-explicit, same-
# dimension mismatches (5.4 см → 54) BEFORE value_id resolution. The conversion
# table is data — app/services/enrichment/strategies/dictionaries/data/
# unit_conversions.json — and only fires on an unambiguous scalar with explicit
# units on both sides (bare numbers, ranges, lists, cross-dimension are left
# untouched). Arithmetic only, no LLM. Set UNIT_NORMALIZE_ENABLED=false to
# revert to raw card values.
UNIT_NORMALIZE_ENABLED: bool = os.getenv(
    "UNIT_NORMALIZE_ENABLED", "true"
).lower() not in ("0", "false", "no")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment (.env)")
if not INTERNAL_SERVICE_SECRET:
    raise RuntimeError("INTERNAL_SERVICE_SECRET is not set in environment (.env)")
