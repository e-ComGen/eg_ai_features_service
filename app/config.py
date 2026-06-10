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

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment (.env)")
if not INTERNAL_SERVICE_SECRET:
    raise RuntimeError("INTERNAL_SERVICE_SECRET is not set in environment (.env)")
