import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
INTERNAL_SERVICE_SECRET = os.getenv("INTERNAL_SERVICE_SECRET", "")

# --- Cheap LLM providers (Tier 1 cost reduction) ---
# DeepSeek: env var is DEEP_SEEK_API_KEY (user convention)
DEEPSEEK_API_KEY = os.getenv("DEEP_SEEK_API_KEY", "")
DEEPSEEK_DEFAULT_MODEL = os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-v4-flash")
DEEPSEEK_PRO_MODEL = os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro")

# OpenRouter: env var is OPEN_ROUTER_API_KEY (user convention)
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY", "")

# Serper web search API
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

WEB_SEARCH_MODEL = os.getenv("WEB_SEARCH_MODEL", "gpt-4o")
WEB_SEARCH_MAX_CONCURRENT = int(os.getenv("WEB_SEARCH_MAX_CONCURRENT", "10"))

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

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment (.env)")
if not INTERNAL_SERVICE_SECRET:
    raise RuntimeError("INTERNAL_SERVICE_SECRET is not set in environment (.env)")
