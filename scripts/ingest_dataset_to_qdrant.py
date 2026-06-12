"""Unified product-dataset → Qdrant indexer.

Streams product records from external datasets, maps them to a common payload
schema, embeds the product NAME with paraphrase-multilingual-MiniLM-L12-v2
(same model as ozon_products), and upserts into a Qdrant collection so that
CompetitorRagSource can read them identically to Ozon cards.

PAYLOAD SCHEMA (matches CompetitorRagSource._coerce_characteristics)
----------------------------------------------------------------------
  name            : str  — embedded text (product name)
  categories      : str  — category path / text (used by MatchText filter)
  characteristics : str  — JSON-encoded dict {attr_name: [val1, val2] | val}
  source          : str  — dataset id ("off" / "obf" / "opff" / "amazon" /
                           "abo" / "ikea" / "rebrickable") — traceable/deletable

SUPPORTED DATASETS
------------------

1. Open Food Facts  (ODbL)   --dataset off
   HF: openfoodfacts/product-database  split=food  (~3.9M rows, ~5 GB parquet)
   Confirmed fields: product_name (list[{lang,text}]), brands (str),
   categories (str), ingredients_text (list[{lang,text}]),
   quantity (str), packaging (str), nutriments (dict|None), code (EAN str)
   Full-run: ~3.9M rows; GPU T4 ≈ 25 min embed; Qdrant ≈ 7 GB

2. Open Beauty Facts  (ODbL)  --dataset obf
   HF: openfoodfacts/product-database  split=beauty  (~900K rows)
   Same schema as OFF. characteristics: INCI/ingredients, brands, quantity,
   packaging.
   Full-run: ~900K rows; GPU T4 ≈ 6 min embed; Qdrant ≈ 1.7 GB

3. Open Pet Food Facts  (ODbL)  --dataset opff
   Source: https://static.openpetfoodfacts.org/data/openpetfoodfacts-products.jsonl.gz
   Confirmed fields: product_name (str), brands (str), categories_tags
   (list[str]), ingredients_text_en / ingredients_text_with_allergens (str),
   quantity (str)
   Full-run: ~300K rows; GPU T4 ≈ 2 min; Qdrant ≈ 0.5 GB

4. Amazon Reviews 2023 product metadata  (non-commercial research)  --dataset amazon
   HF: McAuley-Lab/Amazon-Reviews-2023  raw_meta_<Category>/full-*.parquet
   Confirmed fields: title (str), main_category (str), categories (list[str]),
   features (list[str]), details (JSON str → dict), store (str),
   parent_asin (str)
   Full-run (all 24 cats): ~48M rows; GPU T4 ≈ 5 hours; Qdrant ≈ 74 GB
   Recommended: pick 5–10 categories (10–15M rows)

5. Amazon Berkeley Objects  (CC BY 4.0)  --dataset abo
   Source: s3://amazon-berkeley-objects (--no-sign-request)
     listings/metadata/listings_0.json.gz … listings_9.json.gz  (~10 shards)
   Confirmed fields: item_name (list[{language_tag,value}]),
   brand/color/style/bullet_point (list[{language_tag?,value}]),
   product_type (list[{value}]), item_id (ASIN)
   Full-run: ~147K items; GPU T4 ≈ 1 min; Qdrant ≈ 0.3 GB

6. IKEA US Products 2025  (public)  --dataset ikea
   HF: jeffreyszhou/ikea-us-products-2025  split=train
   Confirmed fields: title (str), materials (str), care_instructions (str),
   category_tree (str), style (str), price (str|float), product_id (str)
   Full-run: ~25K rows; GPU T4 < 1 min; Qdrant ≈ 0.05 GB

7. Rebrickable LEGO sets  (CC BY)  --dataset rebrickable
   Source: https://cdn.rebrickable.com/media/downloads/sets.csv.gz
   Confirmed fields: set_num, name, year, theme_id, num_parts, img_url
   Full-run: ~20K rows; GPU T4 < 1 min; Qdrant ≈ 0.04 GB

8. GSMArena phones  --dataset gsmarena
   Source: Kaggle arwinneil/gsmarena-phone-dataset  (requires KAGGLE_USERNAME +
   KAGGLE_KEY env vars OR ~/.kaggle/kaggle.json)
   SKIPPED GRACEFULLY if credentials absent.

RUNPOD FULL-RUN PLAN
--------------------
See the RUNPOD_RUN_PLAN section at the bottom of this docstring.

USAGE
-----
# Sample 200 rows → throwaway test collection (safe, won't touch ozon_products):
python scripts/ingest_dataset_to_qdrant.py \\
    --dataset off --collection rag_ingest_test --sample 200

# Sample beauty:
python scripts/ingest_dataset_to_qdrant.py \\
    --dataset obf --collection rag_ingest_test --sample 200

# Sample Amazon:
python scripts/ingest_dataset_to_qdrant.py \\
    --dataset amazon --amazon-category All_Beauty \\
    --collection rag_ingest_test --sample 200

# Sample ABO:
python scripts/ingest_dataset_to_qdrant.py \\
    --dataset abo --collection rag_ingest_test --sample 200

# Sample IKEA:
python scripts/ingest_dataset_to_qdrant.py \\
    --dataset ikea --collection rag_ingest_test --sample 200

# Full OFF into ozon_products (RunPod overnight):
python scripts/ingest_dataset_to_qdrant.py \\
    --dataset off --collection ozon_products --full --batch 512 --encode-batch 256

# Validate + clean test collection after run:
python scripts/ingest_dataset_to_qdrant.py \\
    --dataset off --collection rag_ingest_test --sample 50 --validate-only

# Delete the throwaway collection:
python scripts/ingest_dataset_to_qdrant.py --delete-collection rag_ingest_test

REQUIREMENTS
------------
    pip install datasets sentence-transformers qdrant-client python-dotenv
    # Optional for ABO:
    pip install boto3   # or use no-sign requests directly (default)
    # HF_TOKEN in .env or env var  (faster downloads, gated datasets)

RUNPOD FULL-RUN PLAN
--------------------
Pod recommended: RunPod GPU pod, RTX 3090 or A40 (24 GB VRAM), 100 GB disk.
Qdrant: must be running at localhost:6333 (set QDRANT_URL=http://localhost:6333).
Collection: use ozon_products (existing) — new points are UPSERTED, not overwriting.

STEP 1 — start the pod and set up environment:
    git clone <repo>
    cd cpAiFeatures
    python -m venv venv && source venv/bin/activate
    pip install datasets sentence-transformers qdrant-client python-dotenv
    # Restore ozon_products snapshot first (from scripts/qdrant_restore.sh)
    bash scripts/qdrant_restore.sh

STEP 2 — verify Qdrant is healthy:
    curl http://localhost:6333/healthz

STEP 3 — run each dataset (can be parallelised across multiple pods):

Dataset         Rows     GPU T4 estimate  Qdrant size  Command
----------      ------   ---------------  -----------  -------
OFF (food)      3.9M     ~25 min          ~7 GB        --dataset off  --full
OBF (beauty)    0.9M     ~6 min           ~1.7 GB      --dataset obf  --full
OPFF (pet)      0.3M     ~2 min           ~0.5 GB      --dataset opff --full
Amazon-Beauty   0.2M     ~2 min           ~0.4 GB      --dataset amazon --amazon-category All_Beauty --full
Amazon-Electr.  3.0M     ~20 min          ~6 GB        --dataset amazon --amazon-category Electronics --full
Amazon-Fashion  1.5M     ~10 min          ~3 GB        --dataset amazon --amazon-category Amazon_Fashion --full
ABO             0.15M    <1 min           ~0.3 GB      --dataset abo  --full
IKEA            0.025M   <1 min           ~0.05 GB     --dataset ikea --full
Rebrickable     0.02M    <1 min           ~0.04 GB     --dataset rebrickable --full

STEP 4 — build text index on categories field (ONCE after all ingest):
    python scripts/build_rag_text_index.py  (or use --no-text-index and run separately)

STEP 5 — snapshot and download:
    bash scripts/qdrant_extract.sh   # creates .tar.gz snapshot
    # Download to local WSL: rsync / scp

All datasets except GSMArena need only public network (no creds).
GSMArena requires KAGGLE_USERNAME + KAGGLE_KEY env vars.

Total for recommended subset (OFF+OBF+OPFF+Amazon 3 cats+ABO+IKEA+Rebrickable):
  ~10M rows, ~1 hour on RTX 3090, ~20 GB Qdrant storage added.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import uuid
import zlib
from pathlib import Path
from typing import Any, Iterator, Optional

# Windows DLL ordering fix.
try:
    import pyarrow  # noqa: F401
    import pandas   # noqa: F401
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows: force UTF-8 on stdout/stderr so non-ASCII product names print safely.
if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

# ── Constants ─────────────────────────────────────────────────────────────────

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DIM = 384
QDRANT_URL_DEFAULT = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION = "ozon_products"
DEFAULT_SAMPLE = 200

AMAZON_CATEGORIES = [
    "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing",
    "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books",
    "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry",
    "Electronics", "Grocery_and_Gourmet_Food", "Health_and_Household",
    "Home_and_Kitchen", "Industrial_and_Scientific", "Musical_Instruments",
    "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies",
    "Software", "Sports_and_Outdoors", "Tools_and_Home_Improvement",
    "Toys_and_Games", "Video_Games",
]

ABO_SHARDS = [
    f"https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_{i}.json.gz"
    for i in range(10)
]
OPFF_URL = "https://static.openpetfoodfacts.org/data/openpetfoodfacts-products.jsonl.gz"
REBRICKABLE_URL = "https://cdn.rebrickable.com/media/downloads/sets.csv.gz"


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified product dataset → Qdrant ingest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        choices=["off", "obf", "opff", "amazon", "abo", "ikea", "rebrickable", "gsmarena"],
        help="Dataset id to ingest.",
    )
    p.add_argument(
        "--amazon-category", default="All_Beauty",
        choices=AMAZON_CATEGORIES, metavar="CATEGORY",
        help="Amazon category (used with --dataset amazon).",
    )
    p.add_argument(
        "--collection", default=DEFAULT_COLLECTION, metavar="NAME",
        help=f"Qdrant collection name. Default: {DEFAULT_COLLECTION}",
    )
    p.add_argument(
        "--qdrant-url", default=QDRANT_URL_DEFAULT,
        help=f"Qdrant server URL. Default: {QDRANT_URL_DEFAULT}",
    )
    p.add_argument(
        "--sample", type=int, default=DEFAULT_SAMPLE, metavar="N",
        help=f"Rows to ingest in sample mode (default: {DEFAULT_SAMPLE}). Ignored with --full.",
    )
    p.add_argument(
        "--full", action="store_true",
        help="Full ingest (no row limit). Overnight job — see module docstring for scale.",
    )
    p.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help="Hard cap on rows ingested even in --full mode (0 = no cap). "
             "Useful for Phase-1 proof runs: e.g. --full --limit 400000.",
    )
    p.add_argument(
        "--batch", type=int, default=256,
        help="Qdrant upsert batch size. Default: 256",
    )
    p.add_argument(
        "--encode-batch", type=int, default=128,
        help="Sentence-transformer encode batch size. Default: 128",
    )
    p.add_argument(
        "--recreate", action="store_true",
        help="Delete and recreate the collection before ingest.",
    )
    p.add_argument(
        "--no-text-index", action="store_true",
        help="Skip creating payload text index on 'categories' field.",
    )
    p.add_argument(
        "--validate-only", action="store_true",
        help="Run only the post-ingest query validation (no ingest).",
    )
    p.add_argument(
        "--delete-collection", metavar="NAME",
        help="Delete the named collection and exit.",
    )
    p.add_argument(
        "--print-sample", type=int, default=1, metavar="N",
        help="Print N mapped+embedded points to stdout for quality inspection. Default: 1",
    )
    p.add_argument(
        "--out-parquet", metavar="DIR", default=None,
        help=(
            "Output mode: write embedded rows to parquet shards in DIR instead of "
            "upserting to Qdrant. Columns: id(str), vector(list[float] len=384), "
            "name(str), categories(str), characteristics(str JSON), source(str). "
            "Shards named file_0000.parquet, file_0001.parquet, … every 100k rows. "
            "Requires pyarrow. Device='cuda' if available (GPU pod)."
        ),
    )
    p.add_argument(
        "--parquet-shard-size", type=int, default=100_000, metavar="N",
        help="Rows per parquet shard file. Default: 100000.",
    )
    return p.parse_args()


# ── ID helpers ────────────────────────────────────────────────────────────────

def _stable_uuid(namespace: str, key: str) -> str:
    """Deterministic UUID5 from namespace:key — idempotent upserts."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}:{key}"))


def _hash_uuid(value: str) -> str:
    """Fallback: UUID from SHA-1 of the string (for records without natural IDs)."""
    return str(uuid.UUID(hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()[:32]))


# ── Embedding ─────────────────────────────────────────────────────────────────

_model_cache = None
_embed_device: Optional[str] = None


def get_model(device: Optional[str] = None):
    global _model_cache, _embed_device
    if _model_cache is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            sys.exit("[Error] sentence-transformers not installed.")
        # Auto-select device: explicit arg > cuda if available > cpu
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        _embed_device = device
        print(f"[Ingest] Loading embed model: {EMBED_MODEL_NAME} (device={device}) ...")
        _model_cache = SentenceTransformer(EMBED_MODEL_NAME, device=device)
        get_dim = getattr(
            _model_cache,
            "get_embedding_dimension",
            getattr(_model_cache, "get_sentence_embedding_dimension", lambda: VECTOR_DIM),
        )
        dim = get_dim()
        print(f"[Ingest] Model ready (dim={dim})")
    return _model_cache


def embed_batch(texts: list[str]) -> list[list[float]]:
    vecs = get_model().encode(texts, normalize_embeddings=True,
                              batch_size=len(texts), show_progress_bar=False)
    return [v.tolist() for v in vecs]


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def get_client(qdrant_url: str):
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except ImportError:
        sys.exit("[Error] qdrant-client not installed.")
    print(f"[Ingest] Connecting to Qdrant at {qdrant_url}")
    return QdrantClient(url=qdrant_url, timeout=60)


def ensure_collection(client, name: str, recreate: bool) -> None:
    from qdrant_client.models import Distance, VectorParams  # type: ignore
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        if recreate:
            print(f"[Ingest] Recreating '{name}' ...")
            client.delete_collection(name)
        else:
            info = client.get_collection(name)
            print(f"[Ingest] '{name}' exists ({info.points_count or 0} pts). Resuming (upsert).")
            return
    print(f"[Ingest] Creating '{name}' (dim={VECTOR_DIM}, COSINE) ...")
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )


def create_text_index(client, collection: str, field: str = "categories") -> None:
    from qdrant_client.models import PayloadSchemaType  # type: ignore
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=PayloadSchemaType.TEXT,
        )
        print(f"[Ingest] Text index on '{field}' created.")
    except Exception as e:
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            print(f"[Ingest] Text index already exists (idempotent).")
        else:
            print(f"[Ingest] Warning: text index failed: {e}")


def delete_collection(client, name: str) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        client.delete_collection(name)
        print(f"[Ingest] Deleted collection '{name}'.")
    else:
        print(f"[Ingest] Collection '{name}' does not exist — nothing to delete.")


# ── Common field helpers ──────────────────────────────────────────────────────

def _str(v: Any, maxlen: int = 300) -> str:
    if v is None:
        return ""
    return str(v).strip()[:maxlen]


def _list_str(field: Any, max_items: int = 5, maxlen: int = 100) -> str:
    """Flatten list[str|dict] → joined str."""
    if not field:
        return ""
    if isinstance(field, str):
        return field[:500]
    if isinstance(field, list):
        parts = []
        for item in field[:max_items]:
            if isinstance(item, dict):
                t = item.get("text") or item.get("value") or ""
                if t:
                    parts.append(str(t)[:maxlen])
            elif item:
                parts.append(str(item)[:maxlen])
        return " | ".join(parts)
    return str(field)[:300]


def _encode_characteristics(d: dict) -> str:
    """JSON-encode characteristics dict so _coerce_characteristics can parse it.

    Format: {attr_name: [val1, val2, ...] | scalar_val}
    Values that are lists are kept as lists; scalars are passed through.
    Matches what CompetitorRagSource._coerce_characteristics expects:
      - isinstance(raw, str) → json.loads(raw) → dict
    """
    return json.dumps(d, ensure_ascii=False)


# ── OFF / OBF adapter ─────────────────────────────────────────────────────────

def _off_product_name(row: dict) -> str:
    """product_name is list[{lang, text}] — prefer ru then en then main."""
    names = row.get("product_name") or []
    if isinstance(names, str):
        return names.strip()[:300]
    if not isinstance(names, list):
        return ""
    for lang in ("ru", "en", "main"):
        for item in names:
            if isinstance(item, dict) and item.get("lang") == lang:
                t = item.get("text", "").strip()
                if t:
                    return t[:300]
    for item in names:
        if isinstance(item, dict):
            t = item.get("text", "").strip()
            if t:
                return t[:300]
    return ""


def _off_ingredients(row: dict) -> str:
    """ingredients_text is list[{lang, text}] in OFF HF parquet."""
    field = row.get("ingredients_text")
    if not field:
        return ""
    if isinstance(field, str):
        return field.strip()[:500]
    if isinstance(field, list):
        # Prefer ru then en then main
        for lang in ("ru", "en", "main"):
            for item in field:
                if isinstance(item, dict) and item.get("lang") == lang:
                    t = item.get("text", "").strip()
                    if t:
                        return t[:500]
        # Any non-empty
        for item in field:
            if isinstance(item, dict):
                t = item.get("text", "").strip()
                if t:
                    return t[:500]
    return str(field)[:300]


def _off_nutriments(nutriments: Any, max_keys: int = 8) -> dict:
    """nutriments is a dict or None; pick the most nutritionally informative keys."""
    if not isinstance(nutriments, dict):
        return {}
    KEEP = {"energy-kcal_100g", "fat_100g", "saturated-fat_100g",
            "carbohydrates_100g", "sugars_100g", "proteins_100g",
            "salt_100g", "fiber_100g"}
    out = {}
    for k, v in nutriments.items():
        if k in KEEP and v is not None:
            out[k] = str(v)
        if len(out) >= max_keys:
            break
    return out


def stream_off(split: str, limit: int) -> Iterator[tuple[str, str, dict]]:
    """Yield (point_id, embed_text, payload) for OFF/OBF rows."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        sys.exit("[Error] datasets not installed.")

    source_id = "off" if split == "food" else "obf"
    print(f"[{source_id.upper()}] Loading openfoodfacts/product-database split={split} streaming ...")
    ds = load_dataset(
        "openfoodfacts/product-database",
        split=split,
        streaming=True,
        token=HF_TOKEN or None,
    )

    count = 0
    for row in ds:
        if limit and count >= limit:
            break
        name = _off_product_name(row)
        if not name:
            continue

        brands = _str(row.get("brands"), 200)
        categories = _str(row.get("categories"), 400)
        ingredients = _off_ingredients(row)
        quantity = _str(row.get("quantity"), 100)
        packaging = _str(row.get("packaging"), 200)
        nutriments_dict = _off_nutriments(row.get("nutriments"))
        code = _str(row.get("code"), 50)

        embed_text = " | ".join(p for p in [
            name, brands, categories[:200], ingredients[:200]
        ] if p)

        characteristics = {}
        if ingredients:
            characteristics["Состав"] = [ingredients]
        if brands:
            characteristics["Бренд"] = [brands]
        if quantity:
            characteristics["Количество"] = [quantity]
        if packaging:
            characteristics["Упаковка"] = [packaging]
        for k, v in nutriments_dict.items():
            characteristics[k] = [v]

        point_id = _stable_uuid(source_id, code) if code else _hash_uuid(embed_text)

        payload = {
            "name": name,
            "categories": categories,
            "characteristics": _encode_characteristics(characteristics),
            "source": source_id,
        }

        yield point_id, embed_text, payload
        count += 1


# ── Open Pet Food Facts adapter ───────────────────────────────────────────────

def _stream_jsonl_gz(url: str) -> Iterator[dict]:
    """Stream line-by-line from a remote jsonl.gz via zlib streaming decompressor."""
    req = urllib.request.Request(url, headers={"User-Agent": "eg-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = zlib.decompressobj(zlib.MAX_WBITS | 16)  # gzip mode
        buf = ""
        while True:
            chunk = resp.read(65536)
            if not chunk:
                # Flush remaining decompressor output
                try:
                    remainder = d.flush()
                    if remainder:
                        buf += remainder.decode("utf-8", errors="replace")
                except Exception:
                    pass
                break
            try:
                buf += d.decompress(chunk).decode("utf-8", errors="replace")
            except Exception:
                break
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        # Trailing line without newline
        if buf.strip():
            try:
                yield json.loads(buf.strip())
            except Exception:
                pass


def stream_opff(limit: int) -> Iterator[tuple[str, str, dict]]:
    """Yield (point_id, embed_text, payload) for Open Pet Food Facts."""
    print(f"[OPFF] Streaming from {OPFF_URL} ...")
    count = 0
    for row in _stream_jsonl_gz(OPFF_URL):
        if limit and count >= limit:
            break
        # product_name is a plain str in OPFF (unlike OFF HF parquet)
        name = _str(row.get("product_name") or row.get("product_name_en"), 300)
        if not name:
            continue

        brands = _str(row.get("brands"), 200)
        categories_tags = row.get("categories_tags") or []
        categories = " | ".join(str(t) for t in categories_tags[:6])
        ingredients = (
            _str(row.get("ingredients_text_en"), 500)
            or _str(row.get("ingredients_text"), 500)
            or _str(row.get("ingredients_text_with_allergens"), 500)
        )
        quantity = _str(row.get("quantity"), 100)
        code = _str(row.get("code"), 50)

        embed_text = " | ".join(p for p in [name, brands, categories[:200]] if p)

        characteristics = {}
        if ingredients:
            characteristics["Состав"] = [ingredients]
        if brands:
            characteristics["Бренд"] = [brands]
        if quantity:
            characteristics["Количество"] = [quantity]
        # Species/type from categories
        species = [t.replace("en:", "") for t in categories_tags if "cat-food" in t or "dog-food" in t]
        if species:
            characteristics["Вид животного"] = species[:3]

        point_id = _stable_uuid("opff", code) if code else _hash_uuid(embed_text)

        payload = {
            "name": name,
            "categories": categories,
            "characteristics": _encode_characteristics(characteristics),
            "source": "opff",
        }

        yield point_id, embed_text, payload
        count += 1


# ── Amazon Reviews 2023 adapter ───────────────────────────────────────────────

def _amz_details_dict(details: Any) -> dict:
    """details field is a JSON string in the parquet — parse it."""
    if not details:
        return {}
    if isinstance(details, str):
        try:
            parsed = json.loads(details)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    if isinstance(details, dict):
        return details
    return {}


def stream_amazon(category: str, limit: int) -> Iterator[tuple[str, str, dict]]:
    """Yield (point_id, embed_text, payload) for Amazon Reviews 2023 metadata."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        sys.exit("[Error] datasets not installed.")

    parquet_shard = (
        f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/"
        f"raw_meta_{category}/full-00000-of-00001.parquet"
    )
    parquet_glob = (
        f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/"
        f"raw_meta_{category}/full-*.parquet"
    )
    print(f"[Amazon] Loading raw_meta_{category} streaming ...")
    try:
        ds = load_dataset(
            "parquet",
            data_files={"full": parquet_shard},
            split="full",
            streaming=True,
            token=HF_TOKEN or None,
        )
    except Exception:
        ds = load_dataset(
            "parquet",
            data_files={"full": parquet_glob},
            split="full",
            streaming=True,
            token=HF_TOKEN or None,
        )

    count = 0
    for row in ds:
        if limit and count >= limit:
            break
        title = _str(row.get("title"), 300)
        if not title:
            continue

        main_category = _str(row.get("main_category"), 100)
        categories_list = row.get("categories") or []
        categories = " | ".join(str(c) for c in categories_list[:5])
        features = _list_str(row.get("features"), max_items=5)
        store = _str(row.get("store"), 200)
        details = _amz_details_dict(row.get("details"))
        parent_asin = _str(row.get("parent_asin"), 20)

        embed_text = " | ".join(p for p in [title, main_category, store, features[:200]] if p)

        # characteristics: details dict + features list
        characteristics: dict[str, Any] = {}
        if store:
            characteristics["Бренд"] = [store]
        if main_category:
            characteristics["Категория"] = [main_category]
        features_list = row.get("features") or []
        if features_list:
            characteristics["Особенности"] = [str(f)[:100] for f in features_list[:5]]
        for k, v in details.items():
            if v and str(v).strip():
                characteristics[k] = [str(v)[:100]]

        point_id = _stable_uuid("amazon", parent_asin) if parent_asin else _hash_uuid(embed_text)

        payload = {
            "name": title,
            "categories": categories or main_category,
            "characteristics": _encode_characteristics(characteristics),
            "source": "amazon",
        }

        yield point_id, embed_text, payload
        count += 1


# ── Amazon Berkeley Objects (ABO) adapter ────────────────────────────────────

def _abo_multilang_value(field: Any, prefer_langs: tuple = ("en_US", "en_GB", "en")) -> str:
    """Extract best value from list[{language_tag?, value}]."""
    if not isinstance(field, list) or not field:
        return _str(field)
    # Try preferred languages first
    for lang in prefer_langs:
        for item in field:
            if isinstance(item, dict) and item.get("language_tag", "").startswith(lang[:2]):
                v = item.get("value", "")
                if v:
                    return str(v).strip()[:200]
    # Fallback: first with a value
    for item in field:
        if isinstance(item, dict):
            v = item.get("value", "")
            if v:
                return str(v).strip()[:200]
    return ""


def stream_abo(limit: int) -> Iterator[tuple[str, str, dict]]:
    """Yield (point_id, embed_text, payload) from ABO S3 shards."""
    count = 0
    for shard_url in ABO_SHARDS:
        if limit and count >= limit:
            break
        print(f"[ABO] Streaming shard: {shard_url} ...")
        try:
            for row in _stream_jsonl_gz(shard_url):
                if limit and count >= limit:
                    break
                name = _abo_multilang_value(row.get("item_name"))
                if not name:
                    continue

                brand = _abo_multilang_value(row.get("brand"))
                color = _abo_multilang_value(row.get("color"))
                style = _abo_multilang_value(row.get("style"))
                product_type_list = row.get("product_type") or []
                product_type = (product_type_list[0].get("value", "") if product_type_list else "")
                bullet = _abo_multilang_value(row.get("bullet_point"))
                item_id = _str(row.get("item_id"), 20)
                node_list = row.get("node") or []
                categories = " > ".join(
                    str(n.get("node_name", "")) for n in node_list[:4] if isinstance(n, dict)
                )

                embed_text = " | ".join(p for p in [name, brand, product_type, color] if p)

                characteristics: dict[str, Any] = {}
                if brand:
                    characteristics["Бренд"] = [brand]
                if color:
                    characteristics["Цвет"] = [color]
                if style:
                    characteristics["Стиль"] = [style]
                if product_type:
                    characteristics["Тип продукта"] = [product_type]
                if bullet:
                    characteristics["Описание"] = [bullet[:200]]

                point_id = _stable_uuid("abo", item_id) if item_id else _hash_uuid(embed_text)

                payload = {
                    "name": name,
                    "categories": categories,
                    "characteristics": _encode_characteristics(characteristics),
                    "source": "abo",
                }

                yield point_id, embed_text, payload
                count += 1
        except Exception as e:
            print(f"[ABO] Shard {shard_url} failed: {e} — skipping.")
            continue


# ── IKEA adapter ──────────────────────────────────────────────────────────────

def _ikea_list_field(field: Any, maxlen: int = 400) -> str:
    """IKEA materials/care_instructions/category_tree are list[str]."""
    if isinstance(field, list):
        return " | ".join(str(x)[:100] for x in field if x)[:maxlen]
    return _str(field, maxlen)


def stream_ikea(limit: int) -> Iterator[tuple[str, str, dict]]:
    """Yield (point_id, embed_text, payload) from IKEA HF dataset.

    Confirmed fields (jeffreyszhou/ikea-us-products-2025):
      title (str), materials (list[str]), care_instructions (list[str]),
      category_tree (list[str]), style (str), price (str|float), product_id (str)
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        sys.exit("[Error] datasets not installed.")

    print("[IKEA] Loading jeffreyszhou/ikea-us-products-2025 streaming ...")
    ds = load_dataset(
        "jeffreyszhou/ikea-us-products-2025",
        split="train",
        streaming=True,
        token=HF_TOKEN or None,
    )

    count = 0
    for row in ds:
        if limit and count >= limit:
            break
        # Confirmed field: 'title' (not 'name')
        name = _str(row.get("title"), 300)
        if not name:
            continue

        # materials and care_instructions are list[str]; category_tree is list[str]
        materials_list: list[str] = row.get("materials") or []
        care_list: list[str] = row.get("care_instructions") or []
        category_tree_list: list[str] = row.get("category_tree") or []

        materials = _ikea_list_field(materials_list)
        care = _ikea_list_field(care_list)
        # category_tree[:-1] drops the leaf node (product name repeated there)
        cat_path = category_tree_list[:-1] if len(category_tree_list) > 1 else category_tree_list
        categories = " > ".join(cat_path)

        style = _str(row.get("style"), 200)
        price = _str(row.get("price"), 50)
        product_id = _str(row.get("product_id"), 20)

        embed_text = " | ".join(p for p in [name, categories[:200], materials[:100]] if p)

        characteristics: dict[str, Any] = {}
        if materials_list:
            characteristics["Материал"] = [m for m in materials_list if m][:5]
        if care_list:
            characteristics["Уход"] = [c for c in care_list if c][:5]
        if style:
            characteristics["Стиль"] = [style]
        if price:
            characteristics["Цена"] = [price]

        point_id = _stable_uuid("ikea", product_id) if product_id else _hash_uuid(embed_text)

        payload = {
            "name": name,
            "categories": categories,
            "characteristics": _encode_characteristics(characteristics),
            "source": "ikea",
        }

        yield point_id, embed_text, payload
        count += 1


# ── Rebrickable adapter ───────────────────────────────────────────────────────

def stream_rebrickable(limit: int) -> Iterator[tuple[str, str, dict]]:
    """Yield (point_id, embed_text, payload) from Rebrickable sets.csv.gz."""
    print(f"[Rebrickable] Downloading sets.csv.gz from {REBRICKABLE_URL} ...")
    try:
        req = urllib.request.Request(
            REBRICKABLE_URL, headers={"User-Agent": "eg-ingest/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = gzip.decompress(resp.read())
        reader = csv.DictReader(io.StringIO(data.decode("utf-8", errors="replace")))
    except Exception as e:
        print(f"[Rebrickable] Download failed: {e}")
        return

    count = 0
    for row in reader:
        if limit and count >= limit:
            break
        name = _str(row.get("name"), 300)
        if not name:
            continue

        set_num = _str(row.get("set_num"), 30)
        year = _str(row.get("year"), 10)
        theme_id = _str(row.get("theme_id"), 20)
        num_parts = _str(row.get("num_parts"), 20)

        embed_text = f"LEGO {name} {year}"
        categories = f"LEGO | Конструкторы | {year}"

        characteristics: dict[str, Any] = {
            "Год": [year],
            "Деталей": [num_parts],
            "Тема": [theme_id],
            "Артикул": [set_num],
        }

        point_id = _stable_uuid("rebrickable", set_num) if set_num else _hash_uuid(embed_text)

        payload = {
            "name": name,
            "categories": categories,
            "characteristics": _encode_characteristics(characteristics),
            "source": "rebrickable",
        }

        yield point_id, embed_text, payload
        count += 1


# ── GSMArena adapter ──────────────────────────────────────────────────────────

def stream_gsmarena(limit: int) -> Iterator[tuple[str, str, dict]]:
    """Yield from GSMArena Kaggle dataset — skips gracefully if creds absent."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    kaggle_user = os.environ.get("KAGGLE_USERNAME", "")
    kaggle_key = os.environ.get("KAGGLE_KEY", "")
    if not kaggle_json.exists() and not (kaggle_user and kaggle_key):
        print(
            "[GSMArena] SKIP: Kaggle credentials not found.\n"
            "  Provide ~/.kaggle/kaggle.json or set KAGGLE_USERNAME + KAGGLE_KEY env vars.\n"
            "  Dataset: https://www.kaggle.com/datasets/arwinneil/gsmarena-phone-dataset"
        )
        return
    try:
        import kaggle  # type: ignore
    except ImportError:
        print("[GSMArena] SKIP: 'kaggle' package not installed. Run: pip install kaggle")
        return

    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "arwinneil/gsmarena-phone-dataset", path=tmpdir, unzip=True
            )
            csv_files = list(Path(tmpdir).glob("*.csv"))
            if not csv_files:
                print("[GSMArena] No CSV files found after download.")
                return
            csv_path = csv_files[0]
            print(f"[GSMArena] Reading {csv_path.name} ...")
            with open(csv_path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    if limit and count >= limit:
                        break
                    name = _str(row.get("Model") or row.get("name"), 300)
                    if not name:
                        continue
                    brand = _str(row.get("Brand") or row.get("brand"), 100)
                    oem = _str(row.get("OEM") or row.get("oem"), 100)

                    embed_text = " | ".join(p for p in [name, brand] if p)
                    categories = "Смартфоны | Мобильные телефоны"

                    characteristics: dict[str, Any] = {}
                    if brand:
                        characteristics["Бренд"] = [brand]
                    # Include all non-empty spec columns
                    for k, v in row.items():
                        if v and k not in ("Model", "Brand", "name", "brand") and _str(v, 1):
                            characteristics[k] = [_str(v, 100)]

                    model_id = _str(row.get("Model") or row.get("name"), 100)
                    point_id = _stable_uuid("gsmarena", f"{brand}:{model_id}") if model_id else _hash_uuid(embed_text)

                    payload = {
                        "name": name,
                        "categories": categories,
                        "characteristics": _encode_characteristics(characteristics),
                        "source": "gsmarena",
                    }

                    yield point_id, embed_text, payload
                    count += 1
    except Exception as e:
        print(f"[GSMArena] Error: {e}")


# ── Parquet export (pod GPU path) ─────────────────────────────────────────────

def run_parquet_export(
    source_stream: Iterator[tuple[str, str, dict]],
    out_dir: str,
    encode_batch_size: int,
    shard_size: int,
    limit: int,
    print_sample: int = 1,
) -> None:
    """Embed + write to parquet shards; no Qdrant required.

    Parquet schema
    --------------
    id            : str   — stable UUID5 / SHA-1 hash
    vector        : list[float]  length 384
    name          : str
    categories    : str
    characteristics : str  — JSON
    source        : str
    """
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        sys.exit("[Error] pyarrow not installed — needed for --out-parquet mode.")

    # Pre-load model now (uses cuda if available)
    get_model()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("vector", pa.list_(pa.float32())),
        pa.field("name", pa.string()),
        pa.field("categories", pa.string()),
        pa.field("characteristics", pa.string()),
        pa.field("source", pa.string()),
    ])

    shard_idx = 0
    total_written = 0
    total_skipped = 0
    printed = 0

    # Buffers for the current batch
    id_buf: list[str] = []
    text_buf: list[str] = []
    payload_buf: list[dict] = []

    # Rows for the current shard
    shard_rows: list[dict] = []

    t0 = time.time()

    def flush_embed_to_shard() -> None:
        nonlocal total_written
        if not text_buf:
            return
        vecs = embed_batch(text_buf)
        for pid, vec, pl in zip(id_buf, vecs, payload_buf):
            shard_rows.append({
                "id": pid,
                "vector": vec,
                "name": pl["name"],
                "categories": pl["categories"],
                "characteristics": pl["characteristics"],
                "source": pl["source"],
            })
            total_written += 1
        id_buf.clear()
        text_buf.clear()
        payload_buf.clear()

    def flush_shard() -> None:
        nonlocal shard_idx
        if not shard_rows:
            return
        shard_file = out_path / f"file_{shard_idx:04d}.parquet"
        table = pa.table(
            {
                "id":              [r["id"] for r in shard_rows],
                "vector":          [r["vector"] for r in shard_rows],
                "name":            [r["name"] for r in shard_rows],
                "categories":      [r["categories"] for r in shard_rows],
                "characteristics": [r["characteristics"] for r in shard_rows],
                "source":          [r["source"] for r in shard_rows],
            },
            schema=schema,
        )
        pq.write_table(table, shard_file, compression="snappy")
        elapsed = time.time() - t0
        speed = total_written / elapsed if elapsed > 0 else 0
        print(
            f"[Parquet] Shard {shard_idx:04d} → {shard_file.name} "
            f"({len(shard_rows):,} rows | {total_written:,} total | {speed:.0f} rows/s)"
        )
        shard_rows.clear()
        shard_idx += 1

    for point_id, embed_text, payload in source_stream:
        if limit and total_written + len(text_buf) >= limit:
            break

        if not embed_text.strip():
            total_skipped += 1
            continue

        if printed < print_sample:
            print(f"\n[Sample point #{printed + 1}]")
            print(f"  id         : {point_id}")
            print(f"  embed_text : {embed_text[:120]}")
            print(f"  name       : {payload.get('name', '')[:80]}")
            print(f"  source     : {payload.get('source', '')}")
            printed += 1

        id_buf.append(point_id)
        text_buf.append(embed_text)
        payload_buf.append(payload)

        if len(text_buf) >= encode_batch_size:
            flush_embed_to_shard()
            rows_done = total_written + total_skipped
            if rows_done % 10_000 == 0 and rows_done > 0:
                elapsed = time.time() - t0
                speed = total_written / elapsed if elapsed > 0 else 0
                print(f"[Parquet] {total_written:,} rows embedded | {speed:.0f} rows/s | {elapsed:.0f}s")

        if len(shard_rows) >= shard_size:
            flush_shard()

    # Flush remaining buffer and final shard
    flush_embed_to_shard()
    flush_shard()

    elapsed = time.time() - t0
    speed = total_written / elapsed if elapsed > 0 else 0
    print(f"\n[Parquet] === Done ===")
    print(f"  Written  : {total_written:,} rows in {shard_idx} shard(s)")
    print(f"  Skipped  : {total_skipped:,}")
    print(f"  Time     : {elapsed:.1f}s  ({speed:.0f} rows/s)")
    print(f"  Out dir  : {out_path}")
    for f in sorted(out_path.glob("file_*.parquet")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name}  {size_mb:.1f} MB")


# ── Core ingest loop ──────────────────────────────────────────────────────────

def run_ingest(
    source_stream: Iterator[tuple[str, str, dict]],
    collection: str,
    qdrant_url: str,
    limit: int,
    encode_batch_size: int,
    upsert_batch_size: int,
    recreate: bool,
    no_text_index: bool,
    print_sample: int = 1,
) -> None:
    client = get_client(qdrant_url)
    ensure_collection(client, collection, recreate)

    from qdrant_client.models import PointStruct  # type: ignore

    t0 = time.time()
    total_indexed = 0
    total_skipped = 0
    printed = 0

    id_buf: list[str] = []
    text_buf: list[str] = []
    payload_buf: list[dict] = []

    def flush():
        nonlocal total_indexed
        if not text_buf:
            return
        vecs = embed_batch(text_buf)
        points = [
            PointStruct(id=pid, vector=vec, payload=pl)
            for pid, vec, pl in zip(id_buf, vecs, payload_buf)
        ]
        for i in range(0, len(points), upsert_batch_size):
            client.upsert(collection_name=collection, points=points[i:i + upsert_batch_size])
        total_indexed += len(points)
        id_buf.clear()
        text_buf.clear()
        payload_buf.clear()

    for point_id, embed_text, payload in source_stream:
        if limit and (total_indexed + len(text_buf)) >= limit:
            break

        if not embed_text.strip():
            total_skipped += 1
            continue

        # Print sample for quality inspection
        if printed < print_sample:
            print(f"\n[Sample point #{printed + 1}]")
            print(f"  id         : {point_id}")
            print(f"  embed_text : {embed_text[:120]}")
            print(f"  name       : {payload.get('name', '')[:80]}")
            print(f"  categories : {payload.get('categories', '')[:80]}")
            chars_raw = payload.get("characteristics", "{}")
            try:
                chars = json.loads(chars_raw)
            except Exception:
                chars = chars_raw
            print(f"  characteristics: {json.dumps(chars, ensure_ascii=False)[:300]}")
            print(f"  source     : {payload.get('source', '')}")
            printed += 1

        id_buf.append(point_id)
        text_buf.append(embed_text)
        payload_buf.append(payload)

        if len(text_buf) >= encode_batch_size:
            flush()
            rows_done = total_indexed + total_skipped
            if rows_done % 1000 == 0 and rows_done > 0:
                elapsed = time.time() - t0
                speed = total_indexed / elapsed if elapsed > 0 else 0
                print(f"[Ingest] {total_indexed:,} indexed | {speed:.0f} rows/s | {elapsed:.0f}s")

    flush()

    elapsed = time.time() - t0
    info = client.get_collection(collection)
    print(f"\n[Ingest] Done!")
    print(f"  Indexed   : {total_indexed:,}")
    print(f"  Skipped   : {total_skipped:,}")
    print(f"  Total pts : {info.points_count or 0:,}")
    print(f"  Time      : {elapsed:.1f}s")
    if elapsed > 0 and total_indexed > 0:
        print(f"  Speed     : {total_indexed / elapsed:.0f} rows/s")

    if not no_text_index:
        create_text_index(client, collection, "categories")

    print("\n[Ingest] === Validation: query-back ===")
    _validate(client, collection)


def _validate(client, collection: str) -> None:
    sample_queries = [
        "chocolate hazelnut spread",
        "wireless earbuds bluetooth",
        "wooden chair furniture",
        "LEGO brick set",
        "cat food tuna",
    ]
    for q in sample_queries[:3]:
        print(f"[Validate] Query: '{q}'")
        try:
            vec = embed_batch([q])[0]
            result = client.query_points(
                collection_name=collection,
                query=vec,
                limit=3,
                with_payload=True,
            )
            for i, hit in enumerate(result.points):
                p = hit.payload or {}
                name = _str(p.get("name", ""), 70)
                source = p.get("source", "")
                print(f"  [{i}] score={hit.score:.3f} | {source} | {name}")
            print(f"  -> {len(result.points)} results")
        except Exception as e:
            print(f"  ERROR: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Delete-only mode
    if args.delete_collection:
        client = get_client(args.qdrant_url)
        delete_collection(client, args.delete_collection)
        return

    if not args.dataset:
        print("[Error] --dataset is required (unless --delete-collection is used).")
        sys.exit(1)

    collection = args.collection
    limit = 0 if args.full else args.sample
    # --limit overrides even --full; 0 means no cap
    if args.full and args.limit:
        limit = args.limit

    print("[Ingest] === Unified Dataset -> Qdrant ===")
    print(f"  dataset    : {args.dataset}")
    if args.dataset == "amazon":
        print(f"  category   : {args.amazon_category}")
    print(f"  collection : {collection}")
    print(f"  qdrant_url : {args.qdrant_url}")
    limit_label = "unlimited (--full)" if args.full and not args.limit else str(limit)
    print(f"  limit      : {limit_label}")

    # Validate-only mode: skip ingest, just query
    if args.validate_only:
        client = get_client(args.qdrant_url)
        _validate(client, collection)
        return

    dataset_map = {
        "off": lambda: stream_off(split="food", limit=limit),
        "obf": lambda: stream_off(split="beauty", limit=limit),
        "opff": lambda: stream_opff(limit=limit),
        "amazon": lambda: stream_amazon(category=args.amazon_category, limit=limit),
        "abo": lambda: stream_abo(limit=limit),
        "ikea": lambda: stream_ikea(limit=limit),
        "rebrickable": lambda: stream_rebrickable(limit=limit),
        "gsmarena": lambda: stream_gsmarena(limit=limit),
    }

    source = dataset_map[args.dataset]()

    if args.out_parquet:
        print(f"  mode       : parquet export → {args.out_parquet}")
        print(f"  shard_size : {args.parquet_shard_size:,}")
        run_parquet_export(
            source_stream=source,
            out_dir=args.out_parquet,
            encode_batch_size=args.encode_batch,
            shard_size=args.parquet_shard_size,
            limit=limit,
            print_sample=args.print_sample,
        )
    else:
        run_ingest(
            source_stream=source,
            collection=collection,
            qdrant_url=args.qdrant_url,
            limit=limit,
            encode_batch_size=args.encode_batch,
            upsert_batch_size=args.batch,
            recreate=args.recreate,
            no_text_index=args.no_text_index,
            print_sample=args.print_sample,
        )


if __name__ == "__main__":
    main()
