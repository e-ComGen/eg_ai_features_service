"""Generic product-dataset → Qdrant ingest scaffold.

Streams a SMALL sample (few hundred rows) from a HuggingFace dataset,
builds an embed text, embeds via paraphrase-multilingual-MiniLM-L12-v2
(same model as ozon_products), and upserts into a named Qdrant collection.

SUPPORTED DATASETS
------------------

1. Open Food Facts  (ODbL)
   HF id   : openfoodfacts/product-database
   Config  : (no config arg; splits are 'food' and 'beauty')
   Split   : food  (beauty is cosmetics)
   Schema  : 111 fields; key fields for embedding:
     product_name  – list[{lang, text}]; first EN or 'main' entry
     brands        – str
     categories    – str  (EN taxonomy, comma-separated)
     ingredients_text – list[{lang, text}]
     quantity      – str  (e.g. "350 g")
     packaging     – str
     nutriments    – list[{name, 100g, unit}]
     code          – EAN barcode (used as point ID)
   Scale   : ~3.9M food products + ~900K beauty (Parquet, ~6 GB total)
   Full-run plan:
     ~4.8M rows × ~0.005s/row CPU embed = ~6.7 hours CPU
     GPU (T4, RunPod): ~40 min
     Qdrant storage: ~384×4×4.8M ≈ 7.4 GB vectors + ~3 GB payload
     Disk total: ~11 GB

2. Amazon Reviews 2023 – product metadata  (non-commercial research)
   HF id   : McAuley-Lab/Amazon-Reviews-2023
   NOTE    : The dataset's loading script is broken in datasets>=3.x
             (RuntimeError: Dataset scripts are no longer supported).
             Load directly from parquet via data_files= (implemented here).
   Parquet : raw_meta_<Category>/full-*.parquet  (per-category splits)
   Schema  : 16 fields:
     title          – str  (product title, EN)
     main_category  – str
     categories     – list[str]
     description    – list[str]
     features       – list[str]
     details        – dict  {"Package Dimensions": "...", "UPC": "...",
                             "Item model number": "...", "Color": "...",
                             "Material": "..."}  — key attribute source
     store          – str  (brand/seller)
     price          – str  (may be None)
     parent_asin    – str  (used as point ID)
   Scale   : ~48M products across 34 categories (~4 GB parquet total)
   Full-run plan (all categories):
     ~48M rows × ~0.005s/row CPU embed = ~67 hours CPU
     Better: run per-category (e.g. Electronics = ~3M rows ≈ 4 hours)
     GPU (T4): ~30 min per large category
     Qdrant storage: ~384×4×48M ≈ 74 GB vectors + ~20 GB payload
     Realistic scope: pick 5–10 most relevant categories ≈ 10–15M rows

USAGE
-----
# Open Food Facts, sample 300 rows:
python scripts/ingest_dataset_to_qdrant.py --dataset off --sample-size 300

# Amazon Reviews (All_Beauty category), sample 300:
python scripts/ingest_dataset_to_qdrant.py \\
    --dataset amazon --amazon-category All_Beauty --sample-size 300

# Full OFT ingest (overnight):
python scripts/ingest_dataset_to_qdrant.py --dataset off --full

# Full Amazon Electronics ingest:
python scripts/ingest_dataset_to_qdrant.py \\
    --dataset amazon --amazon-category Electronics --full

REQUIREMENTS
------------
    pip install datasets sentence-transformers qdrant-client python-dotenv
    # HF_TOKEN in .env (faster downloads, gated datasets)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

# Windows DLL ordering fix.
try:
    import pyarrow  # noqa: F401
    import pandas   # noqa: F401
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

# ── Constants ─────────────────────────────────────────────────────────────────

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DIM = 384
QDRANT_URL_DEFAULT = os.environ.get("QDRANT_URL", "http://localhost:6333")

DEFAULT_COLLECTION_OFT = "open_food_facts"
DEFAULT_COLLECTION_AMZ = "amazon_products"
DEFAULT_SAMPLE_SIZE = 300

# Full list of Amazon raw_meta categories (subset most relevant to product attributes):
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


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generic product dataset → Qdrant ingest scaffold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset", choices=["off", "amazon"], required=True,
        help="'off' = Open Food Facts, 'amazon' = Amazon Reviews 2023 metadata",
    )
    p.add_argument(
        "--amazon-category", default="All_Beauty",
        choices=AMAZON_CATEGORIES,
        metavar="CATEGORY",
        help=(f"Amazon category to ingest. Default: All_Beauty. "
              f"Choices: {', '.join(AMAZON_CATEGORIES)}"),
    )
    p.add_argument(
        "--collection", metavar="NAME",
        help="Qdrant collection name override. Default: open_food_facts or amazon_products",
    )
    p.add_argument(
        "--qdrant-url", default=QDRANT_URL_DEFAULT,
        help=f"Qdrant server URL. Default: {QDRANT_URL_DEFAULT}",
    )
    p.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Rows to ingest in sample mode (default: {DEFAULT_SAMPLE_SIZE}). Ignored with --full.",
    )
    p.add_argument(
        "--full", action="store_true",
        help="Full ingest (no row limit). OVERNIGHT JOB — see module docstring for scale.",
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
        help="Skip creating payload text index on the primary text field.",
    )
    p.add_argument(
        "--off-split", choices=["food", "beauty"], default="food",
        help="Open Food Facts split. Default: food",
    )
    return p.parse_args()


# ── Embedding ─────────────────────────────────────────────────────────────────

_model_cache = None


def get_model():
    global _model_cache
    if _model_cache is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            sys.exit("[Error] sentence-transformers not installed.")
        print(f"[Ingest] Loading embed model: {EMBED_MODEL_NAME} ...")
        _model_cache = SentenceTransformer(EMBED_MODEL_NAME)
        dim = getattr(_model_cache, "get_embedding_dimension",
                      _model_cache.get_sentence_embedding_dimension)()
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
            print(f"[Ingest] '{name}' exists ({info.points_count or 0} pts). Resume.")
            return
    print(f"[Ingest] Creating '{name}' (dim={VECTOR_DIM}, COSINE) ...")
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )


def create_text_index(client, collection: str, field: str) -> None:
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


def _stable_uuid(namespace: str, key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}:{key}"))


def _hash_uuid(value: str) -> str:
    """Generate a UUID from any string via SHA-1 (fallback for missing IDs)."""
    return str(uuid.UUID(hashlib.sha1(value.encode()).hexdigest()[:32]))


# ── Open Food Facts adapter ───────────────────────────────────────────────────

def _oft_product_name(row: dict) -> str:
    """Extract best product name: prefer English, then 'main', then any."""
    names = row.get("product_name") or []
    if not isinstance(names, list):
        return str(names)[:300]
    for lang_pref in ("en", "main"):
        for item in names:
            if isinstance(item, dict) and item.get("lang") == lang_pref:
                text = item.get("text", "")
                if text and text.strip():
                    return text.strip()[:300]
    # Any non-empty
    for item in names:
        if isinstance(item, dict):
            text = item.get("text", "")
            if text and text.strip():
                return text.strip()[:300]
    return ""


def _oft_text_list(field: Any) -> str:
    """Flatten a list[{lang, text}] or list[str] into a plain string."""
    if not field:
        return ""
    if isinstance(field, str):
        return field[:300]
    if isinstance(field, list):
        texts = []
        for item in field:
            if isinstance(item, dict):
                t = item.get("text", "")
                if t:
                    texts.append(str(t)[:200])
            elif item:
                texts.append(str(item)[:200])
        return " | ".join(texts)[:500]
    return str(field)[:300]


def _oft_nutriments_str(nutriments: Any) -> str:
    if not isinstance(nutriments, list):
        return ""
    parts = []
    for n in nutriments[:8]:
        if isinstance(n, dict):
            name = n.get("name", "")
            val = n.get("100g")
            unit = n.get("unit", "")
            if name and val is not None:
                parts.append(f"{name}:{val}{unit}")
    return " ".join(parts)


def stream_off(split: str, limit: int) -> Iterator[tuple[str, str, dict]]:
    """Yield (point_id, embed_text, payload) for Open Food Facts rows."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        sys.exit("[Error] datasets not installed.")

    print(f"[OFT] Loading openfoodfacts/product-database split={split} streaming ...")
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
        name = _oft_product_name(row)
        if not name:
            continue

        brands = str(row.get("brands") or "")
        categories = str(row.get("categories") or "")
        ingredients = _oft_text_list(row.get("ingredients_text"))
        quantity = str(row.get("quantity") or "")
        packaging = str(row.get("packaging") or "")
        nutriments = _oft_nutriments_str(row.get("nutriments"))

        embed_text = " | ".join(p for p in [
            name, brands, categories[:200], ingredients[:200], quantity, packaging
        ] if p)

        code = str(row.get("code") or "")
        point_id = _stable_uuid("off", code) if code else _hash_uuid(embed_text)

        payload = {
            "source": "open_food_facts",
            "product_name": name,
            "brands": brands[:200],
            "categories": categories[:300],
            "ingredients_text": ingredients[:400],
            "quantity": quantity,
            "packaging": packaging[:200],
            "nutriments_summary": nutriments,
            "code": code,
            "countries": str(row.get("countries_tags") or "")[:200],
            "labels": str(row.get("labels") or "")[:200],
        }

        yield point_id, embed_text, payload
        count += 1


# ── Amazon Reviews 2023 adapter ───────────────────────────────────────────────

def _amz_details_str(details: Any) -> str:
    """Convert the details dict/str into a compact attribute string."""
    if not details:
        return ""
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except Exception:
            return details[:300]
    if isinstance(details, dict):
        parts = []
        for k, v in details.items():
            if v and str(v).strip():
                parts.append(f"{k}: {str(v)[:80]}")
        return " | ".join(parts[:15])
    return str(details)[:300]


def _amz_list_str(field: Any, max_items: int = 5) -> str:
    if not field:
        return ""
    if isinstance(field, list):
        return " | ".join(str(x)[:100] for x in field[:max_items] if x)
    return str(field)[:300]


def stream_amazon(category: str, limit: int) -> Iterator[tuple[str, str, dict]]:
    """Yield (point_id, embed_text, payload) for Amazon Reviews 2023 metadata."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        sys.exit("[Error] datasets not installed.")

    # The dataset's loading script fails in datasets>=3.x; load parquet directly.
    parquet_url = (
        f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/"
        f"raw_meta_{category}/full-00000-of-*"
    )
    # For streaming we load the first shard explicitly (avoids glob resolution latency).
    parquet_shard_0 = (
        f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/"
        f"raw_meta_{category}/full-00000-of-00001.parquet"
    )
    # Try single-shard first; some categories have multiple shards.
    print(f"[Amazon] Loading raw_meta_{category} via parquet streaming ...")
    try:
        ds = load_dataset(
            "parquet",
            data_files={"full": parquet_shard_0},
            split="full",
            streaming=True,
            token=HF_TOKEN or None,
        )
    except Exception:
        # Fallback: glob pattern for multi-shard categories
        parquet_glob = (
            f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/"
            f"raw_meta_{category}/full-*.parquet"
        )
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

        title = str(row.get("title") or "").strip()
        if not title:
            continue

        main_category = str(row.get("main_category") or "")
        categories = _amz_list_str(row.get("categories"), max_items=4)
        description = _amz_list_str(row.get("description"), max_items=2)
        features = _amz_list_str(row.get("features"), max_items=3)
        details_str = _amz_details_str(row.get("details"))
        store = str(row.get("store") or "")
        price = str(row.get("price") or "")

        embed_text = " | ".join(p for p in [
            title, main_category, store, categories, features[:200], details_str[:200]
        ] if p)

        parent_asin = str(row.get("parent_asin") or "")
        point_id = _stable_uuid("amazon", parent_asin) if parent_asin else _hash_uuid(embed_text)

        payload = {
            "source": "amazon_2023",
            "category": main_category,
            "title": title[:300],
            "store": store[:200],
            "price": price,
            "categories": categories[:300],
            "description": description[:400],
            "features": features[:400],
            "details": details_str[:500],
            "parent_asin": parent_asin,
            "average_rating": row.get("average_rating"),
            "rating_number": row.get("rating_number"),
        }

        yield point_id, embed_text, payload
        count += 1


# ── Core ingest loop ──────────────────────────────────────────────────────────

def run_ingest(
    source_stream: Iterator[tuple[str, str, dict]],
    collection: str,
    qdrant_url: str,
    limit: int,
    encode_batch_size: int,
    upsert_batch_size: int,
    recreate: bool,
    text_index_field: str,
    no_text_index: bool,
) -> None:
    client = get_client(qdrant_url)
    ensure_collection(client, collection, recreate)

    from qdrant_client.models import PointStruct  # type: ignore

    t0 = time.time()
    total_indexed = 0
    total_skipped = 0

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
        # Check limit BEFORE accumulating (avoids over-shoot by encode_batch)
        if limit and (total_indexed + len(text_buf) + total_skipped) >= limit:
            break

        if not embed_text.strip():
            total_skipped += 1
            continue

        id_buf.append(point_id)
        text_buf.append(embed_text)
        payload_buf.append(payload)

        if len(text_buf) >= encode_batch_size:
            flush()
            rows_done = total_indexed + total_skipped
            if rows_done % 500 == 0 and rows_done > 0:
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
    if elapsed > 0:
        print(f"  Speed     : {total_indexed / elapsed:.0f} rows/s")

    if not no_text_index:
        create_text_index(client, collection, text_index_field)

    print("\n[Ingest] === Validation: query-back ===")
    _validate(client, collection)


def _validate(client, collection: str) -> None:
    sample_queries = [
        "chocolate spread hazelnut",
        "wireless earbuds bluetooth",
        "leather conditioner",
    ]
    for q in sample_queries[:2]:
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
                name = (p.get("product_name") or p.get("title") or "")[:70]
                brand = (p.get("brands") or p.get("store") or "")[:30]
                print(f"  [{i}] score={hit.score:.3f} | {name} | {brand}")
            print(f"  -> {len(result.points)} results")
        except Exception as e:
            print(f"  ERROR: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    collection = args.collection or (
        DEFAULT_COLLECTION_OFT if args.dataset == "off" else DEFAULT_COLLECTION_AMZ
    )
    limit = 0 if args.full else args.sample_size

    print("[Ingest] === Dataset -> Qdrant Scaffold ===")
    print(f"  dataset    : {args.dataset}")
    if args.dataset == "amazon":
        print(f"  category   : {args.amazon_category}")
    print(f"  collection : {collection}")
    print(f"  qdrant_url : {args.qdrant_url}")
    print(f"  limit      : {'unlimited (--full)' if args.full else limit}")

    if args.dataset == "off":
        source = stream_off(split=args.off_split, limit=limit)
        text_field = "product_name"
    else:  # amazon
        source = stream_amazon(category=args.amazon_category, limit=limit)
        text_field = "title"

    run_ingest(
        source_stream=source,
        collection=collection,
        qdrant_url=args.qdrant_url,
        limit=limit,
        encode_batch_size=args.encode_batch,
        upsert_batch_size=args.batch,
        recreate=args.recreate,
        text_index_field=text_field,
        no_text_index=args.no_text_index,
    )


if __name__ == "__main__":
    main()
