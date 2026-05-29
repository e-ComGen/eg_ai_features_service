"""Скрипт индексации датасета evgmaslov/ozon_ecup в локальный Qdrant-индекс.

Создаёт file-based Qdrant коллекцию для CompetitorRagSource:
- Vector: 384-dim эмбеддинг поля `name` через sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Payload: name, description[:500], categories, characteristics, variantid

Использование:
    python scripts/build_ozon_rag_index.py [--limit 200000] [--batch 1000]

По умолчанию --limit 100000 (первые 100K строк). Для полного датасета: --limit 0.

Требования:
    pip install qdrant-client datasets sentence-transformers python-dotenv

HF_TOKEN читается из .env файла в корне проекта.
"""
from __future__ import annotations

import sys
# Отключить буферизацию stdout для корректного вывода в лог-файл
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Windows DLL fix: pyarrow/pandas должны загружаться ДО torch/sentence_transformers,
# иначе происходит access violation при загрузке pyarrow DLL после torch.
try:
    import pyarrow  # noqa: F401
    import pandas   # noqa: F401
except ImportError:
    pass

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Загрузить .env до импортов которые могут читать переменные среды
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN", "")

# Путь к Qdrant-индексу
INDEX_PATH = str(
    PROJECT_ROOT
    / "app" / "services" / "enrichment" / "strategies" / "dictionaries" / "data"
    / "ozon_rag.qdrant"
)
COLLECTION_NAME = "ozon_products"
VECTOR_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2 output dim

# Датасет — полный train-сплит (2.25M продуктов), конфиг default
DATASET_NAME = "evgmaslov/ozon_ecup"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"

# Модель для эмбеддингов — та же что и в CompetitorRagSource
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Размер батча — 256 для GPU A4000/A5000 (16GB+ VRAM есть запас)
ENCODE_BATCH_SIZE = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Ozon RAG index from HuggingFace dataset")
    p.add_argument("--limit", type=int, default=3_000_000,
                   help="Max rows to index (0 = no limit). По умолчанию 3M — перекрывает весь train-сплит 2.25M.")
    p.add_argument("--batch", type=int, default=1000,
                   help="Qdrant batch insert size")
    p.add_argument("--encode-batch", type=int, default=ENCODE_BATCH_SIZE,
                   help="sentence-transformers encode batch size")
    p.add_argument("--index-path", default=INDEX_PATH,
                   help="Path to Qdrant file-based storage")
    p.add_argument("--recreate", action="store_true",
                   help="Delete existing collection and recreate")
    return p.parse_args()


def parse_json_field(raw) -> dict | list | None:
    """Распарсить JSON поле из датасета (может быть str, dict, list или None)."""
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def safe_str(val, max_len: int = 500) -> str:
    """Безопасно преобразовать значение в строку с ограничением длины."""
    if val is None:
        return ""
    s = str(val)
    return s[:max_len] if len(s) > max_len else s


def build_payload(row: dict) -> dict:
    """Построить payload для Qdrant из строки датасета."""
    return {
        "variantid": row.get("variantid"),
        "name": safe_str(row.get("name"), 300),
        "description": safe_str(row.get("description"), 500),
        "categories": parse_json_field(row.get("categories")),
        "characteristics": parse_json_field(row.get("characteristic_attributes_mapping")),
    }


def load_model(model_name: str):
    """Lazy-load sentence-transformers модель один раз."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        print("[Error] sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)
    print(f"[Index] Loading embed model: {model_name} ...")
    model = SentenceTransformer(model_name)
    # get_embedding_dimension — новый API; get_sentence_embedding_dimension устарел в ST 5.x
    dim = getattr(model, 'get_embedding_dimension', model.get_sentence_embedding_dimension)()
    print(f"[Index] Model loaded (output dim={dim})")
    return model


def main() -> None:
    args = parse_args()

    print(f"[Index] Project root: {PROJECT_ROOT}")
    print(f"[Index] Index path: {args.index_path}")
    print(f"[Index] Limit: {args.limit or 'unlimited'}")
    print(f"[Index] Qdrant batch size: {args.batch}")
    print(f"[Index] Encode batch size: {args.encode_batch}")
    print(f"[Index] Embed model: {EMBED_MODEL_NAME}")
    print(f"[Index] Vector dim: {VECTOR_DIM}")

    # Инициализация Qdrant клиента
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct, OptimizersConfigDiff, HnswConfigDiff
    except ImportError:
        print("[Error] qdrant-client not installed. Run: pip install qdrant-client")
        sys.exit(1)

    os.makedirs(args.index_path, exist_ok=True)
    client = QdrantClient(path=args.index_path)

    # Проверить / создать коллекцию
    collections = [c.name for c in client.get_collections().collections]
    existing_points = 0

    if COLLECTION_NAME in collections:
        if args.recreate:
            print(f"[Index] Recreating collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)
        else:
            info = client.get_collection(COLLECTION_NAME)
            existing_points = info.points_count or 0
            print(f"[Index] Collection '{COLLECTION_NAME}' exists with {existing_points} points.")
            print(f"[Index] Resume mode: варианты уже в индексе будут пропущены.")

    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        print(f"[Index] Creating collection '{COLLECTION_NAME}' (dim={VECTOR_DIM})...")
        # HNSW + indexing disabled during bulk upsert — single rebuild at end (10× faster)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=2_000_000_000),
            hnsw_config=HnswConfigDiff(m=0),
        )
        print(f"[Index] HNSW + indexer threshold disabled during bulk (will rebuild at end)")

    # Получить множество уже проиндексированных variantid для resume
    existing_ids: set[int] = set()
    if existing_points > 0:
        print(f"[Index] Fetching existing IDs for resume (this may take a moment)...")
        offset = None
        while True:
            result, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=10_000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for p in result:
                existing_ids.add(p.id)
            if next_offset is None:
                break
            offset = next_offset
        print(f"[Index] Resume: {len(existing_ids)} existing point IDs loaded.")

    # Lazy-load модели
    model = load_model(EMBED_MODEL_NAME)

    # Загрузка датасета
    print(f"[Index] Loading dataset {DATASET_NAME} config={DATASET_CONFIG} split={DATASET_SPLIT}...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("[Error] datasets not installed. Run: pip install datasets")
        sys.exit(1)

    t0 = time.time()
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,             # конфиг "default" — полный train с 2.25M строк
        split=DATASET_SPLIT,
        streaming=True,  # streaming чтобы не качать весь датасет в память
        token=HF_TOKEN or None,
    )

    total_indexed = 0
    total_skipped = 0
    total_resumed = 0

    # Накопитель для батч-кодирования
    encode_buffer_names: list[str] = []
    encode_buffer_rows: list[dict] = []

    print(f"[Index] Streaming rows (limit={args.limit or 'unlimited'})...")

    def flush_encode_buffer():
        """Закодировать и залить накопленный буфер в Qdrant."""
        nonlocal total_indexed
        if not encode_buffer_names:
            return

        # Batch encode
        vecs = model.encode(
            encode_buffer_names,
            normalize_embeddings=True,
            batch_size=len(encode_buffer_names),
            show_progress_bar=False,
        )

        # Формируем qdrant points
        qdrant_batch: list = []
        from qdrant_client.models import PointStruct
        for row, vec in zip(encode_buffer_rows, vecs):
            try:
                point_id = int(row["variantid"])
            except (TypeError, ValueError):
                continue
            payload = build_payload(row)
            qdrant_batch.append(PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload=payload,
            ))

        if qdrant_batch:
            client.upsert(collection_name=COLLECTION_NAME, points=qdrant_batch)
            total_indexed += len(qdrant_batch)

        encode_buffer_names.clear()
        encode_buffer_rows.clear()

    for row in dataset:
        # Проверить лимит (считаем indexed + resumed + skipped)
        if args.limit and (total_indexed + total_resumed + total_skipped) >= args.limit:
            break

        variant_id = row.get("variantid")
        if variant_id is None:
            total_skipped += 1
            continue

        try:
            point_id = int(variant_id)
        except (TypeError, ValueError):
            total_skipped += 1
            continue

        name = row.get("name", "") or ""
        if not name.strip():
            total_skipped += 1
            continue

        # Resume: пропускаем уже проиндексированные
        if point_id in existing_ids:
            total_resumed += 1
            continue

        encode_buffer_names.append(name)
        encode_buffer_rows.append(row)

        # Когда накоплен полный батч — кодируем и заливаем
        if len(encode_buffer_names) >= args.encode_batch:
            flush_encode_buffer()

            # Периодический flush каждые 5K строк для защиты от падений
            rows_processed = total_indexed + total_skipped + total_resumed
            if rows_processed % 5_000 == 0 and rows_processed > 0:
                # Принудительный сброс остатка буфера раз в 5K строк
                flush_encode_buffer()

            if total_indexed % 10_000 == 0 and total_indexed > 0:
                elapsed = time.time() - t0
                speed = total_indexed / elapsed if elapsed > 0 else 0
                print(
                    f"[Index] {total_indexed:,} indexed, {total_skipped:,} skipped, {total_resumed:,} resumed"
                    f" | {speed:.0f} rows/s | {elapsed:.0f}s elapsed"
                )

    # Залить остаток
    flush_encode_buffer()

    # Re-enable HNSW + indexing (one-time build at end)
    print(f"\n[Index] Bulk upsert done. Rebuilding HNSW index...")
    rebuild_start = time.time()
    client.update_collection(
        collection_name=COLLECTION_NAME,
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
        hnsw_config=HnswConfigDiff(m=16),  # default Qdrant HNSW value
    )
    print(f"[Index] HNSW rebuild triggered in {time.time()-rebuild_start:.1f}s")

    elapsed = time.time() - t0
    info = client.get_collection(COLLECTION_NAME)
    disk_mb = _estimate_disk_mb(args.index_path)

    print(f"\n[Index] Done!")
    print(f"  Indexed this run: {total_indexed:,} points")
    print(f"  Resumed (skipped): {total_resumed:,} rows already in index")
    print(f"  Skipped (no name/id): {total_skipped:,} rows")
    print(f"  Total in collection: {info.points_count or 0:,}")
    print(f"  Disk usage: ~{disk_mb:.1f} MB")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Index path: {args.index_path}")


def _estimate_disk_mb(path: str) -> float:
    """Оценить размер директории на диске в MB."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 * 1024)


if __name__ == "__main__":
    main()
