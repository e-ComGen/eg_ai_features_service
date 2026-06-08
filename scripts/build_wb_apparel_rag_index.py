"""Скрипт индексации датасета nyuuzyou/wb-products (apparel-only) в локальный Qdrant-индекс.

Зеркалит scripts/build_ozon_rag_index.py, чтобы тонкий WbApparelRagSource
(subclass of CompetitorRagSource) мог запрашивать коллекцию идентично.

Создаёт file-based Qdrant коллекцию `wb_apparel_rag`:
- Vector: 384-dim эмбеддинг поля `name` (title) через
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (normalize_embeddings=True)
- Payload: {variantid, name, description, categories, characteristics}
  characteristics — dict {charc_name: value} (тот же shape, что читает CompetitorRagSource
  через neighbor.get("characteristics") и _find_attr_value).

Фильтрация: только одежда/обувь/аксессуары. Точное поле категории в WB неизвестно,
поэтому при --inspect выводим ключи первой строки, а фильтр matched как
case-insensitive substring против реального категорийного поля (candidates:
subj_root_name, subj_name, category). Allow-list по умолчанию {Одежда,Обувь,Аксессуары}.

Использование:
    # Посмотреть реальные поля датасета и выйти (НИЧЕГО не индексирует):
    python scripts/build_wb_apparel_rag_index.py --inspect

    # Тестовый прогон на 5000 строк:
    python scripts/build_wb_apparel_rag_index.py --limit 5000

    # Полный ингест (RunPod, может занять часы):
    python scripts/build_wb_apparel_rag_index.py --limit 0

    # Из локального jsonl/zst (без HF streaming):
    python scripts/build_wb_apparel_rag_index.py --input /path/to/wb-products.jsonl

Требования:
    pip install qdrant-client datasets sentence-transformers python-dotenv zstandard

HF_TOKEN читается из .env файла в корне проекта.
"""
from __future__ import annotations

import sys
# Отключить буферизацию stdout для корректного вывода в лог-файл
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Windows DLL fix: pyarrow/pandas должны загружаться ДО torch/sentence_transformers,
# иначе происходит access violation при загрузке pyarrow DLL после torch.
try:
    import pyarrow  # noqa: F401
    import pandas   # noqa: F401
except ImportError:
    pass

import argparse
import gzip
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Загрузить .env до импортов которые могут читать переменные среды
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

HF_TOKEN = os.getenv("HF_TOKEN", "")

# Путь к Qdrant-индексу (зеркалит ozon: data/<name>.qdrant)
INDEX_PATH = str(
    PROJECT_ROOT
    / "app" / "services" / "enrichment" / "strategies" / "dictionaries" / "data"
    / "wb_apparel_rag.qdrant"
)
COLLECTION_NAME = "wb_apparel_rag"
VECTOR_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2 output dim

# Датасет — WB products dump на HuggingFace
DATASET_NAME = "nyuuzyou/wb-products"
DATASET_SPLIT = "train"

# Модель для эмбеддингов — та же что и в CompetitorRagSource
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Размер батча для batch_encode модели
ENCODE_BATCH_SIZE = 100

# Allow-list категорий (одежда/обувь/аксессуары), substring-match, lower-case.
DEFAULT_CATEGORIES = ["Одежда", "Обувь", "Аксессуары"]

# ВОЗМОЖНЫЕ поля датасета (реально проверяются в runtime через --inspect).
# Имя категории/предмета (WB: subj_root_name — корневой раздел, subj_name — предмет).
CATEGORY_FIELD_CANDIDATES = ["subj_root_name", "subj_name", "category", "subject"]
# Title / имя товара.
NAME_FIELD_CANDIDATES = ["imt_name", "name", "title", "goods_name"]
# Описание.
DESCRIPTION_FIELD_CANDIDATES = ["description", "descr", "desc"]
# ID товара/варианта.
ID_FIELD_CANDIDATES = ["variantid", "nm_id", "nmId", "id", "imt_id"]
# Характеристики — WB обычно отдаёт options/characteristics, состав — compositions.
CHARC_LIST_FIELDS = ["options", "characteristics", "grouped_options"]
COMPOSITION_FIELDS = ["compositions", "composition"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build WB apparel RAG index from nyuuzyou/wb-products"
    )
    p.add_argument("--limit", type=int, default=0,
                   help="Max rows to scan from the stream (0 = no limit). "
                        "Тестовый прогон: --limit 5000.")
    p.add_argument("--batch-size", type=int, default=1000,
                   help="Qdrant batch insert size")
    p.add_argument("--encode-batch", type=int, default=ENCODE_BATCH_SIZE,
                   help="sentence-transformers encode batch size")
    p.add_argument("--out", default=INDEX_PATH,
                   help="Path to Qdrant file-based storage")
    p.add_argument("--collection", default=COLLECTION_NAME,
                   help="Qdrant collection name")
    p.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES,
                   help="Allow-list of apparel category substrings (case-insensitive)")
    p.add_argument("--input", default=None,
                   help="Local jsonl/jsonl.gz/jsonl.zst file (bypass HF streaming)")
    p.add_argument("--inspect", action="store_true",
                   help="Print first row keys + sample and exit (no indexing)")
    p.add_argument("--recreate", action="store_true",
                   help="Delete existing collection and recreate")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pure helpers (covered by tests/test_wb_apparel_ingest.py)
# ---------------------------------------------------------------------------

def safe_str(val, max_len: int = 500) -> str:
    """Безопасно преобразовать значение в строку с ограничением длины."""
    if val is None:
        return ""
    s = str(val)
    return s[:max_len] if len(s) > max_len else s


def _first_present(row: dict, candidates: list[str]):
    """Вернуть (field_name, value) для первого поля из candidates, что есть в row."""
    for f in candidates:
        if f in row and row[f] not in (None, ""):
            return f, row[f]
    return None, None


def pick_name(row: dict) -> str:
    """Достать title из imt_name/name/title/... — whichever exists."""
    _, val = _first_present(row, NAME_FIELD_CANDIDATES)
    return safe_str(val, 300)


def pick_id(row: dict):
    """Достать id товара (variantid/nm_id/id/...). Возвращает первое валидное значение."""
    _, val = _first_present(row, ID_FIELD_CANDIDATES)
    return val


def pick_category_value(row: dict, category_field: str | None = None) -> str:
    """Собрать строку категории из субъектных полей для substring-фильтра.

    Если задано конкретное category_field — используем его. Иначе конкатенируем
    все присутствующие subj_root_name/subj_name/category — чтобы фильтр срабатывал,
    даже если apparel-маркер лежит не в том поле, что мы ожидали.
    """
    if category_field and category_field in row:
        return safe_str(row.get(category_field), 200)
    parts = []
    for f in CATEGORY_FIELD_CANDIDATES:
        v = row.get(f)
        if v:
            parts.append(str(v))
    return " ".join(parts)[:200]


def is_apparel(row: dict, allow_lower: list[str], category_field: str | None = None) -> bool:
    """True если категорийная строка содержит любой из allow-substrings (lower-case)."""
    cat_text = pick_category_value(row, category_field).lower()
    if not cat_text:
        return False
    return any(sub in cat_text for sub in allow_lower)


def _coerce_value(val) -> str | None:
    """Стрингифицировать значение характеристики (скаляр / список / dict)."""
    if val is None:
        return None
    if isinstance(val, str):
        v = val.strip()
        return v or None
    if isinstance(val, (int, float, bool)):
        return str(val)
    if isinstance(val, list):
        parts = [_coerce_value(x) for x in val]
        parts = [p for p in parts if p]
        return ", ".join(parts) if parts else None
    if isinstance(val, dict):
        # WB option-объект часто {"name": ..., "value": ...}
        if "value" in val:
            return _coerce_value(val.get("value"))
        return None
    return safe_str(val, 200) or None


def extract_characteristics(row: dict) -> dict:
    """Собрать {charc_name: value} из WB-полей options/characteristics/compositions.

    WB отдаёт характеристики разными способами:
      - options: список объектов {"name": "Цвет", "value": "красный"}
                 (иногда value — список значений)
      - characteristics: словарь {name: value} или список объектов как options
      - compositions: список составов [{"name": "хлопок", "percentage": 80}] либо
                      [{"name": "Состав", "value": "хлопок 80%"}] — join readably.

    Возвращает плоский dict {имя_характеристики: строковое_значение}, тот же shape,
    что читает CompetitorRagSource._find_attr_value.
    """
    chars: dict[str, str] = {}

    def _ingest_option_list(items):
        for opt in items:
            if not isinstance(opt, dict):
                continue
            name = opt.get("name") or opt.get("charc_name") or opt.get("key")
            if not name:
                continue
            raw_val = opt.get("value")
            if raw_val is None:
                raw_val = opt.get("val") or opt.get("values")
            coerced = _coerce_value(raw_val)
            if coerced:
                chars[str(name).strip()] = coerced

    # options / characteristics / grouped_options — список или dict
    for field in CHARC_LIST_FIELDS:
        block = row.get(field)
        if block is None:
            continue
        if isinstance(block, list):
            _ingest_option_list(block)
        elif isinstance(block, dict):
            # Может быть {name: value} напрямую
            for k, v in block.items():
                coerced = _coerce_value(v)
                if coerced:
                    chars[str(k).strip()] = coerced

    # compositions — состав ткани, join readably в "Состав"
    for field in COMPOSITION_FIELDS:
        comp = row.get(field)
        if not comp:
            continue
        if isinstance(comp, list):
            parts = []
            for c in comp:
                if isinstance(c, dict):
                    cname = c.get("name") or c.get("material")
                    pct = c.get("percentage") or c.get("percent") or c.get("value")
                    if cname and pct not in (None, ""):
                        parts.append(f"{cname} {pct}%" if str(pct).isdigit() else f"{cname} {pct}")
                    elif cname:
                        parts.append(str(cname))
                elif c:
                    parts.append(str(c))
            if parts:
                chars.setdefault("Состав", ", ".join(parts))
        elif isinstance(comp, str) and comp.strip():
            chars.setdefault("Состав", comp.strip())

    return chars


def build_payload(row: dict, category_field: str | None = None) -> dict:
    """Построить payload для Qdrant из строки WB-датасета.

    Shape ИДЕНТИЧЕН ozon: {variantid, name, description, categories, characteristics},
    чтобы CompetitorRagSource читал коллекцию без изменений.
    """
    _, desc = _first_present(row, DESCRIPTION_FIELD_CANDIDATES)
    cat_value = pick_category_value(row, category_field)
    return {
        "variantid": pick_id(row),
        "name": pick_name(row),
        "description": safe_str(desc, 500),
        # categories — список строк (subj_root > subj), как ожидает source.
        "categories": [c for c in (cat_value,) if c],
        "characteristics": extract_characteristics(row),
    }


# ---------------------------------------------------------------------------
# Dataset iteration
# ---------------------------------------------------------------------------

def _open_local(path: str):
    """Итератор по локальному jsonl / jsonl.gz / jsonl.zst файлу — по одной dict-строке."""
    lower = path.lower()
    if lower.endswith(".zst"):
        try:
            import zstandard  # type: ignore
        except ImportError:
            print("[Error] zstandard not installed. Run: pip install zstandard")
            sys.exit(1)
        import io
        fh = open(path, "rb")
        dctx = zstandard.ZstdDecompressor()
        stream = io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8")
        return stream
    if lower.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_rows(args):
    """Унифицированный построчный итератор: локальный файл или HF streaming."""
    if args.input:
        print(f"[Index] Reading local file: {args.input}")
        with _open_local(args.input) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("[Error] `datasets` not installed. Run: pip install datasets")
        print("        (или используйте --input <local jsonl/zst file>)")
        sys.exit(1)

    print(f"[Index] Loading dataset {DATASET_NAME} split={DATASET_SPLIT} (streaming)...")
    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        streaming=True,  # streaming чтобы не качать 160M строк в RAM
        token=HF_TOKEN or None,
    )
    for row in dataset:
        yield row


def _detect_category_field(row: dict) -> str | None:
    """Выбрать реальное поле категории из row по списку кандидатов."""
    field, _ = _first_present(row, CATEGORY_FIELD_CANDIDATES)
    return field


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    """Lazy-load sentence-transformers модель один раз."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        print("[Error] sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)
    print(f"[Index] Loading embed model: {model_name} ...")
    model = SentenceTransformer(model_name)
    dim = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)()
    print(f"[Index] Model loaded (output dim={dim})")
    return model


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


def run_inspect(args) -> None:
    """Вывести ключи и sample первой строки датасета, затем выйти."""
    print("[Inspect] Fetching first row...")
    it = iter_rows(args)
    try:
        first = next(it)
    except StopIteration:
        print("[Inspect] Dataset is empty / no rows streamed.")
        return
    print("[Inspect] First row KEYS:")
    for k in first.keys():
        print(f"    - {k}: {type(first[k]).__name__}")
    cat_field = _detect_category_field(first)
    print(f"\n[Inspect] Detected category field: {cat_field}")
    print(f"[Inspect] Detected name: {pick_name(first)[:80]!r}")
    print(f"[Inspect] Detected id: {pick_id(first)!r}")
    print("[Inspect] Built characteristics:")
    for k, v in list(extract_characteristics(first).items())[:15]:
        print(f"    {k}: {v}")
    print("\n[Inspect] Sample row (truncated to 2000 chars):")
    print(json.dumps(first, ensure_ascii=False, default=str)[:2000])


def main() -> None:
    args = parse_args()

    allow_lower = [c.lower() for c in args.categories]

    print(f"[Index] Project root: {PROJECT_ROOT}")
    print(f"[Index] Out path: {args.out}")
    print(f"[Index] Collection: {args.collection}")
    print(f"[Index] Limit: {args.limit or 'unlimited'}")
    print(f"[Index] Qdrant batch size: {args.batch_size}")
    print(f"[Index] Encode batch size: {args.encode_batch}")
    print(f"[Index] Apparel allow-list: {args.categories}")
    print(f"[Index] Embed model: {EMBED_MODEL_NAME}")
    print(f"[Index] Vector dim: {VECTOR_DIM}")

    if args.inspect:
        run_inspect(args)
        return

    # Инициализация Qdrant клиента
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct  # noqa: F401
    except ImportError:
        print("[Error] qdrant-client not installed. Run: pip install qdrant-client")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    client = QdrantClient(path=args.out)

    collections = [c.name for c in client.get_collections().collections]
    existing_points = 0
    if args.collection in collections:
        if args.recreate:
            print(f"[Index] Recreating collection '{args.collection}'...")
            client.delete_collection(args.collection)
        else:
            info = client.get_collection(args.collection)
            existing_points = info.points_count or 0
            print(f"[Index] Collection '{args.collection}' exists with {existing_points} points.")
            print("[Index] Resume mode: уже проиндексированные id будут пропущены.")

    if args.collection not in [c.name for c in client.get_collections().collections]:
        print(f"[Index] Creating collection '{args.collection}' (dim={VECTOR_DIM})...")
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )

    # Resume: множество уже проиндексированных id
    existing_ids: set[int] = set()
    if existing_points > 0:
        print("[Index] Fetching existing IDs for resume (this may take a moment)...")
        offset = None
        while True:
            result, next_offset = client.scroll(
                collection_name=args.collection,
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

    model = load_model(EMBED_MODEL_NAME)

    t0 = time.time()
    total_indexed = 0
    total_skipped = 0     # malformed / no name / no id
    total_filtered = 0    # not apparel
    total_resumed = 0
    total_scanned = 0
    category_field: str | None = None  # детектится по первой строке

    encode_buffer_names: list[str] = []
    encode_buffer_rows: list[dict] = []

    from qdrant_client.models import PointStruct

    def flush_encode_buffer():
        nonlocal total_indexed
        if not encode_buffer_names:
            return
        vecs = model.encode(
            encode_buffer_names,
            normalize_embeddings=True,
            batch_size=len(encode_buffer_names),
            show_progress_bar=False,
        )
        qdrant_batch: list = []
        for row, vec in zip(encode_buffer_rows, vecs):
            pid = pick_id(row)
            try:
                point_id = int(pid)
            except (TypeError, ValueError):
                continue
            payload = build_payload(row, category_field)
            qdrant_batch.append(PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload=payload,
            ))
        if qdrant_batch:
            client.upsert(collection_name=args.collection, points=qdrant_batch)
            total_indexed += len(qdrant_batch)
        encode_buffer_names.clear()
        encode_buffer_rows.clear()

    print(f"[Index] Streaming rows (limit={args.limit or 'unlimited'})...")
    for row in iter_rows(args):
        total_scanned += 1
        if args.limit and total_scanned > args.limit:
            break

        if not isinstance(row, dict):
            total_skipped += 1
            continue

        # Детект поля категории по первой валидной строке
        if category_field is None:
            category_field = _detect_category_field(row)
            if category_field:
                print(f"[Index] Detected category field: '{category_field}'")

        try:
            if not is_apparel(row, allow_lower, category_field):
                total_filtered += 1
                continue

            pid = pick_id(row)
            try:
                point_id = int(pid)
            except (TypeError, ValueError):
                total_skipped += 1
                continue

            name = pick_name(row)
            if not name.strip():
                total_skipped += 1
                continue

            if point_id in existing_ids:
                total_resumed += 1
                continue

            encode_buffer_names.append(name)
            encode_buffer_rows.append(row)
        except Exception as e:  # noqa: BLE001 — skip malformed row, keep going
            total_skipped += 1
            if total_skipped <= 5:
                print(f"[Index] skip malformed row: {e}")
            continue

        if len(encode_buffer_names) >= args.encode_batch:
            flush_encode_buffer()

        if total_scanned % 50_000 == 0:
            elapsed = time.time() - t0
            speed = total_scanned / elapsed if elapsed > 0 else 0
            print(
                f"[Index] scanned={total_scanned:,} kept={total_indexed:,} "
                f"filtered(non-apparel)={total_filtered:,} skipped={total_skipped:,} "
                f"resumed={total_resumed:,} | {speed:.0f} rows/s | {elapsed:.0f}s"
            )

    flush_encode_buffer()

    elapsed = time.time() - t0
    info = client.get_collection(args.collection)
    disk_mb = _estimate_disk_mb(args.out)

    print("\n[Index] Done!")
    print(f"  Scanned: {total_scanned:,} rows")
    print(f"  Indexed this run: {total_indexed:,} points")
    print(f"  Filtered (non-apparel): {total_filtered:,}")
    print(f"  Resumed (already in index): {total_resumed:,}")
    print(f"  Skipped (no name/id/malformed): {total_skipped:,}")
    print(f"  Total in collection: {info.points_count or 0:,}")
    print(f"  Disk usage: ~{disk_mb:.1f} MB")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Index path: {args.out}")


if __name__ == "__main__":
    main()
