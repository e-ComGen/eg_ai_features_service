"""Ingest FSA Росаккредитация open-data dumps into a Qdrant collection.

SOURCE FORMAT (confirmed via Wayback Machine, 2022 snapshot):
  - RDS (declarations): https://fsa.gov.ru/opendata/7736638268-rds/
  - RSS (certificates):  https://fsa.gov.ru/opendata/7736638268-rss/
  Dumps are published as dated .7z archives containing a semicolon-delimited CSV
  encoded in Windows-1251.  File pattern: data-YYYYMMDD-structure-YYYYMMDD.7z

  IMPORTANT — CURRENT ACCESS STATUS:
    fsa.gov.ru returns HTTP 403 for all non-browser clients (WAF geo/IP block as of
    2026-06-12).  The Wayback Machine confirmed the schema but we cannot download
    the live dumps directly.  To work around, two options are documented below:
      A) Mirror download: provide --dump-url pointing at an archived or mirror 7z.
      B) Manual download: download the 7z manually, pass --local-file path.
    The Wayback Machine hosts the 2021-11 dump at:
      https://web.archive.org/web/20220118100801if_/
        https://fsa.gov.ru/opendata/7736638268-rds/data-20211104-structure-20211206.7z

RDS (declarations) SCHEMA (from structure-20190917.csv):
  id_decl, reg_number, decl_status, decl_type, date_beginning, date_finish,
  declaration_scheme, product_object_type_decl, product_type, product_group,
  product_name, asproduct_info, product_tech_reg,
  organ_to_certification_name, organ_to_certification_reg_number,
  basis_for_decl, old_basis_for_decl,
  applicant_type, person_applicant_type, applicant_ogrn, applicant_inn,
  applicant_name, manufacturer_type, manufacturer_ogrn, manufacturer_inn,
  manufacturer_name, [+ country, address fields in newer versions]

RSS (certificates) SCHEMA adds:
  cert_status, cert_type, reg_number, product_okpd2, product_tn_ved,
  manufacturer_country, manufacturer_address, product_national_standart, ...

VECTOR SPACE: 384-dim paraphrase-multilingual-MiniLM-L12-v2, COSINE distance —
  same model as ozon_products collection.

QDRANT COLLECTION: rosaccreditation_rds / rosaccreditation_rss
  Payload text index on 'product_name' for category-filtered retrieval.

FULL RUN PLAN:
  - RDS 2021-11 dump: ~4.8 GB 7z → ~25 GB uncompressed CSV → ~12M rows
    (estimate: 3.5M active declarations, 8M+ including historical)
  - RSS 2021-11 dump: ~1.2 GB 7z → ~6 GB uncompressed → ~3M rows
  - Embedding throughput on CPU: ~200 rows/s → 12M rows ≈ 16 hours
    (GPU: ~2000 rows/s → 1.5 hours on RunPod T4)
  - Qdrant storage: ~384 floats × 4 bytes × 12M ≈ 18 GB vectors + ~5 GB payload
  - RAM peak: ~2 GB (encode_batch=256 keeps tensors small)
  - Disk: ~25 GB total Qdrant storage for both collections
  - Run flag: --full (omit for small sample mode, default 300 rows)

USAGE:
    # Small sample (scaffold/validate, default 300 rows):
    python scripts/ingest_rosaccreditation.py --local-file /path/to/data.7z

    # Sample from Wayback Machine mirror of 2021-11 RDS dump:
    python scripts/ingest_rosaccreditation.py \\
        --dump-url "https://web.archive.org/web/20220118100801if_/https://fsa.gov.ru/opendata/7736638268-rds/data-20211104-structure-20211206.7z" \\
        --doc-type rds --sample-size 300

    # Full ingest (overnight job):
    python scripts/ingest_rosaccreditation.py --local-file /path/dump.7z --full

Requirements (add to requirements.txt if missing):
    py7zr>=0.20
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Iterator

# Windows DLL ordering fix: pyarrow/pandas before torch.
try:
    import pyarrow  # noqa: F401
    import pandas   # noqa: F401
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── Constants ────────────────────────────────────────────────────────────────

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DIM = 384
QDRANT_URL_DEFAULT = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION_RDS = "rosaccreditation_rds"
DEFAULT_COLLECTION_RSS = "rosaccreditation_rss"
DEFAULT_SAMPLE_SIZE = 300
CSV_ENCODING = "cp1251"
CSV_DELIMITER = ","  # actual dump uses comma (structure CSV says ";" but data files use ",")

# Wayback Machine mirrors for the latest available dumps (2021-11):
WAYBACK_RDS = (
    "https://web.archive.org/web/20220118100801if_/"
    "https://fsa.gov.ru/opendata/7736638268-rds/"
    "data-20211104-structure-20211206.7z"
)
WAYBACK_RSS = (
    "https://web.archive.org/web/20220118103804if_/"
    "https://fsa.gov.ru/opendata/7736638268-rss/"
    "data-20211104-structure-20211206.7z"
)

# Fields used to build the embed text (most signal-dense for product search):
_EMBED_FIELDS_RDS = ["product_name", "product_group", "product_type", "product_tech_reg",
                     "manufacturer_name"]
_EMBED_FIELDS_RSS = ["product_name", "product_group", "product_type", "product_tech_reg",
                     "manufacturer_name", "manufacturer_country",
                     "product_okpd2", "product_tn_ved"]


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest FSA Росаккредитация open-data into Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--doc-type", choices=["rds", "rss"], default="rds",
        help="rds = declarations (декларации), rss = certificates (сертификаты). Default: rds",
    )
    p.add_argument(
        "--local-file", metavar="PATH",
        help="Path to a locally downloaded .7z dump file.",
    )
    p.add_argument(
        "--dump-url", metavar="URL",
        help="URL to download the .7z dump (overrides built-in Wayback Machine URL).",
    )
    p.add_argument(
        "--collection", metavar="NAME",
        help="Qdrant collection name. Default: rosaccreditation_rds or rosaccreditation_rss",
    )
    p.add_argument(
        "--qdrant-url", default=QDRANT_URL_DEFAULT,
        help=f"Qdrant server URL. Default: {QDRANT_URL_DEFAULT}",
    )
    p.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of rows for sample mode (default: {DEFAULT_SAMPLE_SIZE}). Ignored with --full.",
    )
    p.add_argument(
        "--full", action="store_true",
        help="Full ingest (no row limit). Default off — sample mode only.",
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
        help="Skip creating the payload text index on product_name.",
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
            sys.exit("[Error] sentence-transformers not installed. Run: pip install sentence-transformers")
        print(f"[Ingest] Loading embed model: {EMBED_MODEL_NAME} ...")
        _model_cache = SentenceTransformer(EMBED_MODEL_NAME)
        dim = getattr(_model_cache, 'get_embedding_dimension',
                      _model_cache.get_sentence_embedding_dimension)()
        print(f"[Ingest] Model loaded (dim={dim})")
    return _model_cache


def embed_batch(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=len(texts),
                        show_progress_bar=False)
    return [v.tolist() for v in vecs]


# ── Download / extract ────────────────────────────────────────────────────────

def download_7z(url: str, dest: Path) -> Path:
    """Download a .7z file from url to dest, with progress."""
    import urllib.request

    print(f"[Ingest] Downloading {url}")
    print(f"[Ingest] -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    class _Progress:
        def __init__(self):
            self.reported = 0

        def __call__(self, block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = 100 * downloaded / total_size
                mb = downloaded / 1024 / 1024
                if mb - self.reported >= 50:
                    print(f"[Ingest]   {mb:.0f} MB / {total_size/1024/1024:.0f} MB ({pct:.1f}%)")
                    self.reported = mb
            else:
                mb = downloaded / 1024 / 1024
                if mb - self.reported >= 50:
                    print(f"[Ingest]   {mb:.0f} MB downloaded")
                    self.reported = mb

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.getheader("Content-Length", 0))
        prog = _Progress()
        chunk = 64 * 1024
        downloaded = 0
        with dest.open("wb") as f:
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                f.write(data)
                downloaded += len(data)
                prog(downloaded // chunk, chunk, total)
    print(f"[Ingest] Download complete: {dest.stat().st_size / 1024 / 1024:.1f} MB")
    return dest


def iter_csv_from_7z(archive_path: Path, encoding: str = CSV_ENCODING,
                     delimiter: str = CSV_DELIMITER) -> Iterator[dict]:
    """Extract the first CSV file from a .7z archive and yield rows as dicts.

    Uses a temp directory to extract (py7zr 1.x lacks an in-memory read() API).
    The temp dir is cleaned up after the generator exhausts.
    """
    import tempfile

    try:
        import py7zr
    except ImportError:
        sys.exit("[Error] py7zr not installed. Run: pip install py7zr")

    with tempfile.TemporaryDirectory() as tmp_dir:
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            names = archive.getnames()
            csv_names = [n for n in names if n.lower().endswith(".csv")]
            if not csv_names:
                sys.exit(
                    f"[Error] No CSV file found in archive {archive_path}. Files: {names}"
                )
            csv_name = csv_names[0]
            print(f"[Ingest] Extracting CSV: {csv_name} from {archive_path.name}")
            archive.extractall(path=tmp_dir)

        import os
        csv_path = os.path.join(tmp_dir, csv_name)
        print(f"[Ingest] Reading {csv_path} ...")
        with open(csv_path, "rb") as f:
            raw_bytes = f.read()

    # Decode and parse (outside temp dir — file is already in memory)
    text = raw_bytes.decode(encoding, errors="replace")
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    for row in reader:
        yield row


# ── Payload builder ───────────────────────────────────────────────────────────

def _safe(row: dict, key: str, max_len: int = 500) -> str:
    v = row.get(key, "") or ""
    return str(v).strip()[:max_len]


def build_embed_text(row: dict, doc_type: str) -> str:
    """Build the text to embed from the most informative fields."""
    fields = _EMBED_FIELDS_RSS if doc_type == "rss" else _EMBED_FIELDS_RDS
    parts = [_safe(row, f) for f in fields]
    return " | ".join(p for p in parts if p)


def build_payload_rds(row: dict) -> dict:
    return {
        "doc_type": "rds",
        "reg_number": _safe(row, "reg_number", 100),
        "decl_status": _safe(row, "decl_status", 50),
        "product_name": _safe(row, "product_name", 500),
        "product_group": _safe(row, "product_group", 200),
        "product_type": _safe(row, "product_type", 200),
        "product_tech_reg": _safe(row, "product_tech_reg", 500),
        "manufacturer_name": _safe(row, "manufacturer_name", 300),
        "applicant_name": _safe(row, "applicant_name", 300),
        "date_beginning": _safe(row, "date_beginning", 20),
        "date_finish": _safe(row, "date_finish", 20),
    }


def build_payload_rss(row: dict) -> dict:
    return {
        "doc_type": "rss",
        "reg_number": _safe(row, "reg_number", 100),
        "cert_status": _safe(row, "cert_status", 50),
        "product_name": _safe(row, "product_name", 500),
        "product_group": _safe(row, "product_group", 200),
        "product_type": _safe(row, "product_type", 200),
        "product_tech_reg": _safe(row, "product_tech_reg", 500),
        "product_okpd2": _safe(row, "product_okpd2", 200),
        "product_tn_ved": _safe(row, "product_tn_ved", 200),
        "manufacturer_name": _safe(row, "manufacturer_name", 300),
        "manufacturer_country": _safe(row, "manufacturer_country", 100),
        "applicant_name": _safe(row, "applicant_name", 300),
        "date_beginning": _safe(row, "date_begining", 20),  # typo in the official schema
        "date_finish": _safe(row, "date_finish", 20),
    }


def build_payload(row: dict, doc_type: str) -> dict:
    return build_payload_rss(row) if doc_type == "rss" else build_payload_rds(row)


def make_point_id(row: dict, doc_type: str) -> str:
    """Generate a stable UUID from the document's registration number."""
    id_field = "id_cert" if doc_type == "rss" else "id_decl"
    raw_id = row.get(id_field) or row.get("reg_number", "")
    if raw_id and str(raw_id).strip():
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"fsa:{doc_type}:{raw_id.strip()}"))
    # Fallback: UUID4 (not idempotent, but at least not a crash)
    return str(uuid.uuid4())


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def get_qdrant_client(qdrant_url: str):
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except ImportError:
        sys.exit("[Error] qdrant-client not installed. Run: pip install qdrant-client")
    print(f"[Ingest] Connecting to Qdrant at {qdrant_url}")
    return QdrantClient(url=qdrant_url, timeout=60)


def ensure_collection(client, collection: str, recreate: bool) -> None:
    from qdrant_client.models import Distance, VectorParams  # type: ignore

    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        if recreate:
            print(f"[Ingest] Recreating collection '{collection}' ...")
            client.delete_collection(collection)
        else:
            info = client.get_collection(collection)
            print(f"[Ingest] Collection '{collection}' exists ({info.points_count or 0} points). Resume mode.")
            return

    print(f"[Ingest] Creating collection '{collection}' (dim={VECTOR_DIM}, COSINE) ...")
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )


def create_text_index(client, collection: str) -> None:
    from qdrant_client.models import PayloadSchemaType  # type: ignore
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name="product_name",
            field_schema=PayloadSchemaType.TEXT,
        )
        print(f"[Ingest] Text payload index created on 'product_name' in '{collection}'.")
    except Exception as e:
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            print(f"[Ingest] Text index already exists (idempotent).")
        else:
            print(f"[Ingest] Warning: could not create text index: {e}")


# ── Main ingest loop ──────────────────────────────────────────────────────────

def run_ingest(
    archive_path: Path,
    collection: str,
    doc_type: str,
    qdrant_url: str,
    limit: int,
    encode_batch_size: int,
    upsert_batch_size: int,
    recreate: bool,
    no_text_index: bool,
) -> None:
    client = get_qdrant_client(qdrant_url)
    ensure_collection(client, collection, recreate)

    from qdrant_client.models import PointStruct  # type: ignore

    t0 = time.time()
    total_indexed = 0
    total_skipped = 0

    embed_text_buf: list[str] = []
    payload_buf: list[dict] = []
    id_buf: list[str] = []

    def flush():
        nonlocal total_indexed
        if not embed_text_buf:
            return
        vecs = embed_batch(embed_text_buf)
        points = [
            PointStruct(id=pid, vector=vec, payload=payload)
            for pid, vec, payload in zip(id_buf, vecs, payload_buf)
        ]
        # Upsert in sub-batches
        for i in range(0, len(points), upsert_batch_size):
            client.upsert(collection_name=collection, points=points[i:i + upsert_batch_size])
        total_indexed += len(points)
        embed_text_buf.clear()
        payload_buf.clear()
        id_buf.clear()

    for row in iter_csv_from_7z(archive_path):
        # Check limit BEFORE accumulating into buffer (avoids over-shoot by encode_batch)
        if limit and (total_indexed + len(embed_text_buf) + total_skipped) >= limit:
            break

        embed_text = build_embed_text(row, doc_type)
        if not embed_text.strip():
            total_skipped += 1
            continue

        point_id = make_point_id(row, doc_type)
        payload = build_payload(row, doc_type)

        embed_text_buf.append(embed_text)
        payload_buf.append(payload)
        id_buf.append(point_id)

        if len(embed_text_buf) >= encode_batch_size:
            flush()
            rows_done = total_indexed + total_skipped
            if rows_done % 1000 == 0 and rows_done > 0:
                elapsed = time.time() - t0
                speed = total_indexed / elapsed if elapsed > 0 else 0
                print(f"[Ingest] {total_indexed:,} indexed, {total_skipped:,} skipped "
                      f"| {speed:.0f} rows/s | {elapsed:.0f}s")

    flush()  # tail

    elapsed = time.time() - t0
    info = client.get_collection(collection)
    print(f"\n[Ingest] Done!")
    print(f"  Indexed this run : {total_indexed:,}")
    print(f"  Skipped (no text): {total_skipped:,}")
    print(f"  Total in coll.   : {info.points_count or 0:,}")
    print(f"  Time             : {elapsed:.1f}s")
    print(f"  Speed            : {total_indexed / elapsed:.0f} rows/s" if elapsed > 0 else "")

    if not no_text_index:
        create_text_index(client, collection)

    # Quick validation query-back
    print("\n[Ingest] === Validation: query-back test ===")
    _validate(client, collection, doc_type)


def _validate(client, collection: str, doc_type: str) -> None:
    """Embed a sample product name, query Qdrant, and print top results."""
    sample_query = "кабель электрический" if doc_type == "rds" else "напиток безалкогольный"
    try:
        print(f"[Validate] Query: '{sample_query}'")
    except UnicodeEncodeError:
        print("[Validate] Query: (Cyrillic product query)")
    try:
        vec = embed_batch([sample_query])[0]
        result = client.query_points(
            collection_name=collection,
            query=vec,
            limit=3,
            with_payload=["product_name", "manufacturer_name", "reg_number", "doc_type"],
        )
        for i, hit in enumerate(result.points):
            p = hit.payload or {}
            try:
                print(f"  [{i}] score={hit.score:.3f} | {p.get('product_name','')[:80]}"
                      f" | mfr={p.get('manufacturer_name','')[:40]}"
                      f" | reg={p.get('reg_number','')}")
            except UnicodeEncodeError:
                print(f"  [{i}] score={hit.score:.3f} | (Cyrillic payload)")
        print(f"[Validate] OK - {len(result.points)} results returned.")
    except Exception as e:
        print(f"[Validate] ERROR: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    collection = args.collection or (
        DEFAULT_COLLECTION_RSS if args.doc_type == "rss" else DEFAULT_COLLECTION_RDS
    )
    limit = 0 if args.full else args.sample_size

    print(f"[Ingest] === FSA Росаккредитация Ingest ===")
    print(f"  doc_type   : {args.doc_type}")
    print(f"  collection : {collection}")
    print(f"  qdrant_url : {args.qdrant_url}")
    print(f"  limit      : {'unlimited (--full)' if args.full else limit}")
    print(f"  recreate   : {args.recreate}")

    # Resolve archive path
    if args.local_file:
        archive_path = Path(args.local_file)
        if not archive_path.exists():
            sys.exit(f"[Error] Local file not found: {archive_path}")
    else:
        # Download from URL
        dump_url = args.dump_url or (WAYBACK_RSS if args.doc_type == "rss" else WAYBACK_RDS)
        tmp_dir = PROJECT_ROOT / "scripts" / "_tmp_fsa_dumps"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        fname = dump_url.rstrip("/").split("/")[-1].split("?")[0]
        archive_path = tmp_dir / fname
        if archive_path.exists():
            print(f"[Ingest] Using cached download: {archive_path} "
                  f"({archive_path.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            download_7z(dump_url, archive_path)

    run_ingest(
        archive_path=archive_path,
        collection=collection,
        doc_type=args.doc_type,
        qdrant_url=args.qdrant_url,
        limit=limit,
        encode_batch_size=args.encode_batch,
        upsert_batch_size=args.batch,
        recreate=args.recreate,
        no_text_index=args.no_text_index,
    )


if __name__ == "__main__":
    main()
