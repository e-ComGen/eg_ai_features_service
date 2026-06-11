"""One-time script: create the 'categories' text payload index in ozon_rag.qdrant.

IMPORTANT: qdrant-client local/embedded mode IGNORES payload indexes
(it warns "Payload indexes have no effect in the local Qdrant").
This script is only effective when run against a qdrant-server instance
(set QDRANT_URL=http://localhost:6333 and point it at the collection).

Until then, CompetitorRagSource handles the missing index via graceful-degrade:
filtered query -> 4xx -> retry unfiltered. Fills work; noise-reduction is off.

To activate category-filter noise-reduction:
  1. Run qdrant-server Docker: docker run -p 6333:6333 qdrant/qdrant
  2. Upload the collection (qdrant upload or restore from snapshot)
  3. Run this script with QDRANT_URL set
  4. Set QDRANT_URL in the worker environment

Usage (against live server):
    QDRANT_URL=http://localhost:6333 venv/Scripts/python.exe scripts/build_rag_text_index.py
"""
from __future__ import annotations

import os
import sys

COLLECTION = "ozon_products"
FIELD = "categories"

INDEX_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..",
    "app",
    "services",
    "enrichment",
    "strategies",
    "dictionaries",
    "data",
    "ozon_rag.qdrant",
))


def main() -> None:
    qdrant_url = os.environ.get("QDRANT_URL")

    try:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.models import PayloadSchemaType  # type: ignore
    except ImportError as e:
        print(f"[build_rag_text_index] ERROR: qdrant-client not installed: {e}", file=sys.stderr)
        sys.exit(1)

    if qdrant_url:
        print(f"[build_rag_text_index] Connecting to Qdrant server at {qdrant_url} ...")
        try:
            client = QdrantClient(url=qdrant_url, timeout=30)
        except Exception as e:
            print(f"[build_rag_text_index] ERROR connecting to server: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(
            "[build_rag_text_index] WARNING: QDRANT_URL not set — using embedded local mode.\n"
            "  Payload indexes have NO EFFECT in local mode (qdrant-client limitation).\n"
            "  Set QDRANT_URL to point at a running qdrant-server for this to work.\n"
            "  Proceeding anyway for diagnostic purposes ...",
        )
        if not os.path.isdir(INDEX_PATH):
            print(f"[build_rag_text_index] ERROR: local index not found at {INDEX_PATH}", file=sys.stderr)
            sys.exit(1)
        try:
            client = QdrantClient(path=INDEX_PATH)
        except Exception as e:
            print(
                f"[build_rag_text_index] ERROR opening local index (locked?): {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"[build_rag_text_index] Creating TEXT payload index on '{FIELD}' in '{COLLECTION}' ...")
    try:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=FIELD,
            field_schema=PayloadSchemaType.TEXT,
        )
        print(f"[build_rag_text_index] SUCCESS: TEXT index on '{FIELD}' created/confirmed.")
    except Exception as e:
        err_lower = str(e).lower()
        if "already exists" in err_lower or "conflict" in err_lower:
            print(f"[build_rag_text_index] Index already exists (idempotent): {e}")
        elif "no effect" in err_lower or "local" in err_lower:
            print(
                f"[build_rag_text_index] WARNING: local mode ignores this call: {e}\n"
                "  Run with QDRANT_URL set against a real server.",
            )
        else:
            print(f"[build_rag_text_index] ERROR: {e}", file=sys.stderr)
            client.close()
            sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
