"""Build Ozon dictionary via official Seller API.

Использует OZON_CLIENT_ID + OZON_API_KEY из .env.

Endpoints:
- POST https://api-seller.ozon.ru/v1/description-category/tree — все категории + types
- POST https://api-seller.ozon.ru/v1/description-category/attribute — characteristics per (category, type)

Сохраняет в app/services/enrichment/strategies/dictionaries/data/ozon_dictionary.json
формат совместимый с ozon_loader.
"""
import os
import json
import time
import asyncio
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


load_dotenv()
CLIENT_ID = os.getenv("OZON_CLIENT_ID")
API_KEY = os.getenv("OZON_API_KEY")
BASE_URL = "https://api-seller.ozon.ru"

OUTPUT_PATH = Path("app/services/enrichment/strategies/dictionaries/data/ozon_dictionary.json")
RATE_LIMIT_SLEEP = 0.3  # seconds between requests


def headers() -> dict[str, str]:
    if not CLIENT_ID or not API_KEY:
        raise RuntimeError("OZON_CLIENT_ID or OZON_API_KEY not set in .env")
    return {
        "Client-Id": CLIENT_ID,
        "Api-Key": API_KEY,
        "Content-Type": "application/json",
    }


async def fetch_tree(client: httpx.AsyncClient) -> list[dict]:
    """POST /v1/description-category/tree → list of categories with types."""
    print("[tree] Fetching category tree...", flush=True)
    r = await client.post(
        f"{BASE_URL}/v1/description-category/tree",
        headers=headers(),
        json={"language": "DEFAULT"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["result"]


def flatten_tree(tree: list[dict]) -> list[tuple[int, int, str, list[str]]]:
    """Recursively walk tree, yield (description_category_id, type_id, name, path).

    description_category_id берётся от ближайшего НЕНУЛЕВОГО parent в tree
    (leafs обычно имеют cat_id=0, но endpoint требует parent's id).
    """
    out = []

    def walk(nodes: list[dict], path: list[str], inherited_cat_id: int):
        for node in nodes:
            raw_cat = node.get("description_category_id", 0) or 0
            effective_cat = raw_cat if raw_cat != 0 else inherited_cat_id
            type_id = node.get("type_id", 0) or 0
            name = node.get("category_name") or node.get("type_name") or ""
            new_path = path + [name]
            if type_id and type_id != 0:
                # leaf — use inherited parent cat id
                out.append((effective_cat, type_id, name, new_path))
            children = node.get("children") or []
            if children:
                walk(children, new_path, effective_cat)
    walk(tree, [], 0)
    return out


async def fetch_attributes(client: httpx.AsyncClient, cat_id: int, type_id: int) -> list[dict]:
    """POST /v1/description-category/attribute → characteristics for (cat, type)."""
    r = await client.post(
        f"{BASE_URL}/v1/description-category/attribute",
        headers=headers(),
        json={
            "description_category_id": cat_id,
            "type_id": type_id,
            "language": "DEFAULT",
        },
        timeout=30,
    )
    if r.status_code == 429:
        # rate limit — wait and retry
        await asyncio.sleep(5)
        return await fetch_attributes(client, cat_id, type_id)
    r.raise_for_status()
    return r.json().get("result", [])


async def main():
    if not CLIENT_ID or not API_KEY:
        raise SystemExit("ERROR: OZON_CLIENT_ID or OZON_API_KEY not set in .env")

    print(f"[init] Client-Id={CLIENT_ID[:6]}...  Api-Key=<HIDDEN>", flush=True)

    async with httpx.AsyncClient() as client:
        tree = await fetch_tree(client)
        print(f"[tree] Got {len(tree)} top-level categories", flush=True)

        leafs = flatten_tree(tree)
        print(f"[tree] Flattened to {len(leafs)} (category, type) leaf nodes", flush=True)

        result: dict[str, dict[str, Any]] = {}
        # Partial save for resume
        partial_path = OUTPUT_PATH.with_suffix(".partial.json")
        if partial_path.exists():
            result = json.loads(partial_path.read_text(encoding="utf-8"))
            print(f"[resume] Loaded {len(result)} entries from partial", flush=True)

        for i, (cat_id, type_id, name, path) in enumerate(leafs):
            key = f"{cat_id}:{type_id}"
            if key in result:
                continue
            try:
                attrs = await fetch_attributes(client, cat_id, type_id)
                characteristics = [
                    {
                        "id": a.get("id"),
                        "name": a.get("name"),
                        "type": a.get("type"),
                        "is_required": a.get("is_required", False),
                        "is_collection": a.get("is_collection", False),
                        "description": a.get("description", "") or "",
                    }
                    for a in attrs
                ]
                result[key] = {
                    "description_category_id": cat_id,
                    "type_id": type_id,
                    "name": name,
                    "path": path,
                    "characteristics": characteristics,
                }
            except Exception as e:
                print(f"[warn] Failed {key} ({name}): {type(e).__name__}: {e}", flush=True)
                continue

            if (i + 1) % 25 == 0:
                # Persist partial
                OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                partial_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                total_chars = sum(len(v["characteristics"]) for v in result.values())
                print(f"      {i+1}/{len(leafs)} types — {len(result)} done, {total_chars} characteristics total", flush=True)
            await asyncio.sleep(RATE_LIMIT_SLEEP)

        # Final save
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        wrapped = {
            "schema_version": 2,
            "source": "ozon_seller_api",
            "generated_at": time.strftime("%Y-%m-%d"),
            "categories": result,
        }
        OUTPUT_PATH.write_text(
            json.dumps(wrapped, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        total_chars = sum(len(v["characteristics"]) for v in result.values())
        print(f"[done] Saved {len(result)} categories ({total_chars} characteristics) to {OUTPUT_PATH}", flush=True)
        if partial_path.exists():
            partial_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())
