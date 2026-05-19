"""Загрузка готовых Ozon категорий из welel/ozon-scraper github или встроенного seed.

Если нет интернета — fallback на встроенный мини-seed (топ-50 категорий).

welel/ozon-scraper schema (verified 2026-05):
  Each file is named <id>.json and has the shape:
    {
      "data": {
        "id": "15500",
        "title": "...",
        "url": "/category/elektronika-15500/",
        "columns": [
          {
            "categories": [
              {
                "title": "...",
                "url": "/category/slug-id/",
                "categories": [ ... ]   # nested, same shape
              }
            ]
          }
        ]
      }
    }
  Titles may contain replacement characters (Cyrillic corrupted in the repo);
  the `url` field is always ASCII and carries the slug+id.
"""
import json
import re
import httpx
from pathlib import Path


WELEL_API_URL = (
    "https://api.github.com/repos/welel/ozon-scraper/contents/data/categories"
)
WELEL_RAW_BASE = (
    "https://raw.githubusercontent.com/welel/ozon-scraper/main/data/categories"
)

# Кэш рядом с этим модулем (пишется после первого успешного fetch)
_CACHE_PATH = Path(__file__).parent / "_welel_cache.json"

# Минимальный встроенный seed — топ-категории Ozon для bootstrap
# когда welel/ozon-scraper недоступен
MINIMAL_SEED: list[dict] = [
    {"id": 6000,  "name": "Аптека",                  "path": ["Аптека"]},
    {"id": 6500,  "name": "Бытовая техника",          "path": ["Бытовая техника"]},
    {"id": 15500, "name": "Электроника",              "path": ["Электроника"]},
    {"id": 7500,  "name": "Одежда",                   "path": ["Одежда"]},
    {"id": 8500,  "name": "Обувь",                    "path": ["Обувь"]},
    {"id": 200000,"name": "Товары для дома",          "path": ["Товары для дома"]},
    {"id": 93726, "name": "Красота и здоровье",       "path": ["Красота и здоровье"]},
    {"id": 7811,  "name": "Спорт и отдых",            "path": ["Спорт и отдых"]},
    {"id": 7224,  "name": "Детские товары",           "path": ["Детские товары"]},
    {"id": 7340,  "name": "Книги",                    "path": ["Книги"]},
    {"id": 6124,  "name": "Продукты питания",         "path": ["Продукты питания"]},
    {"id": 90829, "name": "Зоотовары",                "path": ["Зоотовары"]},
    {"id": 14500, "name": "Автотовары",               "path": ["Автотовары"]},
    {"id": 8229,  "name": "Ювелирные украшения",      "path": ["Ювелирные украшения"]},
    {"id": 7337,  "name": "Канцтовары",               "path": ["Канцтовары"]},
    {"id": 7330,  "name": "Музыка",                   "path": ["Музыка"]},
    {"id": 500,   "name": "Игры и консоли",           "path": ["Электроника", "Игры и консоли"]},
    {"id": 502,   "name": "Смартфоны",                "path": ["Электроника", "Смартфоны"]},
    {"id": 7313,  "name": "Ноутбуки",                 "path": ["Электроника", "Ноутбуки"]},
    {"id": 7315,  "name": "Планшеты",                 "path": ["Электроника", "Планшеты"]},
]


def load_seed_categories() -> list[dict]:
    """Return category list with id, name, path, and (optionally) sample_urls.

    Priority:
    1. Disk cache (_welel_cache.json) — written after first successful fetch.
    2. welel/ozon-scraper GitHub repository.
    3. MINIMAL_SEED built into this file.
    """
    if _CACHE_PATH.exists():
        try:
            cats = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            # Reject poisoned cache: require at least one entry with a valid url or id
            if cats and any(c.get("url") or c.get("id") is not None for c in cats):
                return cats
        except Exception:
            pass

    # Try fetch from GitHub
    try:
        cats = _fetch_from_welel()
        if cats:
            _CACHE_PATH.write_text(
                json.dumps(cats, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return cats
    except Exception as exc:
        print(f"[seed_loader] welel fetch failed: {exc}, falling back to MINIMAL_SEED")

    return MINIMAL_SEED


def _fetch_from_welel() -> list[dict]:
    """Fetch and flatten the welel/ozon-scraper category tree.

    The repo stores one JSON file per top-level category in data/categories/.
    We list the directory via GitHub API and download each file.

    TODO: Verify real file structure once accessible — filenames and JSON schema
          may differ from what is assumed here.  The _flatten_tree() helper must be
          adapted to the actual key names used in the repo.
    """
    categories: list[dict] = []
    with httpx.Client(timeout=15) as client:
        # List files in data/categories/
        resp = client.get(
            WELEL_API_URL,
            headers={"Accept": "application/vnd.github+json"},
        )
        resp.raise_for_status()
        entries = resp.json()

        json_files = [e for e in entries if e.get("name", "").endswith(".json")]
        if not json_files:
            raise RuntimeError("No JSON files found in welel/ozon-scraper data/categories/")

        for entry in json_files:
            filename = entry["name"]
            r = client.get(f"{WELEL_RAW_BASE}/{filename}")
            if r.status_code != 200:
                continue
            try:
                file_json = r.json()
                # Real schema: {"data": {"id": "...", "url": "...", "columns": [{"categories": [...]}]}}
                data_node = file_json.get("data", {}) if isinstance(file_json, dict) else {}
                columns = data_node.get("columns") or []
                # Flatten every category subtree in every column
                for col in columns:
                    for cat_node in (col.get("categories") or []):
                        categories.extend(_flatten_welel_node(cat_node, path=[]))
                # If no columns, fall back to legacy recursive flatten
                if not columns:
                    categories.extend(_flatten_tree(file_json))
            except Exception:
                continue

    return categories


def _id_from_url(url: str) -> int | None:
    """Extract numeric category id from an Ozon category URL slug.

    Examples:
      /category/smartfony-15502/  -> 15502
      /category/15500/            -> 15500
    """
    m = re.search(r"-(\d+)/?$", url.rstrip("/"))
    if m:
        return int(m.group(1))
    m = re.search(r"/category/(\d+)/?$", url.rstrip("/"))
    if m:
        return int(m.group(1))
    return None


def _flatten_welel_node(node: dict, path: list[str]) -> list[dict]:
    """Recursively flatten a welel/ozon-scraper category node.

    Schema: {"title": "...", "url": "/category/slug-id/", "categories": [...]}
    """
    nodes: list[dict] = []
    if not isinstance(node, dict):
        return nodes

    url = node.get("url", "")
    title = node.get("title", "")
    children = node.get("categories") or []

    # Titles may be garbled; fall back to slug extracted from url
    if title and "�" not in title:
        name = title
    else:
        # Derive readable name from url slug (e.g. "smartfony-15502" -> "smartfony")
        slug_match = re.search(r"/category/([^/]+)/?$", url.rstrip("/"))
        name = slug_match.group(1) if slug_match else title

    current_path = path + [name] if name else path
    cat_id = _id_from_url(url) if url else None

    if not children:
        nodes.append({
            "id": cat_id,
            "name": name,
            "path": current_path,
            "url": url,
        })
    else:
        for child in children:
            nodes.extend(_flatten_welel_node(child, current_path))

    return nodes


def _flatten_tree(tree: dict | list, path: list[str] | None = None) -> list[dict]:
    """Recursively flatten an Ozon category tree to leaf nodes.

    TODO: Adapt field names to the actual welel/ozon-scraper JSON structure.
          Assumed schema (placeholder):
            {
              "id": 123,
              "title": "...",
              "children": [...],
              "url": "https://www.ozon.ru/category/..."  # optional
            }
    """
    path = path or []
    nodes: list[dict] = []

    items = tree if isinstance(tree, list) else [tree]
    for item in items:
        if not isinstance(item, dict):
            continue

        # Try common field names used in Ozon scrapers
        cat_id = item.get("id") or item.get("category_id")
        name = item.get("title") or item.get("name") or item.get("category_name", "")
        children = item.get("children") or item.get("child") or []
        url = item.get("url", "")

        current_path = path + [name] if name else path

        if not children:
            # Leaf node
            node: dict = {
                "id": cat_id,
                "name": name,
                "path": current_path,
            }
            if url:
                # Build a few sample product URL patterns from category URL
                # Ozon product pages live at /product/<slug>-<id>/
                # We leave sample_urls empty — product_parser will receive real
                # URLs from category pages scraped by category_fetcher (future).
                node["sample_urls"] = []
            nodes.append(node)
        else:
            nodes.extend(_flatten_tree(children, current_path))

    return nodes
