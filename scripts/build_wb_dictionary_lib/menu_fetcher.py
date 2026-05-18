"""WB main menu fetcher.

Downloads the WB category tree from the public static JSON and extracts
all leaf categories (subjects) with their metadata.
"""
import httpx

MENU_URL = "https://static-basket-01.wb.ru/vol0/data/main-menu-ru-ru-v2.json"


async def fetch_main_menu() -> list[dict]:
    """Fetch top-level WB menu tree."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(MENU_URL)
        r.raise_for_status()
        return r.json()


def extract_all_subjects(menu: list[dict]) -> list[dict]:
    """Walk the tree recursively and collect all leaf categories.

    Each leaf is returned as:
        {
            "id": int,
            "name": str,
            "shard": str | None,
            "url": str | None,
            "query": str | None,
            "path": list[str],   # breadcrumb from root to leaf
        }
    """
    subjects: list[dict] = []

    def walk(node: dict, path: list[str]) -> None:
        new_path = path + [node["name"]]
        children = node.get("childs") or []
        if children:
            for child in children:
                walk(child, new_path)
        else:
            # Leaf node
            subjects.append(
                {
                    "id": node["id"],
                    "name": node["name"],
                    "shard": node.get("shard"),
                    "url": node.get("url"),
                    "query": node.get("query"),
                    "path": new_path,
                }
            )

    for top in menu:
        walk(top, [])

    return subjects
