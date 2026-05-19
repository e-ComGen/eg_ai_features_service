"""Use HuggingFace nyuuzyou/wb-products as offline seed для nm_ids per subj_id.

Скачивает dataset один раз (~5-10 GB), затем выбирает N sample nm_ids per subject.
Полностью offline — не зависит от WB endpoints для sampling.
"""
import json
from pathlib import Path
from collections import defaultdict


CACHE_PATH = Path(__file__).parent / "_hf_seed_cache.json"


def build_seed_from_hf(samples_per_subj: int = 30, max_subjects: int = None) -> dict[int, list[int]]:
    """Group HF wb-products dataset by subj_id, sample N nm_ids per group.

    Returns: {subj_id: [nm_id1, nm_id2, ...]}

    Если cache есть — используется без re-download.
    """
    if CACHE_PATH.exists():
        cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return {int(k): v for k, v in cached.items()}

    print(f"[hf_seed] Loading HuggingFace nyuuzyou/wb-products...")
    from datasets import load_dataset

    # Streaming mode — не загружает всё в память
    ds = load_dataset("nyuuzyou/wb-products", split="train", streaming=True)

    by_subj = defaultdict(list)
    counter = 0
    for row in ds:
        subj = row.get("subj_id")
        nm = row.get("nm_id")
        if not subj or not nm:
            continue
        if len(by_subj[subj]) < samples_per_subj:
            by_subj[subj].append(nm)
        counter += 1
        if counter % 100_000 == 0:
            print(f"[hf_seed]   processed {counter} rows, {len(by_subj)} subjects")
        # Stop когда у всех subjects достаточно samples
        if max_subjects and len(by_subj) >= max_subjects:
            if all(len(v) >= samples_per_subj for v in by_subj.values()):
                break

    # Cache
    CACHE_PATH.write_text(
        json.dumps({str(k): v for k, v in by_subj.items()}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[hf_seed] Cached {len(by_subj)} subjects to {CACHE_PATH}")
    return dict(by_subj)
