"""Real-world acceptance tests для PipelineOrchestrator.

Прогоняет 10 реальных товаров через pipeline, сравнивает с ground truth,
выводит metrics (coverage, accuracy, source distribution, cost).

Opt-in через RUN_LIVE_TESTS=1.
"""
import json
import pytest
from pathlib import Path
from collections import defaultdict
from tests.integration.conftest import skip_unless_live

from app.services.enrichment.base import (
    Source, ExtractionContext, TargetAttribute, AttributeValue,
)
from app.services.enrichment.pipeline import PipelineOrchestrator


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "products"


def load_all_fixtures() -> list[dict]:
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(FIXTURES_DIR.glob("*.json"))]


def value_matches(extracted, expected_entry) -> bool:
    """Compare extracted value with ground truth (with tolerance/synonyms)."""
    expected = expected_entry["value"]
    tolerance = expected_entry.get("tolerance")
    accept_values = expected_entry.get("accept_values", [])

    # Numeric with tolerance
    if tolerance is not None and isinstance(expected, (int, float)):
        try:
            return abs(float(extracted) - float(expected)) <= tolerance
        except (ValueError, TypeError):
            return False

    # Text — strict OR in accept_values
    extracted_str = str(extracted).strip().lower()
    if extracted_str == str(expected).strip().lower():
        return True
    for synonym in accept_values:
        if extracted_str == str(synonym).strip().lower():
            return True
    # Partial match for text — "iPhone 15 Pro" matches "Apple iPhone 15 Pro"
    if str(expected).strip().lower() in extracted_str:
        return True
    return False


@pytest.fixture(scope="module")
def orchestrator():
    return PipelineOrchestrator()


@skip_unless_live
@pytest.mark.asyncio
@pytest.mark.parametrize("fixture", load_all_fixtures(), ids=lambda f: f["id"])
async def test_real_world_product(fixture, orchestrator):
    """Прогон одного реального товара. Output — metrics."""
    p = fixture["product"]
    context = ExtractionContext(
        product_id=hash(fixture["id"]) & 0xFFFFFF,
        product_name=p["name"],
        product_description=p.get("description", "") or None,
        category_id=p["category_id"],
        category_path=p.get("category_path", []),
        brand=p.get("brand"),
        ean=p.get("ean"),
        source_urls=p.get("source_urls", []),
        image_urls=p.get("image_urls", []),
        max_cost_usd=0.30,
    )
    targets = [TargetAttribute(**t) for t in fixture["targets"]]
    ground_truth = fixture["ground_truth"]

    result = await orchestrator.enrich(context, targets)

    # Metrics
    by_id = {v.attribute_id: v for v in result}
    matched = 0
    mismatched = []
    missing = []
    source_count = defaultdict(int)

    for target in targets:
        gt = ground_truth.get(str(target.id))
        if gt is None:
            continue
        extracted = by_id.get(target.id)
        if extracted is None:
            missing.append((target.id, target.name, gt["value"]))
            continue
        source_count[extracted.source.value] += 1
        if value_matches(extracted.value, gt):
            matched += 1
        else:
            mismatched.append((target.id, target.name, extracted.value, gt["value"]))

    total = len(ground_truth)
    coverage_pct = 100 * (total - len(missing)) / total if total else 0
    accuracy_pct = 100 * matched / total if total else 0

    print(f"\n{'='*70}")
    print(f"Product: {fixture['id']}")
    print(f"Coverage:  {coverage_pct:.0f}% ({total - len(missing)}/{total} filled)")
    print(f"Accuracy:  {accuracy_pct:.0f}% ({matched}/{total} match ground truth)")
    print(f"LLM calls: {context.llm_calls_so_far}")
    print(f"Sources:   {dict(source_count)}")
    if mismatched:
        print(f"Mismatches:")
        for attr_id, name, got, expected in mismatched:
            print(f"  - {name}: got {got!r}, expected {expected!r}")
    if missing:
        print(f"Missing: {[name for _, name, _ in missing]}")
    print(f"{'='*70}")

    # Lenient assertions — лучше показать metrics чем fail
    assert coverage_pct >= 30, f"Coverage too low: {coverage_pct}%"


@skip_unless_live
@pytest.mark.asyncio
async def test_summary_all_products(orchestrator, capsys):
    """Aggregate report по всем товарам."""
    fixtures = load_all_fixtures()

    total_coverage = []
    total_accuracy = []
    total_calls = 0
    total_source_count = defaultdict(int)
    per_product_results = []

    for fixture in fixtures:
        p = fixture["product"]
        context = ExtractionContext(
            product_id=hash(fixture["id"]) & 0xFFFFFF,
            product_name=p["name"],
            product_description=p.get("description", "") or None,
            category_id=p["category_id"],
            category_path=p.get("category_path", []),
            brand=p.get("brand"),
            ean=p.get("ean"),
            source_urls=p.get("source_urls", []),
            image_urls=p.get("image_urls", []),
            max_cost_usd=0.30,
        )
        targets = [TargetAttribute(**t) for t in fixture["targets"]]
        gt = fixture["ground_truth"]

        result = await orchestrator.enrich(context, targets)
        by_id = {v.attribute_id: v for v in result}

        matched = sum(1 for t in targets if str(t.id) in gt and t.id in by_id
                      and value_matches(by_id[t.id].value, gt[str(t.id)]))
        filled = sum(1 for t in targets if str(t.id) in gt and t.id in by_id)
        total = len(gt)

        cov = 100 * filled / total if total else 0
        acc = 100 * matched / total if total else 0
        total_coverage.append(cov)
        total_accuracy.append(acc)
        total_calls += context.llm_calls_so_far

        for v in result:
            total_source_count[v.source.value] += 1

        per_product_results.append((fixture["id"], cov, acc, context.llm_calls_so_far))

    avg_cov = sum(total_coverage) / len(total_coverage)
    avg_acc = sum(total_accuracy) / len(total_accuracy)

    print(f"\n{'='*70}")
    print(f"REAL-WORLD ACCEPTANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Products tested:    {len(fixtures)}")
    print(f"Avg coverage:       {avg_cov:.0f}%")
    print(f"Avg accuracy:       {avg_acc:.0f}%")
    print(f"Total LLM calls:    {total_calls}")
    print(f"Avg calls/product:  {total_calls / len(fixtures):.1f}")
    print(f"Source distribution: {dict(total_source_count)}")
    print(f"\nPer-product:")
    for pid, cov, acc, calls in per_product_results:
        print(f"  {pid}: cov {cov:.0f}% acc {acc:.0f}% calls {calls}")
    print(f"{'='*70}")
