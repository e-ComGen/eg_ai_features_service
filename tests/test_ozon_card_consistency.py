"""Unit tests for the GENERAL brand/wrong-SKU consistency gate in OzonCardSource.

Covers _brand_conflict and its application inside _pick_best_match
(app/services/enrichment/sources/ozon_card_source.py). The gate exists because
Ozon search sometimes returns a card of a DIFFERENT brand than the query
(«Футболка Nike» → matched a «Shilla» card), whose attrs then overwrite correct
values. The gate applies a large score penalty so the wrong-brand card drops
below _BRAND_LINE_THRESHOLD → "skip" → its values are never copied.

Mirrors the style of the existing gender-guard (_gender_conflict + score penalty).
"""
import pytest

from app.services.enrichment.sources.ozon_card_source import (
    OzonCardSource,
    _brand_conflict,
    _BRAND_MISMATCH_PENALTY,
    _BRAND_LINE_THRESHOLD,
)

_classify_match = OzonCardSource._classify_match


# ---------------------------------------------------------------------------
# _brand_conflict — pure predicate
# ---------------------------------------------------------------------------

def test_brand_conflict_different_brand_in_card():
    """(a) query Nike, card titled Shilla (no Nike) → CONFLICT (reject)."""
    assert _brand_conflict(
        query_brand="Nike",
        query_name="Футболка мужская Nike Sportswear Club",
        card_title="Футболка мужская Shilla оверсайз",
    ) is True


def test_brand_conflict_genuine_same_brand_kept():
    """(b) query Nike, genuine Nike card → NO conflict (accept)."""
    assert _brand_conflict(
        query_brand="Nike",
        query_name="Футболка мужская Nike Sportswear Club",
        card_title="Футболка мужская Nike Dri-FIT спортивная",
    ) is False


def test_brand_conflict_same_brand_different_model_kept():
    """(d) same brand, different model → NO conflict (do not over-reject)."""
    assert _brand_conflict(
        query_brand="Nike",
        query_name="Футболка Nike Sportswear Club",
        card_title="Футболка Nike Air Max другая модель",
    ) is False


def test_brand_conflict_no_query_brand_not_rejected():
    """(c) query brand undeterminable (empty) → NOT rejected."""
    assert _brand_conflict(
        query_brand="",
        query_name="Футболка мужская оверсайз",
        card_title="Футболка мужская Shilla оверсайз",
    ) is False


def test_brand_conflict_cyrillic_query_brand_not_judged():
    """Cyrillic query brand (тип/описание-like) → not judged → NOT rejected."""
    assert _brand_conflict(
        query_brand="Спортмастер",
        query_name="Футболка Спортмастер",
        card_title="Футболка Shilla",
    ) is False


def test_brand_conflict_card_brand_undeterminable_not_rejected():
    """(c-2) card has no latin brand candidate → NOT rejected (no false positive)."""
    assert _brand_conflict(
        query_brand="Nike",
        query_name="Футболка Nike",
        card_title="Футболка мужская хлопковая оверсайз",
    ) is False


def test_brand_conflict_card_contains_query_brand_among_others_kept():
    """Card mentions query brand (plus other latin words) → NO conflict."""
    assert _brand_conflict(
        query_brand="Nike",
        query_name="Футболка Nike Sportswear",
        card_title="Футболка Nike Sportswear Club Dri-FIT",
    ) is False


def test_brand_conflict_multiword_brand():
    """Multi-word brand present as contiguous tokens → NO conflict."""
    assert _brand_conflict(
        query_brand="The North Face",
        query_name="Куртка The North Face Resolve",
        card_title="Куртка The North Face Triclimate",
    ) is False
    # ...and a different brand card → conflict.
    assert _brand_conflict(
        query_brand="The North Face",
        query_name="Куртка The North Face Resolve",
        card_title="Куртка Columbia Watertight",
    ) is True


def test_brand_conflict_champion_card_with_extra_latin_words_kept():
    """Regression: «Толстовка худи Champion Reverse Weave».

    "Champion" is both a brand AND a common English word, and the legit card
    title carries other latin tokens ("Reverse", "Weave"). The gate must NOT
    treat those generic latin words as a conflicting brand: as soon as the
    query brand "Champion" is present in the title, it returns False (kept).
    This guards against over-rejecting Champion's own card.
    """
    qn = "Толстовка худи Champion Reverse Weave"
    # Legit Champion card (brand present) + extra latin words → NOT rejected.
    assert _brand_conflict("Champion", qn, "Толстовка Champion Reverse Weave оверсайз") is False
    assert _brand_conflict("Champion", qn, "Худи Champion Reverse Weave мужское") is False
    # Brand absent but no OTHER latin brand present → still NOT rejected.
    assert _brand_conflict("Champion", qn, "Толстовка Reverse Weave оверсайз") is False
    # Genuinely different brand (Champion absent, Nike present) → rejected.
    assert _brand_conflict("Champion", qn, "Толстовка Nike Sportswear мужская") is True


def test_pick_best_match_champion_card_not_skipped():
    """Full path: a genuine Champion card with extra latin words clears the
    gate (not skipped) — the false-reject the brand gate was suspected of."""
    tiles = [{
        "title": "Толстовка худи Champion Reverse Weave оверсайз",
        "slug": "tolstovka-champion", "pid": "222",
    }]
    _tile, score = OzonCardSource._pick_best_match(
        query="Champion Reverse Weave",
        tiles=tiles,
        category_leaf="Толстовка",
        query_brand="Champion",
        query_name="Толстовка худи Champion Reverse Weave",
    )
    assert _classify_match(score) != "skip", (
        f"genuine Champion card scored {score:.1f}, must NOT be skipped by brand gate"
    )


# ---------------------------------------------------------------------------
# _pick_best_match — penalty wired through scoring → "skip"
# ---------------------------------------------------------------------------

def test_pick_best_match_wrong_brand_rejected_via_skip():
    """Wrong-brand card: penalty drops score below threshold → classified skip."""
    tiles = [{
        "title": "Футболка мужская Shilla оверсайз хлопок",
        "slug": "futbolka-shilla", "pid": "1623022699",
    }]
    _tile, score = OzonCardSource._pick_best_match(
        query="Футболка Nike Sportswear Club",
        tiles=tiles,
        category_leaf="Футболка",
        query_brand="Nike",
        query_name="Футболка мужская Nike Sportswear Club",
    )
    assert _classify_match(score) == "skip", (
        f"wrong-brand card scored {score:.1f}, expected skip (<{_BRAND_LINE_THRESHOLD})"
    )


def test_pick_best_match_genuine_brand_not_penalized():
    """Genuine Nike card keeps a copy-worthy score (not skipped by brand gate)."""
    tiles = [{
        "title": "Футболка мужская Nike Sportswear Club спортивная",
        "slug": "futbolka-nike", "pid": "111",
    }]
    _tile, score = OzonCardSource._pick_best_match(
        query="Футболка Nike Sportswear Club",
        tiles=tiles,
        category_leaf="Футболка",
        query_brand="Nike",
        query_name="Футболка мужская Nike Sportswear Club",
    )
    assert _classify_match(score) != "skip", (
        f"genuine brand card scored {score:.1f}, must NOT be skipped"
    )


def test_pick_best_match_penalty_flips_passing_score_to_skip():
    """A title matching the query EXCEPT the brand clears the threshold without
    brand judgement, but the -60 brand penalty flips it to "skip" → values are
    never copied. Penalty magnitude is exactly _BRAND_MISMATCH_PENALTY.
    """
    tiles = [{
        "title": "Футболка Shilla Sportswear Club",
        "slug": "s", "pid": "1",
    }]
    _tw, with_brand = OzonCardSource._pick_best_match(
        query="Футболка Nike Sportswear Club", tiles=tiles,
        category_leaf="Футболка", query_brand="Nike",
        query_name="Футболка Nike Sportswear Club",
    )
    _t2, no_brand = OzonCardSource._pick_best_match(
        query="Футболка Nike Sportswear Club", tiles=tiles,
        category_leaf="Футболка", query_brand="", query_name="Футболка",
    )
    # Baseline (no brand judgement) clears the threshold; the gate flips it to skip.
    assert _classify_match(no_brand) != "skip"
    assert _classify_match(with_brand) == "skip"
    assert no_brand - with_brand == pytest.approx(_BRAND_MISMATCH_PENALTY, abs=0.01)


# ---------------------------------------------------------------------------
# best-of-top-N: SSR ordering jitter must not starve the exact card
# ---------------------------------------------------------------------------

def test_pick_best_match_selects_exact_card_not_first_tile():
    """Ozon SSR tile ORDERING jitters run-to-run: the exact card is present every
    run but not always ranked #1. _pick_best_match must score ALL top-N tiles and
    pick the HIGHEST-scoring one, so the exact card (high score) wins even when a
    weak decoy is ranked first.

    Setup: a low-score decoy is tile #1, the exact-match card is tile #3.
    The source must select the exact card (best of top-N), not the decoy.
    """
    query = "Толстовка женская Adidas Originals Trefoil"
    tiles = [
        # #1 decoy: same type/brand-family but clearly a different product (low score)
        {"title": "Носки Adidas комплект 3 пары", "slug": "noski-adidas", "pid": "1"},
        # #2 another weak neighbour
        {"title": "Футболка мужская Adidas Performance", "slug": "futbolka-adidas", "pid": "2"},
        # #3 the EXACT card (high score) — ranked third due to SSR jitter
        {
            "title": "Толстовка женская Adidas Originals Trefoil",
            "slug": "tolstovka-adidas-trefoil", "pid": "3",
        },
        {"title": "Шапка Adidas зимняя", "slug": "shapka-adidas", "pid": "4"},
    ]
    best_tile, score = OzonCardSource._pick_best_match(
        query=query,
        tiles=tiles,
        category_leaf="Толстовка",
        query_brand="Adidas",
        query_name=query,
    )
    assert best_tile is not None
    assert best_tile["pid"] == "3", (
        f"expected the exact card (pid=3) to win best-of-top-N, got pid="
        f"{best_tile.get('pid')} (title={best_tile.get('title')!r}, score={score:.1f})"
    )
    assert _classify_match(score) != "skip", (
        f"exact card scored {score:.1f}, must clear the threshold"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
