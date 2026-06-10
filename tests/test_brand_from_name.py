"""Unit tests for the GENERAL post-merge brand-from-name resolver.

Covers _apply_brand_from_name in app/services/enrichment/pipeline.py — the product
NAME is authoritative for brand. If EXACTLY ONE allowed-enum brand appears in the
name as a contiguous token-run (>=3 chars), an empty brand field is filled and a
garbage (wrong allowed-enum) brand is OVERRIDDEN. Zero or ambiguous matches leave
the field untouched (anti-garbage guard). No per-brand hardcode.
"""
import pytest

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.pipeline import (
    _apply_brand_from_name,
    _is_brand_target_name,
    _BRAND_TARGET_ATTR_ID,
)

_BRAND_ID = _BRAND_TARGET_ATTR_ID  # 31


def _ctx(name: str, category_path=None) -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name=name,
        category_id=1,
        category_path=category_path or [],
    )


def _brand_target(allowed, *, id=_BRAND_ID, name="Бренд") -> TargetAttribute:
    return TargetAttribute(id=id, name=name, type="enum", allowed_values=allowed)


def _brand_value(value, source=Source.WEB_SEARCH) -> AttributeValue:
    return AttributeValue(
        attribute_id=_BRAND_ID, value=value, confidence=0.85, source=source
    )


def _val_for(values, attr_id=_BRAND_ID):
    return next((v for v in values if v.attribute_id == attr_id), None)


# ── helper-level ──────────────────────────────────────────────────────────────

def test_is_brand_target_name():
    assert _is_brand_target_name("Бренд")
    assert _is_brand_target_name("Бренд в одежде")
    assert _is_brand_target_name("Brand")
    assert _is_brand_target_name("Торговая марка")
    assert not _is_brand_target_name("Цвет")
    assert not _is_brand_target_name("Пол")


# ── (a) unfilled → filled ─────────────────────────────────────────────────────

def test_unfilled_filled_from_name():
    """(a) name contains 'Champion', brand UNFILLED → filled 'Champion'."""
    ctx = _ctx("Толстовка худи Champion Reverse Weave")
    target = _brand_target(["Nike", "Champion", "Adidas", "Reebok"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Champion"
    assert v.source == Source.DESCRIPTION
    assert v.evidence == "brand_from_name"


# ── (b) garbage → overridden ──────────────────────────────────────────────────

def test_garbage_overridden_from_name():
    """(b) name 'Levi's', brand wrongly 'HUGO' → OVERRIDDEN to "Levi's"."""
    ctx = _ctx("Джинсы мужские Levi's 501")
    target = _brand_target(["HUGO", "Levi's", "Wrangler", "Lee"])
    out = _apply_brand_from_name([_brand_value("HUGO")], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Levi's"
    assert v.evidence == "brand_from_name"
    assert v.value_id is None  # re-resolved by strategy downstream


# ── (c) no allowed brand in name → untouched ──────────────────────────────────

def test_no_brand_in_name_untouched_unfilled():
    """(c) no allowed brand present → unfilled stays unfilled (no guess)."""
    ctx = _ctx("Толстовка худи мужская оверсайз чёрная")
    target = _brand_target(["Nike", "Champion", "Adidas"])
    out = _apply_brand_from_name([], [target], ctx)
    assert _val_for(out) is None


def test_no_brand_in_name_keeps_existing():
    """(c') no allowed brand in name → existing value NOT touched."""
    ctx = _ctx("Толстовка худи мужская оверсайз чёрная")
    target = _brand_target(["Nike", "Champion", "HUGO"])
    out = _apply_brand_from_name([_brand_value("HUGO")], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "HUGO"  # untouched, no name evidence


# ── (d) two brands in name → untouched (ambiguous) ────────────────────────────

def test_two_brands_leftmost_wins_unfilled():
    """(d) TWO allowed brands in name → leftmost (Nike) wins (PART 2 tiebreak).

    RU marketplace titles list the real brand first; leftmost occurrence resolves
    the otherwise-ambiguous pair to a single brand rather than skipping.
    """
    ctx = _ctx("Кроссовки Nike x Adidas коллаборация")
    target = _brand_target(["Nike", "Adidas", "Puma"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Nike"  # leftmost wins


def test_two_brands_leftmost_overrides_existing():
    ctx = _ctx("Кроссовки Nike x Adidas коллаборация")
    target = _brand_target(["Nike", "Adidas", "HUGO"])
    out = _apply_brand_from_name([_brand_value("HUGO")], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Nike"  # leftmost overrides garbage


# ── (e) multi-word brand ──────────────────────────────────────────────────────

def test_multiword_brand_matched():
    """(e) multi-word brand 'The North Face' matched as contiguous tokens."""
    ctx = _ctx("Куртка The North Face Resolve мужская")
    target = _brand_target(["Columbia", "The North Face", "Patagonia"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "The North Face"


def test_multiword_brand_with_punctuation():
    """Punctuation-bearing brand 'be quiet!' matched via tokenization."""
    ctx = _ctx("Блок питания be quiet! Pure Power 12")
    target = _brand_target(["Corsair", "be quiet!", "Seasonic"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "be quiet!"


def test_apostrophe_brand_matched():
    """Apostrophe brand 'Levi's' matched even though name has it as 'Levi's'."""
    ctx = _ctx("Джинсы Levi's классические")
    target = _brand_target(["Levi's", "Lee", "Wrangler"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Levi's"


# ── (f) short brand not false-matched ─────────────────────────────────────────

def test_short_two_char_brand_not_false_matched():
    """(f) 2-char brand 'GG' must not false-match on incidental letters."""
    ctx = _ctx("Сумка GG женская кожаная")  # 'GG' present but <3 chars → ignored
    target = _brand_target(["GG", "Gucci", "Prada"])
    out = _apply_brand_from_name([], [target], ctx)
    # 'GG' is too short to be trusted; 'Gucci'/'Prada' absent → nothing filled.
    assert _val_for(out) is None


# ── invariants / scope ────────────────────────────────────────────────────────

def test_never_writes_brand_absent_from_allowed():
    """B is always chosen from allowed_values AND present in name."""
    ctx = _ctx("Джинсы Levi's 501")
    target = _brand_target(["HUGO", "Lee"])  # Levi's NOT in allowed
    out = _apply_brand_from_name([], [target], ctx)
    # Levi's in name but not allowed; HUGO/Lee not in name → nothing.
    assert _val_for(out) is None


def test_free_text_brand_target_skipped_without_dict():
    """Brand target with empty allowed_values AND no dict resolver → skipped.

    Without a brand_options_fn there is no list to match against, so the resolver
    leaves the field untouched (no guess).
    """
    ctx = _ctx("Джинсы Levi's 501")
    target = TargetAttribute(id=_BRAND_ID, name="Бренд", type="text")
    out = _apply_brand_from_name([], [target], ctx)
    assert _val_for(out) is None  # no list → nothing to match


# ── REGRESSION: truncated «Бренд» enum (empty allowed_values + full dict) ──────

def test_truncated_brand_resolved_from_full_dict_unfilled():
    """REGRESSION: «Бренд» is a huge truncated enum → target.allowed_values is [].

    The full brand list is supplied via brand_options_fn (the dict path). With the
    OLD code (which required t.allowed_values) this was skipped and Бренд stayed
    empty. Now it MUST resolve 'Champion' from the name against the full dict list.
    """
    ctx = _ctx("Толстовка худи Champion Reverse Weave")
    target = _brand_target([])  # truncated enum: allowed_values NOT carried
    full_dict = ["Nike", "Champion", "Adidas", "Reebok", "Puma"]
    out = _apply_brand_from_name(
        [], [target], ctx, brand_options_fn=lambda attr_id: full_dict
    )
    v = _val_for(out)
    assert v is not None and v.value == "Champion"
    assert v.source == Source.DESCRIPTION
    assert v.evidence == "brand_from_name"


def test_truncated_brand_overrides_garbage_from_full_dict():
    """REGRESSION: garbage (LEGO/HUGO) on a truncated «Бренд» enum is overridden.

    allowed_values=[] (truncated); full brand list via dict. Name says Levi's,
    field wrongly 'HUGO' → overridden to 'Levi's'.
    """
    ctx = _ctx("Джинсы мужские Levi's 501")
    target = _brand_target([])
    full_dict = ["HUGO", "Levi's", "Wrangler", "Lee", "LEGO"]
    out = _apply_brand_from_name(
        [_brand_value("HUGO")], [target], ctx,
        brand_options_fn=lambda attr_id: full_dict,
    )
    v = _val_for(out)
    assert v is not None and v.value == "Levi's"
    assert v.evidence == "brand_from_name"


def test_truncated_brand_dict_empty_skips():
    """allowed_values=[] AND dict returns [] → nothing to match, untouched."""
    ctx = _ctx("Джинсы Levi's 501")
    target = _brand_target([])
    out = _apply_brand_from_name(
        [], [target], ctx, brand_options_fn=lambda attr_id: []
    )
    assert _val_for(out) is None


def test_truncated_brand_two_brands_leftmost_from_dict():
    """Two dict brands in name → leftmost (Nike) wins via PART 2 tiebreak."""
    ctx = _ctx("Кроссовки Nike x Adidas коллаборация")
    target = _brand_target([])
    full_dict = ["Nike", "Adidas", "Puma"]
    out = _apply_brand_from_name(
        [_brand_value("HUGO")], [target], ctx,
        brand_options_fn=lambda attr_id: full_dict,
    )
    v = _val_for(out)
    assert v is not None and v.value == "Nike"  # leftmost wins


def test_truncated_brand_multiword_from_dict():
    """Multi-word brand resolved from full dict list against truncated enum."""
    ctx = _ctx("Куртка The North Face Resolve мужская")
    target = _brand_target([])
    full_dict = ["Columbia", "The North Face", "Patagonia"]
    out = _apply_brand_from_name(
        [], [target], ctx, brand_options_fn=lambda attr_id: full_dict
    )
    v = _val_for(out)
    assert v is not None and v.value == "The North Face"


def test_allowed_values_preferred_over_dict_when_present():
    """Short allowed_values (≤100) → dict fn consulted and used if it has more options.

    Change: brand targets with short allowed_values (e.g. eval-truncated [:50] slice)
    now consult brand_options_fn for the full dict. If the full dict has more options,
    it replaces the short slice. The final brand is still chosen by name-match.
    Champion matches in both the short list and the full dict → result is Champion.
    """
    ctx = _ctx("Толстовка Champion Reverse Weave")
    target = _brand_target(["Champion", "Nike"])

    # brand_options_fn returns the same small set → no upgrade; short list is used.
    # Still fills Champion correctly from the name.
    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda _attr_id: ["Champion", "Nike"],
    )
    v = _val_for(out)
    assert v is not None and v.value == "Champion"


def test_large_allowed_values_not_replaced_by_dict():
    """Large allowed_values (>100 items) are used as-is WITHOUT consulting the dict.

    A target with >100 allowed_values is treated as a real full enum (not truncated).
    brand_options_fn (which would raise here) must not be called.
    """
    ctx = _ctx("Толстовка Champion Reverse Weave")
    # 101 items → above _BRAND_MAX_INLINE threshold
    big_allowed = ["Champion"] + ["Brand" + str(i) for i in range(100)]
    target = _brand_target(big_allowed)

    def _boom(_attr_id):
        raise AssertionError("brand_options_fn must not be called with large allowed_values")

    out = _apply_brand_from_name([], [target], ctx, brand_options_fn=_boom)
    v = _val_for(out)
    assert v is not None and v.value == "Champion"


# ── value_id attach (drain-C fix) ─────────────────────────────────────────────

def test_brand_from_name_attaches_value_id_unfilled():
    """FIX: brand filled from name carries the dict value_id (not None).

    The truncated «Бренд» enum dropped brand-from-name values because value_id
    stayed None (drain C). brand_id_fn supplies {brand: id}; on an EXACT match the
    emitted AttributeValue MUST carry both value='Nike' AND value_id=12345.
    """
    ctx = _ctx("Кроссовки Nike Air мужские")
    target = _brand_target([])  # truncated enum
    full_dict = ["Nike", "Adidas", "Puma"]
    id_map = {"Nike": 12345, "Adidas": 999}
    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda attr_id: full_dict,
        brand_id_fn=lambda attr_id: id_map,
    )
    v = _val_for(out)
    assert v is not None and v.value == "Nike"
    assert v.value_id == 12345  # survives drain-C (was None before the fix)


def test_brand_from_name_attaches_value_id_overwrite():
    """Overriding a garbage brand also attaches the chosen brand's dict value_id."""
    ctx = _ctx("Джинсы мужские Levi's 501")
    target = _brand_target([])
    full_dict = ["HUGO", "Levi's", "Wrangler"]
    id_map = {"Levi's": 555, "HUGO": 111}
    out = _apply_brand_from_name(
        [_brand_value("HUGO")], [target], ctx,
        brand_options_fn=lambda attr_id: full_dict,
        brand_id_fn=lambda attr_id: id_map,
    )
    v = _val_for(out)
    assert v is not None and v.value == "Levi's"
    assert v.value_id == 555


def test_brand_value_id_exact_only_no_fuzzy():
    """No EXACT key in the id-map → value_id stays None (no fuzzy/invented ids).

    Owner is sensitive to wrong ids: a brand present in the options list but absent
    from the {brand: id} map must fill the string with value_id=None, not guess.
    """
    ctx = _ctx("Кроссовки Nike Air")
    target = _brand_target([])
    full_dict = ["Nike", "Adidas"]
    id_map = {"Adidas": 999}  # Nike intentionally missing
    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda attr_id: full_dict,
        brand_id_fn=lambda attr_id: id_map,
    )
    v = _val_for(out)
    assert v is not None and v.value == "Nike"
    assert v.value_id is None


def test_brand_value_id_case_yo_insensitive():
    """Exact match is case + ё/е insensitive (dict key casing differs from name)."""
    ctx = _ctx("Пуховик Тёма зимний")
    target = _brand_target([])
    full_dict = ["Тема", "Other"]
    id_map = {"Тема": 7}  # name has ё, dict key has е
    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda attr_id: full_dict,
        brand_id_fn=lambda attr_id: id_map,
    )
    v = _val_for(out)
    assert v is not None and v.value == "Тема"
    assert v.value_id == 7


def test_brand_id_fn_failure_does_not_crash():
    """brand_id_fn raising → resolver still fills the brand, value_id None."""
    ctx = _ctx("Кроссовки Nike Air")
    target = _brand_target([])
    full_dict = ["Nike"]

    def _boom(_attr_id):
        raise RuntimeError("dict unavailable")

    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda attr_id: full_dict,
        brand_id_fn=_boom,
    )
    v = _val_for(out)
    assert v is not None and v.value == "Nike"
    assert v.value_id is None


def test_detect_by_id_when_name_differs():
    """Target with id==31 but odd name still detected as brand target."""
    ctx = _ctx("Толстовка Champion Reverse Weave")
    target = _brand_target(["Champion", "Nike"], name="Производитель")
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Champion"


def test_already_correct_brand_left_intact():
    """Filled with the right brand already → kept (no needless override)."""
    ctx = _ctx("Толстовка Champion Reverse Weave")
    target = _brand_target(["Champion", "Nike"])
    existing = _brand_value("Champion", source=Source.OZON_CARD)
    out = _apply_brand_from_name([existing], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Champion"
    assert v.source == Source.OZON_CARD  # untouched, not rewritten


def test_non_brand_attribute_untouched():
    ctx = _ctx("Толстовка Champion Reverse Weave")
    color = TargetAttribute(id=10, name="Цвет", type="enum", allowed_values=["Чёрный"])
    cand = [AttributeValue(attribute_id=10, value="Чёрный", confidence=0.9, source=Source.WEB_SEARCH)]
    out = _apply_brand_from_name(cand, [color], ctx)
    assert len(out) == 1 and out[0].value == "Чёрный"


def test_yo_normalization_in_brand():
    """ё/е equivalence in brand vs name."""
    ctx = _ctx("Конфеты Алёнка молочные")
    target = _brand_target(["Аленка", "Рот Фронт"])  # 'Аленка' with е
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Аленка"


# ── APPAREL DISAMBIGUATION: generic enum «Бренд в одежде и обуви» (id 31) ─────
# That enum embeds GENERIC words as fake "brands" (футболка, Мужская, NORTH,
# Original...). Without filtering, «Футболка мужская Nike» matched
# {Nike, футболка, Мужская} = 3 → ambiguous SKIP → REQUIRED brand stayed empty
# and garbage from other sources survived. The disambiguator must collapse
# containment and drop gender/type noise so EXACTLY ONE real brand remains.


def test_apparel_gender_and_type_noise_filtered_to_single():
    """(1) {Nike, футболка, Мужская} → noise dropped → resolves to Nike."""
    ctx = _ctx("Футболка мужская Nike Sportswear Club", category_path=["Одежда", "Футболки"])
    target = _brand_target(["Nike", "футболка", "Мужская", "Adidas"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Nike"
    assert v.evidence == "brand_from_name"


def test_apparel_containment_collapse_plus_gender():
    """(2) {The North Face, NORTH, Мужская} → NORTH⊂The North Face + gender drop."""
    ctx = _ctx("Куртка мужская The North Face Resolve 2", category_path=["Одежда", "Куртки"])
    target = _brand_target(["The North Face", "NORTH", "Мужская", "куртка"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "The North Face"


def test_apparel_overwrite_garbage_after_noise_filter():
    """(3) garbage LEGO + 'Джинсы мужские Levi's 501' {Levi's, Мужские} → Levi's."""
    ctx = _ctx("Джинсы мужские Levi's 501", category_path=["Одежда", "Джинсы"])
    target = _brand_target(["Levi's", "Мужские", "джинсы", "LEGO"])
    out = _apply_brand_from_name([_brand_value("LEGO")], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Levi's"
    assert v.evidence == "brand_from_name"
    assert v.value_id is None


def test_apparel_two_real_brands_leftmost_wins():
    """(4) TWO real brands remain after noise filter → leftmost (Nike) wins.

    PART 2: after gender/type noise is dropped, a still-ambiguous pair is resolved
    by leftmost occurrence in the title (the real brand leads the descriptive part).
    """
    ctx = _ctx("Футболка мужская Nike x Adidas", category_path=["Одежда", "Футболки"])
    target = _brand_target(["Nike", "Adidas", "футболка", "Мужская"])
    out = _apply_brand_from_name([_brand_value("HUGO")], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Nike"  # leftmost after noise filter


def test_apparel_regression_clean_single_token_fills():
    """(5a) clean 'Кроссовки Adidas Ultraboost 22' still fills Adidas."""
    ctx = _ctx("Кроссовки Adidas Ultraboost 22", category_path=["Обувь", "Кроссовки"])
    target = _brand_target(["Adidas", "Nike", "кроссовки"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Adidas"


def test_apparel_regression_no_brand_stays_empty():
    """(5b) no real brand in title → field stays empty (only noise present)."""
    ctx = _ctx("Футболка мужская оверсайз чёрная", category_path=["Одежда", "Футболки"])
    target = _brand_target(["футболка", "Мужская", "Nike", "Adidas"])
    out = _apply_brand_from_name([], [target], ctx)
    assert _val_for(out) is None


def test_disambiguation_does_not_break_non_apparel_no_category_path():
    """No category_path (type_words empty) → behaves as before for clean cases."""
    ctx = _ctx("Толстовка худи Champion Reverse Weave")
    target = _brand_target(["Nike", "Champion", "Adidas"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Champion"


# ── PART 1: brand-source-guard (brand is IDENTITY, never guessed) ─────────────
# vision/web_search/llm_knowledge/competitor_rag GUESS the brand (HUGO/LEGO/
# Великобритания on Nike/Levi's/Adidas). Those candidates are DROPPED before
# merge; authoritative cards/description/icecat are KEPT.

from app.services.enrichment.pipeline import (
    _apply_brand_source_guard,
    _BRAND_GUESS_SOURCES,
)


@pytest.mark.parametrize(
    "guess_src, garbage",
    [
        (Source.VISION, "HUGO"),
        (Source.WEB_SEARCH, "LEGO"),
        (Source.LLM_KNOWLEDGE, "Великобритания"),
        (Source.COMPETITOR_RAG, "Reebok"),
    ],
)
def test_guard_drops_brand_from_guess_sources(guess_src, garbage):
    """(1) Brand value from a guess source → DROPPED (HUGO/LEGO/Великобритания gone)."""
    target = _brand_target(["Nike", "The North Face"])
    vals = [_brand_value(garbage, source=guess_src)]
    out = _apply_brand_source_guard(vals, [target])
    assert _val_for(out) is None  # guess-source brand dropped


@pytest.mark.parametrize("card_src", [Source.OZON_CARD, Source.WB_CARD])
def test_guard_keeps_brand_from_cards(card_src):
    """(2) Brand value from ozon_card/wb_card → KEPT (authoritative identity)."""
    target = _brand_target(["The North Face", "Nike"])
    vals = [_brand_value("The North Face", source=card_src)]
    out = _apply_brand_source_guard(vals, [target])
    v = _val_for(out)
    assert v is not None and v.value == "The North Face"
    assert v.source == card_src


@pytest.mark.parametrize(
    "auth_src",
    [
        Source.DESCRIPTION,
        Source.ICECAT,
        Source.PDF_DATASHEET,
    ],
)
def test_guard_keeps_brand_from_other_authoritative(auth_src):
    """description/icecat/pdf_datasheet are authoritative → KEPT.

    (tnved emits Source.LLM_KNOWLEDGE but never produces brand candidates, so it
    is not a real source of brand values; no Source.TNVED enum exists.)
    """
    target = _brand_target(["Nike"])
    out = _apply_brand_source_guard([_brand_value("Nike", source=auth_src)], [target])
    v = _val_for(out)
    assert v is not None and v.value == "Nike"


def test_guard_set_membership():
    """The guess set is exactly the four identity-unsafe sources."""
    assert _BRAND_GUESS_SOURCES == {
        Source.VISION, Source.WEB_SEARCH, Source.LLM_KNOWLEDGE, Source.COMPETITOR_RAG,
    }


def test_guard_leaves_non_brand_attrs_alone():
    """Guard only touches brand targets; other attrs from guess sources pass."""
    color = TargetAttribute(id=10, name="Цвет", type="enum", allowed_values=["Чёрный"])
    cand = [AttributeValue(attribute_id=10, value="Чёрный", confidence=0.9,
                           source=Source.WEB_SEARCH)]
    out = _apply_brand_source_guard(cand, [color])
    assert len(out) == 1 and out[0].value == "Чёрный"


# ── PART 2: leftmost-occurrence recovery on real traced products ──────────────


def test_part2_nike_sportswear_club_leftmost():
    """(3) 'Футболка мужская Nike Sportswear Club' → Nike (leftmost after noise)."""
    ctx = _ctx("Футболка мужская Nike Sportswear Club", category_path=["Одежда", "Футболки"])
    target = _brand_target(["Nike", "Sportswear", "Club", "Мужская", "футболка"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Nike"
    assert v.evidence == "brand_from_name"


def test_part2_levis_501_original():
    """(4) 'Джинсы мужские Levis 501 Original' → Levis."""
    ctx = _ctx("Джинсы мужские Levis 501 Original", category_path=["Одежда", "Джинсы"])
    target = _brand_target(["Levis", "Original", "Мужские", "джинсы"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Levis"


def test_part2_adidas_ultraboost():
    """(4) 'Кроссовки Adidas Ultraboost 22' → Adidas."""
    ctx = _ctx("Кроссовки Adidas Ultraboost 22", category_path=["Обувь", "Кроссовки"])
    target = _brand_target(["Adidas", "Ultraboost", "кроссовки"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Adidas"


def test_part2_wrangler_texas():
    """(4) 'Джинсы Wrangler Texas' → Wrangler (Texas is a model, leftmost wins)."""
    ctx = _ctx("Джинсы Wrangler Texas", category_path=["Одежда", "Джинсы"])
    target = _brand_target(["Wrangler", "Texas", "джинсы"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Wrangler"


def test_part2_no_brand_only_guess_values_ends_empty():
    """(5) No brand in title + only guess-source garbage → brand ends EMPTY (honest).

    Combines PART 1 (guard drops the guess value) + PART 2 (no name match → no fill).
    """
    ctx = _ctx("Футболка мужская оверсайз чёрная", category_path=["Одежда", "Футболки"])
    target = _brand_target(["Nike", "Adidas", "футболка", "Мужская"])
    # guess-source garbage that the guard must drop, then brand-from-name finds nothing
    guarded = _apply_brand_source_guard([_brand_value("LEGO", source=Source.WEB_SEARCH)],
                                        [target])
    out = _apply_brand_from_name(guarded, [target], ctx)
    assert _val_for(out) is None  # empty, not garbage


# ── REGRESSION FIX 1: apostrophe-insensitive brand matching (Levi's ↔ Levis) ──
# "Levi's" in dict splits to ['levi','s'] which never matched title token 'levis'.
# Canon-fallback: strip apostrophes before tokenising → both collapse to 'levis'.


def test_levis_apostrophe_insensitive_match_unfilled():
    """FIX1: 'Levis' in title matches dict brand 'Levi's' (apostrophe-insensitive)."""
    ctx = _ctx("Джинсы мужские Levis 501 Original", category_path=["Одежда", "Джинсы"])
    target = _brand_target([], name="Бренд")
    id_map = {"Levi's": 971812807, "ORIGINAL": 971939550}
    out = _apply_brand_from_name(
        [_brand_value("ORIGINAL")],
        [target],
        ctx,
        brand_options_fn=lambda aid: ["Levi's", "ORIGINAL", "Мужские", "джинсы"],
        brand_id_fn=lambda aid: id_map,
    )
    v = _val_for(out)
    assert v is not None and v.value == "Levi's", f"expected Levi's, got {v}"
    assert v.value_id == 971812807, f"expected vid 971812807, got {v.value_id}"
    assert v.evidence == "brand_from_name"


def test_levis_apostrophe_insensitive_value_id_correct():
    """FIX1: value_id must be 971812807 (Levi's), NOT 971939550 (ORIGINAL)."""
    ctx = _ctx("Джинсы мужские Levis 501 Original", category_path=["Одежда", "Джинсы"])
    target = _brand_target(["Levi's", "ORIGINAL"])
    id_map = {"Levi's": 971812807, "ORIGINAL": 971939550}
    out = _apply_brand_from_name(
        [], [target], ctx, brand_id_fn=lambda aid: id_map
    )
    v = _val_for(out)
    assert v is not None and v.value == "Levi's"
    assert v.value_id == 971812807


# ── REGRESSION FIX 2: Russian descriptor-adjective masquerading as brand ──────
# "Спортивные" is a real dict brand token sitting before "Nike" in the title.
# The adjective morphology filter drops single Cyrillic tokens with adj. endings.


def test_adjective_noise_dropped_nike_wins():
    """FIX2: 'Спортивные' (adj ending -ые) dropped → Nike wins correctly."""
    ctx = _ctx("Шорты мужские спортивные Nike Dri-FIT", category_path=["Одежда", "Шорты"])
    target = _brand_target([], name="Бренд")
    id_map = {"Nike": 971812808, "Спортивные": 971939550}
    out = _apply_brand_from_name(
        [_brand_value("Спортивные")],
        [target],
        ctx,
        brand_options_fn=lambda aid: ["Спортивные", "Nike", "Мужские", "шорты"],
        brand_id_fn=lambda aid: id_map,
    )
    v = _val_for(out)
    assert v is not None and v.value == "Nike", f"expected Nike, got {v}"
    assert v.value_id == 971812808
    assert v.evidence == "brand_from_name"


def test_all_adjective_title_returns_empty():
    """FIX2 safety: if ONLY adjective-looking tokens match, return EMPTY not garbage.

    Title: 'Чёрные спортивные прямые' — all three are adjectives by morphology.
    No real brand → brand field must stay EMPTY (empty > wrong descriptor).
    """
    ctx = _ctx("Чёрные спортивные прямые", category_path=["Одежда", "Джинсы"])
    # All three happen to be in the brand dict as fake brands (common in the enum)
    target = _brand_target(["Чёрные", "Спортивные", "Прямые"])
    out = _apply_brand_from_name([], [target], ctx)
    assert _val_for(out) is None, "all-adjective title must yield empty, not a descriptor adjective"


# ── NEEDLE fallback: context.brand present in name when dict is empty ─────────
# The static dict returns only first ~5000 brands; many real brands are absent.
# When the dict (options list) is empty AND context.brand is set AND present in
# the product name — accept it directly (same _brand_in_name token rules).
# value_id stays None and is resolved later by resolve_value_ids_async (search API).


def test_needle_fallback_fills_brand_when_dict_empty():
    """NEEDLE: dict empty + context.brand in name → filled from context.brand."""
    ctx = ExtractionContext(
        product_id=1,
        product_name="Кроссовки Nike Air мужские",
        category_id=1,
        brand="Nike",  # known from product metadata
    )
    target = TargetAttribute(id=_BRAND_ID, name="Бренд", type="enum", allowed_values=[])
    # brand_options_fn returns [] (truncated dict, brand not in first 5000)
    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda attr_id: [],
    )
    v = _val_for(out)
    assert v is not None and v.value == "Nike"
    assert v.source == Source.DESCRIPTION
    assert v.evidence == "brand_from_name"
    assert v.value_id is None  # resolved later by async path


def test_needle_fallback_no_options_fn_uses_context_brand():
    """NEEDLE: no brand_options_fn (no dict) + context.brand in name → filled."""
    ctx = ExtractionContext(
        product_id=1,
        product_name="Джинсы Wrangler Texas",
        category_id=1,
        brand="Wrangler",
    )
    target = TargetAttribute(id=_BRAND_ID, name="Бренд", type="enum", allowed_values=[])
    out = _apply_brand_from_name([], [target], ctx)  # no brand_options_fn at all
    v = _val_for(out)
    assert v is not None and v.value == "Wrangler"
    assert v.evidence == "brand_from_name"


def test_needle_fallback_brand_not_in_name_stays_empty():
    """NEEDLE: context.brand is set but NOT in the product name → no fill."""
    ctx = ExtractionContext(
        product_id=1,
        product_name="Кроссовки мужские спортивные",
        category_id=1,
        brand="Nike",  # brand set but absent from name
    )
    target = TargetAttribute(id=_BRAND_ID, name="Бренд", type="enum", allowed_values=[])
    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda attr_id: [],
    )
    assert _val_for(out) is None


def test_needle_fallback_skipped_when_dict_has_options():
    """NEEDLE not used when dict returns non-empty options (dict path takes priority)."""
    ctx = ExtractionContext(
        product_id=1,
        product_name="Кроссовки Adidas Ultraboost",
        category_id=1,
        brand="Nike",  # context.brand differs from name brand
    )
    target = TargetAttribute(id=_BRAND_ID, name="Бренд", type="enum", allowed_values=[])
    # Dict has options — Adidas matched, NOT Nike (context.brand)
    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda attr_id: ["Adidas", "Puma", "Reebok"],
    )
    v = _val_for(out)
    assert v is not None and v.value == "Adidas"  # dict path wins, not needle


def test_needle_fallback_none_context_brand_skips():
    """NEEDLE: context.brand is None → no fill (dict also empty)."""
    ctx = ExtractionContext(
        product_id=1,
        product_name="Кроссовки Nike мужские",
        category_id=1,
        brand=None,  # brand unknown
    )
    target = TargetAttribute(id=_BRAND_ID, name="Бренд", type="enum", allowed_values=[])
    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda attr_id: [],
    )
    assert _val_for(out) is None


# ── NEEDLE FIX: category-noun guard (MUD prevention) ─────────────────────────
# When dict is empty, the needle fallback uses context.brand. If context.brand
# is a category noun (e.g. eval heuristic takes "колонка" from "Умная колонка"),
# it must be REJECTED — a category type word is not a brand.


def test_needle_category_noun_rejected_kolonka():
    """NEEDLE: 'колонка' (2nd word of 'Умная колонка') is a category noun → EMPTY."""
    ctx = ExtractionContext(
        product_id=1,
        product_name="Умная колонка Яндекс Станция Мини 2",
        category_id=1,
        category_path=["Электроника", "Умная колонка"],
        brand="колонка",  # eval heuristic: words[1] of product name
    )
    target = TargetAttribute(id=85, name="Бренд", type="enum", allowed_values=[])
    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda attr_id: [],
    )
    assert _val_for(out, 85) is None  # rejected: category noun


def test_needle_category_noun_rejected_kniga():
    """NEEDLE: 'книга' from 'Электронная книга PocketBook' → EMPTY (category noun)."""
    ctx = ExtractionContext(
        product_id=1,
        product_name="Электронная книга PocketBook 629 Verse",
        category_id=1,
        category_path=["Электроника", "Электронная книга"],
        brand="книга",  # eval heuristic: words[1]
    )
    target = TargetAttribute(id=85, name="Бренд", type="enum", allowed_values=[])
    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda attr_id: [],
    )
    assert _val_for(out, 85) is None


def test_needle_category_noun_rejected_mashina():
    """NEEDLE: 'машина' from 'Стиральная машина Bosch' → EMPTY (category noun)."""
    ctx = ExtractionContext(
        product_id=1,
        product_name="Стиральная машина Bosch WGG2540MOE",
        category_id=1,
        category_path=["Бытовая техника", "Стиральная машина"],
        brand="машина",
    )
    target = TargetAttribute(id=85, name="Бренд", type="enum", allowed_values=[])
    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda attr_id: [],
    )
    assert _val_for(out, 85) is None


def test_needle_category_noun_rejected_pech():
    """NEEDLE: 'печь' from 'Микроволновая печь Samsung' → EMPTY (category noun)."""
    ctx = ExtractionContext(
        product_id=1,
        product_name="Микроволновая печь Samsung MS23K3513AK",
        category_id=1,
        category_path=["Бытовая техника", "Микроволновая печь"],
        brand="печь",
    )
    target = TargetAttribute(id=85, name="Бренд", type="enum", allowed_values=[])
    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda attr_id: [],
    )
    assert _val_for(out, 85) is None


# ── NEEDLE FIX: 2-char brand (LG, HP) must fill ──────────────────────────────
# context.brand='LG' from "Монитор LG UltraGear" — 'LG' is 2 chars, which was
# previously rejected by _BRAND_MIN_LEN=3. Needle path now uses _NEEDLE_BRAND_MIN_LEN=2.


def test_needle_two_char_brand_lg_fills():
    """NEEDLE: 'LG' (2 chars) in name + dict empty → fills brand with 'LG'."""
    ctx = ExtractionContext(
        product_id=1,
        product_name="Монитор LG UltraGear 27GP850-B 27 дюймов",
        category_id=1,
        category_path=["Электроника", "Мониторы"],
        brand="LG",
    )
    target = TargetAttribute(id=85, name="Бренд", type="enum", allowed_values=[])
    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda attr_id: [],
    )
    v = _val_for(out, 85)
    assert v is not None and v.value == "LG"
    assert v.source.value == "description"
    assert v.evidence == "brand_from_name"


def test_needle_two_char_brand_hp_fills():
    """NEEDLE: 'HP' (2 chars) in name → fills."""
    ctx = ExtractionContext(
        product_id=1,
        product_name="Ноутбук HP Pavilion 15 Core i5",
        category_id=1,
        category_path=["Электроника", "Ноутбуки"],
        brand="HP",
    )
    target = TargetAttribute(id=85, name="Бренд", type="enum", allowed_values=[])
    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda attr_id: [],
    )
    v = _val_for(out, 85)
    assert v is not None and v.value == "HP"


def test_needle_real_brand_not_category_noun_yandex():
    """NEEDLE: 'Яндекс' is NOT a category noun for 'Умная колонка' path → fills."""
    ctx = ExtractionContext(
        product_id=1,
        product_name="Умная колонка Яндекс Станция Мини 2",
        category_id=1,
        category_path=["Электроника", "Умная колонка"],
        brand="Яндекс",  # correctly extracted brand
    )
    target = TargetAttribute(id=85, name="Бренд", type="enum", allowed_values=[])
    out = _apply_brand_from_name(
        [], [target], ctx,
        brand_options_fn=lambda attr_id: [],
    )
    v = _val_for(out, 85)
    assert v is not None and v.value == "Яндекс"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
