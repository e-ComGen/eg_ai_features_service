"""Unit tests for YandexMarketSource (prototype, no network).

Covers:
  • _build_ym_query: type-word-preserving Serper query building.
  • _parse_card / _parse_state_specs: characteristics extraction from a
    hand-made HTML sample containing JSON-LD (title/brand/category) + an
    embedded __INITIAL_STATE__ specs blob.
  • _YM_PRODUCT_RE: product-URL recognition.
  • _classify_match / _type_compatible: scoring gates.

No live network: _fetch_card_html is never called here.
"""
from __future__ import annotations

import pytest

from app.services.enrichment.sources.yandex_market_source import (
    YandexMarketSource,
    _build_ym_query,
    _YM_PRODUCT_RE,
)


# ---------------------------------------------------------------------------
# Hand-made sample HTML (mirrors market.yandex.ru shape closely enough)
# ---------------------------------------------------------------------------

SAMPLE_HTML = """
<!DOCTYPE html><html><head>
<title>Куртка The North Face Resolve мужская — купить на Яндекс Маркете</title>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "Куртка The North Face Resolve",
  "brand": {"@type": "Brand", "name": "The North Face"},
  "category": "Куртки",
  "offers": {"@type": "Offer", "price": "12990", "priceCurrency": "RUB"}
}
</script>
</head><body>
<script>
window.__INITIAL_STATE__ = {"widgets":{"product":{"specs":[
  {"name":"Цвет","value":"Чёрный"},
  {"name":"Материал","values":["Полиэстер","Нейлон"]},
  {"name":"Сезон","value":"Демисезон"},
  {"name":"Артикул","value":"NF0A2VD5"}
]}}};
</script>
</body></html>
"""

# Alternate shape: __NEXT_DATA__ with grouped specs and {key,value} pairs.
SAMPLE_HTML_NEXTDATA = """
<html><head>
<script type="application/ld+json">
{"@graph":[{"@type":"Product","name":"Футболка Nike Sportswear","brand":"Nike","category":"Футболки"}]}
</script>
</head><body>
<script id="__NEXT_DATA__" type="application/json">
{"props":{"pageProps":{"specifications":[
  {"groupName":"Основные","specs":[
     {"key":"Цвет","value":"Белый"},
     {"key":"Состав","value":"Хлопок 100%"}
  ]}
]}}}
</script>
</body></html>
"""


# ---------------------------------------------------------------------------
# URL recognition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://market.yandex.ru/product--kurtka/123456789", True),
    ("https://market.yandex.ru/product/987654321", True),
    ("https://market.yandex.ru/card/55512345", True),
    ("https://market.yandex.ru/catalog/some-listing", False),
    ("https://www.ozon.ru/product/foo-12345/", False),
])
def test_product_url_recognition(url, expected):
    assert bool(_YM_PRODUCT_RE.search(url)) is expected


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------

def test_build_ym_query_preserves_type_word():
    q = _build_ym_query(
        "Куртка The North Face Resolve мужская",
        brand="The North Face",
        cat_leaf="Куртки",
        max_tokens=5,
    )
    low = q.lower()
    # Type word (from category leaf) must be present so Serper stays in-class.
    assert "куртк" in low
    # Brand survives.
    assert "north" in low and "face" in low


def test_build_ym_query_nonempty_for_plain_name():
    q = _build_ym_query("Samsung Galaxy A55 8/256GB", brand="Samsung",
                        cat_leaf="Смартфоны", max_tokens=5)
    assert q.strip()
    assert "samsung" in q.lower()


# ---------------------------------------------------------------------------
# Characteristics parsing — __INITIAL_STATE__ shape
# ---------------------------------------------------------------------------

def test_parse_card_initial_state():
    parsed = YandexMarketSource._parse_card(SAMPLE_HTML)
    assert parsed["title"] == "Куртка The North Face Resolve"
    assert parsed["brand"] == "The North Face"
    assert parsed["card_type"] == "Куртки"

    chars = {c["name"].lower(): c["value"] for c in parsed["chars"]}
    assert chars["цвет"] == "Чёрный"
    # list value joined
    assert "Полиэстер" in chars["материал"] and "Нейлон" in chars["материал"]
    assert chars["сезон"] == "Демисезон"
    assert chars["артикул"] == "NF0A2VD5"


def test_parse_card_next_data_grouped():
    parsed = YandexMarketSource._parse_card(SAMPLE_HTML_NEXTDATA)
    assert parsed["title"] == "Футболка Nike Sportswear"
    assert parsed["brand"] == "Nike"
    assert parsed["card_type"] == "Футболки"
    chars = {c["name"].lower(): c["value"] for c in parsed["chars"]}
    # nested group specs flattened
    assert chars["цвет"] == "Белый"
    assert chars["состав"] == "Хлопок 100%"


def test_parse_state_specs_dedup_and_empty_safe():
    # Empty / garbage HTML must not raise and must return [].
    assert YandexMarketSource._parse_state_specs("<html>no state here</html>") == []
    assert YandexMarketSource._parse_card("")["chars"] == []


# ---------------------------------------------------------------------------
# Format 3: data-auto="product-spec" (SSR spec table, 2024-2026 product pages)
# ---------------------------------------------------------------------------

SAMPLE_HTML_SSR_SPECS = """
<!DOCTYPE html><html lang="ru"><head>
<script type="application/ld+json">
{"@type":"Product","name":"Джинсы Levi's 511 Slim","brand":{"@type":"Brand","name":"Levi's"}}
</script>
</head><body>
<div data-zone-name="fullSpecs">
  <div class="_2jsum">
    <div class="j0C6X"><div class="_16c58">
      <span data-auto="product-spec" class="ds-text">Состав материала</span>
    </div></div>
    <div class="_1_zPW"><div class="_19UG7">
      <div class="ds-text"><span>Хлопок 98%, Эластан 2%</span></div>
    </div></div>
  </div>
  <div class="_2jsum">
    <div class="j0C6X"><div class="_16c58">
      <span data-auto="product-spec" class="ds-text">Пол</span>
    </div></div>
    <div class="_1_zPW"><div class="_19UG7">
      <div class="" data-zone-name="specLink" data-node-id="x1"
           data-zone-data="{&quot;url&quot;:&quot;/catalog&quot;,&quot;urlTargetType&quot;:&quot;catalog_list&quot;,&quot;isUrlCanonical&quot;:false,&quot;text&quot;:&quot;мужской&quot;}"
           data-baobab-name="specLink"><a href="/catalog">мужской</a></div>
    </div></div>
  </div>
  <div class="_2jsum">
    <div class="j0C6X"><div class="_16c58">
      <span data-auto="product-spec" class="ds-text">Сезон</span>
    </div></div>
    <div class="_1_zPW"><div class="_19UG7">
      <div class="ds-text"><span>всесезон</span></div>
    </div></div>
  </div>
  <div class="_2jsum">
    <div class="j0C6X"><div class="_16c58">
      <span data-auto="product-spec" class="ds-text">Артикул Маркета</span>
    </div></div>
    <div class="_1_zPW"><div class="_19UG7">
      <div class="ds-text"><span>103064272661</span></div>
    </div></div>
  </div>
</div>
</body></html>
"""


def test_parse_card_ssr_format3_spec_table():
    """Format 3: data-auto=product-spec gives the full SSR spec table."""
    parsed = YandexMarketSource._parse_card(SAMPLE_HTML_SSR_SPECS)
    assert parsed["title"] == "Джинсы Levi's 511 Slim"
    assert parsed["brand"] == "Levi's"
    chars = {c["name"].lower(): c["value"] for c in parsed["chars"]}
    # plain text span value
    assert "хлопок 98%" in chars["состав материала"].lower()
    assert "эластан 2%" in chars["состав материала"].lower()
    # specLink text value
    assert chars["пол"] == "мужской"
    # numeric-value (artikul)
    assert chars["артикул маркета"] == "103064272661"
    # another plain value
    assert chars["сезон"] == "всесезон"
    # dedup: same name should appear only once
    names = [c["name"].lower() for c in parsed["chars"]]
    assert len(names) == len(set(names)), "No duplicates expected"


def test_parse_state_specs_format3_dedup():
    """Format 3 deduplication: same spec name from multiple zones appears once."""
    # Spec appears twice (ProductSpecsList + fullSpecs both rendered server-side)
    double_html = SAMPLE_HTML_SSR_SPECS + """
    <div data-zone-name="ProductSpecsList">
      <div class="_2jsum">
        <div class="j0C6X"><div class="_16c58">
          <span data-auto="product-spec" class="ds-text">Состав материала</span>
        </div></div>
        <div class="_1_zPW"><div class="_19UG7">
          <div class="ds-text"><span>Хлопок 98%, Эластан 2%</span></div>
        </div></div>
      </div>
    </div>
    """
    parsed = YandexMarketSource._parse_card(double_html)
    names = [c["name"].lower() for c in parsed["chars"]]
    assert names.count("состав материала") == 1, "Duplicate spec name must be deduped"


def test_pairs_from_container_flat_dict():
    pairs = dict(YandexMarketSource._pairs_from_container(
        {"Цвет": "Синий", "Вес": "0.5"}
    ))
    assert pairs["Цвет"] == "Синий"
    assert pairs["Вес"] == "0.5"


# ---------------------------------------------------------------------------
# Scoring gates
# ---------------------------------------------------------------------------

def test_classify_match_buckets():
    assert YandexMarketSource._classify_match(90.0) == "exact"
    assert YandexMarketSource._classify_match(65.0) == "brand_line"
    assert YandexMarketSource._classify_match(40.0) == "skip"


def test_type_compatible():
    assert YandexMarketSource._type_compatible("куртка", "куртка") is True
    # same root, different form
    assert YandexMarketSource._type_compatible("футболка", "футболочка") is True
    # different type → reject
    assert YandexMarketSource._type_compatible("куртка", "шорты") is False


def test_pick_best_match_prefers_closer_title():
    tiles = [
        {"title": "Куртка The North Face Resolve"},
        {"title": "Шорты Adidas Running"},
    ]
    best, score = YandexMarketSource._pick_best_match(
        "Куртка The North Face Resolve", tiles, category_leaf="Куртки",
    )
    assert best is not None
    assert "North Face" in best["title"]
    assert score > 60.0
