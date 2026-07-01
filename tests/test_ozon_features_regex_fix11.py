"""FIX-11: _FEATURES_STATE_RE / _parse_characteristics_html accept BOTH quote
styles Ozon uses for the `data-state` attribute of the characteristics widget.

Root cause (rooted 2026-07-01, live smoke of FIX-10 against real Ozon pages):
Ozon changed the widget markup from
    data-state='<raw JSON>'                (single quote, raw JSON)
to
    data-state="<HTML-escaped JSON>"       (double quote, internal `"` as `&quot;`)
The old regex matched ONLY the single-quote form -> 0 characteristics extracted
even though the transport (scrape.do, FIX-10) successfully fetched the page.

INV-11a: old single-quote markup -> same characteristics as before (regression).
INV-11b: new double-quote HTML-escaped markup -> characteristics extracted.
INV-11c: both markups present in one HTML -> both parsed.
INV-11d: html.unescape() applied ONLY to the double-quote branch; the
          single-quote branch is NOT mutated (a literal `&` in a raw-JSON value
          must survive unchanged).
INV-11e: broken JSON in data-state does not crash extract (existing error
          handling preserved) for both quote styles.

Real-world reference fragment (captured live via scrape.do, iPhone 15 card,
scratchpad `diag_features_out3.txt`, 2026-07-01):
    <div id="state-webCharacteristics-903456-default-1" data-state="{&quot;link&quot;:
    &quot;https:\\u002F\\u002Fwww.ozon.ru\\u002Fproduct\\u002F...&quot;,&quot;characteristics&quot;:
    [{&quot;short&quot;:[{&quot;key&quot;:&quot;RequiredApps&quot;,&quot;name&quot;:
    &quot;Обязательные программы предустановлены&quot;,&quot;values&quot;:[{&q...
This confirms the production shape: double-quote wrapper, `&quot;`-escaped
internals, JSON `\\u002F`-style unicode escapes left untouched by html.unescape
(json.loads handles those natively).
"""
from __future__ import annotations

import html as _html
import json
import textwrap

from app.services.enrichment.sources.ozon_card_source import OzonCardSource


def _characteristics_json(*, names_values: list[tuple[str, str, str]]) -> str:
    """Builds the widget JSON payload (characteristics[].short[].{name,values})."""
    items = [
        {"name": name, "values": [{"text": value, "id": vid}]}
        for name, value, vid in names_values
    ]
    return json.dumps(
        {"characteristics": [{"short": items, "long": [], "full": []}]},
        ensure_ascii=False,
    )


def _single_quote_div(widget_id: str, raw_json: str) -> str:
    return f'<div id="state-webCharacteristics-{widget_id}" data-state=\'{raw_json}\'></div>'


def _double_quote_escaped_div(widget_id: str, raw_json: str) -> str:
    escaped = _html.escape(raw_json, quote=True)
    return f'<div id="state-webCharacteristics-{widget_id}" data-state="{escaped}"></div>'


class TestInv11aOldSingleQuoteRegression:
    def test_old_single_quote_markup_still_parses(self):
        raw = _characteristics_json(
            names_values=[("Цвет", "Чёрный", "123"), ("Материал", "Металл", "456")]
        )
        html = textwrap.dedent(f"""
            <html><body>
            {_single_quote_div("111", raw)}
            </body></html>
        """).strip()

        chars = OzonCardSource._parse_characteristics_html(html)

        names = {c["name"]: c["value"] for c in chars}
        assert names == {"Цвет": "Чёрный", "Материал": "Металл"}


class TestInv11bNewDoubleQuoteEscaped:
    def test_new_double_quote_escaped_markup_extracts(self):
        raw = _characteristics_json(
            names_values=[("Вес", "120 г", "789"), ("Бренд", "Bosch", "321")]
        )
        html = textwrap.dedent(f"""
            <html><body>
            {_double_quote_escaped_div("903456", raw)}
            </body></html>
        """).strip()

        chars = OzonCardSource._parse_characteristics_html(html)

        assert len(chars) >= 1
        names = {c["name"]: c["value"] for c in chars}
        assert names == {"Вес": "120 г", "Бренд": "Bosch"}

    def test_real_world_fragment_shape_parses(self):
        """Reconstructs the REAL captured fragment shape (diag_features_out3.txt,
        live scrape.do fetch, 2026-07-01) — double-quote wrapper + &quot;-escaped
        internals + JSON unicode escapes untouched by html.unescape.
        """
        raw = json.dumps(
            {
                "link": "https://www.ozon.ru/product/apple-smartfon-iphone-15/features/",
                "characteristics": [
                    {
                        "short": [
                            {
                                "key": "RequiredApps",
                                "name": "Обязательные программы предустановлены",
                                "values": [{"text": "Да", "id": "1"}],
                            }
                        ],
                        "long": [],
                        "full": [],
                    }
                ],
            },
            ensure_ascii=False,
        )
        html = textwrap.dedent(f"""
            <html><body>
            {_double_quote_escaped_div("903456-default-1", raw)}
            </body></html>
        """).strip()

        chars = OzonCardSource._parse_characteristics_html(html)

        assert any(
            c["name"] == "Обязательные программы предустановлены" and c["value"] == "Да"
            for c in chars
        )


class TestInv11cBothMarkupsInOneHtml:
    def test_both_quote_styles_in_same_html_both_parsed(self):
        raw_single = _characteristics_json(names_values=[("Цвет", "Чёрный", "1")])
        raw_double = _characteristics_json(names_values=[("Вес", "200 г", "2")])
        html = textwrap.dedent(f"""
            <html><body>
            {_single_quote_div("111", raw_single)}
            {_double_quote_escaped_div("222", raw_double)}
            </body></html>
        """).strip()

        chars = OzonCardSource._parse_characteristics_html(html)

        names = {c["name"]: c["value"] for c in chars}
        assert names == {"Цвет": "Чёрный", "Вес": "200 г"}


class TestInv11dUnescapeOnlyDoubleQuoteBranch:
    def test_single_quote_branch_literal_ampersand_survives_unmutated(self):
        # A raw (single-quote, un-escaped) JSON value containing a literal '&'.
        # The single-quote branch must NOT run html.unescape on it — an
        # accidental unescape would be a no-op here (no entity present) but a
        # BROKEN unescape-everything implementation would still leave it
        # untouched only by luck; this test pins the actual invariant: the
        # raw-quote group is used AS-IS, never passed through _html.unescape.
        raw = _characteristics_json(
            names_values=[("Совместимость", "AT&T / Wi-Fi", "1")]
        )
        html = textwrap.dedent(f"""
            <html><body>
            {_single_quote_div("111", raw)}
            </body></html>
        """).strip()

        chars = OzonCardSource._parse_characteristics_html(html)

        names = {c["name"]: c["value"] for c in chars}
        assert names == {"Совместимость": "AT&T / Wi-Fi"}

    def test_double_quote_branch_html_entities_are_unescaped(self):
        # The escaped JSON contains &amp; (from a literal '&' in the value)
        # ON TOP of the &quot;-escaped quotes wrapping the whole payload.
        raw = _characteristics_json(
            names_values=[("Совместимость", "AT&T / Wi-Fi", "1")]
        )
        html = textwrap.dedent(f"""
            <html><body>
            {_double_quote_escaped_div("222", raw)}
            </body></html>
        """).strip()

        chars = OzonCardSource._parse_characteristics_html(html)

        names = {c["name"]: c["value"] for c in chars}
        assert names == {"Совместимость": "AT&T / Wi-Fi"}


class TestInv11eBrokenJsonDoesNotCrash:
    def test_broken_json_single_quote_skipped_not_raised(self):
        html = textwrap.dedent("""
            <html><body>
            <div id="state-webCharacteristics-111" data-state='{not valid json'></div>
            </body></html>
        """).strip()

        chars = OzonCardSource._parse_characteristics_html(html)
        assert chars == []

    def test_broken_json_double_quote_skipped_not_raised(self):
        html = textwrap.dedent("""
            <html><body>
            <div id="state-webCharacteristics-222" data-state="{not valid json"></div>
            </body></html>
        """).strip()

        chars = OzonCardSource._parse_characteristics_html(html)
        assert chars == []

    def test_broken_and_valid_mixed_only_valid_survives(self):
        raw_valid = _characteristics_json(names_values=[("Цвет", "Чёрный", "1")])
        html = textwrap.dedent(f"""
            <html><body>
            <div id="state-webCharacteristics-111" data-state='{{broken'></div>
            {_double_quote_escaped_div("222", raw_valid)}
            </body></html>
        """).strip()

        chars = OzonCardSource._parse_characteristics_html(html)

        names = {c["name"]: c["value"] for c in chars}
        assert names == {"Цвет": "Чёрный"}
