"""UgcSource — извлекает характеристики из отзывов и Q&A на Ozon/WB.

User-generated content часто содержит compat/physical-атрибуты которых нет
в datasheet производителя:
  - «реальная длина 60см, в кейс не влез»
  - «работает с mATX материнками»
  - «есть PCIe 5.0 разъём»
  - «греется до 75°C под нагрузкой»

Эти данные ценны потому, что покупатели физически держали товар в руках
и описывают реальное поведение, а не спеки на коробке.

Источники:
  - WB feedbacks v1 JSON (ПУБЛИЧНЫЙ, без auth):
    https://feedbacks{1|2}.wb.ru/feedbacks/v1/{nm_id}
    Поля: text, pros, cons (всё пользовательский текст).
  - Ozon reviews HTML:
    https://www.ozon.ru/product/{slug}-{pid}/reviews/
    DataDome protected → требуется Scrappey.
  - Ozon questions HTML (Q&A):
    https://www.ozon.ru/product/{slug}-{pid}/questions/
    DataDome protected → требуется Scrappey.

Алгоритм:
  1. is_applicable: True если есть product_name И есть already_filled
     (UGC — дополняющий источник, бессмысленно как первый и единственный).
  2. extract:
     a. Найти nm_id (WB) и/или (slug, pid) (Ozon) — TODO: пока через
        existing context.ean / source_urls; в будущем — через search.
     b. WB: дёрнуть feedbacks JSON (обе ветки fb1/fb2), собрать text+pros+cons.
     c. Ozon: Scrappey GET /reviews/ и /questions/, собрать видимый текст.
     d. Конкатенировать все тексты, фильтр regex по числам+единицам
        (это где обычно прячется спецификация).
     e. DeepSeek extraction с явным указанием noisy-нативы UGC.
     f. Confidence cap 0.75 (UGC всегда дополняет, не заменяет).

Cost:
  - WB feedbacks: бесплатно (открытый API).
  - Ozon reviews/questions: 2 credits Scrappey (~$0.0004 total).
  - DeepSeek extraction: ~$0.0002 за вызов.

Anti-block:
  - WB API без auth — никаких anti-bot.
  - Ozon Scrappey HTML — те же DataDome обходы, что в OzonCardSource.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field, AliasChoices

from app.services.enrichment.base import (
    AttributeSource,
    AttributeValue,
    ExtractionContext,
    LlmJudge,
    Source,
    TargetAttribute,
)
from app.services.enrichment.judges.ugc_judge import UgcJudge
from app.services.enrichment.prompt_router import (
    build_already_filled_block,
    build_meta_guidance,
    filter_already_filled_targets,
    format_target_line,
)
from app.services.providers.factory import get_main_manager
from app.services.providers.structured_adapter import StructuredLlmManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

_SCRAPPEY_ENDPOINT = "https://publisher.scrappey.com/api/v1"
_OZON_PRODUCT_BASE = "https://www.ozon.ru/product/"
_HTTP_TIMEOUT = 180.0

# Confidence cap для UGC — noisy источник, никогда не пускаем выше.
_CONF_CAP = 0.75

# Минимальная длина текстового фрагмента, который имеет смысл скармливать LLM.
_MIN_TEXT_LEN = 50
# Максимальная длина combined-текста, скармливаемого LLM (token budget).
_MAX_TEXT_LEN = 12_000
# Максимум фрагментов после фильтрации.
_MAX_KEPT_SNIPPETS = 60

# Regex: значимые упоминания спеков — где есть число + единица измерения.
# Без числа отзыв обычно не несёт factual информации для extraction.
_SPEC_HINT_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*"
    r"(?:см|мм|м(?!\w)|кг|г(?!\w)|мг|"
    r"вт|кВт|в(?!\w)|а(?!\w)|мА|"
    r"ггц|мгц|кГц|"
    r"гб|мб|тб|"
    r"°c|°с|c°|с°|"
    r"мл|л(?!\w)|"
    r"вольт|ампер|герц|ватт|"
    r"шт|штук"
    r")",
    re.IGNORECASE,
)

# Regex: ключевые слова compat/совместимость (даже без чисел).
_COMPAT_HINT_RE = re.compile(
    r"(?:совмест|подход|поместил|влез|размер|габарит|"
    r"matx|atx|itx|pcie|usb|hdmi|displayport|"
    r"оператив|память|разъём|разъем|порт|"
    r"ddr3|ddr4|ddr5|nvme|sata|m\.2)",
    re.IGNORECASE,
)

# WB feedbacks endpoint — пробуем обе ветки.
_WB_FEEDBACKS_HOSTS = ("feedbacks1.wb.ru", "feedbacks2.wb.ru")


def _is_datadome_block(content: str) -> bool:
    """Detect DataDome challenge."""
    if not content:
        return False
    return "incidentId" in content[:500]


def _extract_visible_text(html: str) -> str:
    """Грубое выделение текста из HTML.

    Убираем <script>, <style>, теги. Не идеально, но для regex-фильтра
    по числам+единицам достаточно. BeautifulSoup тут излишен — мы всё равно
    дальше будем дробить на куски и фильтровать regex'ом.
    """
    if not html:
        return ""
    txt = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    txt = re.sub(r"<style[^>]*>.*?</style>", " ", txt, flags=re.DOTALL | re.IGNORECASE)
    txt = re.sub(r"<[^>]+>", " ", txt)
    # Decode common entities.
    txt = (txt.replace("&nbsp;", " ").replace("&quot;", '"')
              .replace("&amp;", "&").replace("&#39;", "'")
              .replace("&lt;", "<").replace("&gt;", ">"))
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def _filter_snippets(texts: list[str]) -> list[str]:
    """Оставить только фрагменты с числами+единицами или compat-кейвордами.

    UGC обычно содержит много «огонь, рекомендую» — это шум. Нам нужны
    куски с factual информацией: «60см длина», «matx материнка», «PCIe 5.0».
    """
    kept: list[str] = []
    seen: set[str] = set()
    for t in texts:
        t = t.strip()
        if not t or len(t) < _MIN_TEXT_LEN:
            continue
        # Грубая дедупликация по нормализованному prefix.
        key = re.sub(r"\s+", " ", t.lower())[:120]
        if key in seen:
            continue
        if _SPEC_HINT_RE.search(t) or _COMPAT_HINT_RE.search(t):
            seen.add(key)
            kept.append(t[:600])  # хвост обычно вода
            if len(kept) >= _MAX_KEPT_SNIPPETS:
                break
    return kept


# ---------------------------------------------------------------------------
# LLM response schema
# ---------------------------------------------------------------------------


class _ExtractedAttr(BaseModel):
    model_config = {"populate_by_name": True}
    attribute_id: int = Field(..., validation_alias=AliasChoices("attribute_id", "id"))
    value: str | int | float | bool | list[str | int | float | bool] = Field(
        ..., validation_alias=AliasChoices("value", "attribute_value", "extracted_value"),
    )
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    evidence: Optional[str] = None
    mention_count: int = Field(default=1, ge=1, description="Сколько отзывов упомянули это значение")


class _ExtractionResponse(BaseModel):
    extracted: list[_ExtractedAttr]


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class UgcSource(AttributeSource):
    """Извлекает characteristics из отзывов и Q&A покупателей на Ozon/WB.

    UGC — дополняющий источник: имеет смысл только тогда, когда основные
    spec-источники (description/datasheet/icecat) уже что-то нашли, и нужно
    докинуть compat/реальные-размеры attrs, которых нет в datasheet.

    Cost: ~$0.0004 Scrappey + ~$0.0002 LLM = ~$0.0006/product.
    Latency: 10-30s (Scrappey overhead, WB API быстрый).
    """

    def __init__(
        self,
        scrappey_key: Optional[str] = None,
        llm_manager: Optional[StructuredLlmManager] = None,
        **kwargs: Any,
    ):
        _ = kwargs  # backward-compat
        self._scrappey_key = scrappey_key or os.environ.get("SCRAPPEY_KEY")
        self._llm = llm_manager or get_main_manager()
        self._judge = UgcJudge()

    @property
    def source_type(self) -> Source:
        return Source.UGC

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """UGC применим если есть product_name. is_applicable один target за раз —
        skip-логика «только когда уже что-то filled» делается в extract() через
        already_filled, а pipeline.py решает порядок вызова source'ов.
        """
        return bool(context.product_name and len(context.product_name.strip()) >= 5)

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        if not targets or not context.product_name:
            return []

        already_filled = already_filled or []

        # UGC — дополняющий источник. Если уже ничего не filled — значит
        # основные источники не нашли товар или description слишком плохое;
        # UGC сам по себе не вытянет, лучше не тратить деньги.
        if not already_filled:
            logger.debug("[UGC] skip: no already_filled — UGC дополняет, не заменяет")
            return []

        effective = filter_already_filled_targets(targets, already_filled)
        if not effective:
            logger.debug("[UGC] skip: все targets уже filled с high confidence")
            return []

        # ----- собираем тексты из доступных источников -----
        raw_texts: list[str] = []

        try:
            wb_texts = await self._fetch_wb_texts(context)
            raw_texts.extend(wb_texts)
        except Exception as exc:
            logger.info("[UGC] WB fetch failed: %s", exc)

        try:
            ozon_texts = await self._fetch_ozon_texts(context)
            raw_texts.extend(ozon_texts)
        except Exception as exc:
            logger.info("[UGC] Ozon fetch failed: %s", exc)

        if not raw_texts:
            logger.debug("[UGC] no UGC texts собрано для '%s'", context.product_name[:60])
            return []

        # ----- фильтр по числам+единицам -----
        snippets = _filter_snippets(raw_texts)
        if not snippets:
            logger.debug("[UGC] нет snippets со spec-hints после фильтра")
            return []

        combined = "\n---\n".join(snippets)[:_MAX_TEXT_LEN]
        logger.info(
            "[UGC] LLM extraction: %d snippets (%d chars) → %d targets",
            len(snippets), len(combined), len(effective),
        )

        # ----- LLM extraction -----
        try:
            values = await self._llm_extract(context, effective, combined, already_filled)
        except Exception as exc:
            logger.warning("[UGC] LLM extraction failed: %s", exc)
            return []

        target_by_id = {t.id: t for t in targets}
        out: list[AttributeValue] = []
        for v in values:
            target = target_by_id.get(v.attribute_id)
            if target is None:
                continue
            # Confidence cap — UGC noisy, никогда выше 0.75.
            # Если 2+ отзыва упоминали — оставляем как есть (но всё равно ≤ cap).
            # Если 1 отзыв — режем на 0.6 как floor для judge.
            base_conf = min(v.confidence, _CONF_CAP)
            if v.mention_count < 2:
                base_conf = min(base_conf, 0.65)
            out.append(AttributeValue(
                attribute_id=v.attribute_id,
                value=v.value,
                confidence=base_conf,
                source=Source.UGC,
                evidence=v.evidence,
                semantic_type=target.semantic_type,
                is_collection=target.is_collection,
            ))
        return out

    def get_judge(self) -> LlmJudge:
        return self._judge

    # ------------------------------------------------------------------
    # WB feedbacks (открытый API, без auth)
    # ------------------------------------------------------------------

    async def _fetch_wb_texts(self, context: ExtractionContext) -> list[str]:
        """Дёрнуть WB feedbacks JSON если в context есть nm_id.

        TODO: формула WB basket-номера + nm_id discovery через WB search не
        реализованы — другой агент работает над WbCardSource. Пока полагаемся
        на context.ean (если совпадает с nm_id численно) или будущее
        context-поле. Если nm_id нет — возвращаем [], это OK.
        """
        nm_id = self._extract_wb_nm_id(context)
        if not nm_id:
            return []

        texts: list[str] = []
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for host in _WB_FEEDBACKS_HOSTS:
                url = f"https://{host}/feedbacks/v1/{nm_id}"
                try:
                    r = await client.get(url, headers={
                        "Accept": "application/json",
                        "User-Agent": "Mozilla/5.0",
                    })
                except (httpx.TimeoutException, httpx.HTTPError) as exc:
                    logger.debug("[UGC/WB] %s err: %s", host, exc)
                    continue
                if r.status_code != 200:
                    logger.debug("[UGC/WB] %s HTTP %s", host, r.status_code)
                    continue
                try:
                    data = r.json()
                except (ValueError, json.JSONDecodeError):
                    continue
                feedbacks = data.get("feedbacks") or data.get("data") or []
                if not isinstance(feedbacks, list):
                    continue
                for fb in feedbacks:
                    if not isinstance(fb, dict):
                        continue
                    for key in ("text", "pros", "cons"):
                        val = fb.get(key)
                        if isinstance(val, str) and val.strip():
                            texts.append(val.strip())
                if texts:
                    logger.info("[UGC/WB] %s → %d feedback fragments", host, len(texts))
                    break  # одной ветки достаточно
        return texts

    @staticmethod
    def _extract_wb_nm_id(context: ExtractionContext) -> Optional[str]:
        """Попытаться вытащить WB nm_id из context.

        Стратегии:
          1. context.source_urls — ищем https://www.wildberries.ru/catalog/{id}/...
          2. context.ean — если выглядит как WB nm_id (8-10 цифр).

        TODO: integrate WB search когда WbCardSource будет готов — он сделает
        nm_id discovery и положит в context (предположительно через новое поле).
        """
        for url in context.source_urls or []:
            if not isinstance(url, str):
                continue
            m = re.search(r"wildberries\.ru/catalog/(\d+)/", url, re.IGNORECASE)
            if m:
                return m.group(1)
        # ean не равно nm_id концептуально, но иногда в context.ean кладут nm_id
        # как identifier для конкретного товара. Если есть и выглядит как nm_id —
        # пробуем. False positive безвреден: WB вернёт 200 с пустыми feedbacks.
        ean = (context.ean or "").strip()
        if ean.isdigit() and 6 <= len(ean) <= 11:
            return ean
        return None

    # ------------------------------------------------------------------
    # Ozon reviews + Q&A через Scrappey
    # ------------------------------------------------------------------

    async def _fetch_ozon_texts(self, context: ExtractionContext) -> list[str]:
        """Скачать Ozon reviews + questions HTML через Scrappey.

        TODO: pid+slug discovery — пока ожидаем что они есть в context.source_urls
        как ссылка на товар. В будущем — OzonCardSource положит pid в context
        после своей search-фазы (другой агент работает над интеграцией).
        Если pid нет — возвращаем [], это OK.
        """
        if not self._scrappey_key:
            logger.debug("[UGC/Ozon] нет SCRAPPEY_KEY — skip")
            return []

        slug_pid = self._extract_ozon_slug_pid(context)
        if not slug_pid:
            return []
        slug, pid = slug_pid

        texts: list[str] = []
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            for endpoint in ("reviews", "questions"):
                url = f"{_OZON_PRODUCT_BASE}{slug}-{pid}/{endpoint}/"
                html = await self._scrappey_get(client, url)
                if not html:
                    continue
                visible = _extract_visible_text(html)
                if not visible:
                    continue
                # Бьём на куски по абзацам — стопить весь текст в один кусок плохо
                # для regex-фильтра snippets.
                chunks = re.split(r"(?:\.\s|\n|\r)+", visible)
                texts.extend(c for c in chunks if c.strip())
                logger.info(
                    "[UGC/Ozon] %s → %d chunks из %d chars HTML",
                    endpoint, len(chunks), len(html),
                )
        return texts

    async def _scrappey_get(
        self,
        client: httpx.AsyncClient,
        target_url: str,
    ) -> Optional[str]:
        """Один POST к Scrappey без retry — экономим credits на UGC."""
        payload = {"cmd": "request.get", "url": target_url}
        try:
            r = await client.post(
                _SCRAPPEY_ENDPOINT,
                params={"key": self._scrappey_key},
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            logger.info("[UGC/Ozon] Scrappey err: %s", exc)
            return None
        if r.status_code >= 400:
            logger.info("[UGC/Ozon] Scrappey HTTP %s", r.status_code)
            return None
        try:
            envelope = r.json()
        except (ValueError, json.JSONDecodeError):
            return None
        solution = envelope.get("solution") or {}
        if solution.get("statusCode") != 200:
            return None
        content = solution.get("response") or ""
        if not content or _is_datadome_block(content):
            return None
        return content

    @staticmethod
    def _extract_ozon_slug_pid(context: ExtractionContext) -> Optional[tuple[str, str]]:
        """Вытащить (slug, pid) Ozon из context.source_urls.

        Формат: https://www.ozon.ru/product/{slug}-{pid}/ (или с /reviews/ etc.)
        """
        ozon_url_re = re.compile(
            r"ozon\.ru/product/([a-z0-9\-]+?)-(\d+)/?",
            re.IGNORECASE,
        )
        for url in context.source_urls or []:
            if not isinstance(url, str):
                continue
            m = ozon_url_re.search(url)
            if m:
                return m.group(1), m.group(2)
        return None

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    async def _llm_extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        ugc_text: str,
        already_filled: list[AttributeValue],
    ) -> list[_ExtractedAttr]:
        """Один LLM call с явным указанием noisy-нативы UGC."""
        targets_block = "\n".join([format_target_line(t) for t in targets])
        already_preamble, already_rule = build_already_filled_block(already_filled)

        system_prompt = (
            "You extract product characteristics from USER REVIEWS and Q&A "
            "on Russian marketplaces (Ozon, Wildberries). "
            "These texts are NOISY: each review is ONE buyer's opinion, not a verdict. "
            "Rules:\n"
            "- Extract only attributes for which the text gives a CONCRETE value "
            "  (number+unit, model name, compatibility statement).\n"
            "- If 2+ reviews mention the same value — confidence higher (≥0.7), "
            "  set mention_count accordingly.\n"
            "- If only 1 review mentions it — confidence ≤0.65, mention_count=1.\n"
            "- Ignore generic praise («огонь», «рекомендую», «качество топ»).\n"
            "- Ignore complaints without specifics («сломалось», «не работает»).\n"
            "- Provide a short evidence quote (max 120 chars) — the actual review fragment.\n"
            "- If attribute is not mentioned, do NOT include it in the response.\n"
            "- If target has is_collection=true, return a JSON array; otherwise a single scalar.\n"
            + build_meta_guidance()
            + already_rule
        )
        user_text = (
            f"Product name: {context.product_name}\n"
            f"Category: {' / '.join(context.category_path) or context.category_id}\n\n"
            f"User reviews and Q&A (combined, noisy):\n{ugc_text}\n\n"
            + already_preamble
            + f"Target attributes (only fill these, skip everything else):\n{targets_block}\n\n"
            "Return JSON with field 'extracted' = list of "
            "{attribute_id, value, confidence, evidence, mention_count}."
        )

        parsed, _tokens = await self._llm.structured_request(
            system_prompt=system_prompt,
            user_text=user_text,
            response_model=_ExtractionResponse,
        )
        if parsed is None:
            return []
        context.llm_calls_so_far += 1
        return parsed.extracted
