"""image_card_search — IMAGE-BASED card finder (DESIGN + prototype).

Идея: фото товара — гораздо более сильный матч-сигнал, чем generic-заголовок.
Title-поиск (Ozon/WB card sources) промахивается на товарах с шумным именем
("Куртка мужская зимняя 2024 NEW премиум") — фото же ведёт прямо к карточке-
близнецу. Этот модуль закрывает card=N gap, где текстовый поиск отдаёт 0.

ПОТОК (по шагам):
  1. reverse_image_search(image_url) → кандидатные карточки маркетплейса
     (URL/id + visual_score). **STUB** — см. _reverse_image_search (TODO).
  2. GATING LAYER (полностью реализован, юнит-тестируем без сети/LLM):
     слоистый гейт, который НЕ ДАЁТ ложному кандидату загрязнить атрибуты.
     Owner гиперчувствителен к неверным REQUIRED-fill'ам → empty > wrong.

⚠️ Этот модуль НЕ подключён в pipeline. Reverse-image вызов — STUB.
   Реальный вызов делает платный API (SerpApi) — НЕ дёргаем в тестах.

────────────────────────────────────────────────────────────────────────────
FALSE-POSITIVE MITIGATION (ранжировано по эффективности)
────────────────────────────────────────────────────────────────────────────
Reverse-image-поиск ВСЕГДА вернёт похожие-но-чужие товары (look-alikes:
та же куртка другого бренда, тот же силуэт). Слои гейта (короткий путь —
дешёвые детерминированные проверки первыми, дорогой LLM последним):

  [MUST-HAVE]
  G1. BRAND cross-check (детерминированный, ~0 cost). Бренд из имени товара
      (context.brand) ОБЯЗАН совпасть с брендом кандидата. Mismatch → reject.
      Самый дешёвый и самый сильный фильтр: image-поиск тащит чужие бренды
      того же силуэта — бренд их режет мгновенно.
  G2. TYPE cross-check (детерминированный, ~0 cost). Тип товара (лемма из
      category leaf / имени, _target_type_lemma) ОБЯЗАН совпасть с типом
      карточки. Цель «куртка» vs карточка «жилет/брюки» → reject. Ловит
      visual look-alikes одного класса-картинки но другого товара.
  G3. VISUAL-SCORE threshold + top-K agreement. visual_score < порога →
      reject. Доп.: если ДВА разных движка (Yandex+Lens) сошлись на одной
      карточке — сильный сигнал; одиночное совпадение требует выше порога.
  G6. FALLBACK to no-fill. Любое сомнение на любом слое → НЕ заполняем.
      Пустое поле всегда лучше неверного (особенно required). Это не «слой»,
      а инвариант: гейт fail-closed.

  [NICE-TO-HAVE]
  G4. FINAL-LLM verification. Кандидат-заголовок + ключевые атрибуты (+опц.
      его картинка) vs ИМЯ нашего товара → LLM «тот же товар? yes/no +
      confidence» со строгим rubric. Ловит то, что прошло G1-G3 (правильный
      бренд+тип, но другая модель/поколение). Дорогой (1 LLM call) → ПОСЛЕ
      дешёвых гейтов, только для выживших кандидатов. Hook здесь, промпт
      реализован; сам LLM-manager инъектируется (в тестах — мок).
  G5. ATTRIBUTE-LEVEL enum gate + confidence cap. Значение из image-карточки
      заполняет атрибут ТОЛЬКО если резолвится в реальный Ozon value_id
      (resolve_value_id). Confidence капается НИЖЕ card/description источников
      (IMAGE_CARD_CONF_CAP) — при конфликте merge предпочтёт надёжный источник.

Почему такой порядок: G1/G2 — детерминированы и почти бесплатны, отсекают
большинство мусора до того, как тратим деньги на LLM (G4). G3 — числовой
порог, тоже дёшев. G4 — единственный платный, гейтит остаток. G5 —
последний предохранитель на уровне записи значения.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Конфигурация гейта (env-overridable)
# ---------------------------------------------------------------------------

# Минимальный visual_score для ОДИНОЧНОГО движка. Сошлись два движка на одной
# карточке (top-K agreement) → достаточно VISUAL_SCORE_MIN_MULTI.
VISUAL_SCORE_MIN_SINGLE = float(os.getenv("EG_IMG_VISUAL_MIN_SINGLE", "0.82"))
VISUAL_SCORE_MIN_MULTI = float(os.getenv("EG_IMG_VISUAL_MIN_MULTI", "0.65"))

# Порог уверенности финального LLM-verify (G4): ниже → reject.
LLM_VERIFY_CONF_MIN = float(os.getenv("EG_IMG_LLM_VERIFY_MIN", "0.75"))

# Confidence cap для значений из image-карточки (G5). Намеренно НИЖЕ
# description/card/icecat порогов в base.SOURCE_CONFIDENCE_THRESHOLDS, чтобы при
# конфликте merge/judge предпочёл более надёжный источник, а не картинку.
IMAGE_CARD_CONF_CAP = float(os.getenv("EG_IMG_CONF_CAP", "0.70"))


# ---------------------------------------------------------------------------
# Типы данных
# ---------------------------------------------------------------------------

@dataclass
class ImageCandidate:
    """Кандидатная карточка маркетплейса, найденная по фото."""
    url: str
    marketplace: str                       # "ozon" | "wb" | "yandex_market"
    title: str = ""                        # заголовок карточки (для brand/type/LLM)
    brand: Optional[str] = None            # бренд карточки (если движок отдал)
    card_id: Optional[str] = None          # nm_id / SKU / артикул
    visual_score: float = 0.0              # 0..1 визуальная близость от движка
    engines: set[str] = field(default_factory=set)  # какие движки нашли (для top-K)


@dataclass
class GateResult:
    """Результат прохождения кандидата через слоистый гейт."""
    candidate: ImageCandidate
    accepted: bool
    reason: str                            # почему принят/отвергнут (audit)
    match_confidence: float = 0.0          # итоговая уверенность «тот же товар»


class LlmVerifier(Protocol):
    """Интерфейс финального LLM-verify (G4). В тестах — мок.

    Реализация должна вернуть (same_product: bool, confidence: float).
    """
    async def verify_same_product(
        self, product_name: str, candidate_title: str, candidate_attrs: str
    ) -> tuple[bool, float]:
        ...


# ---------------------------------------------------------------------------
# G1/G2 helpers — brand & type cross-check (детерминированные)
# ---------------------------------------------------------------------------

def _norm_brand(b: Optional[str]) -> str:
    return (b or "").strip().lower()


def brand_matches(our_brand: Optional[str], candidate: ImageCandidate) -> Optional[bool]:
    """G1: совпадает ли наш бренд с брендом кандидата.

    Возвращает:
      True  — бренды совпали (подстрока в любую сторону),
      False — оба известны и НЕ совпали → reject,
      None  — недостаточно данных (наш бренд или бренд кандидата неизвестен).
              None НЕ режет сам по себе (бренд может быть в заголовке) —
              решение принимает gate с учётом title.
    """
    ob = _norm_brand(our_brand)
    if not ob:
        return None
    cb = _norm_brand(candidate.brand)
    title = (candidate.title or "").lower()
    if cb:
        return ob in cb or cb in ob
    # Бренд кандидата явно не задан. G1 режет ТОЛЬКО на позитивном конфликте
    # брендов; отсутствие бренда в заголовке — НЕ конфликт (маркетплейсы часто
    # не пишут бренд в title, держа его в отдельном поле, которое движок не
    # отдал). Поэтому: нашли наш бренд в title → True; не нашли → None
    # (неопределённость, решат G2/G4), а не False.
    if title and ob in title:
        return True
    return None


def type_matches(target_type: Optional[str], candidate: ImageCandidate) -> Optional[bool]:
    """G2: совпадает ли тип нашего товара (лемма) с типом кандидата.

    Сравнение по леммам токенов заголовка кандидата (использует тот же
    _type_lemma из wb_card_source, что и тип-гейт карточных источников —
    единый словарь синонимики). Возвращает True/False/None (None = нет данных).
    """
    if not target_type:
        return None
    title = (candidate.title or "").strip()
    if not title:
        return None
    try:
        from app.services.enrichment.sources.wb_card_source import (
            _type_lemma, _TYPE_TOKEN_RE,
        )
        cand_lemmas = {
            _type_lemma(tok)
            for tok in _TYPE_TOKEN_RE.findall(title.lower())
            if len(tok) >= 3
        }
        if not cand_lemmas:
            return None
        return _lemmas_compatible(target_type, cand_lemmas)
    except Exception:  # noqa: BLE001 — морфология опциональна
        # Fallback без pymorphy: грубое вхождение типа в заголовок.
        return target_type.lower() in title.lower()


def _lemmas_compatible(target_type: str, cand_lemmas: set[str]) -> bool:
    """Совместимость типа (лемма) с леммами заголовка кандидата.

    Та же логика, что WbCardSource._type_compatible (НЕ импортируем приватный
    метод класса — дублируем компактно): точное совпадение лемм ИЛИ длинная
    общая основа (однокоренные формы куртка↔курточка). Разные корни → False.
    """
    if target_type in cand_lemmas:
        return True
    for subj in cand_lemmas:
        common = 0
        for x, y in zip(target_type, subj):
            if x == y:
                common += 1
            else:
                break
        shorter = min(len(target_type), len(subj))
        if shorter and common >= 5 and common / shorter >= 0.7:
            return True
    return False


# ---------------------------------------------------------------------------
# Marketplace domain → canonical name
# ---------------------------------------------------------------------------

_MARKETPLACE_DOMAINS: dict[str, str] = {
    "ozon.ru": "ozon",
    "wildberries.ru": "wb",
    "wb.ru": "wb",
    "market.yandex.ru": "yandex_market",
}

# Card-id extractors for each marketplace (cheap regex, no network).
# Ozon: /product/<slug>-<digits>/ or /product/<digits>/
_OZON_CARD_ID_RE = re.compile(r"/product/(?:[a-z0-9\-]+-)?(\d{5,})", re.IGNORECASE)
# WB: /catalog/<nm_id>/  or /product/<nm_id>
_WB_CARD_ID_RE = re.compile(r"(?:/catalog/|/product/)(\d{5,})", re.IGNORECASE)
# Yandex Market: /product--<slug>/<id> or /product/<id>
_YM_CARD_ID_RE = re.compile(r"/product(?:--[^/?#]+)?/(\d{4,})", re.IGNORECASE)

_CARD_ID_RES: dict[str, re.Pattern[str]] = {
    "ozon": _OZON_CARD_ID_RE,
    "wb": _WB_CARD_ID_RE,
    "yandex_market": _YM_CARD_ID_RE,
}


def _extract_marketplace(link: str) -> Optional[str]:
    """Return canonical marketplace name from a URL, or None if not a target domain."""
    for domain, name in _MARKETPLACE_DOMAINS.items():
        if domain in link:
            return name
    return None


def _extract_card_id(link: str, marketplace: str) -> Optional[str]:
    """Extract a card identifier from a marketplace product URL. Returns None if not found."""
    pattern = _CARD_ID_RES.get(marketplace)
    if pattern is None:
        return None
    m = pattern.search(link)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Serper /lens reverse-image-search implementation
# ---------------------------------------------------------------------------

async def _reverse_image_search(
    image_url: str,
    *,
    marketplaces: tuple[str, ...] = ("ozon", "wb", "yandex_market"),
) -> list[ImageCandidate]:
    """Call Serper /lens (Google Lens) with image_url, return marketplace candidates.

    Uses the existing SerperClient.lens() method (same key, same httpx pattern).
    Only links whose domain is ozon.ru / wildberries.ru / market.yandex.ru are kept.
    visual_score is normalized from rank position: 1.0 - idx * 0.05, clamped to [0.1, 1.0].

    Returns [] on any error (fail-closed — the gate is fail-closed downstream).
    """
    try:
        from app.services.providers.factory import get_web_search_client
        client = get_web_search_client()
    except Exception as exc:
        logger.warning("[ImageCardSearch] SerperClient unavailable: %s → []", exc)
        return []

    try:
        organic = await client.lens(image_url)
    except Exception as exc:
        logger.warning("[ImageCardSearch] /lens error for %s: %s → []", image_url[:80], exc)
        return []

    if not organic:
        logger.info("[ImageCardSearch] /lens returned 0 results for %s", image_url[:80])
        return []

    candidates: list[ImageCandidate] = []
    step = 0.05  # rank-based visual_score decay
    for idx, item in enumerate(organic):
        link = (item.get("link") or "").strip()
        if not link:
            continue
        mp = _extract_marketplace(link)
        if mp is None or mp not in marketplaces:
            continue
        visual_score = max(0.1, 1.0 - idx * step)
        card_id = _extract_card_id(link, mp)
        title = (item.get("title") or "").strip()
        candidates.append(ImageCandidate(
            url=link,
            marketplace=mp,
            title=title,
            brand=None,  # Serper /lens doesn't expose brand separately
            card_id=card_id,
            visual_score=visual_score,
            engines={"serper_lens"},
        ))

    logger.info(
        "[ImageCardSearch] /lens → %d marketplace candidates (of %d total results) for %s",
        len(candidates), len(organic), image_url[:80],
    )
    return candidates


# ---------------------------------------------------------------------------
# Слоистый гейт (G1→G2→G3→G4) — полностью реализован
# ---------------------------------------------------------------------------

async def gate_candidate(
    candidate: ImageCandidate,
    *,
    our_brand: Optional[str],
    target_type: Optional[str],
    product_name: str,
    candidate_attrs_text: str = "",
    llm_verifier: Optional[LlmVerifier] = None,
) -> GateResult:
    """Прогнать ОДНОГО кандидата через слоистый гейт. fail-closed (G6).

    Порядок дёшево→дорого: G1 brand → G2 type → G3 visual-score → G4 LLM.
    Любой fail → accepted=False с reason. Проходит — match_confidence
    собирается из visual_score и (если был) LLM-confidence.
    """
    # --- G1: BRAND cross-check ---
    bm = brand_matches(our_brand, candidate)
    if bm is False:
        return GateResult(candidate, False, f"G1 brand-mismatch (our={our_brand!r})")
    # bm is None → бренд неизвестен, не режем здесь (полагаемся на G2/G4).

    # --- G2: TYPE cross-check ---
    tm = type_matches(target_type, candidate)
    if tm is False:
        return GateResult(candidate, False, f"G2 type-mismatch (target={target_type!r})")

    # --- G3: VISUAL-SCORE threshold + top-K agreement ---
    multi_engine = len(candidate.engines) >= 2
    threshold = VISUAL_SCORE_MIN_MULTI if multi_engine else VISUAL_SCORE_MIN_SINGLE
    if candidate.visual_score < threshold:
        return GateResult(
            candidate, False,
            f"G3 visual-score {candidate.visual_score:.2f} < {threshold:.2f} "
            f"(multi={multi_engine})",
        )

    match_conf = candidate.visual_score

    # --- G4: FINAL-LLM verification (опц., дорогой, только для выживших) ---
    if llm_verifier is not None:
        try:
            same, llm_conf = await llm_verifier.verify_same_product(
                product_name=product_name,
                candidate_title=candidate.title,
                candidate_attrs=candidate_attrs_text,
            )
        except Exception as exc:  # noqa: BLE001 — LLM-сбой = сомнение = reject (G6)
            return GateResult(candidate, False, f"G4 llm-error: {exc}")
        if not same or llm_conf < LLM_VERIFY_CONF_MIN:
            return GateResult(
                candidate, False,
                f"G4 llm-verify reject (same={same}, conf={llm_conf:.2f})",
            )
        # Итоговая уверенность — консервативный min(visual, llm).
        match_conf = min(match_conf, llm_conf)

    return GateResult(candidate, True, "accepted", match_confidence=match_conf)


async def find_matching_card(
    image_url: str,
    product_name: str,
    *,
    our_brand: Optional[str] = None,
    target_type: Optional[str] = None,
    llm_verifier: Optional[LlmVerifier] = None,
    candidate_attrs_provider: Optional[Any] = None,
) -> Optional[GateResult]:
    """Главный вход: фото → лучшая ВЕРИФИЦИРОВАННАЯ карточка (или None).

    1. reverse-image-поиск (STUB) → кандидаты.
    2. merge дубликатов по card_id (накопить .engines для top-K agreement).
    3. каждый кандидат через gate_candidate (fail-closed).
    4. вернуть принятого кандидата с наибольшей match_confidence (или None).

    None означает «карточка по фото не подтверждена» → downstream НЕ заполняет
    ничего (empty > wrong). enum-gate + confidence-cap (G5) применяются уже на
    этапе записи значений (см. enum_gate_value / IMAGE_CARD_CONF_CAP).
    """
    candidates = await _reverse_image_search(image_url)
    if not candidates:
        return None

    merged = _merge_by_card_id(candidates)

    results: list[GateResult] = []
    for cand in merged:
        attrs_text = ""
        if candidate_attrs_provider is not None:
            try:
                attrs_text = await candidate_attrs_provider(cand)  # type: ignore[misc]
            except Exception:  # noqa: BLE001
                attrs_text = ""
        res = await gate_candidate(
            cand,
            our_brand=our_brand,
            target_type=target_type,
            product_name=product_name,
            candidate_attrs_text=attrs_text,
            llm_verifier=llm_verifier,
        )
        if res.accepted:
            results.append(res)
        else:
            logger.info("[ImageCardSearch] reject %s: %s", cand.url, res.reason)

    if not results:
        return None
    return max(results, key=lambda r: r.match_confidence)


def _merge_by_card_id(cands: list[ImageCandidate]) -> list[ImageCandidate]:
    """Слить кандидатов с одинаковым (marketplace, card_id): накопить движки и
    взять максимальный visual_score. Это база top-K agreement (G3)."""
    by_key: dict[tuple[str, str], ImageCandidate] = {}
    passthrough: list[ImageCandidate] = []
    for c in cands:
        if not c.card_id:
            passthrough.append(c)
            continue
        key = (c.marketplace, c.card_id)
        if key not in by_key:
            # копия, чтобы не мутировать вход
            by_key[key] = ImageCandidate(
                url=c.url, marketplace=c.marketplace, title=c.title,
                brand=c.brand, card_id=c.card_id, visual_score=c.visual_score,
                engines=set(c.engines),
            )
        else:
            m = by_key[key]
            m.visual_score = max(m.visual_score, c.visual_score)
            m.engines |= c.engines
            if not m.title and c.title:
                m.title = c.title
            if not m.brand and c.brand:
                m.brand = c.brand
    return list(by_key.values()) + passthrough


# ---------------------------------------------------------------------------
# G5: attribute-level enum gate + confidence cap
# ---------------------------------------------------------------------------

def enum_gate_value(
    cat_id: Optional[int],
    type_id: Optional[int],
    attribute_id: int,
    value: str,
) -> Optional[int]:
    """G5 (enum-gate): значение из image-карточки филлит атрибут ТОЛЬКО если
    резолвится в реальный Ozon value_id. Возвращает value_id или None (None →
    значение НЕ записываем для enum-таргета — empty > wrong).

    Тонкая обёртка над resolve_value_id с graceful-fallback (без падения).
    """
    if not cat_id or not type_id:
        return None
    try:
        from app.services.enrichment.strategies.dictionaries.ozon_loader import (
            resolve_value_id,
        )
        return resolve_value_id(cat_id, type_id, attribute_id, value)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[ImageCardSearch] enum_gate_value failed: %s", exc)
        return None


def capped_confidence(raw: float) -> float:
    """G5 (confidence cap): значения image-карточки не должны перебивать
    надёжные источники — капаем под IMAGE_CARD_CONF_CAP."""
    return min(raw, IMAGE_CARD_CONF_CAP)


# Строгий rubric для G4 LLM-verify (используется реальной реализацией LlmVerifier).
LLM_VERIFY_SYSTEM_PROMPT = (
    "You verify whether a marketplace product card refers to the EXACT SAME "
    "product as ours. Be STRICT: same product means same brand, same model/line, "
    "same generation. Different color/size of the SAME model = same product. "
    "Different model, generation, brand, or product type = NOT the same. "
    "If unsure, answer NO. Reply JSON: {\"same_product\": bool, \"confidence\": 0..1}. "
    "A wrong YES pollutes required fields — when in doubt, say NO."
)


__all__ = [
    "ImageCandidate", "GateResult", "LlmVerifier",
    "brand_matches", "type_matches",
    "gate_candidate", "find_matching_card",
    "enum_gate_value", "capped_confidence",
    "VISUAL_SCORE_MIN_SINGLE", "VISUAL_SCORE_MIN_MULTI",
    "LLM_VERIFY_CONF_MIN", "IMAGE_CARD_CONF_CAP",
    "LLM_VERIFY_SYSTEM_PROMPT",
]
