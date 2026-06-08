"""WbApparelRagSource — тонкий subclass CompetitorRagSource для apparel-RAG из WB.

Запрашивает Qdrant-коллекцию `wb_apparel_rag` (built by
scripts/build_wb_apparel_rag_index.py из nyuuzyou/wb-products), payload которой
зеркалит ozon_rag ({variantid, name, description, categories, characteristics}).
Вся query/consensus/LLM-filter логика наследуется без изменений.

ВЫБОР enum-vs-reuse: НЕ добавляем отдельный Source.WB_APPAREL_RAG. Это разнесло бы
изменения по SOURCE_PRIORITY, SOURCE_CONFIDENCE_THRESHOLDS, judge dispatch, merge и
тестам ради источника, который по умолчанию ВЫКЛЮЧЕН. Вместо этого переиспользуем
Source.COMPETITOR_RAG (source_type наследуется) — judge/priority/threshold работают
без правок. Меняем только коллекцию и путь к индексу.

Safety-гейты (по требованию владельца — источник не должен ошибочно заполнять
REQUIRED-поля):
  (a) emit ТОЛЬКО если qdrant-коллекция реально существует (graceful no-op, пока
      индекс не построен — ничего не ломается);
  (b) для хрупкого "Состав"/material emit ТОЛЬКО когда значение резолвится в
      реальный Ozon enum value_id (иначе skip);
  (c) сохраняем родительский consensus≥ceil(n/2) gate.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    TargetAttribute,
)
from app.services.enrichment.sources.competitor_rag_source import CompetitorRagSource

logger = logging.getLogger(__name__)

# Коллекция и путь к индексу — зеркало build_wb_apparel_rag_index.py.
WB_APPAREL_COLLECTION = "wb_apparel_rag"
_WB_APPAREL_INDEX_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "strategies",
    "dictionaries",
    "data",
    "wb_apparel_rag.qdrant",
)

# Имена целевых полей, для которых требуем резолв в реальный Ozon enum value_id.
# "Состав"/материал — хрупкий: WB-карточки шумят, а REQUIRED material легко испортить.
_MATERIAL_TARGET_NAMES = {"состав", "материал", "материал изделия", "основной материал"}


class WbApparelRagSource(CompetitorRagSource):
    """RAG из WB-apparel карточек. Subclass CompetitorRagSource: только коллекция/путь.

    Дополнительно гейтит результат: no-op при отсутствии коллекции, и для material
    эмитит лишь значения, резолвящиеся в Ozon enum value_id.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        collection_name: str = WB_APPAREL_COLLECTION,
        **kwargs,
    ):
        super().__init__(
            index_path=index_path or _WB_APPAREL_INDEX_PATH,
            collection_name=collection_name,
            **kwargs,
        )

    def _list_collections(self) -> Optional[set[str]]:
        """Вернуть множество имён коллекций; None → недоступно.

        Зеркалит fallback-логику родителя: если задан QDRANT_URL, но сервер
        недоступен (connection/5xx), грациозно падаем на embedded-индекс и
        повторяем (как _search_neighbors). Так гейт не уходит в ложную дормантность,
        когда индекс лежит на диске, а сервер просто не поднят.
        """
        try:
            client = self._get_client()
            return {c.name for c in client.get_collections().collections}
        except Exception as e:  # noqa: BLE001
            if self._is_connection_error(e) and self._fallback_to_embedded(e):
                try:
                    client = self._get_client()
                    return {c.name for c in client.get_collections().collections}
                except Exception as e2:  # noqa: BLE001
                    logger.info(
                        "[WbApparelRag] collection check failed after fallback (%s) "
                        "→ source dormant", e2,
                    )
                    return None
            logger.info(
                "[WbApparelRag] collection check failed (%s) → source dormant", e
            )
            return None

    def _collection_exists(self) -> bool:
        """Gate (a): True только если qdrant-коллекция реально доступна.

        Graceful no-op, пока индекс не построен: в embedded-режиме (нет QDRANT_URL)
        директории индекса нет → False. С QDRANT_URL полагаемся на _list_collections,
        которая сама делает fallback на embedded при недоступном сервере.
        Любая недоступность → False (источник дормантен).
        """
        # Чистый embedded-режим: индекс — директория на диске. Нет её → нет коллекции.
        if not self._qdrant_url and not os.path.isdir(self._index_path):
            return False
        names = self._list_collections()
        if names is None:
            return False
        return self._collection_name in names

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Как у родителя, но с gate (a) перед запросом и gate (b) над material.

        Gate (a): no-op если коллекции нет (индекс ещё не построен).
        Gate (b): material-значения, не резолвящиеся в Ozon enum value_id, выкидываем.
        Gate (c): наследуется — consensus≥ceil(n/2) внутри родительского extract.
        """
        if not targets:
            return []

        # Gate (a): дормантность пока коллекция/индекс не существуют.
        if not self._collection_exists():
            return []

        values = await super().extract(context, targets, already_filled=already_filled)
        if not values:
            return values

        # Gate (b): хрупкий material — оставляем только value_id-резолвимые.
        targets_by_id = {t.id: t for t in targets}
        return [v for v in values if self._material_gate_ok(v, targets_by_id, context)]

    def _material_gate_ok(
        self,
        value: AttributeValue,
        targets_by_id: dict[int, TargetAttribute],
        context: ExtractionContext,
    ) -> bool:
        """Gate (b): для material-таргета требуем резолв в реальный Ozon enum value_id.

        Не-material значения пропускаем без изменений. Для material — резолвим строку
        через ozon_loader.resolve_value_id; если id найден → проставляем его в AV и
        пропускаем, иначе drop (чтобы не заполнять REQUIRED "Состав" мусором).
        """
        target = targets_by_id.get(value.attribute_id)
        if target is None:
            return True
        if (target.name or "").lower().strip() not in _MATERIAL_TARGET_NAMES:
            return True

        try:
            from app.services.enrichment.strategies.dictionaries.ozon_loader import (
                resolve_value_id,
            )
        except Exception:  # noqa: BLE001 — нет словаря → не можем верифицировать → drop
            return False

        type_id = getattr(context, "ozon_type_id", None)
        raw = value.value
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        if raw is None:
            return False

        try:
            vid = resolve_value_id(context.category_id, type_id, value.attribute_id, str(raw))
        except Exception as e:  # noqa: BLE001
            logger.info("[WbApparelRag] material resolve_value_id failed (%s) → drop", e)
            return False

        if vid is None:
            logger.info(
                "[WbApparelRag] material '%s' did not resolve to Ozon value_id → drop",
                str(raw)[:60],
            )
            return False

        # Резолвилось — проставим id, чтобы downstream писал enum напрямую.
        value.value_id = vid
        return True
