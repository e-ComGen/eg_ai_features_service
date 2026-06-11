"""CompetitorRagSource — RAG-источник характеристик из реальных карточек Ozon.

Использует локальный Qdrant-индекс (file-based) для поиска top-K похожих товаров
по 384-dim эмбеддингу названия через sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
Каждый сосед уже прошёл модерацию Ozon.

Алгоритм (с LLM-фильтрацией):
  1. Vector search: top-10 кандидатов по cosine similarity.
  2. LLM relevance filter: DeepSeek проверяет, все ли кандидаты того же типа товара.
  3. Consensus voting только на отфильтрованных кандидатах (≥2 кандидатов требуется).
  4. Confidence: 0.7 + 0.05 * agree_count (capped at 0.95) — выше из-за LLM-фильтрации.

При сбое LLM-фильтра (timeout/error) — graceful degradation: старый consensus на raw top-5.

Spec: docs/architecture/pipeline.md (CompetitorRagSource — cheap pre-LLM stage).
"""
from __future__ import annotations

# Windows DLL fix: pyarrow/pandas должны загружаться ДО torch/sentence_transformers,
# иначе происходит access violation при загрузке pyarrow DLL после torch.
try:
    import pyarrow  # noqa: F401
    import pandas   # noqa: F401
except ImportError:
    pass

import logging
import math
import os
import socket
from collections import Counter
from typing import Optional, TYPE_CHECKING

from pydantic import BaseModel

from app.services.enrichment.base import (
    AttributeSource,
    AttributeValue,
    ExtractionContext,
    LlmJudge,
    Source,
    TargetAttribute,
)
from app.services.enrichment.judges.competitor_rag_judge import CompetitorRagJudge

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Путь к локальному Qdrant-индексу по умолчанию
_DEFAULT_INDEX_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "strategies",
    "dictionaries",
    "data",
    "ozon_rag.qdrant",
)

# Имя коллекции в Qdrant
_COLLECTION_NAME = "ozon_products"

# Top-K соседей для поиска — увеличен до 10 для LLM-фильтрации
_TOP_K = 10

# Fallback top-K при сбое LLM-фильтра
_FALLBACK_TOP_K = 5

# Минимальное количество отфильтрованных кандидатов для consensus.
# Снижено с 2 → 1: с category filter в retrieval шум резко падает,
# можно принимать одиночные голоса от категорийно-точных кандидатов.
_MIN_FILTERED_CANDIDATES = 1

# Модель эмбеддингов — ДОЛЖНА совпадать с build_ozon_rag_index.py.
# paraphrase-multilingual-MiniLM-L12-v2: 384-dim, быстрая, хорошо работает с русским.
_EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_EMBED_DIM = 384  # Размерность выходного вектора модели

# Системный промпт для LLM-фильтрации релевантности
_RELEVANCE_FILTER_SYSTEM = (
    "Ты определяешь является ли каждый из кандидатов тем же типом товара что запрос. "
    "Возвращай только индексы кандидатов которые точно того же типа."
)


class _RelevanceFilter(BaseModel):
    """Ответ LLM-фильтра: индексы релевантных кандидатов."""
    relevant_indices: list[int]


def _get_embedding(text: str) -> list[float]:
    """Вычислить 384-dim L2-нормированный эмбеддинг текста через sentence-transformers.

    Lazy-load: модель загружается только при первом вызове и кэшируется в _model_cache.
    Нормализация: normalize_embeddings=True, чтобы cosine search работал корректно.
    """
    from sentence_transformers import SentenceTransformer  # type: ignore
    global _model_cache
    if _model_cache is None:
        logger.info("[CompetitorRag] Loading embed model %s (dim=%d)", _EMBED_MODEL_NAME, _EMBED_DIM)
        _model_cache = SentenceTransformer(_EMBED_MODEL_NAME)
    vec = _model_cache.encode(text, normalize_embeddings=True)
    return vec.tolist()


# Кэш загруженной модели (singleton per process)
_model_cache = None

# ── Process-wide Qdrant embedded singleton ────────────────────────────────────
# Embedded local mode loads ALL vectors into numpy RAM (~3.7 GB for ozon_rag).
# Creating multiple QdrantClient(path=...) instances multiplies that footprint.
# This singleton ensures one shared client for the lifetime of the process.
_embedded_client_singleton = None


def _get_embedded_client_singleton(index_path: str):
    """Return (or lazily create) the process-wide embedded Qdrant client.

    Thread-safety: Python GIL protects the assignment; the client itself is
    NOT thread-safe for concurrent writes but reads are safe in practice.
    """
    global _embedded_client_singleton
    if _embedded_client_singleton is None:
        from qdrant_client import QdrantClient  # type: ignore
        logger.info(
            "[CompetitorRag] Creating process-wide embedded Qdrant client at %s "
            "(first call — vectors load into RAM, ~1-2 min for large index)",
            index_path,
        )
        _embedded_client_singleton = QdrantClient(path=index_path)
    return _embedded_client_singleton


# ── One-time server reachability probe ────────────────────────────────────────
# We probe the Qdrant server URL ONCE per process with a cheap TCP connect
# (1-second timeout).  Result is cached so we never do a per-product dead-server
# roundtrip again.
_server_reachable_cache: Optional[bool] = None


def _probe_server_reachable(url: str, timeout: float = 1.0) -> bool:
    """TCP-probe url (http[s]://host:port) — returns True if port is open.

    This is intentionally cheap: one connect(), no HTTP, no TLS handshake.
    Called at most once per process.
    """
    try:
        # Strip scheme
        host_port = url.split("://", 1)[-1].rstrip("/").split("/")[0]
        if ":" in host_port:
            host, port_str = host_port.rsplit(":", 1)
            port = int(port_str)
        else:
            host = host_port
            port = 443 if url.startswith("https") else 80
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _build_relevance_user_text(query_name: str, candidates: list[dict]) -> str:
    """Собрать user-текст для LLM-фильтра: запрос + список кандидатов.

    Для каждого кандидата показываем imt_name + категорию из поля categories (если есть).
    """
    lines = [f"Запрос: {query_name}", "", "Кандидаты:"]
    for i, c in enumerate(candidates):
        imt_name = c.get("imt_name") or c.get("name") or "(без имени)"
        # Категорийный путь из поля categories (список строк или dict)
        cats_raw = c.get("categories")
        cat_path = ""
        if isinstance(cats_raw, list) and cats_raw:
            # Список строк вида ["Электроника", "Цепи"]
            cat_path = " > ".join(str(x) for x in cats_raw)
        elif isinstance(cats_raw, dict):
            # Может быть {id: name} или подобное — берём значения
            cat_path = " > ".join(str(v) for v in cats_raw.values())
        if cat_path:
            lines.append(f"  [{i}] {imt_name} (категория: {cat_path})")
        else:
            lines.append(f"  [{i}] {imt_name}")
    return "\n".join(lines)


class CompetitorRagSource(AttributeSource):
    """Извлекает характеристики по consensus из top-K похожих карточек Ozon.

    Алгоритм с LLM-фильтрацией:
      1. Vector search: top-10 кандидатов.
      2. LLM relevance filter (DeepSeek): отсеиваем нерелевантные товары.
      3. Consensus voting на отфильтрованном подмножестве (≥2 кандидатов).

    При сбое LLM-фильтра — graceful degradation на raw top-5 с прежним min_consensus=2.
    Cache: LLM_CACHE_ENABLED=1 гарантирует бесплатные повторные запуски.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        collection_name: str = _COLLECTION_NAME,
        top_k: int = _TOP_K,
        min_consensus: int = _MIN_FILTERED_CANDIDATES,
        embed_model_name: str = _EMBED_MODEL_NAME,
        llm_manager=None,
        qdrant_url: Optional[str] = None,
    ):
        # QDRANT_URL env → HTTP server mode (preferred for 2M+ points).
        # Falls back to embedded local mode using index_path.
        self._qdrant_url = qdrant_url or os.environ.get("QDRANT_URL")
        self._index_path = os.path.abspath(index_path or _DEFAULT_INDEX_PATH)
        self._collection_name = collection_name
        self._top_k = top_k
        self._min_consensus = min_consensus
        self._embed_model_name = embed_model_name
        self._client = None   # lazy-init при первом использовании
        # True после graceful fallback на embedded-индекс (сервер был недоступен).
        self._fellback_to_embedded = False
        self._judge = CompetitorRagJudge()
        # LLM-менеджер для relevance filter — инициализируется лениво
        self._llm_manager = llm_manager
        # Server reachability is decided ONCE per-instance at first _get_client call.
        # (The process-wide _server_reachable_cache covers the common case of one instance.)

    def _get_client(self):
        """Lazy-init Qdrant client. HTTP server mode if QDRANT_URL set AND reachable, else embedded singleton."""
        if self._client is not None:
            return self._client

        global _server_reachable_cache

        use_server = False
        if self._qdrant_url and not self._fellback_to_embedded:
            # Probe once per process — avoids per-product dead-server roundtrip.
            if _server_reachable_cache is None:
                _server_reachable_cache = _probe_server_reachable(self._qdrant_url)
                if not _server_reachable_cache:
                    logger.warning(
                        "[CompetitorRagSource] QDRANT_URL=%s is unreachable (TCP probe failed) — "
                        "using embedded singleton for the whole run. "
                        "RAM residual: full vector set stays in numpy (~3.7 GB). "
                        "For true RAM reduction run qdrant-server out-of-process.",
                        self._qdrant_url,
                    )
            use_server = _server_reachable_cache

        if use_server:
            from qdrant_client import QdrantClient  # type: ignore
            self._client = QdrantClient(url=self._qdrant_url, timeout=60)
            logger.info("[CompetitorRagSource] using Qdrant server at %s", self._qdrant_url)
        else:
            self._client = self._build_embedded_client()

        return self._client

    def _build_embedded_client(self):
        """Return the process-wide embedded Qdrant singleton for _index_path.

        IMPORTANT — RAM reality check:
          qdrant-client local mode loads ALL vectors into numpy arrays in RAM.
          For the ozon_rag index (384-dim, ~2M points) that is ~3 GB.
          There is NO mmap/on_disk option in QdrantClient(path=...) v1.18.
          The only way to avoid this RAM cost is to run qdrant-server as a
          separate OS process (which uses its own mmap). This singleton at
          least ensures the cost is paid ONCE per Python process instead of
          once per CompetitorRagSource instance / product.
        """
        return _get_embedded_client_singleton(self._index_path)

    def _fallback_to_embedded(self, exc: Exception) -> bool:
        """Переключиться на embedded-индекс при недоступности Qdrant-сервера.

        Вызывается, когда запрос к серверному клиенту упал по сетевой причине.
        Если локальный индекс существует на диске — пересоздаём клиент в local mode
        и возвращаем True (можно повторить запрос). Иначе — False (фейл наверх).
        """
        if self._fellback_to_embedded:
            # Уже в embedded-режиме — повторный fallback невозможен, это уже не серверная ошибка.
            return False
        if not os.path.isdir(self._index_path):
            logger.error(
                "[CompetitorRagSource] Qdrant server unreachable (%s) and no local "
                "embedded index at %s — cannot fallback. Запусти Qdrant (QDRANT_URL=%s) "
                "или положи индекс на диск.",
                exc, self._index_path, self._qdrant_url,
            )
            return False
        logger.warning(
            "[CompetitorRagSource] Qdrant server unreachable (%s) -> fallback to embedded "
            "singleton at %s.",
            exc, self._index_path,
        )
        # Update process-wide cache so no other instance probes again.
        global _server_reachable_cache
        _server_reachable_cache = False

        old = self._client
        self._fellback_to_embedded = True
        self._client = None  # will be recreated via _build_embedded_client → singleton
        try:
            if old is not None:
                old.close()
        except Exception:
            pass
        self._client = self._build_embedded_client()
        return True

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """Эвристика: ошибка похожа на недоступность Qdrant-сервера (сеть/timeout/5xx).

        Покрывает requests/httpx connection errors, qdrant ResponseHandlingException,
        UnexpectedResponse с 5xx и таймауты — всё, что значит «сервер не отвечает».
        """
        from qdrant_client.http.exceptions import (  # type: ignore
            ResponseHandlingException,
            UnexpectedResponse,
        )
        if isinstance(exc, ResponseHandlingException):
            return True
        if isinstance(exc, UnexpectedResponse):
            # 5xx / 503 — серверная недоступность; 4xx — это уже наша ошибка запроса.
            status = getattr(exc, "status_code", None)
            return status is None or status >= 500
        text = f"{type(exc).__name__}: {exc}".lower()
        markers = (
            "connection", "connect", "timed out", "timeout", "refused",
            "unreachable", "503", "502", "504", "max retries", "newconnectionerror",
        )
        return any(m in text for m in markers)

    def _get_llm_manager(self):
        """Lazy-инициализация LLM-менеджера (DeepSeek по умолчанию)."""
        if self._llm_manager is None:
            from app.services.providers.factory import get_main_manager
            self._llm_manager = get_main_manager()
        return self._llm_manager

    @property
    def source_type(self) -> Source:
        return Source.COMPETITOR_RAG

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Применим для любого товара с product_name (не требует description/images)."""
        return bool(context.product_name and context.product_name.strip())

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Найти похожие Ozon-карточки, отфильтровать LLM и извлечь consensus характеристики.

        Алгоритм:
        1. Вычислить эмбеддинг product_name.
        2. Найти top-10 ближайших в Qdrant.
        3. LLM relevance filter: отсеять нерелевантные товары другого типа.
        4. Если отфильтрованных < 2 → return [] (нет надёжного консенсуса).
        5. Для каждого target.name собрать «голоса» из отфильтрованных кандидатов.
        6. Консенсус ≥ ceil(len(filtered)/2) → emit AV(confidence=0.7+0.05*agree_count).
        """
        if not targets:
            return []

        # Определяем атрибуты которые уже заполнены
        already_filled_ids: set[int] = set()
        if already_filled:
            already_filled_ids = {av.attribute_id for av in already_filled}

        effective_targets = [t for t in targets if t.id not in already_filled_ids]
        if not effective_targets:
            return []

        # Вычислить эмбеддинг
        try:
            query_vector = _get_embedding(context.product_name)
        except Exception as e:
            logger.warning("[CompetitorRag] embed failed: %s", e)
            return []

        # Категорийный фильтр для отсечения мусора (EVGA→замки EVVA, ASUS→мыши/ноутбуки).
        # Используем самую глубокую категорию из category_path как text-match.
        cat_filter_text: Optional[str] = None
        if context.category_path:
            for level in reversed(context.category_path):
                if level and len(level) >= 4:
                    cat_filter_text = level
                    break

        # Поиск в Qdrant (top-10)
        try:
            neighbors = self._search_neighbors(query_vector, cat_filter_text)
        except Exception as e:
            logger.warning("[CompetitorRag] qdrant search failed: %s", e)
            return []

        if not neighbors:
            return []

        # LLM relevance filter — отсеиваем нерелевантные товары другого типа
        filtered_neighbors = await self._apply_relevance_filter(
            context.product_name, neighbors
        )

        if filtered_neighbors is None:
            # Graceful degradation: LLM-фильтр упал → старый consensus на raw top-5
            logger.warning(
                "[CompetitorRag] LLM relevance filter failed, falling back to raw top-5 consensus"
            )
            fallback = neighbors[:_FALLBACK_TOP_K]
            return self._aggregate_consensus_legacy(fallback, effective_targets, min_consensus=2)

        if len(filtered_neighbors) < _MIN_FILTERED_CANDIDATES:
            # Нет надёжного консенсуса — лучше пропустить, чем добавить мусор
            logger.info(
                "[CompetitorRag] Only %d/%d candidates passed relevance filter for '%s' → skip",
                len(filtered_neighbors), len(neighbors), context.product_name[:60],
            )
            return []

        # Consensus voting на отфильтрованных кандидатах
        return self._aggregate_consensus(filtered_neighbors, effective_targets)

    async def _apply_relevance_filter(
        self,
        query_name: str,
        candidates: list[dict],
    ) -> Optional[list[dict]]:
        """LLM-фильтр релевантности: оставить только кандидатов того же типа товара.

        Использует DeepSeek (дешевый, ~$0.0001/вызов).
        Кэшируется через LLM_CACHE_ENABLED=1 — бесплатен при повторных запусках.

        Returns:
            Отфильтрованный список кандидатов, или None при ошибке (fallback).
        """
        try:
            manager = self._get_llm_manager()
            user_text = _build_relevance_user_text(query_name, candidates)
            result, _tokens = await manager.structured_request(
                _RELEVANCE_FILTER_SYSTEM,
                user_text,
                _RelevanceFilter,
            )
            if result is None:
                return None

            # Фильтруем кандидатов по возвращённым индексам
            valid_indices = [
                i for i in result.relevant_indices
                if 0 <= i < len(candidates)
            ]
            logger.info(
                "[CompetitorRag] Relevance filter: %d/%d candidates kept for '%s' (indices=%s)",
                len(valid_indices), len(candidates), query_name[:60], valid_indices,
            )
            return [candidates[i] for i in valid_indices]

        except Exception as e:
            logger.warning("[CompetitorRag] relevance filter exception: %s", e)
            return None

    @staticmethod
    def _is_filter_index_error(exc: Exception) -> bool:
        """Return True when a 4xx UnexpectedResponse signals a missing payload index.

        This happens when MatchText is applied to a field that has no text index.
        Qdrant returns 400 Bad Request in that case.  It is NOT a connection problem —
        the server is alive; we just need to retry without the filter.
        """
        from qdrant_client.http.exceptions import UnexpectedResponse  # type: ignore
        if isinstance(exc, UnexpectedResponse):
            status = getattr(exc, "status_code", None)
            return status is not None and 400 <= status < 500
        # Local embedded mode raises ValueError for unsupported filter ops
        return isinstance(exc, (ValueError, TypeError)) and "index" in str(exc).lower()

    def _run_query(
        self,
        query_vector: list[float],
        query_filter,
    ):
        """Execute query_points, falling back to embedded on connection error."""
        try:
            return self._get_client().query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=self._top_k,
                with_payload=True,
                query_filter=query_filter,
            )
        except Exception as exc:
            if self._is_connection_error(exc) and self._fallback_to_embedded(exc):
                return self._get_client().query_points(
                    collection_name=self._collection_name,
                    query=query_vector,
                    limit=self._top_k,
                    with_payload=True,
                    query_filter=query_filter,
                )
            raise

    def _search_neighbors(
        self,
        query_vector: list[float],
        category_filter_text: Optional[str] = None,
    ) -> list[dict]:
        """Vector search в Qdrant с опциональным category text-фильтром.

        Если задан category_filter_text — фильтруем кандидатов чьё поле
        `categories` содержит эту строку (через text-payload-index). Это резко
        снижает шум на запросах типа "EVGA SuperNOVA" (иначе ловит замки EVVA).

        Graceful degrade: if the filtered query fails with a 4xx (missing text index),
        automatically retries WITHOUT the category filter so fills are never blocked
        by an unbuilt index.  A warning is logged once so the missing index is obvious.
        """
        query_filter = None
        if category_filter_text:
            from qdrant_client.models import Filter, FieldCondition, MatchText  # type: ignore
            query_filter = Filter(
                must=[FieldCondition(
                    key="categories",
                    match=MatchText(text=category_filter_text),
                )]
            )

        already_retried_unfiltered = False
        try:
            response = self._run_query(query_vector, query_filter)
        except Exception as exc:
            if query_filter is not None and self._is_filter_index_error(exc):
                # Missing text index on `categories` — retry without filter.
                # Fills are restored; noise suppression is just disabled until index is built.
                logger.warning(
                    "[CompetitorRag] category filter failed (%s: %s) — "
                    "retrying without filter. Build the text index to restore noise-reduction "
                    "(run scripts/build_rag_text_index.py once).",
                    type(exc).__name__, exc,
                )
                response = self._run_query(query_vector, None)
                already_retried_unfiltered = True
            else:
                raise

        neighbors = []
        for hit in response.points:
            if hit.payload:
                neighbors.append(hit.payload)

        # Empty-result fallback: category taxonomy mismatch (Ozon leaf vs EPG) causes
        # MatchText to return 0 results silently — no 4xx, just an empty list.
        # If a filter WAS applied AND we haven't already retried (exception path above),
        # AND results are below the floor, retry without the filter.
        # The downstream LLM relevance filter + consensus already remove off-topic noise.
        _EMPTY_FALLBACK_FLOOR = max(_MIN_FILTERED_CANDIDATES, 3)
        if query_filter is not None and not already_retried_unfiltered and len(neighbors) < _EMPTY_FALLBACK_FLOOR:
            logger.info(
                "[CompetitorRag] category filter '%s' returned %d results (< floor %d) — "
                "taxonomy mismatch suspected; retrying without filter.",
                category_filter_text, len(neighbors), _EMPTY_FALLBACK_FLOOR,
            )
            unfiltered_response = self._run_query(query_vector, None)
            neighbors = [
                hit.payload for hit in unfiltered_response.points if hit.payload
            ]

        return neighbors

    def _aggregate_consensus(
        self,
        neighbors: list[dict],
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """Найти consensus значения характеристик среди LLM-отфильтрованных кандидатов.

        Консенсус = ≥ ceil(len(filtered)/2) кандидатов с одинаковым значением.
        Confidence = 0.7 + 0.05 * agree_count (capped at 0.95).
        Повышенная уверенность относительно legacy-версии: кандидаты уже проверены LLM.
        """
        results: list[AttributeValue] = []
        n = len(neighbors)
        min_agree = math.ceil(n / 2)  # минимум половина отфильтрованных согласны

        for target in targets:
            # Собираем голоса: value → count
            votes: Counter[str] = Counter()
            for neighbor in neighbors:
                characteristics = self._coerce_characteristics(neighbor.get("characteristics"))
                value = self._find_attr_value(characteristics, target.name)
                if value is not None:
                    votes[str(value)] += 1

            if not votes:
                continue

            best_value, best_count = votes.most_common(1)[0]
            if best_count < min_agree:
                continue  # консенсуса нет

            # Уверенность выше, т.к. кандидаты отфильтрованы LLM
            confidence = min(0.7 + 0.05 * best_count, 0.95)
            evidence = f"seen in {best_count}/{n} similar Ozon cards (LLM-filtered)"

            results.append(AttributeValue(
                attribute_id=target.id,
                value=best_value,
                confidence=confidence,
                source=Source.COMPETITOR_RAG,
                evidence=evidence,
                semantic_type=target.semantic_type,
                is_collection=target.is_collection,
            ))

        return results

    def _aggregate_consensus_legacy(
        self,
        neighbors: list[dict],
        targets: list[TargetAttribute],
        min_consensus: int = 2,
    ) -> list[AttributeValue]:
        """Старый consensus-алгоритм — используется как fallback при сбое LLM-фильтра.

        Минимум min_consensus соседей с одинаковым значением.
        Confidence: 0.6 + 0.4 * (best_count / len(neighbors)).
        """
        results: list[AttributeValue] = []

        for target in targets:
            votes: Counter[str] = Counter()
            for neighbor in neighbors:
                characteristics = self._coerce_characteristics(neighbor.get("characteristics"))
                value = self._find_attr_value(characteristics, target.name)
                if value is not None:
                    votes[str(value)] += 1

            if not votes:
                continue

            best_value, best_count = votes.most_common(1)[0]
            if best_count < min_consensus:
                continue

            confidence = 0.6 + 0.4 * (best_count / len(neighbors))
            evidence = f"seen in {best_count}/{len(neighbors)} similar Ozon cards"

            results.append(AttributeValue(
                attribute_id=target.id,
                value=best_value,
                confidence=confidence,
                source=Source.COMPETITOR_RAG,
                evidence=evidence,
                semantic_type=target.semantic_type,
                is_collection=target.is_collection,
            ))

        return results

    @staticmethod
    def _coerce_characteristics(raw) -> dict:
        """Parse characteristics payload field to dict, handling JSON-string encoding.

        Real Qdrant payloads store characteristics as a JSON-encoded string, e.g.
            '{"Цвет товара": ["белый"], "Бренд": ["1 Toy"]}'
        rather than a nested dict.  This helper normalises both shapes so consensus
        voting works regardless of index format version.

        Returns an empty dict on malformed JSON or unexpected type (never raises).
        """
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            import json as _json
            try:
                parsed = _json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except _json.JSONDecodeError:
                return {}
        return {}

    def _find_attr_value(self, characteristics: dict, target_name: str) -> Optional[str]:
        """Найти значение атрибута по имени цели в словаре characteristics.

        Поиск нечёткий: нормализуем к нижнему регистру, убираем пробелы.
        characteristics формат из датасета: {attr_name: [val1, val2, ...]} или {attr_name: val}.
        """
        target_lower = target_name.lower().strip()

        for key, val in characteristics.items():
            key_lower = key.lower().strip()
            # Прямое совпадение или совпадение с нормализацией
            if key_lower == target_lower or key_lower.replace(" ", "_") == target_lower.replace(" ", "_"):
                # Извлечь первое значение из списка или скаляр
                if isinstance(val, list) and val:
                    return str(val[0])
                elif val is not None:
                    return str(val)

        return None

    def get_judge(self) -> LlmJudge:
        return self._judge
