"""CostPredictor — pre-flight check для дорогих этапов.

Перед запуском WebSearch (самый дорогой stage, ~$0.02-0.05) спрашивает LLM:
"найдём ли мы данные про этот товар в публичном вебе?". Для известных
брендов — да; для уникальных кастомов — нет, не тратим.

Spec: docs/architecture/pipeline.md, section "CostPredictor / Stage 4 gate".
"""
from typing import Optional
from pydantic import BaseModel, Field
from app.services.enrichment.base import TargetAttribute, ExtractionContext
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager


class _WorthVerdict(BaseModel):
    worth_it: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=200)


class CostPredictor:
    """LLM-based predictor: стоит ли запускать дорогой stage."""

    def __init__(self, llm_manager: Optional[StructuredLlmManager] = None):
        self._llm = llm_manager or get_main_manager()

    async def is_web_search_worth(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> bool:
        """True если есть реальная вероятность найти эти атрибуты в публичном вебе."""
        if not targets:
            return False

        # Budget gate (deterministic): если уже потратили >70% budget — не запускаем
        if context.cost_so_far_usd > context.max_cost_usd * 0.7:
            return False

        attrs_summary = ", ".join([a.name for a in targets[:10]])

        system_prompt = (
            "You decide whether running an expensive web search is worth it for a given product.\n"
            "Return worth_it=true if the product is well-known enough that authoritative sources "
            "(manufacturer site, major retailers, established review sites) likely have these specs.\n"
            "Return worth_it=false for:\n"
            "- No-name / generic / handmade products\n"
            "- Custom items or services\n"
            "- Attributes that are typically not published online (e.g., 'описание_продавца')\n"
            "Be honest. False negatives are OK (we'll just lack those attrs)."
        )
        user_text = (
            f"Product: {context.product_name}\n"
            f"Brand: {context.brand or 'unknown'}\n"
            f"Category: {' / '.join(context.category_path) or 'n/a'}\n"
            f"Attributes to find: {attrs_summary}\n\n"
            f"Is web search likely to find these?"
        )

        verdict, _ = await self._llm.structured_request(
            system_prompt=system_prompt,
            user_text=user_text,
            response_model=_WorthVerdict,
        )
        if verdict is None:
            # Default: yes (don't block on LLM failure)
            return True

        context.llm_calls_so_far += 1
        return verdict.worth_it
