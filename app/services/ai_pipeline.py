import csv
import os
import datetime
from typing import Optional, Union
from .llm_manager import OpenAIManager
from .tree_router import TreeRouter
from .web_search import WebSearchService
from ..models import ResearchMode
from ..strategies.base import DeductionResult
from ..judge.judge import HallucinationJudge
from ..config import OPENAI_API_KEY

class AiFeaturePipeline:

    def __init__(self, llm_manager,
                 web_search_model: str = "gpt-4o",
                 web_search_max_concurrent: int = 10):
        self.llm = llm_manager
        # TreeRouter needs a raw AsyncOpenAI client for beta.chat.completions.parse.
        # If llm_manager is a StructuredLlmManager or similar (no .client attr),
        # fall back to a dedicated OpenAIManager for routing only.
        if hasattr(llm_manager, 'client'):
            router_client = llm_manager.client
        else:
            # New provider path: construct a lightweight OpenAIManager just for routing.
            _openai_mgr = OpenAIManager(api_key=OPENAI_API_KEY)
            router_client = _openai_mgr.client
        self.router = TreeRouter(router_client)
        self.deduction_log_file = "deductions_log.csv"
        self.judge = HallucinationJudge(self.llm)
        # WebSearchService also needs raw AsyncOpenAI client (Responses API).
        self.web_search = WebSearchService(
            client=router_client,
            model=web_search_model,
            max_concurrent=web_search_max_concurrent,
        )

    def _log_deduction(self, feature_name: str, confidence: int, context: str):
        file_exists = os.path.isfile(self.deduction_log_file)
        try:
            with open(self.deduction_log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "Feature", "Confidence_Score", "Deduced_Context"])
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, feature_name, confidence, context])
        except Exception as e:
            print(f"⚠️ Could not write to deduction log: {e}")

    @staticmethod
    def _unpack_value(result):
        """Extracts the value from any leaf-specific Pydantic result schema."""
        if not result:
            return None
        if hasattr(result, 'extracted_values') and isinstance(result.extracted_values, list):
            val = {item.language: item.text for item in result.extracted_values}
            if all(str(v).strip().lower() == "none" for v in val.values()):
                return None
            return val
        if hasattr(result, 'extracted_value'):
            return result.extracted_value
        return None

    async def _try_knowledge_extract(self, product_text: str, feature_name: str, matched_leaf,
                                     suffix: str, options: list, target_languages: list[str]) -> tuple:
        """
        Fallback: ask the LLM to recall this exact product from its training knowledge.
        Returns (value, tokens, analysis).
        """
        if matched_leaf is None:
            return None, 0, "No leaf matched — knowledge fallback skipped."

        TargetModel = matched_leaf.get_response_model(options=options)
        domain_instruction = matched_leaf.get_instruction(
            unit=suffix, options=options, target_languages=target_languages
        )

        knowledge_sys_msg = (
            "ROLE: Product Knowledge Researcher.\n"
            "GOAL: The provided product description was insufficient to extract this feature. "
            "Use YOUR TRAINING KNOWLEDGE about this exact product (identified by its name, brand, and model number) "
            "to determine the value.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. Only use knowledge about THIS EXACT product or model. Do NOT generalize from product category.\n"
            "2. If you do not specifically remember this product, return None and set confidence='Low'.\n"
            "3. Do NOT invent. Do NOT guess based on similar products.\n"
            "4. Better to return None than to hallucinate.\n\n"
            f"{domain_instruction}\n\n"
            f"--- EXECUTION CONTEXT ---\n"
            f"TARGET FEATURE: '{feature_name}'\n"
            f"TARGET UNIT/SUFFIX: '{suffix}'\n"
        )

        user_text = (
            f"Identify the product from the snippet below and recall its '{feature_name}' "
            f"strictly from your knowledge (the description itself is too sparse):\n\n"
            f"{product_text}"
        )

        result, tokens = await self.llm.structured_request(
            system_prompt=knowledge_sys_msg,
            user_text=user_text,
            response_model=TargetModel
        )

        if not result or result.confidence == 'Low':
            return None, tokens, getattr(result, 'analysis', 'Knowledge fallback: low confidence or no response')

        val = self._unpack_value(result)
        return val, tokens, getattr(result, 'analysis', 'No analysis')

    async def extract_feature(self, product_text: str, feature_name: str, suffix: str = "",
                              options: list = None, target_languages: list[str] = None,
                              research_mode: ResearchMode = ResearchMode.OFF) -> dict:
        if target_languages is None:
            target_languages = ["en"]

        full_instruction, router_debug, matched_leaf = await self.router.find_instruction(
            product_text, feature_name, unit=suffix
        )

        TargetModel = matched_leaf.get_response_model(options=options)
        dynamic_instruction = matched_leaf.get_instruction(
            unit=suffix, options=options, target_languages=target_languages
        )

        final_sys_msg = (
            f"{dynamic_instruction}\n\n"
            f"--- EXECUTION CONTEXT ---\n"
            f"TARGET FEATURE: '{feature_name}'\n"
            f"TARGET UNIT/SUFFIX: '{suffix}'\n"
        )

        result, total_tokens = await self.llm.structured_request(
            system_prompt=final_sys_msg,
            user_text=product_text,
            response_model=TargetModel
        )

        val = None
        reasoning = "LLM Error or Empty Response"
        source = "description"
        source_urls: Optional[list[str]] = None
        # Audit context grows as fallbacks succeed; judge sees the latest.
        audit_extras: list[tuple[str, str]] = []

        if result and result.confidence != 'Low':
            reasoning = getattr(result, 'analysis', 'No analysis provided')
            val = self._unpack_value(result)

        final_deduced_context = None

        # ==========================================
        # ЭТАП 2: УМНАЯ ДЕДУКЦИЯ (по описанию)
        # ==========================================
        if val is None and matched_leaf and getattr(matched_leaf, 'supports_deduction', False):
            deduction_sys_msg = matched_leaf.get_deduction_instruction(feature_name, suffix)
            deduction_result, deduc_tokens = await self.llm.structured_request(
                system_prompt=deduction_sys_msg,
                user_text=product_text,
                response_model=DeductionResult
            )
            total_tokens += deduc_tokens

            if deduction_result:
                final_deduced_context = deduction_result.context_clues
                self._log_deduction(feature_name, deduction_result.confidence_score, final_deduced_context)

                if deduction_result.confidence_score >= 75:
                    enriched_user_text = (
                        f"--- ORIGINAL PRODUCT TEXT ---\n{product_text}\n\n"
                        f"--- DEDUCED CONTEXT (CRITICAL: MUST USE) ---\n{final_deduced_context}\n"
                        f"Use the DEDUCED CONTEXT to help identify the correct value, but your final answer MUST remain concise (e.g., 1-3 words) and strictly adhere to the expected data format."
                    )

                    retry_result, retry_tokens = await self.llm.structured_request(
                        system_prompt=final_sys_msg,
                        user_text=enriched_user_text,
                        response_model=TargetModel
                    )
                    total_tokens += retry_tokens

                    if retry_result and retry_result.confidence != 'Low':
                        val = self._unpack_value(retry_result)
                        retry_analysis = getattr(retry_result, 'analysis', 'No analysis')
                        reasoning = f"[DEDUCTION SUCCESS - Score {deduction_result.confidence_score}]: {retry_analysis}"
                        if val is not None:
                            audit_extras.append(("APPROVED DEDUCED CONTEXT", final_deduced_context))
                    else:
                        reasoning += f" | [DEDUCTION FAILED]: Strict extraction still failed."
                else:
                    score = deduction_result.confidence_score
                    reasoning += f" | [DEDUCTION REJECTED]: Low confidence score ({score}/100)."

        # ==========================================
        # ЭТАП 3: KNOWLEDGE FALLBACK (по памяти модели)
        # Включается для FALLBACK и DEEP режимов.
        # ==========================================
        if val is None and research_mode != ResearchMode.OFF:
            print(f"🧠 [Knowledge Fallback] Trying LLM memory for '{feature_name}' (mode={research_mode.value})")
            knowledge_val, knowledge_tokens, knowledge_analysis = await self._try_knowledge_extract(
                product_text=product_text,
                feature_name=feature_name,
                matched_leaf=matched_leaf,
                suffix=suffix,
                options=options,
                target_languages=target_languages,
            )
            total_tokens += knowledge_tokens

            if knowledge_val is not None:
                val = knowledge_val
                source = "knowledge"
                reasoning = f"[KNOWLEDGE FALLBACK | mode={research_mode.value}]: {knowledge_analysis}"
            else:
                reasoning += f" | [KNOWLEDGE FALLBACK FAILED]: {knowledge_analysis}"

        # ==========================================
        # ЭТАП 4: WEB SEARCH FALLBACK (только DEEP)
        # Самый дорогой шаг — стреляет только если description+deduction+knowledge все промахнулись.
        # ==========================================
        if val is None and research_mode == ResearchMode.DEEP:
            print(f"🌐 [Web Search] Trying web for '{feature_name}'")
            web_result = await self.web_search.search_product_feature(
                product_text=product_text,
                feature_name=feature_name,
                suffix=suffix,
            )
            total_tokens += web_result.tokens

            if web_result.error:
                reasoning += f" | [WEB SEARCH ERROR]: {web_result.error}"
            elif not web_result.answer_text:
                reasoning += " | [WEB SEARCH]: Empty result from API."
            else:
                citation_block = ""
                if web_result.citations:
                    citation_block = "\n\n--- SOURCES ---\n" + "\n".join(
                        f"[{i+1}] {c.title or 'source'}: {c.url}"
                        for i, c in enumerate(web_result.citations)
                    )
                enriched_text = (
                    f"--- ORIGINAL PRODUCT TEXT ---\n{product_text}\n\n"
                    f"--- WEB RESEARCH FINDINGS ---\n{web_result.answer_text}"
                    f"{citation_block}"
                )

                web_retry, retry_tokens = await self.llm.structured_request(
                    system_prompt=final_sys_msg,
                    user_text=enriched_text,
                    response_model=TargetModel
                )
                total_tokens += retry_tokens

                if web_retry and web_retry.confidence != 'Low':
                    web_val = self._unpack_value(web_retry)
                    if web_val is not None:
                        val = web_val
                        source = "web"
                        source_urls = [c.url for c in web_result.citations]
                        reasoning = f"[WEB SEARCH | mode=deep]: {getattr(web_retry, 'analysis', '')}"
                        # Judge will see the web findings as evidence.
                        audit_extras.append((
                            "WEB RESEARCH FINDINGS",
                            web_result.answer_text + citation_block
                        ))
                    else:
                        reasoning += " | [WEB SEARCH FAILED]: Re-extraction returned None."
                else:
                    reasoning += " | [WEB SEARCH FAILED]: Low confidence in re-extraction."

        judge_payload = None

        # ==========================================
        # ЭТАП 5: СУДЬЯ
        # Для source=knowledge стандартная проверка "значение в тексте" бессмысленна — пропускаем.
        # Для source=description и source=web судья работает с audit_text (с накопленным контекстом).
        # ==========================================
        if val is not None and source in ("description", "web"):
            validation_status = matched_leaf.evaluate_need_for_judgment(val, product_text, options)

            if validation_status == "REJECT" and source == "description":
                # FAST REJECT основан на наличии числа в исходном тексте.
                # Для source=web этот фильтр не применим (число пришло извне), пропускаем сразу к JUDGE.
                val = None
                reasoning += " | ⛔ FAST REJECT: Алгоритм выявил слепую галлюцинацию."

            elif validation_status == "ACCEPT":
                reasoning += " | ✅ FAST ACCEPT: Идеальное совпадение (Судья не потребовался)."

            elif validation_status == "JUDGE" or (validation_status == "REJECT" and source == "web"):
                judge_profile = matched_leaf.get_judge_profile()

                audit_text = product_text
                for label, content in audit_extras:
                    audit_text += f"\n\n--- {label} ---\n{content}"

                raw_verdict, judge_tokens = await self.judge.execute_audit(
                    text=audit_text,
                    feature_name=feature_name,
                    extracted_value=val,
                    profile=judge_profile
                )
                total_tokens += judge_tokens

                if raw_verdict:
                    judge_payload = raw_verdict.model_dump() if hasattr(raw_verdict, 'model_dump') else raw_verdict.dict()
                    judge_thoughts = getattr(raw_verdict, 'analysis', 'No reasoning')
                else:
                    judge_thoughts = "LLM Error"

                judgment = matched_leaf.process_judgment(raw_verdict, judge_tokens)

                if judgment.needs_review:
                    val = None
                    reasoning += f" | ⚖️ СУД ОТКЛОНИЛ: {judgment.status_message} | Мысли: {judge_thoughts}"
                else:
                    reasoning += f" | ⚖️ СУД ОДОБРИЛ: {judgment.status_message} | Мысли: {judge_thoughts}"
        elif val is not None and source == "knowledge":
            reasoning += " | ⏭ Judge skipped (source=knowledge: value not expected to be in text)."

        if val is None:
            source = None
            source_urls = None

        return {
            "value": val,
            "tokens": total_tokens,
            "router_debug": router_debug,
            "extraction_reasoning": reasoning,
            "deduced_context": final_deduced_context,
            "judge_data": judge_payload,
            "source": source,            # "description" | "knowledge" | "web" | None
            "source_urls": source_urls,  # list[str] | None — only for source="web"
        }
