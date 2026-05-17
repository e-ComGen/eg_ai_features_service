import importlib
import pkgutil
import os
import csv
import datetime
from typing import Type
from pydantic import BaseModel, Field
from ..strategies.base import BaseStrategyNode
from ..strategies.definitions.root import RootStrategy
from .providers.structured_adapter import StructuredLlmManager

class RoutingDecision(BaseModel):
    selected_name: str = Field(..., description="The exact name of the selected child node.")
    reasoning: str = Field(..., description="Explain WHY you chose this node based on the product context.")

class TreeRouter:
    def __init__(self, llm_manager: StructuredLlmManager):
        self.llm_manager = llm_manager
        self.log_file = "tree_traversal_log.csv"
        self._load_all_strategies()
        self._init_log()

    def _load_all_strategies(self):
        def_path = os.path.join(os.path.dirname(__file__), '../strategies/definitions')
        for _, name, _ in pkgutil.iter_modules([def_path]):
            try:
                importlib.import_module(f"app.strategies.definitions.{name}")
            except Exception as e:
                print(f"⚠️ Could not load strategy {name}: {e}")

    def _init_log(self):
        with open(self.log_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Product_Snippet", "Feature", "Current_Node", "Selected_Node", "Reasoning"])

    def _log_step(self, text, feature, current, selected, reason):
        try:
            with open(self.log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                snippet = text.split('\n')[0][:50].replace("Title: ", "")
                writer.writerow([timestamp, snippet, feature, current, selected, reason])
        except Exception:
            pass

    async def find_instruction(self, product_text: str, feature_name: str, unit: str = "") -> tuple:
        print(f"🌳 [Router] Start traversal for '{feature_name}'...")
        return await self._traverse(RootStrategy, product_text, feature_name, unit)

    async def _traverse(self, current_cls, text: str, feature: str, unit: str) -> tuple:
        if current_cls.is_leaf():
            if hasattr(current_cls, 'get_task_logic'):
                return current_cls.get_instruction(unit=unit), {}, current_cls
            print(f"📍 [Leaf] Reached: {current_cls.name}")
            return current_cls.get_instruction(unit=unit), {}, current_cls

        children = current_cls.get_children()

        if not children:
            print(f"⚠️ Branch {current_cls.name} has no children! Fallback.")
            return "Extract value directly.", {}, current_cls

        options_str = current_cls.get_options_text()

        sys_msg = (
            f"You are a Decision Tree Router. Select the next step for extracting '{feature}'.\n"
            f"Product Context: {text[:200]}\n\n"
            f"Available Paths:\n{options_str}\n\n"
            "Select the most specific and relevant Child Node."
        )

        try:
            parsed, _tokens = await self.llm_manager.structured_request(
                system_prompt=sys_msg,
                user_text="Analyze and route.",
                response_model=RoutingDecision,
            )

            if parsed is None:
                print(f"⚠️ Router: structured_request returned None. Fallback.")
                return "Extract value directly.", {}, None

            target_name = parsed.selected_name
            reasoning = parsed.reasoning

            print(f"👉 {current_cls.name} -> {target_name} | 💭 {reasoning[:50]}...")
            self._log_step(text, feature, current_cls.name, target_name, reasoning)

            next_cls = next((c for c in children if c.name == target_name), None)

            current_debug = {
                "selected_node": target_name,
                "reasoning": reasoning
            }

            if next_cls:
                instruction, child_debug, matched_leaf = await self._traverse(next_cls, text, feature, unit)
                final_debug = child_debug if child_debug else current_debug
                return instruction, final_debug, matched_leaf

            print(f"⚠️ Invalid selection '{target_name}'. Fallback to first child.")
            return children[0].get_instruction(unit=unit), {}, children[0]

        except Exception as e:
            print(f"❌ Router Error: {e}")
            return "Extract value directly.", {}, None