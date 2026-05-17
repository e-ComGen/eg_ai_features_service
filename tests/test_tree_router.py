"""Unit tests for TreeRouter — migrated to StructuredLlmManager API.

All tests mock StructuredLlmManager.structured_request — no live LLM calls.
Run with: pytest tests/test_tree_router.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from app.services.tree_router import TreeRouter, RoutingDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(
    selected_name: str = "some_child",
    reasoning: str = "test reason",
    return_none: bool = False,
) -> MagicMock:
    """Build a mock StructuredLlmManager whose structured_request returns RoutingDecision."""
    manager = MagicMock()
    if return_none:
        manager.structured_request = AsyncMock(return_value=(None, 0))
    else:
        decision = RoutingDecision(selected_name=selected_name, reasoning=reasoning)
        manager.structured_request = AsyncMock(return_value=(decision, 42))
    return manager


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTreeRouterInit:
    def test_accepts_structured_llm_manager(self):
        """TreeRouter should accept a StructuredLlmManager (not a raw OpenAI client)."""
        manager = _make_manager()
        router = TreeRouter(llm_manager=manager)
        assert router.llm_manager is manager

    def test_does_not_require_openai_client(self):
        """No beta.chat or AsyncOpenAI should be needed to construct TreeRouter."""
        manager = _make_manager()
        # If TreeRouter still held a .client attribute this would fail type checks;
        # we just verify construction succeeds cleanly.
        router = TreeRouter(llm_manager=manager)
        assert not hasattr(router, "client")


class TestTreeRouterTraversal:
    @pytest.mark.asyncio
    async def test_calls_structured_request_with_routing_decision_model(self):
        """_traverse() must call structured_request with RoutingDecision as response_model."""
        manager = _make_manager(selected_name="_invalid_name_so_fallback_", reasoning="x")
        router = TreeRouter(llm_manager=manager)

        # find_instruction starts at RootStrategy which has children
        await router.find_instruction("some product text", "color")

        manager.structured_request.assert_called()
        call_kwargs = manager.structured_request.call_args.kwargs
        assert call_kwargs.get("response_model") is RoutingDecision

    @pytest.mark.asyncio
    async def test_returns_fallback_instruction_when_structured_request_returns_none(self):
        """When structured_request returns (None, 0), _traverse returns a safe fallback."""
        manager = _make_manager(return_none=True)
        router = TreeRouter(llm_manager=manager)

        instruction, debug, leaf = await router.find_instruction("product", "size")

        assert instruction == "Extract value directly."
        assert leaf is None

    @pytest.mark.asyncio
    async def test_returns_fallback_on_invalid_selected_name(self):
        """When LLM returns an unrecognised node name, the first child is used as fallback."""
        manager = _make_manager(selected_name="__nonexistent_node__", reasoning="oops")
        router = TreeRouter(llm_manager=manager)

        instruction, debug, leaf = await router.find_instruction("product text", "weight")

        # Should still return something (first child fallback), not crash
        assert instruction is not None

    @pytest.mark.asyncio
    async def test_traversal_exception_returns_safe_fallback(self):
        """Any exception inside _traverse is caught and returns a safe fallback tuple."""
        manager = MagicMock()
        manager.structured_request = AsyncMock(side_effect=RuntimeError("boom"))
        router = TreeRouter(llm_manager=manager)

        instruction, debug, leaf = await router.find_instruction("product", "color")

        assert instruction == "Extract value directly."
        assert leaf is None

    @pytest.mark.asyncio
    async def test_leaf_node_returns_without_calling_llm(self):
        """A leaf node should be returned immediately without any LLM call."""
        manager = _make_manager()
        router = TreeRouter(llm_manager=manager)

        # Import a known leaf strategy to ensure it is loaded
        from app.strategies.definitions.text import TextBranch  # noqa: F401 — loaded by _load_all_strategies

        # Directly call _traverse with a leaf class
        # We can identify a leaf by checking RootStrategy children for one
        from app.strategies.definitions.root import RootStrategy
        children = RootStrategy.get_children()
        if not children:
            pytest.skip("RootStrategy has no children — strategy definitions not loaded")

        # Walk until we find a leaf
        leaf_cls = None
        for child in children:
            if child.is_leaf():
                leaf_cls = child
                break

        if leaf_cls is None:
            pytest.skip("No leaf found among direct children of RootStrategy")

        instruction, debug, matched = await router._traverse(leaf_cls, "text", "feature", "")

        # LLM should NOT have been called (leaf resolved without routing)
        manager.structured_request.assert_not_called()
        assert instruction is not None
        assert matched is leaf_cls

    @pytest.mark.asyncio
    async def test_system_message_contains_feature_and_product_context(self):
        """The system prompt sent to LLM must include the feature name and product snippet."""
        manager = _make_manager(selected_name="__fallback__", reasoning="r")
        router = TreeRouter(llm_manager=manager)

        product_text = "This is a product description for a red widget"
        feature_name = "color_attribute"

        await router.find_instruction(product_text, feature_name)

        call_kwargs = manager.structured_request.call_args.kwargs
        sys_prompt = call_kwargs.get("system_prompt", "")

        assert feature_name in sys_prompt
        # Product context is truncated to 200 chars in the prompt
        assert product_text[:50] in sys_prompt


class TestTreeRouterDI:
    def test_manager_injected_not_created_internally(self):
        """TreeRouter should use the injected manager, not create its own."""
        manager = _make_manager()
        router = TreeRouter(llm_manager=manager)

        # No OpenAI client should have been instantiated
        assert router.llm_manager is manager
