"""Conftest для live integration tests.

Эти тесты ходят в реальные API и тратят токены user-а.
Запускаются ТОЛЬКО когда RUN_LIVE_TESTS=1 в env.
"""
import os
import pytest

LIVE_TESTS_ENABLED = os.getenv("RUN_LIVE_TESTS") == "1"

skip_unless_live = pytest.mark.skipif(
    not LIVE_TESTS_ENABLED,
    reason="Live integration tests require RUN_LIVE_TESTS=1 env var"
)

# Per-test cost cap (USD). Если live тест потратит больше — assertion fail.
MAX_COST_PER_TEST = 0.05
