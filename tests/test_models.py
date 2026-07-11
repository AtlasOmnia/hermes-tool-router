from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "hermes_tool_router_models_test"
spec = importlib.util.spec_from_file_location(
    PKG,
    ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
module = importlib.util.module_from_spec(spec)
sys.modules[PKG] = module
assert spec.loader is not None
spec.loader.exec_module(module)

from hermes_tool_router_models_test.models import RouteAction, RouteDecision


def test_route_decision_distinguishes_no_tools_from_full_fallback():
    no_tools = RouteDecision.no_tools(reason_code="answer_only")
    full = RouteDecision.full(reason_code="uncertain")

    assert no_tools.action is RouteAction.NO_TOOLS
    assert full.action is RouteAction.FULL
    assert no_tools != full


def test_narrow_route_requires_at_least_one_toolset():
    try:
        RouteDecision.narrow(set(), confidence=0.9, reason_code="bad")
    except ValueError as exc:
        assert "toolset" in str(exc).lower()
    else:
        raise AssertionError("empty narrow route must be rejected")


def test_missing_confidence_is_not_treated_as_perfect_confidence():
    decision = RouteDecision.full(reason_code="missing_confidence")
    assert decision.confidence is None
