from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "hermes_tool_router_classifier_test"
spec = importlib.util.spec_from_file_location(PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[PKG] = module
spec.loader.exec_module(module)

from hermes_tool_router_classifier_test.classifier import call_with_hard_timeout, parse_classifier_output
from hermes_tool_router_classifier_test.models import RouteAction


def test_missing_confidence_fails_open():
    decision = parse_classifier_output('{"action":"narrow","toolsets":["web"]}', {"web"}, 0.9)
    assert decision.action is RouteAction.FULL
    assert decision.reason_code == "missing_confidence"


def test_low_confidence_and_unknown_toolsets_fail_open():
    low = parse_classifier_output('{"action":"narrow","toolsets":["web"],"confidence":0.5}', {"web"}, 0.9)
    unknown = parse_classifier_output('{"action":"narrow","toolsets":["magic"],"confidence":0.99}', {"web"}, 0.9)
    assert low.action is RouteAction.FULL
    assert unknown.action is RouteAction.FULL


def test_valid_structured_output_narrows():
    decision = parse_classifier_output(
        '```json\n{"action":"narrow","toolsets":["web"],"confidence":0.96,"reason_code":"fresh"}\n```',
        {"web", "file"},
        0.9,
    )
    assert decision.action is RouteAction.NARROW
    assert decision.toolsets == frozenset({"web"})
    assert decision.confidence == 0.96


def test_malformed_output_fails_open():
    assert parse_classifier_output("not json", {"web"}, 0.9).action is RouteAction.FULL


def test_hard_timeout_returns_without_waiting_for_hung_worker():
    import time

    started = time.monotonic()
    completed, value = call_with_hard_timeout(lambda: (time.sleep(0.5), "late")[1], 0.03)
    elapsed = time.monotonic() - started
    assert completed is False
    assert value is None
    assert elapsed < 0.15
