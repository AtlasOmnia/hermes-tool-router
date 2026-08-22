"""Contract tests for policy.parse_classifier_payload (strict fail-open path)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "hermes_tool_router_policy_v2_test"
spec = importlib.util.spec_from_file_location(PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[PKG] = module
spec.loader.exec_module(module)

from hermes_tool_router_policy_v2_test.policy import parse_classifier_payload

AVAILABLE = {"web", "browser", "file", "terminal"}


def test_missing_confidence_fails_open():
    assert parse_classifier_payload('{"toolsets": ["web"]}', AVAILABLE, 0.9) == {"action": "full"}


def test_string_percent_confidence_is_tolerated():
    result = parse_classifier_payload(
        '{"toolsets": ["web"], "confidence": "95%"}', AVAILABLE, 0.9
    )
    assert result == {"action": "narrow", "toolsets": ["web"]}


def test_nonfinite_confidence_fails_open():
    assert parse_classifier_payload(
        '{"toolsets": ["web"], "confidence": NaN}', AVAILABLE, 0.9
    ) == {"action": "full"}


def test_below_threshold_fails_open():
    assert parse_classifier_payload(
        '{"toolsets": ["web"], "confidence": 0.5}', AVAILABLE, 0.9
    ) == {"action": "full"}


def test_all_signal_maps_to_full():
    assert parse_classifier_payload(
        '{"toolsets": ["all"], "confidence": 0.99}', AVAILABLE, 0.9
    ) == {"action": "full"}


def test_empty_toolset_list_is_no_tools():
    assert parse_classifier_payload(
        '{"toolsets": [], "confidence": 0.95}', AVAILABLE, 0.9
    ) == {"action": "no_tools"}


def test_unknown_toolsets_fail_open():
    assert parse_classifier_payload(
        '{"toolsets": ["nonexistent"], "confidence": 0.95}', AVAILABLE, 0.9
    ) == {"action": "full"}


def test_invalid_json_fails_open():
    assert parse_classifier_payload("not json at all", AVAILABLE, 0.9) == {"action": "full"}
