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

from hermes_tool_router_policy_v2_test.policy import (
    ROUTER_SYSTEM_PROMPT,
    _extract_confidence,
    _predict_toolsets_by_rules,
)


AVAILABLE = {"web", "browser", "file", "terminal", "vision", "image_gen", "video", "video_gen"}


def test_policy_does_not_route_conceptual_keyword_collisions_to_tools():
    for prompt in ("Explain how to open a file in Python", "What is a Git repository?", "Compare video codecs"):
        predicted, reason = _predict_toolsets_by_rules(prompt, AVAILABLE)
        assert predicted == set(), (prompt, predicted, reason)


def test_policy_uses_minimum_toolset_for_url_summary():
    predicted, _ = _predict_toolsets_by_rules("Summarize https://example.com", AVAILABLE)
    assert predicted == {"web"}


def test_missing_classifier_confidence_is_unknown_not_perfect():
    assert _extract_confidence({"toolsets": ["web"]}) is None


def test_classifier_prompt_formats_literal_json_contract():
    prompt = ROUTER_SYSTEM_PROMPT.format(
        toolset_descriptions="  - web: current web data",
        user_message="What is the weather right now?",
    )

    assert 'Return ONLY JSON: {"toolsets": ["name"], "confidence": 0.0}.' in prompt
    assert '{"toolsets": ["terminal", "file"], "confidence": 0.96}' in prompt
    assert 'Personal calendar, itinerary, reservation, or flight-detail lookups require "skills" and "terminal"' in prompt
