from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "hermes_tool_router_intent_test"
spec = importlib.util.spec_from_file_location(PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[PKG] = module
spec.loader.exec_module(module)

from hermes_tool_router_intent_test.intent import Intent, classify_intent


def test_conceptual_keyword_collisions_do_not_load_tools():
    for prompt in (
        "Explain how to open a file in Python",
        "What is a Git repository?",
        "Compare video codecs",
    ):
        result = classify_intent(prompt)
        assert result.intents == frozenset({Intent.ANSWER_ONLY}), prompt


def test_url_summary_uses_web_but_interaction_uses_browser():
    assert classify_intent("Summarize https://example.com").intents == frozenset({Intent.RESEARCH_WEB})
    assert classify_intent("Log into https://example.com and submit the form").intents == frozenset({Intent.INTERACT_BROWSER})


def test_media_analysis_is_not_confused_with_browser_or_generation():
    assert classify_intent("Review this screenshot").intents == frozenset({Intent.ANALYZE_IMAGE})
    assert classify_intent("Generate a logo image").intents == frozenset({Intent.GENERATE_IMAGE})


def test_desktop_window_capture_routes_to_computer_use():
    result = classify_intent("Capture the Safari window and tell me what is open")
    assert result.intents == frozenset({Intent.CONTROL_DESKTOP})


def test_explicit_tool_intents_beat_generic_action_words():
    cases = {
        "Load the hermes-agent skill before answering, then return the CLI command.": Intent.SKILL_OPERATION,
        "Use the cronjob tool to list scheduled jobs.": Intent.SCHEDULE_OPERATION,
        "Use session search to find the previous session where Tampa weather was tested.": Intent.SESSION_LOOKUP,
        "Use the sandboxed code_execution tool, not terminal, to calculate 7 times 9.": Intent.EXECUTE_CODE,
        "Use the todo tool to create a two-item task list.": Intent.TODO_OPERATION,
        "Use computer_use with action list_apps.": Intent.CONTROL_DESKTOP,
    }
    for prompt, expected in cases.items():
        assert classify_intent(prompt).intents == frozenset({expected}), prompt


def test_multi_intent_requests_return_union():
    result = classify_intent("Search the latest release notes, save them to a file, then run the tests")
    assert Intent.RESEARCH_WEB in result.intents
    assert Intent.WRITE_LOCAL in result.intents
    assert Intent.EXECUTE_LOCAL in result.intents
