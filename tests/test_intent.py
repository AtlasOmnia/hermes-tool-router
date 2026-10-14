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



CAPABILITY_PROMPTS = (
    "What can you do?",
    "What else can you do?",
    "What are your capabilities?",
    "What tools can you use?",
    "What tools do you have access to?",
    "Which tools are available to you?",
    "what toolsets are loaded right now",
    "What do you have access to?",
    "List your tools",
    "Show me your available tools",
    "Tell me your capabilities",
    "What integrations do you have?",
    "What can you help me with?",
)

NOT_CAPABILITY_PROMPTS = (
    "Can you read the file notes.txt?",
    "Can you send a Telegram message to Ana saying hi?",
    "Are you able to run shell commands?",
    "What tools are used in this repo?",
    "What is a Git repository?",
    "Search the web for the latest news",
    "What tools can you use? Also open https://example.com and log in",
    # Review-reported task prompts that must keep their action routing.
    "Search the web for which tools support SQLite.",
    "What tools can you use? Then run pytest.",
    "Which integrations launched today? Search the web.",
    "Inspect a repository and tell me what tools it uses.",
    "What tools can you use to search the web for SQLite news?",
    "Which tools are available for SQLite?",
    "What can you do with the screenshot I just took?",
    "What could you do to speed up the build?",
    "What tools do you have for git? commit my changes",
    "Tell me about your available tools for image generation and then generate a cat picture",
    "What tools do you have access to? Then read notes.txt",
)


def test_capability_questions_are_capability_inquiries():
    for prompt in CAPABILITY_PROMPTS:
        result = classify_intent(prompt)
        assert result.intents == frozenset({Intent.CAPABILITY_INQUIRY}), prompt


def test_requests_and_local_questions_are_not_capability_inquiries():
    for prompt in NOT_CAPABILITY_PROMPTS:
        result = classify_intent(prompt)
        assert Intent.CAPABILITY_INQUIRY not in result.intents, prompt


def test_task_prompts_keep_their_pre_change_routes():
    from hermes_tool_router_intent_test.policy import _predict_toolsets_by_rules

    available = {"web", "file", "terminal", "browser", "vision", "memory", "skills"}
    expectations = {
        "Search the web for which tools support SQLite.": {"web"},
        "What tools can you use? Then run pytest.": {"terminal"},
        "Which integrations launched today? Search the web.": {"web"},
        "Inspect a repository and tell me what tools it uses.": {"file"},
    }
    for prompt, required in expectations.items():
        predicted, reason = _predict_toolsets_by_rules(prompt, available)
        assert predicted is not None and required <= predicted, (prompt, predicted, reason)


def test_capability_questions_route_to_small_surface():
    from hermes_tool_router_intent_test.policy import _predict_toolsets_by_rules

    available = {"web", "file", "terminal", "browser", "vision", "memory", "skills"}
    for prompt in CAPABILITY_PROMPTS:
        predicted, reason = _predict_toolsets_by_rules(prompt, available)
        assert predicted == set(), prompt
        assert reason == "intent:capability_inquiry", prompt
