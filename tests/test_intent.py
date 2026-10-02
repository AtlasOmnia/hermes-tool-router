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
from hermes_tool_router_intent_test import intent as _intent_module

CHINESE_ABSTENTION_REASON = "zh_abstain_no_classifier"


def chinese_abstention_reason(message: str) -> str | None:
    helper = getattr(_intent_module, "chinese_abstention_reason", None)
    return helper(message) if callable(helper) else None


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


CHINESE_ROUTING_AVAILABLE = {"file", "web", "terminal"}


def _predict_chinese_toolsets(prompt: str) -> tuple[set[str] | None, str]:
    from hermes_tool_router_intent_test.policy import _predict_toolsets_by_rules

    return _predict_toolsets_by_rules(prompt, CHINESE_ROUTING_AVAILABLE)


def test_chinese_actions_match_english_controls_and_minimum_toolsets():
    cases = (
        (
            "帮我看看这个文件里写了什么 /root/config.yaml",
            "read the file /root/config.yaml",
            Intent.READ_LOCAL,
            {"file"},
        ),
        (
            "上网查一下 glm-5.3 的最新跑分",
            "search the web for the latest glm-5.3 benchmark",
            Intent.RESEARCH_WEB,
            {"web"},
        ),
        (
            "把这个目录下的测试跑一遍",
            "run the test suite with pytest",
            Intent.EXECUTE_LOCAL,
            {"terminal"},
        ),
        (
            "请读取这个文件：/tmp/notes.txt",
            "read the file /tmp/notes.txt",
            Intent.READ_LOCAL,
            {"file"},
        ),
        (
            "请查看这个文件，/tmp/notes.txt",
            "read the file /tmp/notes.txt",
            Intent.READ_LOCAL,
            {"file"},
        ),
        (
            "帮我读取/root/config.yaml",
            "read the file /root/config.yaml",
            Intent.READ_LOCAL,
            {"file"},
        ),
        (
            "搜索一下glm-5.3最新跑分？",
            "search the web for the latest glm-5.3 benchmark",
            Intent.RESEARCH_WEB,
            {"web"},
        ),
        (
            "上网查一下https://example.com的最新信息？",
            "search the web for the latest information at https://example.com",
            Intent.RESEARCH_WEB,
            {"web"},
        ),
        (
            "运行pytest：tests/test_intent.py",
            "run the test suite with pytest",
            Intent.EXECUTE_LOCAL,
            {"terminal"},
        ),
    )

    for prompt, control, expected_intent, expected_toolsets in cases:
        result = classify_intent(prompt)
        control_result = classify_intent(control)
        assert result.intents == control_result.intents == frozenset({expected_intent}), prompt
        assert result.reason_code == "deterministic_actions_zh", prompt

        predicted, reason = _predict_chinese_toolsets(prompt)
        control_predicted, _ = _predict_chinese_toolsets(control)
        assert predicted == control_predicted == expected_toolsets, (prompt, predicted, reason)
        assert reason == "intent:deterministic_actions_zh", prompt


def test_chinese_conceptual_prompts_remain_answer_only():
    for prompt in (
        "什么是 Git 仓库",
        "解释一下这个插件的原理",
        "介绍一下 Hermes",
        "这两个模型有什么区别",
        "为什么要运行测试",
    ):
        result = classify_intent(prompt)
        assert result.intents == frozenset({Intent.ANSWER_ONLY}), prompt
        assert result.reason_code == "conceptual"
        predicted, reason = _predict_chinese_toolsets(prompt)
        assert predicted == set(), (prompt, predicted, reason)
        assert reason == "intent:conceptual", prompt


def test_chinese_ambiguous_and_keyword_collision_prompts_fail_open():
    for prompt in (
        "ping",
        "hi",
    ):
        result = classify_intent(prompt)
        assert result.intents == frozenset({Intent.FULL_SURFACE}), prompt
        assert result.confidence == 0.0, prompt
        assert result.reason_code == "unresolved", prompt
        predicted, reason = _predict_chinese_toolsets(prompt)
        assert predicted == set(), (prompt, predicted, reason)
        assert reason.startswith("plain_"), (prompt, predicted, reason)

    for prompt in (
        "只回复两个字：成功",
        "检查hermes是否已升级",
        "评估 Hermes tool router这个插件",
        "Hermes内置了调用 codex的能力吗",
        "帮我看看这个",
    ):
        result = classify_intent(prompt)
        assert result.intents == frozenset({Intent.FULL_SURFACE}), prompt
        assert result.confidence == 0.0, prompt
        assert result.reason_code == CHINESE_ABSTENTION_REASON, prompt
        predicted, reason = _predict_chinese_toolsets(prompt)
        assert predicted is None, (prompt, predicted, reason)
        assert reason == CHINESE_ABSTENTION_REASON, (prompt, predicted, reason)


def test_chinese_actions_containing_conceptual_nouns_match_english_controls():
    cases = (
        (
            "上网查一下这两个模型有什么区别",
            "search the web for the difference between these two models",
            Intent.RESEARCH_WEB,
            {"web"},
        ),
        (
            "运行测试，然后看看两个版本的区别",
            "run the tests then show the difference between the two versions",
            Intent.EXECUTE_LOCAL,
            {"terminal"},
        ),
        (
            "跑一下测试，解释一下失败原因",
            "run the tests and explain the failure reason",
            Intent.EXECUTE_LOCAL,
            {"terminal"},
        ),
    )

    for prompt, control, expected_intent, expected_toolsets in cases:
        result = classify_intent(prompt)
        control_result = classify_intent(control)
        assert result.intents == control_result.intents == frozenset({expected_intent}), prompt
        assert result.reason_code == "deterministic_actions_zh", prompt

        predicted, reason = _predict_chinese_toolsets(prompt)
        control_predicted, _ = _predict_chinese_toolsets(control)
        assert predicted == control_predicted == expected_toolsets, (prompt, predicted, reason)
        assert reason == "intent:deterministic_actions_zh", prompt


def test_mixed_language_actions_preserve_baseline_toolsets():
    from hermes_tool_router_intent_test.policy import _predict_toolsets_by_rules

    available = {
        "file",
        "web",
        "terminal",
        "browser",
        "vision",
        "image_gen",
        "git",
        "computer_use",
        "video_gen",
    }
    cases = (
        ("commit 这些改动，解释一下原理", {"file", "git", "terminal"}),
        ("打开 https://example.com 并登录，解释一下原理", {"web"}),
        ("这两个模型有什么区别 https://example.com", {"web"}),
    )

    for prompt, baseline_toolsets in cases:
        predicted, reason = _predict_toolsets_by_rules(prompt, available)
        assert predicted is not None, (prompt, predicted, reason)
        assert baseline_toolsets <= predicted, (prompt, predicted, reason)

    for prompt in ("这两个模型有什么区别", "解释一下这个插件的原理"):
        result = classify_intent(prompt)
        assert result.intents == frozenset({Intent.ANSWER_ONLY}), prompt
        assert result.reason_code == "conceptual", prompt


def test_mixed_language_actions_with_chinese_concepts_preserve_control_toolsets():
    from hermes_tool_router_intent_test.policy import _predict_toolsets_by_rules

    available = {
        "file",
        "web",
        "terminal",
        "browser",
        "vision",
        "image_gen",
        "git",
        "computer_use",
        "video_gen",
    }
    cases = (
        (
            "take a screenshot 并解释一下原理",
            "take a screenshot and explain the principle",
            {"browser", "vision", "web"},
        ),
        (
            "search the web 并解释一下原理",
            "search the web and explain the principle",
            {"web"},
        ),
        (
            "run the tests 并解释一下原理",
            "run the tests and explain the principle",
            {"terminal"},
        ),
    )

    for mixed, control, baseline_toolsets in cases:
        mixed_toolsets, mixed_reason = _predict_toolsets_by_rules(mixed, available)
        control_toolsets, control_reason = _predict_toolsets_by_rules(control, available)
        assert control_toolsets == baseline_toolsets, (control, control_toolsets, control_reason)
        assert mixed_toolsets is not None, (mixed, mixed_toolsets, mixed_reason)
        assert control_toolsets <= mixed_toolsets, (mixed, mixed_toolsets, mixed_reason)
        assert baseline_toolsets <= mixed_toolsets, (mixed, mixed_toolsets, mixed_reason)


def test_chinese_leading_conceptual_frames_match_english_controls():
    cases = (
        ("怎么运行测试", "how do i run the tests"),
        ("怎样查看这个文件的内容", "how do i view the contents of this file"),
        ("如何打开这个配置文件", "how to open this config file"),
        ("比较一下运行测试的两种方式", "compare the two ways to run the tests"),
        ("定义一下执行脚本的流程", "define the process of executing a script"),
        ("谁是这个项目的维护者", "who is the maintainer of this project"),
    )

    for prompt, control in cases:
        result = classify_intent(prompt)
        control_result = classify_intent(control)
        assert result.intents == control_result.intents == frozenset({Intent.ANSWER_ONLY}), prompt
        assert result.reason_code == control_result.reason_code == "conceptual", prompt

        predicted, reason = _predict_chinese_toolsets(prompt)
        control_predicted, control_reason = _predict_chinese_toolsets(control)
        assert predicted == control_predicted == set(), (
            prompt,
            predicted,
            reason,
            control_reason,
        )
        assert reason == control_reason == "intent:conceptual", prompt


def test_chinese_sentence_final_conceptual_frames_match_english_controls():
    cases = (
        ("运行测试的命令是什么", "what is the command to run the tests"),
        ("查看文件内容的方法是什么", "what is the way to view file contents"),
        ("执行这个脚本的风险是什么", "what is the risk of executing this script"),
        ("pytest 是什么", "what is pytest"),
    )

    for prompt, control in cases:
        result = classify_intent(prompt)
        control_result = classify_intent(control)
        assert result.intents == control_result.intents == frozenset({Intent.ANSWER_ONLY}), prompt
        assert result.reason_code == control_result.reason_code == "conceptual", prompt

        predicted, reason = _predict_chinese_toolsets(prompt)
        control_predicted, control_reason = _predict_chinese_toolsets(control)
        assert predicted == control_predicted == set(), (
            prompt,
            predicted,
            reason,
            control_reason,
        )
        assert reason == control_reason == "intent:conceptual", prompt


def test_chinese_sentence_final_meaning_phrase_is_conceptual_despite_english_gap():
    # The English "what does" gap is pre-existing and frozen; Chinese remains answer-only.
    prompt = "这个命令什么意思"
    control = "what does this command mean"
    result = classify_intent(prompt)
    control_result = classify_intent(control)
    assert result.intents == frozenset({Intent.ANSWER_ONLY}), prompt
    assert result.reason_code == "conceptual", prompt
    assert control_result.intents == frozenset({Intent.EXECUTE_LOCAL}), control
    assert control_result.reason_code == "deterministic_actions", control

    predicted, reason = _predict_chinese_toolsets(prompt)
    control_predicted, control_reason = _predict_chinese_toolsets(control)
    assert predicted == set(), (prompt, predicted, reason)
    assert reason == "intent:conceptual", prompt
    assert control_predicted == {"terminal"}, (control, control_predicted, control_reason)
    assert control_reason == "intent:deterministic_actions", control


def test_chinese_sentence_final_conceptual_variants_match_english_controls():
    cases = (
        ("运行测试是干什么的", "what are the tests for"),
        ("读取日志文件有什么用", "what is the use of reading the log file"),
        ("这个插件是怎么回事", "what is going on with this plugin"),
    )

    for prompt, control in cases:
        result = classify_intent(prompt)
        control_result = classify_intent(control)
        assert result == control_result, (prompt, result, control_result)

        predicted, reason = _predict_chinese_toolsets(prompt)
        control_predicted, control_reason = _predict_chinese_toolsets(control)
        assert predicted == control_predicted, (prompt, predicted, control_predicted)
        assert reason == control_reason, (prompt, reason, control_reason)


def test_chinese_web_action_with_principle_or_difference_suffix_keeps_web_routing():
    cases = (
        (
            "上网查一下这个算法的原理",
            "search the web for this algorithm's principle",
        ),
        (
            "上网查一下这两个模型有什么区别",
            "search the web for the difference between these two models",
        ),
    )

    for prompt, control in cases:
        result = classify_intent(prompt)
        control_result = classify_intent(control)
        assert result.intents == control_result.intents, (prompt, result, control_result)

        predicted, reason = _predict_chinese_toolsets(prompt)
        control_predicted, control_reason = _predict_chinese_toolsets(control)
        assert predicted == control_predicted == {"web"}, (
            prompt,
            predicted,
            control_predicted,
        )
        assert reason == "intent:deterministic_actions_zh", prompt
        assert control_reason == "intent:deterministic_actions", control


def test_chinese_difference_suffix_pins_deferred_action_behavior():
    prompt = "运行测试和别的方式有什么区别"
    conceptual_control = "what is the difference between running tests and other approaches"
    action_control = "run the tests and compare with other approaches"

    result = classify_intent(prompt)
    conceptual_result = classify_intent(conceptual_control)
    action_result = classify_intent(action_control)
    assert result.intents == action_result.intents, (prompt, result, action_result)
    assert conceptual_result.intents != result.intents, (prompt, result, conceptual_result)

    predicted, reason = _predict_chinese_toolsets(prompt)
    conceptual_predicted, conceptual_reason = _predict_chinese_toolsets(conceptual_control)
    action_predicted, action_reason = _predict_chinese_toolsets(action_control)
    assert predicted == action_predicted == {"terminal"}, (prompt, predicted, action_predicted)
    assert conceptual_predicted != predicted, (prompt, predicted, conceptual_predicted)
    assert reason == "intent:deterministic_actions_zh", prompt
    assert action_reason == "intent:deterministic_actions", action_control
    assert conceptual_reason == "intent:conceptual", conceptual_control


def test_chinese_leading_freshness_guard_keeps_current_fail_open_behavior():
    prompt = "什么是 glm-5.3 的最新跑分"
    control = "what is the latest glm-5.3 benchmark"

    result = classify_intent(prompt)
    control_result = classify_intent(control)
    predicted, reason = _predict_chinese_toolsets(prompt)
    control_predicted, control_reason = _predict_chinese_toolsets(control)

    assert control_result.intents != result.intents, (prompt, result, control_result)
    assert control_predicted != predicted, (prompt, predicted, control_predicted)
    assert result.intents == frozenset({Intent.FULL_SURFACE}), result
    assert result.confidence == 0.0, result
    assert result.reason_code == CHINESE_ABSTENTION_REASON, result
    assert predicted is None, (prompt, predicted, reason)
    assert reason == CHINESE_ABSTENTION_REASON, (prompt, predicted, reason)
    assert control_reason == "intent:deterministic_actions", control


def test_chinese_residual_principle_frame_remains_conceptual():
    prompt = "这个算法的原理"
    control = "explain the principle of this algorithm"

    result = classify_intent(prompt)
    control_result = classify_intent(control)
    assert result == control_result, (prompt, result, control_result)

    predicted, reason = _predict_chinese_toolsets(prompt)
    control_predicted, control_reason = _predict_chinese_toolsets(control)
    assert predicted == control_predicted, (prompt, predicted, control_predicted)
    assert reason == control_reason == "intent:conceptual", prompt


def test_chinese_sentence_final_conceptual_does_not_override_explicit_read_request():
    # Controller decision: explicit read requests keep file routing despite final 是什么.
    for prompt in (
        "帮我看看这个文件里写的是什么",
        "帮我看看这个文件里写的是什么 /root/config.yaml",
    ):
        result = classify_intent(prompt)
        assert result.intents == frozenset({Intent.READ_LOCAL}), prompt
        assert result.reason_code == "deterministic_actions_zh", prompt
        predicted, reason = _predict_chinese_toolsets(prompt)
        assert predicted == {"file"}, (prompt, predicted, reason)
        assert reason == "intent:deterministic_actions_zh", prompt


def test_chinese_attached_filesystem_paths_keep_explicit_read_routing():
    for prompt in (
        "读取/root/config.yaml是什么",
        "打开/root/config.yaml是什么",
    ):
        result = classify_intent(prompt)
        assert result.intents == frozenset({Intent.READ_LOCAL}), prompt
        assert result.reason_code == "deterministic_actions_zh", prompt
        predicted, reason = _predict_chinese_toolsets(prompt)
        assert predicted == {"file"}, (prompt, predicted, reason)
        assert reason == "intent:deterministic_actions_zh", prompt


def test_chinese_leading_conceptual_prompts_keep_english_control_parity():
    cases = (
        ("什么是 pytest", "what is pytest", {Intent.ANSWER_ONLY}, set()),
        (
            "为什么 run the tests 会失败",
            "why run the tests fail",
            {Intent.ANSWER_ONLY},
            set(),
        ),
        (
            "解释一下这个插件的原理",
            "explain the principle of this plugin",
            {Intent.ANSWER_ONLY},
            set(),
        ),
        (
            "介绍一下，运行 pytest 测试",
            "describe, run the pytest tests",
            {Intent.EXECUTE_LOCAL},
            {"terminal"},
        ),
    )

    for prompt, control, expected_intents, expected_toolsets in cases:
        result = classify_intent(prompt)
        control_result = classify_intent(control)
        assert result.intents == control_result.intents == frozenset(expected_intents), prompt

        predicted, reason = _predict_chinese_toolsets(prompt)
        control_predicted, control_reason = _predict_chinese_toolsets(control)
        assert predicted == control_predicted == expected_toolsets, (
            prompt,
            predicted,
            reason,
            control_reason,
        )
        assert reason == (
            "intent:deterministic_actions_zh"
            if expected_toolsets
            else "intent:conceptual"
        ), prompt


def test_chinese_leading_conceptual_variants_match_english_controls():
    cases = (
        (
            ("为什么运行测试会失败", "为啥运行测试会失败"),
            "why do the tests fail",
        ),
        (("讲讲运行测试的原理",), "explain how running the tests works"),
        (("说说运行测试的原理",), "explain how running the tests works"),
        (("请教运行测试的原理",), "explain how running the tests works"),
        (("请问讲讲运行测试的原理",), "explain how running the tests works"),
    )

    for prompts, control in cases:
        control_result = classify_intent(control)
        control_predicted, control_reason = _predict_chinese_toolsets(control)
        observed = []
        for prompt in prompts:
            result = classify_intent(prompt)
            predicted, reason = _predict_chinese_toolsets(prompt)
            assert result == control_result, (prompt, result, control_result)
            assert predicted == control_predicted, (prompt, predicted, control_predicted)
            assert reason == control_reason, (prompt, reason, control_reason)
            observed.append((result, predicted, reason))
        if len(observed) == 2:
            assert observed[0] == observed[1], prompts


def test_chinese_polite_question_conceptual_frames_match_english_control():
    cases = tuple(
        (prompt, "how do i run the tests")
        for prompt in (
            "请问怎么运行测试",
            "请 问 怎么运行测试",
            "请问，怎么运行测试",
        )
    )

    for prompt, control in cases:
        result = classify_intent(prompt)
        control_result = classify_intent(control)
        assert result.intents == control_result.intents == frozenset({Intent.ANSWER_ONLY}), prompt
        assert result.reason_code == control_result.reason_code == "conceptual", prompt

        predicted, reason = _predict_chinese_toolsets(prompt)
        control_predicted, control_reason = _predict_chinese_toolsets(control)
        assert predicted == control_predicted == set(), (
            prompt,
            predicted,
            reason,
            control_reason,
        )
        assert reason == control_reason == "intent:conceptual", prompt


def test_chinese_introduction_prefix_does_not_hide_actions():
    cases = (
        ("介绍一下，跑一下测试", {"terminal"}),
        ("介绍一下，帮我看看这个文件里写了什么 /root/config.yaml", {"file"}),
        ("介绍一下 build the project", {"terminal"}),
    )

    for prompt, expected_toolsets in cases:
        predicted, reason = _predict_chinese_toolsets(prompt)
        assert predicted == expected_toolsets, (prompt, predicted, reason)
        assert reason in {
            "intent:deterministic_actions_zh",
            "intent:deterministic_actions",
        }, prompt


def test_mixed_chinese_and_english_actions_union_toolsets():
    from hermes_tool_router_intent_test.policy import _predict_toolsets_by_rules

    available = {
        "file",
        "web",
        "terminal",
        "browser",
        "vision",
        "image_gen",
        "git",
        "computer_use",
        "video_gen",
    }
    cases = (
        (
            "帮我读取/root/config.yaml and commit the change",
            "read /root/config.yaml and commit the change",
            {"file", "git", "terminal"},
        ),
        (
            "帮我读取/root/config.yaml，然后 commit the change",
            "read /root/config.yaml, then commit the change",
            {"file", "git", "terminal"},
        ),
        (
            "把这个目录下的测试跑一遍 and commit the change",
            "run the test suite and commit the change",
            {"file", "git", "terminal"},
        ),
        (
            "上网查一下最新信息，然后 analyze this screenshot",
            "search the web for the latest information and analyze this screenshot",
            {"vision"},
        ),
    )

    for mixed, control, baseline_toolsets in cases:
        mixed_toolsets, mixed_reason = _predict_toolsets_by_rules(mixed, available)
        control_toolsets, control_reason = _predict_toolsets_by_rules(control, available)
        assert control_toolsets is not None, (control, control_toolsets, control_reason)
        assert mixed_toolsets is not None, (mixed, mixed_toolsets, mixed_reason)
        assert baseline_toolsets <= mixed_toolsets, (mixed, mixed_toolsets, mixed_reason)
        assert control_toolsets <= mixed_toolsets, (mixed, mixed_toolsets, mixed_reason)


def test_sentence_final_chinese_concepts_do_not_hide_english_imperative_actions():
    from hermes_tool_router_intent_test.policy import _predict_toolsets_by_rules

    available = {
        "file",
        "web",
        "terminal",
        "browser",
        "vision",
        "image_gen",
        "memory",
        "cronjob",
        "delegation",
        "git",
    }
    cases = (
        (
            "Run the pytest command，原理是什么",
            "Run the pytest command",
            {"terminal"},
        ),
        (
            "Generate a logo image，原理是什么",
            "Generate a logo image",
            {"image_gen"},
        ),
        (
            "Remember this fact，原理是什么",
            "Remember this fact",
            {"memory"},
        ),
        (
            "Schedule a cron job，原理是什么",
            "Schedule a cron job",
            {"cronjob"},
        ),
        (
            "Delegate this to a subagent，原理是什么",
            "Delegate this to a subagent",
            {"delegation"},
        ),
        (
            "push 这些改动，原理是什么",
            "push these changes",
            {"file", "git", "terminal"},
        ),
        ("take a screenshot，原理是什么", "take a screenshot", None),
    )

    for prompt, control, expected_toolsets in cases:
        result = classify_intent(prompt)
        control_result = classify_intent(control)
        predicted, reason = _predict_toolsets_by_rules(prompt, available)
        control_predicted, control_reason = _predict_toolsets_by_rules(control, available)
        assert result.intents == control_result.intents, (prompt, result, control_result)
        assert reason == control_reason, (prompt, reason, control_reason)
        assert predicted == control_predicted, (prompt, predicted, control_predicted)
        assert control_predicted is not None, (control, control_predicted, control_reason)
        if expected_toolsets is not None:
            assert control_predicted == expected_toolsets, (control, control_predicted, control_reason)

    for prompt in (
        "pytest 是什么",
        "build 是什么",
        "test 是什么",
        "screenshot 的原理是什么",
    ):
        result = classify_intent(prompt)
        predicted, reason = _predict_toolsets_by_rules(prompt, available)
        assert result.intents == frozenset({Intent.ANSWER_ONLY}), prompt
        assert result.reason_code == "conceptual", prompt
        assert predicted == set(), (prompt, predicted, reason)
        assert reason == "intent:conceptual", prompt


def test_chinese_abstention_reason_corpus_is_exact_and_policy_marked():
    from hermes_tool_router_intent_test.policy import _predict_toolsets_by_rules

    marked = (
        "只回复两个字：成功",
        "检查hermes是否已升级",
        "评估 Hermes tool router这个插件",
        "Hermes内置了调用 codex的能力吗",
        "帮我看看这个",
        "什么是 glm-5.3 的最新跑分",
        "不要运行测试",
        "请不要运行pytest",
        "别读取这个文件 /tmp/notes.txt",
        "不要上网查一下最新信息",
        "无需执行这个脚本",
        "不必运行测试",
        "“运行测试”",
        '"读取这个文件"',
        "'上网查一下最新信息'",
        "`运行pytest`",
        "“上网查一下https://example.com的最新信息”",
        "这句话是：运行测试",
        "他说：读取这个文件",
        "日志显示：运行pytest",
        "文档写着：上网查一下最新信息",
        "这句话是: 运行测试",
        "他说 : 读取这个文件",
        "日志显示:运行pytest",
        "文档写着 : 上网查一下最新信息",
        "不要运行测试，请读取这个文件 /tmp/notes.txt",
        "我知道了。不要运行测试",
    )
    for prompt in marked:
        assert chinese_abstention_reason(prompt) == CHINESE_ABSTENTION_REASON, prompt
        assert chinese_abstention_reason(f"  {prompt}  ") == CHINESE_ABSTENTION_REASON, prompt
        result = classify_intent(prompt)
        assert result.intents == frozenset({Intent.FULL_SURFACE}), prompt
        assert result.confidence == 0.0, prompt
        assert result.reason_code == CHINESE_ABSTENTION_REASON, prompt
        predicted, reason = _predict_toolsets_by_rules(
            prompt,
            {"file", "web", "terminal"},
        )
        assert predicted is None, (prompt, predicted, reason)
        assert reason == CHINESE_ABSTENTION_REASON, (prompt, predicted, reason)


def test_chinese_abstention_marker_is_not_user_text_or_language_detector():
    for prompt in (
        CHINESE_ABSTENTION_REASON,
        "請不要運行測試",
        "テストを実行しないでください",
        "테스트를 실행하지 마세요",
    ):
        assert chinese_abstention_reason(prompt) is None, prompt
        assert classify_intent(prompt).reason_code != CHINESE_ABSTENTION_REASON, prompt


def test_chinese_quoted_operands_and_positive_frames_remain_active():
    from hermes_tool_router_intent_test.policy import _predict_toolsets_by_rules

    positive = (
        "帮我看看这个文件里写了什么 /root/config.yaml",
        "请读取这个文件：/tmp/notes.txt",
        "帮我读取/root/config.yaml",
        "上网查一下 glm-5.3 的最新跑分",
        "上网查一下https://example.com的最新信息？",
        "把这个目录下的测试跑一遍",
        "运行pytest：tests/test_intent.py",
        "运行测试，然后看看两个版本的区别",
        "介绍一下，跑一下测试",
        "请读取这个文件：“/tmp/notes.txt”",
    )
    expected = (
        (Intent.READ_LOCAL, {"file"}),
        (Intent.READ_LOCAL, {"file"}),
        (Intent.READ_LOCAL, {"file"}),
        (Intent.RESEARCH_WEB, {"web"}),
        (Intent.RESEARCH_WEB, {"web"}),
        (Intent.EXECUTE_LOCAL, {"terminal"}),
        (Intent.EXECUTE_LOCAL, {"terminal"}),
        (Intent.EXECUTE_LOCAL, {"terminal"}),
        (Intent.EXECUTE_LOCAL, {"terminal"}),
        (Intent.READ_LOCAL, {"file"}),
    )
    for prompt, (expected_intent, expected_toolsets) in zip(positive, expected):
        assert chinese_abstention_reason(prompt) is None, prompt
        result = classify_intent(prompt)
        assert result.intents == frozenset({expected_intent}), prompt
        assert result.reason_code == "deterministic_actions_zh", prompt
        predicted, reason = _predict_toolsets_by_rules(prompt, {"file", "web", "terminal"})
        assert predicted == expected_toolsets, (prompt, predicted, reason)
        assert reason == "intent:deterministic_actions_zh", prompt
