import json
from types import SimpleNamespace

from agent import router_feedback


def test_router_feedback_records_without_raw_prompt(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setattr(router_feedback, "get_hermes_home", lambda: home)

    agent = SimpleNamespace(
        session_id="s1",
        platform="api_server",
        _router_toolset_telemetry={
            "message_sha256": "abc123",
            "effective_toolsets": ["file"],
            "full_surface": False,
            "loaded_tool_count": 4,
        },
        _router_recalled_toolsets=set(),
    )

    router_feedback.start_turn(agent, user_message="secret repo prompt")
    router_feedback.record_event(agent, "invalid_tool_call", tool="terminal")
    router_feedback.finish_turn(
        agent,
        {
            "completed": True,
            "failed": False,
            "partial": False,
            "api_calls": 1,
            "first_prompt_tokens": 123,
            "prompt_tokens": 456,
            "total_tokens": 789,
        },
    )

    lines = router_feedback.feedback_path().read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    assert {record["type"] for record in records} == {
        "turn_start",
        "invalid_tool_call",
        "turn_finish",
    }
    assert "secret repo prompt" not in router_feedback.feedback_path().read_text(encoding="utf-8")
    assert records[0]["message_sha256"] == "abc123"
    assert records[0]["message_chars"] == len("secret repo prompt")


def test_router_feedback_summary_counts_recalls_and_failures(tmp_path):
    path = tmp_path / "events.jsonl"
    rows = [
        {
            "type": "turn_finish",
            "router": {"effective_toolsets": ["file"]},
            "first_prompt_tokens": 3000,
            "failed": False,
            "partial": False,
            "recalled_toolsets": [],
        },
        {
            "type": "turn_finish",
            "router": {"effective_toolsets": None},
            "first_prompt_tokens": 23000,
            "failed": True,
            "partial": False,
            "recalled_toolsets": ["terminal"],
        },
        {"type": "router_recall", "toolset": "browser"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    summary = router_feedback.summarize_feedback(path)

    assert summary["turns"] == 2
    assert summary["by_effective_toolsets"] == {"file": 1, "full": 1}
    assert summary["avg_first_prompt_tokens"] == {"file": 3000, "full": 23000}
    assert summary["recalls_by_toolset"] == {"terminal": 1, "browser": 1}
    assert summary["failures_by_effective_toolsets"] == {"full": 1}
    assert summary["recommendations"]
