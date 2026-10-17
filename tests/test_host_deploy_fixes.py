"""Coverage for PR #21: agent-first profile binding, classifier extra_body, commit telemetry."""
from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest

from tests.test_capability_envelope import (
    _rev6_agent,
    _rev6_config,
    _rev6_names,
    _rev6_registry_modules,
    _Rev6Registry,
    router,
)

config = sys.modules[router.__name__ + ".config"]
policy = sys.modules[router.__name__ + ".policy"]


def test_profile_name_prefers_agent_over_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_PROFILE", "default")
    monkeypatch.setattr(router, "_RUNTIME_OVERRIDE", None)
    monkeypatch.delenv("HERMES_TOKEN_ROUTER_ENABLED", raising=False)
    cfg = {"global": {"enabled": False}, "profiles": {"ai-os": {"enabled": True, "floor_toolsets": ["web"]}}}
    served = SimpleNamespace(_profile_name="ai-os")
    assert router._profile_name_for(served) == "ai-os"
    assert router._router_enabled(cfg, served) is True
    assert router._routing_settings(cfg, served)["floor_toolsets"] == ["web"]
    # No agent identity: unchanged env-based resolution.
    assert router._profile_name_for(SimpleNamespace(_profile_name="  ")) is None
    assert router._router_enabled(cfg) is False
    assert router._router_enabled(cfg, SimpleNamespace()) is False


def test_classifier_extra_body_sugar_and_merge() -> None:
    assert config._get_classifier_extra_body({}) == {}
    assert config._get_classifier_extra_body({"classifier": "bad"}) == {}
    assert config._get_classifier_extra_body({"classifier": {"disable_thinking": True}}) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    merged = config._get_classifier_extra_body({"classifier": {
        "disable_thinking": True,
        "extra_body": {"chat_template_kwargs": {"foo": 1}, "top_k": 5},
    }})
    assert merged == {"chat_template_kwargs": {"enable_thinking": False, "foo": 1}, "top_k": 5}


def test_classifier_forwards_extra_body_only_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            msg = SimpleNamespace(content='{"toolsets": ["web"], "confidence": 0.99}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    monkeypatch.setattr(policy, "_get_router_client", lambda *a, **k: (client, "m"))
    for extra in (None, {"chat_template_kwargs": {"enable_thinking": False}}):
        policy._predict_toolsets_via_llm(
            "search the web", {"web", "file"}, "m", 0.0, "custom", 2.0, extra_body=extra)
    assert len(calls) == 2
    assert "extra_body" not in calls[0]
    assert calls[1]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_commit_emits_one_info_telemetry_line(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(registry, "telemetry", names=("web_search", "request_toolset"),
                        enabled=("web", "router_recovery"))
    with caplog.at_level(logging.INFO):
        result = router.tool_request_middleware(
            agent=agent, session_id="telemetry", tool_name="read_file", args={})
    assert result == {"args": {}, "router_recovered": "file"}
    assert "read_file" in _rev6_names(agent)
    assert any("surface committed" in r.getMessage() for r in caplog.records)
