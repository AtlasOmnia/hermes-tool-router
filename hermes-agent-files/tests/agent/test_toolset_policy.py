"""Regression tests for router-first toolset policy resolution."""

from agent import toolset_policy, toolset_router


AVAILABLE_TOOLSETS = {
    name: name
    for name in [
        "clarify",
        "file",
        "memory",
        "session_search",
        "todo",
        "web",
        "terminal",
        "skills",
    ]
}


def _disable_external_policy(monkeypatch):
    monkeypatch.setattr(toolset_policy, "_external_policy_path", lambda _config: None)


def test_router_reduction_keeps_safe_core_and_skill_requirements(monkeypatch):
    _disable_external_policy(monkeypatch)
    monkeypatch.setattr(toolset_policy, "_active_skill_toolsets", lambda _disabled: ["skills"])
    monkeypatch.setattr(toolset_router, "predict_toolsets", lambda *_args: (["terminal"], True))

    result = toolset_policy.resolve_effective_toolsets(
        user_message="run tests and fix failures",
        router_config={"enabled": True},
        available_toolsets=AVAILABLE_TOOLSETS,
        platform_toolsets=["clarify"],
        enabled_toolsets_explicit=False,
        disabled_toolsets=[],
    )

    assert result.enabled_toolsets == [
        "web",
        "terminal",
        "skills",
    ]
    assert result.router_predicted_toolsets == result.enabled_toolsets
    assert result.merge_skill_required is False


def test_api_server_filters_clarify_from_router_predictions(monkeypatch):
    _disable_external_policy(monkeypatch)
    monkeypatch.setattr(toolset_policy, "_active_skill_toolsets", lambda _disabled: [])
    monkeypatch.setattr(toolset_router, "predict_toolsets", lambda *_args: (["clarify"], True))

    result = toolset_policy.resolve_effective_toolsets(
        user_message="hello",
        router_config={"enabled": True},
        available_toolsets=AVAILABLE_TOOLSETS,
        platform_toolsets=["clarify"],
        enabled_toolsets_explicit=False,
        disabled_toolsets=[],
        platform="api_server",
    )

    assert result.enabled_toolsets == ["web"]
    assert "clarify" not in result.enabled_toolsets


def test_router_decline_uses_full_tool_surface_for_implicit_platform_defaults(monkeypatch):
    _disable_external_policy(monkeypatch)
    monkeypatch.setattr(toolset_policy, "_active_skill_toolsets", lambda _disabled: [])
    monkeypatch.setattr(toolset_router, "predict_toolsets", lambda *_args: ([], False))

    result = toolset_policy.resolve_effective_toolsets(
        user_message="long multi-step task",
        router_config={"enabled": True},
        available_toolsets=AVAILABLE_TOOLSETS,
        platform_toolsets=["clarify"],
        enabled_toolsets_explicit=False,
        disabled_toolsets=[],
    )

    assert result.enabled_toolsets is None
    assert result.router_predicted_toolsets is None
    assert result.merge_skill_required is True


def test_predict_toolsets_declines_long_messages_without_router_call(monkeypatch):
    called = False

    def _fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("router client should not be resolved for long prompts")

    monkeypatch.setattr(
        toolset_router,
        "_build_user_prompt",
        lambda *_args, **_kwargs: "unused",
    )
    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_provider_client",
        _fail_if_called,
    )

    predicted, should_reduce = toolset_router.predict_toolsets(
        "x" * 600,
        {"enabled": True},
        AVAILABLE_TOOLSETS,
    )

    assert predicted == []
    assert should_reduce is False
    assert called is False


def test_explicit_toolset_allowlist_is_respected(monkeypatch):
    _disable_external_policy(monkeypatch)
    monkeypatch.setattr(toolset_policy, "_active_skill_toolsets", lambda _disabled: [])
    monkeypatch.setattr(toolset_router, "predict_toolsets", lambda *_args: (["terminal"], True))

    result = toolset_policy.resolve_effective_toolsets(
        user_message="run a command",
        router_config={"enabled": True},
        available_toolsets=AVAILABLE_TOOLSETS,
        platform_toolsets=["web"],
        enabled_toolsets_explicit=True,
        disabled_toolsets=[],
    )

    assert result.enabled_toolsets == ["web"]
    assert "terminal" not in result.enabled_toolsets
