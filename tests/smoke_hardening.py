"""Smoke checks for the Hermes token-router hardening layer.

Run from the repository root:
    python tests/smoke_hardening.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
# Tests install a fake Hermes tool registry, so they do not require a live Hermes checkout.

module_name = "hermes_token_router_smoke_pkg"
spec = importlib.util.spec_from_file_location(
    module_name,
    PLUGIN_DIR / "__init__.py",
    submodule_search_locations=[str(PLUGIN_DIR)],
)
plugin = importlib.util.module_from_spec(spec)
sys.modules[module_name] = plugin
assert spec.loader is not None
spec.loader.exec_module(plugin)

# The public sample config is disabled-by-default. Smoke tests patch the
# plugin's config loader in memory so tests exercise routing without requiring
# a user's real Hermes profile/configuration.
_TEST_CONFIG = {
    "global": {
        "enabled": True,
        "floor_toolsets": [],
        "deterministic_rules_enabled": True,
        "confidence_threshold": 0.0,
        "long_message_decline_chars": 12000,
        "short_message_bypass_chars": 0,
        "router_provider": "deepseek",
        "router_model": "deepseek-chat",
    },
    "profiles": {},
}

def _test_load_config():
    return _TEST_CONFIG

plugin._load_config = _test_load_config
_config_mod = sys.modules.get(f"{module_name}.config")
_tools_mod = sys.modules.get(f"{module_name}.tools")
if _config_mod is not None:
    _config_mod._load_config = _test_load_config
if _tools_mod is not None:
    _tools_mod._load_config = _test_load_config


class _Entry:
    def __init__(self, name: str, toolset: str):
        self.name = name
        self.toolset = toolset


class _FakeRegistry:
    TOOLSETS = {
        "web": ["web_search"],
        "browser": ["browser_open"],
        "file": ["read_file"],
        "terminal": ["shell"],
        "git": ["git_status"],
        "vision": ["vision_analyze"],
        "image_gen": ["image_generate"],
        "memory": ["memory_store"],
        "skills": ["skill_manage"],
        "delegation": ["delegate_task"],
        "router_recovery": ["request_toolset"],
    }

    def __init__(self):
        self.entries = {
            tool: _Entry(tool, toolset)
            for toolset, tools in self.TOOLSETS.items()
            for tool in tools
        }

    def get_registered_toolset_names(self):
        return list(self.TOOLSETS)

    def get_tool_names_for_toolset(self, toolset):
        return list(self.TOOLSETS.get(toolset, []))

    def get_definitions(self, names, quiet=True):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"fake {name}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in names
            if name in self.entries
        ]

    def get_toolset_alias_target(self, name):
        return None

    def get_entry(self, name):
        return self.entries.get(name)


_FAKE_REGISTRY = _FakeRegistry()
_fake_registry_module = types.ModuleType("tools.registry")
_fake_registry_module.registry = _FAKE_REGISTRY
sys.modules["tools.registry"] = _fake_registry_module


class _FakeAgent:
    def __init__(self):
        full_names = [
            tool
            for toolset, tools in _FAKE_REGISTRY.TOOLSETS.items()
            if toolset != "router_recovery"
            for tool in tools
        ]
        self.tools = _FAKE_REGISTRY.get_definitions(full_names)
        self.valid_tool_names = set(full_names)
        self.enabled_toolsets = [ts for ts in _FAKE_REGISTRY.TOOLSETS if ts != "router_recovery"]
        self.session_id = "smoke-session"
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.platform = "smoke"
        self._current_turn_id = ""
        self._current_task_id = "smoke-task"
        self._user_id = "smoke-user"
        self.status_messages = []

    def _emit_status(self, message):
        self.status_messages.append(message)


def _toolset_for_tool(tool_name: str) -> str:
    if tool_name == plugin.RECOVERY_TOOL_NAME:
        return "request_toolset"
    entry = _FAKE_REGISTRY.get_entry(tool_name)
    assert entry is not None, f"unknown fake tool in valid set: {tool_name}"
    return entry.toolset


def _exposed_toolsets(agent: _FakeAgent) -> set[str]:
    return {_toolset_for_tool(name) for name in agent.valid_tool_names}


def _route(prompt: str) -> _FakeAgent:
    agent = _FakeAgent()
    turn_id = f"turn-{abs(hash(prompt))}"
    agent._current_turn_id = turn_id
    plugin.pre_turn_context_build(
        agent=agent,
        session_id=agent.session_id,
        task_id=agent._current_task_id,
        turn_id=turn_id,
        user_message=prompt,
        conversation_history=[],
        is_first_turn=True,
        model=agent.model,
        provider=agent.provider,
        platform=agent.platform,
        sender_id=agent._user_id,
        available_toolsets=list(agent.enabled_toolsets),
        available_tool_names=list(agent.valid_tool_names),
    )
    state = plugin._get_router_state(agent)
    assert state.routed_turn_id == turn_id
    assert state.routed_source == "pre_turn_context_build"
    return agent


def _assert_route(prompt: str, expected_toolsets: set[str]) -> None:
    agent = _route(prompt)
    actual = _exposed_toolsets(agent)
    assert actual == expected_toolsets, f"{prompt!r}: expected {expected_toolsets}, got {actual}"
    assert "request_toolset" in actual


def test_recovery_schema_and_core_hook_surface():
    assert "request_toolset" == plugin.RECOVERY_TOOL_NAME
    assert plugin.RECOVERY_TOOL_SCHEMA["parameters"].get("additionalProperties") is False
    assert "toolsets" in plugin.RECOVERY_TOOL_SCHEMA["parameters"]["properties"]
    assert plugin.RECOVERY_TOOL_SCHEMA["parameters"]["properties"]["toolsets"]["items"] == {"type": "string"}
    assert "tool_name" in plugin.RECOVERY_TOOL_SCHEMA["parameters"]["properties"]
    assert callable(plugin.pre_turn_context_build)
    assert callable(plugin.pre_llm_call)


def test_deterministic_routes():
    cases = [
        ("what is photosynthesis", {"request_toolset"}),
        ("check the hermes subreddit", {"web", "browser", "request_toolset"}),
        ("open https://example.com and click the first link", {"browser", "request_toolset"}),
        ("latest weather today", {"web", "request_toolset"}),
        ("read this file", {"file", "request_toolset"}),
        ("run the shell test command", {"terminal", "request_toolset"}),
        ("git diff and commit this repo", {"git", "file", "terminal", "request_toolset"}),
        ("debug this code", {"file", "terminal", "request_toolset"}),
        ("analyze this diagram", {"vision", "request_toolset"}),
        ("generate art for a logo", {"image_gen", "request_toolset"}),
        ("remember this fact", {"memory", "request_toolset"}),
        ("use skills to inspect this", {"skills", "request_toolset"}),
        ("delegate this to a subagent", {"delegation", "request_toolset"}),
    ]
    for prompt, expected in cases:
        _assert_route(prompt, expected)


def test_late_pre_llm_compatibility_routes_before_request_without_explicit_agent():
    agent = _FakeAgent()
    agent._current_turn_id = "late-turn"

    def invoke_from_turn_context_stack():
        # The local variable name intentionally matches Hermes build_turn_context.
        return plugin.pre_llm_call(
            session_id=agent.session_id,
            turn_id=agent._current_turn_id,
            user_message="latest weather today",
            conversation_history=[],
            is_first_turn=True,
        )

    invoke_from_turn_context_stack()
    assert _exposed_toolsets(agent) == {"web", "request_toolset"}


def test_pre_llm_call_skips_after_core_hook_routed_turn():
    prompt = "check the hermes subreddit"
    agent = _route(prompt)
    before_tools = list(agent.tools)
    before_names = set(agent.valid_tool_names)
    original = plugin._predict_toolsets_by_rules

    def fail_if_called(user_message, available):
        raise AssertionError("late pre_llm_call should skip after pre_turn_context_build")

    plugin._predict_toolsets_by_rules = fail_if_called
    try:
        plugin.pre_llm_call(
            session_id=agent.session_id,
            task_id=agent._current_task_id,
            turn_id=agent._current_turn_id,
            user_message=prompt,
            conversation_history=[],
            is_first_turn=True,
            model=agent.model,
            platform=agent.platform,
            sender_id=agent._user_id,
        )
    finally:
        plugin._predict_toolsets_by_rules = original

    assert agent.tools == before_tools
    assert agent.valid_tool_names == before_names


def test_tool_request_middleware_expands_registered_pruned_tool_before_validation():
    agent = _route("what is photosynthesis")
    assert "git_status" not in agent.valid_tool_names
    result = plugin.tool_request_middleware(
        session_id=agent.session_id,
        tool_name="git_status",
        args={"short": True},
    )
    assert result == {"args": {"short": True}, "router_recovered": "git"}
    assert "git_status" in agent.valid_tool_names


def test_request_toolset_git_expansion():
    agent = _route("what is photosynthesis")
    assert _exposed_toolsets(agent) == {"request_toolset"}
    response = json.loads(plugin.request_toolset_handler({"toolset": "git", "reason": "need git status"}))
    assert response["ok"] is True
    assert response["toolset"] == "git"
    assert "git_status" in response["enabled_tools"]
    assert "request_toolset" in response["enabled_tools"]


def test_request_toolset_unknown_suggestion():
    response = json.loads(plugin.request_toolset_handler({"toolset": "gti"}))
    assert response["ok"] is False
    assert response["error"].startswith("unknown toolset")
    assert "suggestions" in response
    assert "git" in response["suggestions"]


def test_first_turn_surface_is_sticky_and_does_not_shrink_or_reclassify():
    agent = _FakeAgent()
    agent._current_turn_id = "turn-one"
    plugin.pre_turn_context_build(
        agent=agent,
        turn_id="turn-one",
        user_message="latest weather today",
        conversation_history=[],
        is_first_turn=True,
    )
    first_names = set(agent.valid_tool_names)
    assert "web_search" in first_names

    agent._current_turn_id = "turn-two"
    plugin.pre_turn_context_build(
        agent=agent,
        turn_id="turn-two",
        user_message="read this file",
        conversation_history=[{"role": "user", "content": "latest weather today"}],
        is_first_turn=False,
    )
    assert agent.valid_tool_names == first_names


if __name__ == "__main__":
    test_recovery_schema_and_core_hook_surface()
    print("token-router hook surface smoke: ok")
    test_deterministic_routes()
    print("deterministic route smoke: ok")
    test_pre_llm_call_skips_after_core_hook_routed_turn()
    print("pre_llm_call skip smoke: ok")
    test_request_toolset_git_expansion()
    print("request_toolset git smoke: ok")
    test_request_toolset_unknown_suggestion()
    print("unknown toolset smoke: ok")
