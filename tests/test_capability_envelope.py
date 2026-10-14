from __future__ import annotations

import copy
import json
import math
import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

PACKAGE_DIR = __import__("pathlib").Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "hermes_token_router_plugin",
    PACKAGE_DIR / "__init__.py",
    submodule_search_locations=[str(PACKAGE_DIR)],
)
assert SPEC is not None and SPEC.loader is not None
router = module_from_spec(SPEC)
sys.modules.setdefault("hermes_token_router_plugin", router)
SPEC.loader.exec_module(router)

from hermes_token_router_plugin.capabilities import (  # noqa: E402
    MISSING,
    EnsureAdmissionResult,
    canonical_json,
    freeze_json,
    thaw_json,
)


class _Registry:
    def __init__(self) -> None:
        self.toolsets = {
            "web": ["web_search"],
            "file": ["read_file"],
            "terminal": ["run_command"],
            "kanban": ["kanban_show", "kanban_complete", "kanban_sibling"],
            "router_recovery": ["request_toolset"],
        }
        self.definitions = {
            name: {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"captured-{name}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for names in self.toolsets.values()
            for name in names
        }
        self.entries = {
            name: SimpleNamespace(toolset=toolset)
            for toolset, names in self.toolsets.items()
            for name in names
        }

    def get_registered_toolset_names(self) -> list[str]:
        return list(self.toolsets)

    def get_tool_names_for_toolset(self, toolset: str) -> list[str]:
        return list(self.toolsets.get(toolset, []))

    def get_entry(self, name: str) -> Any:
        return self.entries.get(name)

    def get_definitions(self, names: set[str], quiet: bool = True) -> list[dict[str, Any]]:
        return [self.definitions[name] for name in sorted(names) if name in self.definitions]


def _registry_modules(monkeypatch: pytest.MonkeyPatch, registry: _Registry) -> None:
    tools_pkg = ModuleType("tools")
    tools_pkg.__path__ = []
    registry_mod = ModuleType("tools.registry")
    registry_mod.registry = registry  # type: ignore[attr-defined]
    tools_pkg.registry = registry_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.registry", registry_mod)


def test_direct_specialist_prediction_and_floor_cannot_create_kanban(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    _registry_modules(monkeypatch, registry)
    config = {
        "enabled": True,
        "long_message_decline_chars": 10_000,
        "short_message_bypass_chars": 0,
        "floor_toolsets": ["kanban"],
        "deterministic_rules_enabled": True,
        "classifier": {"enabled": False},
    }
    monkeypatch.setattr(router, "_load_config", lambda: {"global": config})
    monkeypatch.setattr(router, "_get_profile_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(router, "_get_available_toolsets", lambda: set(registry.toolsets))
    monkeypatch.setattr(
        router,
        "_predict_toolsets_by_rules",
        lambda message, available: ({"kanban"}, "forced-kanban"),
    )
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    definitions = [registry.definitions[name] for name in ("web_search", "read_file", "request_toolset")]
    agent = SimpleNamespace(
        session_id="direct-specialist",
        _current_turn_id="specialist-route",
        tools=definitions,
        valid_tool_names={item["function"]["name"] for item in definitions},
        enabled_toolsets=["web", "file", "router_recovery"],
    )
    before = (list(agent.tools), set(agent.valid_tool_names), list(agent.enabled_toolsets))

    router.pre_turn_context_build(
        agent=agent,
        session_id=agent.session_id,
        turn_id=agent._current_turn_id,
        user_message="use kanban",
    )

    assert (list(agent.tools), set(agent.valid_tool_names), list(agent.enabled_toolsets)) == before
    state = router._get_router_state(agent)
    assert "kanban" not in state.active_toolsets
    assert "kanban_show" not in {
        item.get("function", {}).get("name", "") for item in agent.tools
    }

def test_tagged_freeze_thaw_is_injective_ordered_and_isolated() -> None:
    negative_zero = -0.0
    positive_zero = 0.0
    values = [True, 1, 1.0, negative_zero, positive_zero, None, "héllo"]
    frozen = [freeze_json(value) for value in values]

    assert len({frozen[0], frozen[1], frozen[2]}) == 3
    assert frozen[0][0] == "bool"
    assert frozen[1][0] == "int"
    assert frozen[2][0] == "float"
    assert frozen[3] != frozen[4]
    assert frozen[3][1] == "-0x0.0p+0"
    assert frozen[4][1] == "0x0.0p+0"
    assert len({frozen[3], frozen[4]}) == 2

    nested_left = {"x": [True, 1, 1.0, negative_zero]}
    nested_right = [["x", 1]]
    assert freeze_json(nested_left) != freeze_json(nested_right)

    assert type(thaw_json(frozen[0])) is bool
    assert type(thaw_json(frozen[1])) is int
    assert type(thaw_json(frozen[2])) is float
    assert math.copysign(1.0, thaw_json(frozen[3])) == -1.0
    assert math.copysign(1.0, thaw_json(frozen[4])) == 1.0
    assert canonical_json(frozen[0]) == "true"
    assert canonical_json(frozen[1]) == "1"
    assert canonical_json(frozen[2]) == "1.0"
    assert canonical_json(frozen[3]) == "-0.0"
    assert canonical_json(frozen[4]) == "0.0"

    ordered = {"z": ["β", {"a": 1}], "a": []}
    frozen_ordered = freeze_json(ordered)
    first = thaw_json(frozen_ordered)
    second = thaw_json(frozen_ordered)
    assert list(first) == ["z", "a"]
    assert first == second == ordered
    first["z"][1]["a"] = 99
    first["z"].append("new")
    assert second == ordered
    assert thaw_json(frozen_ordered) == ordered

    assert freeze_json({}) != freeze_json([])
    assert freeze_json({"x": 1}) != freeze_json([["x", 1]])
    assert freeze_json({"nested": [{"u": "✓"}, []]}) == freeze_json(
        {"nested": [{"u": "✓"}, []]}
    )

    class Custom:
        pass

    cyclic: list[object] = []
    cyclic.append(cyclic)
    for invalid in [
        {1: "bad"},
        b"bytes",
        {"bad"},
        Custom(),
        float("nan"),
        float("inf"),
        cyclic,
    ]:
        with pytest.raises((TypeError, ValueError)):
            freeze_json(invalid)

    with pytest.raises((TypeError, ValueError)):
        thaw_json(("float", "1.0"))
    with pytest.raises((TypeError, ValueError)):
        thaw_json(("unknown", None))
    with pytest.raises((TypeError, ValueError)):
        thaw_json(("null", 1))
    with pytest.raises((TypeError, ValueError)):
        thaw_json(("float", "0X0.0P+0"))
    with pytest.raises((TypeError, ValueError)):
        thaw_json(("float", "nan"))

    assert json.loads(canonical_json(frozen[3])) == -0.0
    assert json.loads(canonical_json(frozen[4])) == 0.0


def test_metadata_distinguishes_missing_from_null_and_rejects_malformed_values() -> None:
    from hermes_token_router_plugin.capabilities import MISSING, TrustedHostPolicy, read_trusted_host_policy

    empty = read_trusted_host_policy(MISSING, MISSING)
    assert isinstance(empty, TrustedHostPolicy)
    assert empty.valid is True
    assert empty.protected_toolsets == frozenset()
    assert empty.pinned_tool_names == frozenset()
    assert empty.source_channels == ()

    metadata = {
        "schema_version": 1,
        "protected_toolsets": ["kanban", "web"],
        "pinned_tool_names": ["kanban_show"],
    }
    hook_only = read_trusted_host_policy(metadata, MISSING)
    agent_only = read_trusted_host_policy(MISSING, metadata)
    equal = read_trusted_host_policy(metadata, dict(metadata))
    for parsed in (hook_only, agent_only, equal):
        assert parsed.valid is True
        assert parsed.protected_toolsets == frozenset({"kanban", "web"})
        assert parsed.pinned_tool_names == frozenset({"kanban_show"})
    assert hook_only.source_channels == ("hook",)
    assert agent_only.source_channels == ("agent",)
    assert equal.source_channels == ("hook", "agent")

    invalid_values = [
        (None, MISSING),
        (True, MISSING),
        ({}, MISSING),
        ({"schema_version": True, "protected_toolsets": [], "pinned_tool_names": []}, MISSING),
        ({"schema_version": 2, "protected_toolsets": [], "pinned_tool_names": []}, MISSING),
        ({"schema_version": 1, "protected_toolsets": "kanban", "pinned_tool_names": []}, MISSING),
        ({"schema_version": 1, "protected_toolsets": ["kanban", "kanban"], "pinned_tool_names": []}, MISSING),
        ({"schema_version": 1, "protected_toolsets": [" kanban"], "pinned_tool_names": []}, MISSING),
        ({"schema_version": 1, "protected_toolsets": [""], "pinned_tool_names": []}, MISSING),
        ({"schema_version": 1, "protected_toolsets": [], "pinned_tool_names": ["show", "show"]}, MISSING),
        (metadata, {**metadata, "pinned_tool_names": ["other"]}),
    ]
    for hook_metadata, agent_metadata in invalid_values:
        parsed = read_trusted_host_policy(hook_metadata, agent_metadata)
        assert parsed.valid is False
        assert parsed.source_channels in ((), ("hook",), ("agent",), ("hook", "agent"))
        assert parsed.errors

    for mapping in (
        {"schema_version": 1, "protected_toolsets": [], "pinned_tool_names": [], "extra": 1},
        {"schema_version": 1, "protected_toolsets": ["kanban"], "pinned_tool_names": []},
    ):
        parsed = read_trusted_host_policy(mapping, MISSING)
        assert parsed.valid is ("extra" not in mapping)

def test_adapter_composes_policy_before_one_shot_ensure_and_same_session_reuses_result() -> None:
    from hermes_token_router_plugin.capabilities import (
        MISSING,
        EnsureAdmissionResult,
        OwnerSnapshot,
    )

    class Agent:
        def __init__(self, session_id: str) -> None:
            self.session_id = session_id
            self.tools = [
                {"type": "function", "function": {"name": "web_search"}},
                {"type": "function", "function": {"name": "kanban_show", "description": "host"}},
            ]
            self.valid_tool_names = {"web_search", "kanban_show"}
            self.enabled_toolsets = ["web", "kanban"]

    worker = Agent("session-worker")
    ordinary = Agent("session-ordinary")
    snapshot = OwnerSnapshot(
        incoming_names=("web_search", "kanban_show"),
        owner_by_name=(("web_search", "web"), ("kanban_show", "kanban")),
        registry_import_ok=True,
        registry_lookup_error_names=(),
        agent_local_lookup_error_names=(),
        malformed_definition_indexes=(),
        duplicate_names=(),
        errors=(),
    )
    capture_calls: list[tuple[str, ...]] = []
    composition_calls: list[tuple[object, object]] = []
    task_reads: list[str] = []
    predicate_calls: list[str] = []

    def capture_spy(agent, untouched_surface):
        capture_calls.append(tuple(item["function"]["name"] for item in untouched_surface))
        return snapshot

    def task_spy():
        task_reads.append("task")
        return "task-1"

    def predicate_spy():
        predicate_calls.append("predicate")
        return True

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(router, "_capture_owner_snapshot", capture_spy)
        monkeypatch.setattr(router, "_read_worker_task_id", task_spy)
        monkeypatch.setattr(router, "_is_dispatcher_owned_worker_context", predicate_spy)
        original_compose = router._compose_effective_policy

        def compose_spy(parsed, owner_snapshot, **kwargs):
            composition_calls.append((parsed, owner_snapshot))
            return original_compose(parsed, owner_snapshot, **kwargs)

        monkeypatch.setattr(router, "_compose_effective_policy", compose_spy)

        worker_metadata = {
            "schema_version": 1,
            "protected_toolsets": ["kanban"],
            "pinned_tool_names": [],
        }
        first = router._ensure_host_admission(
            source="pre_turn_context_build",
            agent=worker,
            session_id=worker.session_id,
            untouched_surface=tuple(worker.tools),
            original_enabled_toolsets=tuple(worker.enabled_toolsets),
            hook_metadata=worker_metadata,
            agent_metadata=MISSING,
        )
        assert isinstance(first, EnsureAdmissionResult)
        assert first.status == "READY"
        assert first.owner_snapshot is snapshot
        assert first.effective_policy is not None
        assert "kanban_show" in first.effective_policy.pinned_tool_names
        assert capture_calls == [("web_search", "kanban_show")]
        assert len(task_reads) == 1
        assert len(predicate_calls) == 1
        assert len(composition_calls) == 1
        assert composition_calls[0][1] is snapshot

        second = router._ensure_host_admission(
            source="pre_llm_call",
            agent=worker,
            session_id=worker.session_id,
            untouched_surface=(),
            original_enabled_toolsets=(),
            hook_metadata=None,
            agent_metadata=None,
        )
        assert second is first
        assert len(capture_calls) == 1
        assert len(task_reads) == 1
        assert len(predicate_calls) == 1
        assert len(composition_calls) == 1

        no_envelope = router._ensure_host_admission(
            source="visible_request",
            agent=ordinary,
            session_id=ordinary.session_id,
            untouched_surface=MISSING,
            original_enabled_toolsets=MISSING,
            hook_metadata=MISSING,
            agent_metadata=MISSING,
        )
        assert no_envelope.status == "NO_AUTHORITY"
        assert no_envelope.owner_snapshot is None
        assert no_envelope.effective_policy is not None
        assert no_envelope.effective_policy.protected_toolsets == frozenset({"kanban"})
        assert len(capture_calls) == 1
        assert len(task_reads) == 1
        assert len(predicate_calls) == 1

        mismatch = router._ensure_host_admission(
            source="pre_llm_call",
            agent=worker,
            session_id="other-session",
            untouched_surface=MISSING,
            original_enabled_toolsets=MISSING,
            hook_metadata=MISSING,
            agent_metadata=MISSING,
        )
        assert mismatch.status == "SESSION_MISMATCH"
        router.on_session_end(agent=worker, session_id=worker.session_id)
        fresh = router._ensure_host_admission(
            source="pre_turn_context_build",
            agent=worker,
            session_id="new-session",
            untouched_surface=tuple(worker.tools),
            original_enabled_toolsets=tuple(worker.enabled_toolsets),
            hook_metadata=worker_metadata,
            agent_metadata=MISSING,
        )
        assert fresh.status == "READY"
        assert fresh is not first
    finally:
        monkeypatch.undo()

def test_non_attachable_agents_preserve_surface_without_global_capability_state() -> None:
    from hermes_token_router_plugin.capabilities import MISSING
    from hermes_token_router_plugin.state import _router_state

    class NonAttachable:
        __slots__ = ("session_id", "tools", "valid_tool_names", "enabled_toolsets")

        def __init__(self) -> None:
            object.__setattr__(self, "session_id", "unattachable")
            object.__setattr__(
                self,
                "tools",
                [{"type": "function", "function": {"name": "foo"}}],
            )
            object.__setattr__(self, "valid_tool_names", {"foo"})
            object.__setattr__(self, "enabled_toolsets", ["ordinary"])

    agent = NonAttachable()
    before = (
        list(agent.tools),
        set(agent.valid_tool_names),
        list(agent.enabled_toolsets),
    )
    result = router._ensure_host_admission(
        source="visible_request",
        agent=agent,
        session_id=agent.session_id,
        untouched_surface=MISSING,
        original_enabled_toolsets=MISSING,
        hook_metadata=MISSING,
        agent_metadata=MISSING,
    )
    assert result.status == "NO_AUTHORITY_UNATTACHABLE"
    assert result.owner_snapshot is None
    assert (list(agent.tools), set(agent.valid_tool_names), list(agent.enabled_toolsets)) == before
    assert _router_state._bound_admission_result is None
    assert _router_state._contamination_marker is None

def test_worker_adapter_requires_both_signals_and_safe_no_prunes_on_owner_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_token_router_plugin.capabilities import MISSING, OwnerSnapshot

    metadata = {
        "schema_version": 1,
        "protected_toolsets": [],
        "pinned_tool_names": [],
    }

    def snapshot_for(
        *owners: str | None,
        import_ok: bool = True,
        registry_errors: tuple[str, ...] = (),
        local_errors: tuple[str, ...] = (),
    ) -> OwnerSnapshot:
        names = ("ordinary_tool", "kanban_show")
        return OwnerSnapshot(
            incoming_names=names,
            owner_by_name=tuple(zip(names, owners, strict=True)),
            registry_import_ok=import_ok,
            registry_lookup_error_names=registry_errors,
            agent_local_lookup_error_names=local_errors,
            malformed_definition_indexes=(),
            duplicate_names=(),
            errors=(),
        )

    def agent(session_id: str) -> Any:
        class Agent:
            pass

        value = Agent()
        value.session_id = session_id
        value.tools = [
            {"type": "function", "function": {"name": "ordinary_tool"}},
            {"type": "function", "function": {"name": "kanban_show"}},
        ]
        value.valid_tool_names = {"ordinary_tool", "kanban_show"}
        value.enabled_toolsets = ["ordinary", "kanban"]
        return value

    cases = [
        ("both-signals", "task", True, snapshot_for("ordinary", "kanban"), "READY"),
        ("environment-only", "task", False, snapshot_for("ordinary", "kanban"), "READY"),
        ("predicate-error", "task", RuntimeError("predicate"), snapshot_for("ordinary", "kanban"), "SAFE_NO_PRUNE"),
        ("registry-error", "task", True, snapshot_for("ordinary", None, import_ok=False), "SAFE_NO_PRUNE"),
        ("owner-lookup-error", "task", True, snapshot_for("ordinary", "kanban", registry_errors=("kanban_show",)), "SAFE_NO_PRUNE"),
        ("local-lookup-error", "task", True, snapshot_for(None, "kanban", local_errors=("ordinary_tool",)), "SAFE_NO_PRUNE"),
        ("owner-unmapped", "task", True, snapshot_for("ordinary", None), "SAFE_NO_PRUNE"),
    ]
    for label, task_id, predicate, snapshot, expected_status in cases:
        current = agent(f"worker-{label}")
        monkeypatch.setattr(router, "_capture_owner_snapshot", lambda *_args, snapshot=snapshot: snapshot)
        monkeypatch.setattr(router, "_read_worker_task_id", lambda task_id=task_id: task_id)
        if isinstance(predicate, BaseException):
            def raise_predicate() -> bool:
                raise predicate

            monkeypatch.setattr(router, "_is_dispatcher_owned_worker_context", raise_predicate)
        else:
            monkeypatch.setattr(
                router,
                "_is_dispatcher_owned_worker_context",
                lambda predicate=predicate: predicate,
            )
        result = router._ensure_host_admission(
            source="pre_turn_context_build",
            agent=current,
            session_id=current.session_id,
            untouched_surface=tuple(current.tools),
            original_enabled_toolsets=tuple(current.enabled_toolsets),
            hook_metadata=metadata,
            agent_metadata=MISSING,
        )
        assert result.status == expected_status, label
        assert result.envelope is not None
        assert result.owner_snapshot is snapshot
        if label == "both-signals":
            assert "kanban_show" in result.effective_policy.pinned_tool_names
        else:
            assert "kanban_show" not in result.effective_policy.pinned_tool_names

def test_worker_answer_web_and_early_late_routes_keep_exact_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SimpleNamespace()
    registry.toolsets = {
        "web": ["web_search"],
        "file": ["read_file"],
        "kanban": ["kanban_show", "kanban_complete", "kanban_sibling"],
        "router_recovery": ["request_toolset"],
    }
    registry.definitions = {
        name: {"type": "function", "function": {"name": name, "description": f"{name}-schema"}}
        for names in registry.toolsets.values()
        for name in names
    }
    registry.entries = {
        name: SimpleNamespace(toolset=toolset)
        for toolset, names in registry.toolsets.items()
        for name in names
    }
    registry.get_registered_toolset_names = lambda: list(registry.toolsets)
    registry.get_tool_names_for_toolset = lambda name: list(registry.toolsets.get(name, []))
    registry.get_entry = lambda name: registry.entries.get(name)
    registry.get_definitions = lambda names, quiet=True: [
        registry.definitions[name] for name in sorted(names) if name in registry.definitions
    ]
    registry.get_toolset_alias_target = lambda name: name
    tools_pkg = ModuleType("tools")
    tools_pkg.__path__ = []
    registry_mod = ModuleType("tools.registry")
    registry_mod.registry = registry  # type: ignore[attr-defined]
    tools_pkg.registry = registry_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.registry", registry_mod)
    config = {"enabled": True, "floor_toolsets": [], "classifier": {"enabled": False}}
    monkeypatch.setattr(router, "_load_config", lambda: {"global": config})
    monkeypatch.setattr(router, "_get_profile_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(router, "_get_available_toolsets", lambda: set(registry.toolsets))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task")
    monkeypatch.setattr(router, "_is_dispatcher_owned_worker_context", lambda: True)
    predicted_calls: list[str] = []
    monkeypatch.setattr(
        router,
        "_predict_toolsets_by_rules",
        lambda message, available: (predicted_calls.append(message) or {"web"}, "test"),
    )

    def make_agent(session_id: str) -> Any:
        definitions = [
            registry.definitions[name]
            for name in ("web_search", "read_file", "kanban_show", "kanban_complete", "request_toolset")
        ]
        return SimpleNamespace(
            session_id=session_id,
            _current_turn_id="",
            tools=definitions,
            valid_tool_names={item["function"]["name"] for item in definitions},
            enabled_toolsets=["web", "file", "kanban", "router_recovery"],
        )

    def invoke(source: str, agent: Any) -> None:
        fn = router.pre_turn_context_build if source == "early" else router.pre_llm_call
        fn(agent=agent, session_id=agent.session_id, turn_id=source, user_message="search the web")

    worker = make_agent("slice6-answer")
    monkeypatch.setattr(router, "_predict_toolsets_by_rules", lambda message, available: (set(), "answer"))
    invoke("early", worker)
    answer_names = [item["function"]["name"] for item in worker.tools]
    assert "kanban_show" in answer_names
    assert "kanban_complete" in answer_names
    assert "kanban_sibling" not in answer_names

    monkeypatch.setattr(
        router,
        "_predict_toolsets_by_rules",
        lambda message, available: (predicted_calls.append(message) or {"web"}, "test"),
    )
    worker = make_agent("slice6-worker")
    invoke("early", worker)
    assert predicted_calls == ["search the web"]
    names = [item["function"]["name"] for item in worker.tools]
    assert names == ["web_search", "kanban_show", "kanban_complete", "request_toolset"]
    assert worker.tools[names.index("kanban_show")] == registry.definitions["kanban_show"]

    late = make_agent("slice6-late")
    invoke("late", late)
    late_names = [item["function"]["name"] for item in late.tools]
    assert late_names == names

class _Rev6Registry:
    def __init__(self) -> None:
        self.toolsets = {
            "web": ["web_search"],
            "file": ["read_file"],
            "terminal": ["run_command"],
            "memory": ["memory_search"],
            "skills": ["skill_view"],
            "kanban": ["kanban_show", "kanban_complete", "kanban_sibling"],
            "router_recovery": ["request_toolset"],
        }
        self.definitions = {
            name: {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"captured-{name}",
                    "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
                },
            }
            for names in self.toolsets.values()
            for name in names
        }
        self.entries = {
            name: SimpleNamespace(toolset=toolset)
            for toolset, names in self.toolsets.items()
            for name in names
        }
        self.aliases: dict[str, str] = {}
        self.entry_calls: list[str] = []
        self.definition_calls: list[tuple[str, ...]] = []
        self.fail_entry: set[str] = set()
        self.fail_definitions: set[str] = set()

    def get_registered_toolset_names(self):
        return list(self.toolsets)

    def get_tool_names_for_toolset(self, toolset):
        return list(self.toolsets.get(toolset, []))

    def get_entry(self, name):
        self.entry_calls.append(name)
        if name in self.fail_entry:
            raise RuntimeError(f"entry:{name}")
        return self.entries.get(name)

    def get_definitions(self, names, quiet=True):
        frozen = tuple(sorted(names))
        self.definition_calls.append(frozen)
        if self.fail_definitions.intersection(names):
            raise RuntimeError("definitions")
        return [self.definitions[name] for name in frozen if name in self.definitions]

    def get_toolset_alias_target(self, name):
        return self.aliases.get(name, name)

    def register(self, **kwargs):
        return kwargs


def _rev6_registry_modules(monkeypatch: pytest.MonkeyPatch, registry: _Rev6Registry) -> None:
    tools_pkg = ModuleType("tools")
    tools_pkg.__path__ = []
    registry_mod = ModuleType("tools.registry")
    registry_mod.registry = registry  # type: ignore[attr-defined]
    tools_pkg.registry = registry_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.registry", registry_mod)


def _rev6_config(monkeypatch: pytest.MonkeyPatch, registry: _Rev6Registry, **overrides: Any) -> None:
    config = {
        "enabled": True,
        "long_message_decline_chars": 10_000,
        "short_message_bypass_chars": 0,
        "floor_toolsets": [],
        "deterministic_rules_enabled": True,
        "confidence_threshold": 0.0,
        "router_hard_timeout_ms": 1_200,
        "classifier": {"enabled": False},
    }
    config.update(overrides)
    monkeypatch.setattr(router, "_load_config", lambda: {"global": config})
    monkeypatch.setattr(router, "_get_profile_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(router, "_get_available_toolsets", lambda: set(registry.toolsets))


def _rev6_agent(
    registry: _Rev6Registry,
    session_id: str,
    names: tuple[str, ...] | None = None,
    enabled: tuple[str, ...] | None = None,
) -> Any:
    names = names or (
        "web_search",
        "read_file",
        "run_command",
        "kanban_show",
        "kanban_complete",
        "request_toolset",
    )
    definitions = [
        registry.definitions.get(
            name,
            {"type": "function", "function": {"name": name, "description": f"host-{name}"}},
        )
        for name in names
    ]
    return SimpleNamespace(
        session_id=session_id,
        _current_turn_id="",
        tools=definitions,
        valid_tool_names={item["function"]["name"] for item in definitions},
        enabled_toolsets=list(enabled or ("web", "file", "terminal", "kanban", "router_recovery")),
    )


def _rev6_admit(
    agent: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    task: str = "",
    owned: bool | BaseException | None = False,
    hook_metadata: Any = None,
    agent_metadata: Any = MISSING,
) -> EnsureAdmissionResult:
    if task:
        os.environ["HERMES_KANBAN_TASK"] = task
    else:
        os.environ.pop("HERMES_KANBAN_TASK", None)
    if isinstance(owned, BaseException):
        def raise_owned() -> bool:
            raise owned
        monkeypatch.setattr(router, "_is_dispatcher_owned_worker_context", raise_owned)
    elif owned is not None:
        monkeypatch.setattr(
            router,
            "_is_dispatcher_owned_worker_context",
            lambda owned=owned: bool(owned),
        )
    return router._ensure_host_admission(
        source="pre_turn_context_build",
        agent=agent,
        session_id=agent.session_id,
        untouched_surface=tuple(agent.tools),
        original_enabled_toolsets=tuple(agent.enabled_toolsets),
        hook_metadata=MISSING if hook_metadata is None else hook_metadata,
        agent_metadata=agent_metadata,
    )


def _rev6_names(agent: Any) -> list[str]:
    return [item.get("function", {}).get("name", "") for item in agent.tools]


# Compact redefinition keeps the node focused on the public route contract;
# the preceding draft remains the preserved Slice 6 RED chronology.

def test_direct_orchestrator_routes_and_recovers_only_host_subset_without_auto_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    metadata = {
        "schema_version": 1,
        "protected_toolsets": ["kanban"],
        "pinned_tool_names": [],
    }
    agent = _rev6_agent(
        registry,
        "direct-orchestrator",
        names=("web_search", "kanban_show", "request_toolset"),
        enabled=("web", "kanban", "router_recovery"),
    )
    admission = _rev6_admit(agent, monkeypatch, hook_metadata=metadata, owned=False)
    assert admission.status == "READY"
    assert admission.effective_policy is not None
    assert admission.effective_policy.pinned_tool_names == frozenset()

    router.pre_turn_context_build(
        agent=agent,
        session_id=agent.session_id,
        turn_id="direct-route",
        user_message="search the web",
    )
    assert "web_search" in _rev6_names(agent)
    assert "kanban_show" not in _rev6_names(agent)
    response = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["kanban"]},
            agent=agent,
            session_id=agent.session_id,
        )
    )
    assert response["ok"] is True
    assert response["added_tool_names"] == ["kanban_show"]
    assert "kanban_sibling" not in _rev6_names(agent)

    pinned = _rev6_agent(
        registry,
        "direct-pinned",
        names=("web_search", "kanban_show", "request_toolset"),
        enabled=("web", "kanban", "router_recovery"),
    )
    pinned_admission = _rev6_admit(
        pinned,
        monkeypatch,
        hook_metadata={**metadata, "pinned_tool_names": ["kanban_show"]},
        owned=False,
    )
    assert pinned_admission.status == "READY"
    router.pre_turn_context_build(
        agent=pinned,
        session_id=pinned.session_id,
        turn_id="pinned-route",
        user_message="search the web",
    )
    assert "kanban_show" in _rev6_names(pinned)
    assert "kanban_sibling" not in _rev6_names(pinned)

def test_visible_request_uses_ensure_seam_and_truthful_admission_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)

    attachable = _rev6_agent(
        registry,
        "visible-attachable",
        names=("web_search", "request_toolset"),
        enabled=("web", "router_recovery"),
    )
    ordinary = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["terminal"]},
            agent=attachable,
            session_id=attachable.session_id,
        )
    )
    assert ordinary["ok"] is True
    assert ordinary["expanded_toolsets"] == ["terminal"]
    assert ordinary["denied_toolsets"] == []
    assert ordinary["added_tool_names"] == ["run_command"]
    assert "run_command" in _rev6_names(attachable)

    protected = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["kanban"]},
            agent=attachable,
            session_id=attachable.session_id,
        )
    )
    assert protected["ok"] is False
    assert protected["denied_toolsets"] == ["kanban"]
    assert protected["added_tool_names"] == []
    assert "kanban_show" not in _rev6_names(attachable)

    mixed = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["file", "kanban"]},
            agent=attachable,
            session_id=attachable.session_id,
        )
    )
    assert mixed["ok"] is False
    assert mixed["expanded_toolsets"] == ["file"]
    assert mixed["denied_toolsets"] == ["kanban"]
    assert "read_file" in _rev6_names(attachable)
    assert "kanban_show" not in _rev6_names(attachable)

    worker = _rev6_agent(
        registry,
        "visible-worker",
        names=("web_search", "kanban_show", "request_toolset"),
        enabled=("web", "kanban", "router_recovery"),
    )
    admission = _rev6_admit(worker, monkeypatch, task="worker", owned=True)
    assert admission.status == "READY"
    worker_result = json.loads(
        router.request_toolset_handler(
            {"tool_name": "kanban_show"},
            agent=worker,
            session_id=worker.session_id,
        )
    )
    assert worker_result["ok"] is True
    assert worker_result["denied_toolsets"] == []
    assert "kanban_show" in worker_result["installed_tool_names"]

def test_middleware_recovers_exact_admitted_tool_only_after_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    metadata = {
        "schema_version": 1,
        "protected_toolsets": ["kanban"],
        "pinned_tool_names": [],
    }
    worker = _rev6_agent(
        registry,
        "middleware-worker",
        names=("web_search", "kanban_show", "request_toolset"),
        enabled=("web", "kanban", "router_recovery"),
    )
    admission = _rev6_admit(worker, monkeypatch, task="", owned=False, hook_metadata=metadata)
    assert admission.status == "READY"
    monkeypatch.setattr(router, "_predict_toolsets_by_rules", lambda message, available: ({"web"}, "web"))
    router.pre_turn_context_build(
        agent=worker,
        session_id=worker.session_id,
        turn_id="middleware-route",
        user_message="search the web",
    )
    assert "kanban_show" not in _rev6_names(worker)
    recovered = router.tool_request_middleware(
        agent=worker,
        session_id=worker.session_id,
        tool_name="kanban_show",
        args={"task_id": "t"},
    )
    assert recovered == {"args": {"task_id": "t"}, "router_recovered": "kanban"}
    assert _rev6_names(worker).count("kanban_show") == 1
    assert worker.tools[_rev6_names(worker).index("kanban_show")] == registry.definitions["kanban_show"]

    ordinary = _rev6_agent(
        registry,
        "middleware-ordinary",
        names=("web_search", "request_toolset"),
        enabled=("web", "router_recovery"),
    )
    ordinary_recovered = router.tool_request_middleware(
        agent=ordinary,
        session_id=ordinary.session_id,
        tool_name="read_file",
        args={},
    )
    assert ordinary_recovered == {"args": {}, "router_recovered": "file"}
    assert "read_file" in _rev6_names(ordinary)

    specialist = _rev6_agent(
        registry,
        "middleware-specialist",
        names=("web_search", "request_toolset"),
        enabled=("web", "router_recovery"),
    )
    before = (list(specialist.tools), set(specialist.valid_tool_names), list(specialist.enabled_toolsets))
    assert router.tool_request_middleware(
        agent=specialist,
        session_id=specialist.session_id,
        tool_name="kanban_show",
        args={},
    ) is None
    assert (list(specialist.tools), set(specialist.valid_tool_names), list(specialist.enabled_toolsets)) == before

def test_post_tool_retries_only_after_exact_admitted_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    metadata = {"schema_version": 1, "protected_toolsets": ["kanban"], "pinned_tool_names": []}

    direct = _rev6_agent(
        registry,
        "post-direct",
        names=("web_search", "kanban_show", "request_toolset"),
        enabled=("web", "kanban", "router_recovery"),
    )
    _rev6_admit(direct, monkeypatch, hook_metadata=metadata, owned=False)
    monkeypatch.setattr(router, "_predict_toolsets_by_rules", lambda message, available: ({"web"}, "web"))
    router.pre_turn_context_build(
        agent=direct,
        session_id=direct.session_id,
        turn_id="post-route",
        user_message="search the web",
    )
    assert "kanban_show" not in _rev6_names(direct)
    router.post_tool_call(
        agent=direct,
        session_id=direct.session_id,
        tool_name="kanban_show",
    )
    assert "kanban_show" in _rev6_names(direct)
    assert router._get_router_state(direct)._retry_pending is True
    router.post_tool_call(agent=direct, session_id=direct.session_id, tool_name="kanban_show")
    assert _rev6_names(direct).count("kanban_show") == 1

    ordinary = _rev6_agent(
        registry,
        "post-ordinary",
        names=("web_search", "request_toolset"),
        enabled=("web", "router_recovery"),
    )
    router.post_tool_call(
        agent=ordinary,
        session_id=ordinary.session_id,
        tool_name="read_file",
    )
    assert "read_file" in _rev6_names(ordinary)
    assert router._get_router_state(ordinary)._retry_pending is True

    specialist = _rev6_agent(
        registry,
        "post-specialist",
        names=("web_search", "request_toolset"),
        enabled=("web", "router_recovery"),
    )
    before = (list(specialist.tools), set(specialist.valid_tool_names), list(specialist.enabled_toolsets))
    router.post_tool_call(
        agent=specialist,
        session_id=specialist.session_id,
        tool_name="kanban_show",
    )
    assert (list(specialist.tools), set(specialist.valid_tool_names), list(specialist.enabled_toolsets)) == before
    assert router._get_router_state(specialist)._retry_pending is False
    router.post_tool_call(agent=specialist, session_id=specialist.session_id, tool_name="missing")
    assert (list(specialist.tools), set(specialist.valid_tool_names), list(specialist.enabled_toolsets)) == before

def test_fallback_restores_exact_envelope_or_preserves_current_without_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    metadata = {"schema_version": 1, "protected_toolsets": ["kanban"], "pinned_tool_names": []}
    direct = _rev6_agent(
        registry,
        "fallback-direct",
        names=("web_search", "kanban_show", "request_toolset"),
        enabled=("web", "kanban", "router_recovery"),
    )
    _rev6_admit(direct, monkeypatch, hook_metadata=metadata, owned=False)
    original_schema = json.loads(json.dumps(registry.definitions["kanban_show"]))
    monkeypatch.setattr(router, "_predict_toolsets_by_rules", lambda message, available: ({"web"}, "web"))
    router.pre_turn_context_build(
        agent=direct,
        session_id=direct.session_id,
        turn_id="fallback-route",
        user_message="search the web",
    )
    assert "kanban_show" not in _rev6_names(direct)
    registry.definitions["kanban_show"]["function"]["description"] = "mutated-registry"
    registry.toolsets["kanban"].append("kanban_new")
    registry.definitions["kanban_new"] = {
        "type": "function",
        "function": {"name": "kanban_new", "description": "sibling"},
    }
    router._handle_full_fallback(direct)
    restored = next(item for item in direct.tools if item["function"]["name"] == "kanban_show")
    assert restored == original_schema
    assert "kanban_new" not in _rev6_names(direct)
    assert router._get_router_state(direct)._retry_pending is False

    ordinary = _rev6_agent(
        registry,
        "fallback-no-authority",
        names=("web_search", "request_toolset"),
        enabled=("web", "router_recovery"),
    )
    before = (list(ordinary.tools), set(ordinary.valid_tool_names), list(ordinary.enabled_toolsets))
    state = router._get_router_state(ordinary)
    state._retry_pending = False
    router._handle_full_fallback(ordinary)
    assert (list(ordinary.tools), set(ordinary.valid_tool_names), list(ordinary.enabled_toolsets)) == before
    assert state._retry_pending is False

def test_disable_restores_and_reenable_reuses_original_envelope_until_session_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    metadata = {"schema_version": 1, "protected_toolsets": ["kanban"], "pinned_tool_names": []}
    agent = _rev6_agent(
        registry,
        "disable-lifecycle",
        names=("web_search", "kanban_show", "request_toolset"),
        enabled=("web", "kanban", "router_recovery"),
    )
    _rev6_admit(agent, monkeypatch, hook_metadata=metadata, owned=False)
    monkeypatch.setattr(router, "_predict_toolsets_by_rules", lambda message, available: ({"web"}, "web"))
    router.pre_turn_context_build(
        agent=agent,
        session_id=agent.session_id,
        turn_id="disable-route",
        user_message="search the web",
    )
    assert "kanban_show" not in _rev6_names(agent)
    result_identity = router._get_router_state(agent)._bound_admission_result
    registry.definitions["kanban_show"]["function"]["description"] = "changed-after-capture"
    monkeypatch.setattr(router, "_get_profile_config", lambda *args, **kwargs: {"enabled": False})
    router.pre_turn_context_build(
        agent=agent,
        session_id=agent.session_id,
        turn_id="disabled",
        user_message="search the web",
    )
    assert "kanban_show" in _rev6_names(agent)
    restored = next(item for item in agent.tools if item["function"]["name"] == "kanban_show")
    assert restored["function"]["description"] == "captured-kanban_show"
    monkeypatch.setattr(router, "_get_profile_config", lambda *args, **kwargs: {"enabled": True, "floor_toolsets": []})
    router.pre_turn_context_build(
        agent=agent,
        session_id=agent.session_id,
        turn_id="reenabled",
        user_message="search the web",
    )
    assert router._get_router_state(agent)._bound_admission_result is result_identity
    assert "kanban_show" in _rev6_names(agent)
    router.on_session_end(agent=agent, session_id=agent.session_id)
    assert router._get_router_state(agent)._bound_admission_result is None

def test_delegated_inherited_and_cron_contexts_do_not_gain_worker_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    metadata = {"schema_version": 1, "protected_toolsets": ["kanban"], "pinned_tool_names": []}
    for label in ("delegated", "inherited", "cron"):
        agent = _rev6_agent(
            registry,
            f"{label}-context",
            names=("web_search", "kanban_show", "request_toolset"),
            enabled=("web", "kanban", "router_recovery"),
        )
        admission = _rev6_admit(
            agent,
            monkeypatch,
            task=f"inherited-{label}",
            owned=False,
            hook_metadata=metadata,
        )
        assert admission.status == "READY", label
        assert admission.effective_policy is not None
        assert "kanban_show" not in admission.effective_policy.pinned_tool_names, label
        monkeypatch.setattr(router, "_predict_toolsets_by_rules", lambda message, available: ({"web"}, "web"))
        router.pre_turn_context_build(
            agent=agent,
            session_id=agent.session_id,
            turn_id=f"{label}-route",
            user_message="search the web",
        )
        assert "web_search" in _rev6_names(agent)
        assert "kanban_show" not in _rev6_names(agent)
        response = json.loads(
            router.request_toolset_handler(
                {"toolsets": ["kanban"]},
                agent=agent,
                session_id=agent.session_id,
            )
        )
        assert response["ok"] is True, label
        assert response["added_tool_names"] == ["kanban_show"], label
        assert "kanban_sibling" not in _rev6_names(agent)

def test_agent_local_memory_and_fallback_rebuild_all_effective_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    registry.toolsets.pop("memory")
    registry.entries.pop("memory_search")
    registry.definitions.pop("memory_search")
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(
        registry,
        "agent-local-memory",
        names=("web_search", "memory_search", "request_toolset"),
        enabled=("web", "memory", "router_recovery"),
    )
    agent._memory_manager = SimpleNamespace(
        get_all_tool_names=lambda: ["memory_search"],
        has_tool=lambda name: name == "memory_search",
    )
    metadata = {"schema_version": 1, "protected_toolsets": ["kanban"], "pinned_tool_names": []}
    admission = _rev6_admit(agent, monkeypatch, hook_metadata=metadata, owned=False)
    assert admission.status == "READY"
    state = router._get_router_state(agent)
    state.predicted_toolsets = {"stale"}
    state.initial_toolsets = {"stale"}
    state.active_toolsets = {"stale"}
    monkeypatch.setattr(router, "_predict_toolsets_by_rules", lambda message, available: ({"web"}, "web"))
    router.pre_turn_context_build(
        agent=agent,
        session_id=agent.session_id,
        turn_id="memory-route",
        user_message="search the web",
    )
    assert "memory_search" not in _rev6_names(agent)
    result = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["memory"]},
            agent=agent,
            session_id=agent.session_id,
        )
    )
    assert result["ok"] is True
    assert result["added_tool_names"] == ["memory_search"]
    assert "memory" in state.active_toolsets
    assert "stale" not in state.active_toolsets
    assert state.installed_tool_names == set(_rev6_names(agent))
    router._handle_full_fallback(agent)
    assert "memory_search" in _rev6_names(agent)
    assert "stale" not in state.active_toolsets
    assert state.installed_tool_names == set(_rev6_names(agent))

def test_route_mixed_expansion_and_fallback_have_one_serialized_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    registry.toolsets["file"] = ["z_file", "a_file"]
    registry.toolsets["terminal"] = ["z_terminal", "a_terminal"]
    for name, toolset in (
        ("z_file", "file"),
        ("a_file", "file"),
        ("z_terminal", "terminal"),
        ("a_terminal", "terminal"),
    ):
        registry.entries[name] = SimpleNamespace(toolset=toolset)
        registry.definitions[name] = {
            "type": "function",
            "function": {"name": name, "description": f"{name}-schema"},
        }
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    outputs: list[list[str]] = []
    for requested in (("terminal", "file"), ("file", "terminal"), ("file", "terminal")):
        agent = _rev6_agent(
            registry,
            f"order-{len(outputs)}",
            names=("web_search", "request_toolset"),
            enabled=("web", "router_recovery"),
        )
        response = json.loads(
            router.request_toolset_handler(
                {"toolsets": list(requested)},
                agent=agent,
                session_id=agent.session_id,
            )
        )
        assert response["ok"] is True
        outputs.append(_rev6_names(agent))
        assert response["installed_tool_names"] == sorted(response["installed_tool_names"])
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0][-4:] == ["a_file", "z_file", "a_terminal", "z_terminal"]

def test_empty_envelope_and_missing_pin_fail_without_manufacturing_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    empty = SimpleNamespace(
        session_id="empty-envelope",
        _current_turn_id="",
        tools=[],
        valid_tool_names=set(),
        enabled_toolsets=[],
    )
    empty_admission = _rev6_admit(empty, monkeypatch, owned=False)
    assert empty_admission.status == "READY"
    before_calls = list(registry.definition_calls)
    response = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["kanban"]},
            agent=empty,
            session_id=empty.session_id,
        )
    )
    assert response["ok"] is False
    assert response["added_tool_names"] == []
    assert _rev6_names(empty) == []
    assert registry.definition_calls == before_calls

    missing_pin = SimpleNamespace(
        session_id="missing-pin",
        _current_turn_id="",
        tools=[],
        valid_tool_names=set(),
        enabled_toolsets=[],
    )
    admission = _rev6_admit(
        missing_pin,
        monkeypatch,
        owned=False,
        hook_metadata={
            "schema_version": 1,
            "protected_toolsets": ["kanban"],
            "pinned_tool_names": ["kanban_show"],
        },
    )
    assert admission.status == "CAPTURE_INVALID_NO_MUTATION"
    assert _rev6_names(missing_pin) == []
    assert registry.definition_calls == before_calls

def test_capture_invalid_no_mutation_survives_disable_reenable_and_dies_only_at_session_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    malformed = {"not": "a tool"}
    agent = _rev6_agent(
        registry,
        "capture-invalid",
        names=("web_search", "request_toolset"),
        enabled=("web", "router_recovery"),
    )
    agent.tools.insert(1, malformed)
    agent.valid_tool_names.add("web_search")
    before_tools = list(agent.tools)
    before_enabled = list(agent.enabled_toolsets)
    admission = _rev6_admit(agent, monkeypatch, owned=False)
    assert admission.status == "CAPTURE_INVALID_NO_MUTATION"
    state = router._get_router_state(agent)
    bound = state._bound_admission_result
    assert bound is admission
    before_calls = list(registry.definition_calls)
    for caller in (
        lambda: router.pre_turn_context_build(
            agent=agent,
            session_id=agent.session_id,
            turn_id="invalid-route",
            user_message="search the web",
        ),
        lambda: router.request_toolset_handler(
            {"toolsets": ["file"]}, agent=agent, session_id=agent.session_id
        ),
        lambda: router.tool_request_middleware(
            agent=agent, session_id=agent.session_id, tool_name="read_file", args={}
        ),
        lambda: router.post_tool_call(
            agent=agent, session_id=agent.session_id, tool_name="read_file"
        ),
    ):
        caller()
    router._handle_full_fallback(agent)
    assert state._bound_admission_result is bound
    assert agent.tools == before_tools
    assert agent.enabled_toolsets == before_enabled
    assert registry.definition_calls == before_calls
    monkeypatch.setattr(router, "_get_profile_config", lambda *args, **kwargs: {"enabled": False})
    router.pre_turn_context_build(
        agent=agent,
        session_id=agent.session_id,
        turn_id="invalid-disabled",
        user_message="search the web",
    )
    assert state._bound_admission_result is bound
    assert agent.tools == before_tools
    router.on_session_end(agent=agent, session_id=agent.session_id)
    assert state._bound_admission_result is None
    assert state._contamination_marker is None

    agent.tools = [registry.definitions["web_search"], registry.definitions["request_toolset"]]
    agent.valid_tool_names = {"web_search", "request_toolset"}
    agent.enabled_toolsets = ["web", "router_recovery"]
    monkeypatch.setattr(router, "_get_profile_config", lambda *args, **kwargs: {"enabled": True, "floor_toolsets": []})
    fresh = _rev6_admit(agent, monkeypatch, owned=False)
    assert fresh.status == "READY"
    assert fresh is not bound

def test_invalid_or_conflicting_policy_preserves_route_surface_and_denies_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    valid = {"schema_version": 1, "protected_toolsets": ["kanban"], "pinned_tool_names": []}
    invalid_channels = [
        None,
        {"schema_version": 1, "protected_toolsets": "kanban", "pinned_tool_names": []},
    ]
    for index, channel in enumerate(invalid_channels):
        agent = _rev6_agent(
            registry,
            f"invalid-policy-{index}",
            names=("web_search", "kanban_show", "request_toolset"),
            enabled=("web", "kanban", "router_recovery"),
        )
        before = (list(agent.tools), set(agent.valid_tool_names), list(agent.enabled_toolsets))
        result = router._ensure_host_admission(
            source="pre_turn_context_build",
            agent=agent,
            session_id=agent.session_id,
            untouched_surface=tuple(agent.tools),
            original_enabled_toolsets=tuple(agent.enabled_toolsets),
            hook_metadata=channel,
            agent_metadata=MISSING,
        )
        assert result.status == "SAFE_NO_PRUNE"
        assert result.envelope is not None
        assert result.effective_policy is not None and result.effective_policy.valid is False
        assert (list(agent.tools), set(agent.valid_tool_names), list(agent.enabled_toolsets)) == before

    conflicting = _rev6_agent(
        registry,
        "conflicting-policy",
        names=("web_search", "kanban_show", "request_toolset"),
        enabled=("web", "kanban", "router_recovery"),
    )
    conflicting._hermes_token_router_admission = {
        "schema_version": 1,
        "protected_toolsets": ["file"],
        "pinned_tool_names": [],
    }
    conflict = router._ensure_host_admission(
        source="pre_turn_context_build",
        agent=conflicting,
        session_id=conflicting.session_id,
        untouched_surface=tuple(conflicting.tools),
        original_enabled_toolsets=tuple(conflicting.enabled_toolsets),
        hook_metadata=valid,
        agent_metadata=conflicting._hermes_token_router_admission,
    )
    assert conflict.status == "SAFE_NO_PRUNE"
    assert conflict.effective_policy is not None
    assert "CONFLICTING_CHANNELS" in conflict.effective_policy.errors

    valid_agent = _rev6_agent(
        registry,
        "valid-empty-channel",
        names=("web_search", "kanban_show", "request_toolset"),
        enabled=("web", "kanban", "router_recovery"),
    )
    valid_result = router._ensure_host_admission(
        source="pre_turn_context_build",
        agent=valid_agent,
        session_id=valid_agent.session_id,
        untouched_surface=tuple(valid_agent.tools),
        original_enabled_toolsets=tuple(valid_agent.enabled_toolsets),
        hook_metadata=valid,
        agent_metadata=MISSING,
    )
    assert valid_result.status == "READY"
    assert valid_result.envelope is not None

def test_registry_mutation_after_capture_cannot_change_protected_ceiling_or_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    metadata = {"schema_version": 1, "protected_toolsets": ["kanban"], "pinned_tool_names": []}
    agent = _rev6_agent(
        registry,
        "registry-mutation",
        names=("web_search", "kanban_show", "request_toolset"),
        enabled=("web", "kanban", "router_recovery"),
    )
    _rev6_admit(agent, monkeypatch, hook_metadata=metadata, owned=False)
    original = json.loads(json.dumps(registry.definitions["kanban_show"]))
    registry.definition_calls.clear()
    registry.definitions["kanban_show"]["function"]["description"] = "mutated"
    registry.toolsets["kanban"].append("kanban_created_later")
    registry.entries["kanban_created_later"] = SimpleNamespace(toolset="kanban")
    registry.definitions["kanban_created_later"] = {
        "type": "function",
        "function": {"name": "kanban_created_later", "description": "new"},
    }
    monkeypatch.setattr(router, "_predict_toolsets_by_rules", lambda message, available: ({"web"}, "web"))
    router.pre_turn_context_build(
        agent=agent,
        session_id=agent.session_id,
        turn_id="mutation-route",
        user_message="search the web",
    )
    response = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["kanban"]},
            agent=agent,
            session_id=agent.session_id,
        )
    )
    assert response["ok"] is True
    assert "kanban_created_later" not in _rev6_names(agent)
    recovered = next(item for item in agent.tools if item["function"]["name"] == "kanban_show")
    assert recovered == original
    assert not any("kanban_created_later" in call for call in registry.definition_calls)
    router._handle_full_fallback(agent)
    restored = next(item for item in agent.tools if item["function"]["name"] == "kanban_show")
    assert restored == original

def test_global_recovery_schema_is_discovery_only_and_handler_denies_per_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    schema = router.build_recovery_tool_schema(set(registry.toolsets))
    items = schema["parameters"]["properties"]["toolsets"]["items"]
    assert items == {"type": "string"}
    assert "enum" not in items
    specialist = _rev6_agent(
        registry,
        "schema-specialist",
        names=("web_search", "request_toolset"),
        enabled=("web", "router_recovery"),
    )
    before = (list(specialist.tools), set(specialist.valid_tool_names), list(specialist.enabled_toolsets))
    denied = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["kanban"]},
            agent=specialist,
            session_id=specialist.session_id,
        )
    )
    assert denied["ok"] is False
    assert denied["denied_toolsets"] == ["kanban"]
    assert (list(specialist.tools), set(specialist.valid_tool_names), list(specialist.enabled_toolsets)) == before

    admitted = _rev6_agent(
        registry,
        "schema-direct",
        names=("kanban_show", "request_toolset"),
        enabled=("kanban", "router_recovery"),
    )
    _rev6_admit(
        admitted,
        monkeypatch,
        owned=False,
        hook_metadata={"schema_version": 1, "protected_toolsets": ["kanban"], "pinned_tool_names": []},
    )
    accepted = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["kanban"]},
            agent=admitted,
            session_id=admitted.session_id,
        )
    )
    assert accepted["ok"] is True
    assert accepted["added_tool_names"] == []

def test_total_protected_denial_changes_no_surface_state_counter_fallback_or_retry_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(
        registry,
        "total-denial",
        names=("web_search", "request_toolset"),
        enabled=("web", "router_recovery"),
    )
    admission = _rev6_admit(agent, monkeypatch, owned=False)
    assert admission.status == "READY"
    state = router._get_router_state(agent)
    state.active = True
    state.predicted_toolsets = {"web"}
    state.initial_toolsets = {"web"}
    state.active_toolsets = {"web"}
    state.installed_tool_names = set(_rev6_names(agent))
    state.admitted_toolsets = {"web"}
    state.expansion_count = 7
    state._fallback_triggered = False
    state._retry_pending = False
    state.routed_turn_id = "turn"
    state.routed_source = "test"
    before = (
        list(agent.tools),
        set(agent.valid_tool_names),
        list(agent.enabled_toolsets),
        state.capability_snapshot(),
        state._bound_admission_result,
    )
    result = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["kanban"]},
            agent=agent,
            session_id=agent.session_id,
        )
    )
    assert result["ok"] is False
    assert result["denied_toolsets"] == ["kanban"]
    assert result["added_tool_names"] == []
    assert list(agent.tools) == before[0]
    assert set(agent.valid_tool_names) == before[1]
    assert list(agent.enabled_toolsets) == before[2]
    assert state.capability_snapshot() == before[3]
    assert state._bound_admission_result is before[4]

def test_explicit_unmapped_incoming_pin_survives_without_registry_refill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    registry.entries.pop("kanban_show")
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(
        registry,
        "unmapped-pin",
        names=("web_search", "kanban_show", "request_toolset"),
        enabled=("web", "kanban", "router_recovery"),
    )
    agent.tools[1] = {
        "type": "function",
        "function": {"name": "kanban_show", "description": "host-only-schema", "parameters": {"x": 1}},
    }
    admission = _rev6_admit(
        agent,
        monkeypatch,
        owned=False,
        hook_metadata={
            "schema_version": 1,
            "protected_toolsets": ["kanban"],
            "pinned_tool_names": ["kanban_show"],
        },
    )
    assert admission.status == "READY"
    assert admission.envelope is not None
    assert admission.envelope.owner_snapshot.owner_by_name[1] == ("kanban_show", None)
    registry.definition_calls.clear()
    agent.tools = [agent.tools[0], agent.tools[2]]
    agent.valid_tool_names = {"web_search", "request_toolset"}
    monkeypatch.setattr(router, "_predict_toolsets_by_rules", lambda message, available: ({"web"}, "web"))
    router.pre_turn_context_build(
        agent=agent,
        session_id=agent.session_id,
        turn_id="unmapped-route",
        user_message="search the web",
    )
    assert "kanban_show" in _rev6_names(agent)
    recovered = next(item for item in agent.tools if item["function"]["name"] == "kanban_show")
    assert recovered["function"]["description"] == "host-only-schema"
    agent.tools = [item for item in agent.tools if item["function"]["name"] != "kanban_show"]
    agent.valid_tool_names.discard("kanban_show")
    response = json.loads(
        router.request_toolset_handler(
            {"tool_name": "kanban_show"},
            agent=agent,
            session_id=agent.session_id,
        )
    )
    assert response["ok"] is True
    assert response["added_tool_names"] == ["kanban_show"]
    assert not registry.definition_calls
    router._handle_full_fallback(agent)
    assert next(item for item in agent.tools if item["function"]["name"] == "kanban_show")["function"]["description"] == "host-only-schema"

def test_no_authority_preserves_current_and_only_appends_ordinary_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    clean = _rev6_agent(
        registry,
        "no-authority-clean",
        names=("web_search", "request_toolset"),
        enabled=("web", "router_recovery"),
    )
    first = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["terminal"]}, agent=clean, session_id=clean.session_id
        )
    )
    assert first["ok"] is True
    state = router._get_router_state(clean)
    marker = state._contamination_marker
    assert marker is not None
    assert marker.first_added_tool_names == ("run_command",)
    before_protected = (list(clean.tools), set(clean.valid_tool_names), list(clean.enabled_toolsets))
    denied = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["kanban"]}, agent=clean, session_id=clean.session_id
        )
    )
    assert denied["ok"] is False
    assert denied["denied_toolsets"] == ["kanban"]
    assert (list(clean.tools), set(clean.valid_tool_names), list(clean.enabled_toolsets)) == before_protected
    mixed = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["skills", "kanban"]}, agent=clean, session_id=clean.session_id
        )
    )
    assert mixed["ok"] is False
    assert mixed["expanded_toolsets"] == ["skills"]
    assert mixed["denied_toolsets"] == ["kanban"]
    assert state._contamination_marker is marker

    middleware_agent = _rev6_agent(
        registry,
        "no-authority-middleware",
        names=("web_search", "request_toolset"),
        enabled=("web", "router_recovery"),
    )
    recovered = router.tool_request_middleware(
        agent=middleware_agent,
        session_id=middleware_agent.session_id,
        tool_name="run_command",
        args={"command": "true"},
    )
    assert recovered == {"args": {"command": "true"}, "router_recovered": "terminal"}
    assert router.tool_request_middleware(
        agent=middleware_agent,
        session_id=middleware_agent.session_id,
        tool_name="kanban_show",
        args={},
    ) is None

    post_agent = _rev6_agent(
        registry,
        "no-authority-post",
        names=("web_search", "request_toolset"),
        enabled=("web", "router_recovery"),
    )
    router.post_tool_call(
        agent=post_agent,
        session_id=post_agent.session_id,
        tool_name="read_file",
    )
    assert "read_file" in _rev6_names(post_agent)
    assert router._get_router_state(post_agent)._retry_pending is True

    fallback_agent = _rev6_agent(
        registry,
        "no-authority-fallback",
        names=("web_search", "request_toolset"),
        enabled=("web", "router_recovery"),
    )
    before = (list(fallback_agent.tools), set(fallback_agent.valid_tool_names), list(fallback_agent.enabled_toolsets))
    router._handle_full_fallback(fallback_agent)
    assert (list(fallback_agent.tools), set(fallback_agent.valid_tool_names), list(fallback_agent.enabled_toolsets)) == before

    class Unattachable:
        __slots__ = ("session_id", "tools", "valid_tool_names", "enabled_toolsets")

        def __init__(self) -> None:
            self.session_id = "unattachable-truth-table"
            self.tools = [registry.definitions["web_search"]]
            self.valid_tool_names = {"web_search"}
            self.enabled_toolsets = ["web"]

    unattachable = Unattachable()
    before = (list(unattachable.tools), set(unattachable.valid_tool_names), list(unattachable.enabled_toolsets))
    denied = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["terminal"]},
            agent=unattachable,
            session_id=unattachable.session_id,
        )
    )
    assert denied["ok"] is False
    assert denied["reason"] == "NO_PERSISTENT_CONTAMINATION_GUARD"
    assert (list(unattachable.tools), set(unattachable.valid_tool_names), list(unattachable.enabled_toolsets)) == before

def test_worker_answer_only_keeps_pins_and_recovery_control_in_fixed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(
        registry,
        "worker-answer-order",
        names=("kanban_complete", "web_search", "kanban_show"),
        enabled=("kanban", "web"),
    )
    _rev6_admit(agent, monkeypatch, task="worker", owned=True)
    monkeypatch.setattr(router, "_predict_toolsets_by_rules", lambda message, available: (set(), "answer"))
    router.pre_turn_context_build(
        agent=agent,
        session_id=agent.session_id,
        turn_id="answer-order",
        user_message="hello",
    )
    assert _rev6_names(agent) == ["kanban_complete", "kanban_show", "request_toolset"]
    assert "kanban_sibling" not in _rev6_names(agent)
    assert _rev6_names(agent)[-1] == "request_toolset"
    # The recovery control is an agent-local copy of the registry definition
    # carrying this session's capability catalog; the shared registry
    # definition itself is unchanged.
    recovery = agent.tools[-1]
    shared = registry.definitions["request_toolset"]
    assert recovery["function"]["name"] == shared["function"]["name"]
    assert recovery["function"]["parameters"] == shared["function"]["parameters"]
    assert shared["function"]["description"] == "captured-request_toolset"
    catalog = recovery["function"]["description"]
    assert "kanban (" in catalog and "web (" in catalog
    assert "kanban_sibling" not in catalog

def test_no_authority_append_contaminates_session_and_forbids_later_host_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    strict = {"schema_version": 1, "protected_toolsets": ["terminal"], "pinned_tool_names": []}
    callers = ("route", "visible_request", "middleware", "post_tool")
    for caller in callers:
        agent = _rev6_agent(
            registry,
            f"contamination-{caller}",
            names=("web_search", "request_toolset"),
            enabled=("web", "router_recovery"),
        )
        initial = router._ensure_host_admission(
            source=caller,
            agent=agent,
            session_id=agent.session_id,
            untouched_surface=MISSING,
            original_enabled_toolsets=MISSING,
            hook_metadata=MISSING,
            agent_metadata=MISSING,
        )
        assert initial.status == "NO_AUTHORITY"
        if caller == "visible_request":
            result = json.loads(
                router.request_toolset_handler(
                    {"toolsets": ["terminal"]}, agent=agent, session_id=agent.session_id
                )
            )
        elif caller == "middleware":
            assert router.tool_request_middleware(
                agent=agent, session_id=agent.session_id, tool_name="run_command", args={}
            ) is not None
            result = {"added_tool_names": ["run_command"]}
        elif caller == "post_tool":
            router.post_tool_call(agent=agent, session_id=agent.session_id, tool_name="run_command")
            result = {"added_tool_names": ["run_command"]}
        else:
            result_obj = router.expand_admitted_toolsets(
                agent,
                initial,
                requested_toolsets={"terminal"},
                requested_tool_names=set(),
                caller="route",
                session_id=agent.session_id,
            )
            result = {"added_tool_names": list(result_obj.added_tool_names)}
        assert result["added_tool_names"] == ["run_command"], caller
        state = router._get_router_state(agent)
        marker = state._contamination_marker
        assert marker is not None
        assert marker.first_added_tool_names == ("run_command",)
        before = (list(agent.tools), set(agent.valid_tool_names), list(agent.enabled_toolsets))
        contaminated = router._ensure_host_admission(
            source="pre_turn_context_build",
            agent=agent,
            session_id=agent.session_id,
            untouched_surface=MISSING,
            original_enabled_toolsets=MISSING,
            hook_metadata=strict,
            agent_metadata=MISSING,
        )
        assert contaminated.status == "NO_AUTHORITY_CONTAMINATED"
        assert contaminated.contamination is marker
        denied = json.loads(
            router.request_toolset_handler(
                {"toolsets": ["terminal"]},
                agent=agent,
                session_id=agent.session_id,
                hermes_token_router_admission=strict,
            )
        )
        assert denied["ok"] is False
        assert denied["added_tool_names"] == []
        assert (list(agent.tools), set(agent.valid_tool_names), list(agent.enabled_toolsets)) == before
        router.on_session_end(agent=agent, session_id=agent.session_id)
        assert state._contamination_marker is None

    class Unattachable:
        __slots__ = ("session_id", "tools", "valid_tool_names", "enabled_toolsets")

        def __init__(self) -> None:
            self.session_id = "contamination-unattachable"
            self.tools = [registry.definitions["web_search"]]
            self.valid_tool_names = {"web_search"}
            self.enabled_toolsets = ["web"]

    unattachable = Unattachable()
    before = (list(unattachable.tools), set(unattachable.valid_tool_names), list(unattachable.enabled_toolsets))
    response = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["terminal"]},
            agent=unattachable,
            session_id=unattachable.session_id,
        )
    )
    assert response["ok"] is False
    assert response["added_tool_names"] == []
    assert (list(unattachable.tools), set(unattachable.valid_tool_names), list(unattachable.enabled_toolsets)) == before

def test_fail_open_and_bypass_paths_restore_the_exact_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    cases = [
        ("long", {"long_message_decline_chars": 3}, "this message is long", None),
        ("short", {"short_message_bypass_chars": 20}, "short", None),
        ("classifier-disabled", {"deterministic_rules_enabled": False}, "ambiguous", None),
        ("classifier-none", {"deterministic_rules_enabled": False, "classifier": {"enabled": True}}, "ambiguous", "none"),
        ("classifier-exception", {"deterministic_rules_enabled": False, "classifier": {"enabled": True}}, "ambiguous", RuntimeError("classifier")),
        ("available-empty", {}, "search the web", "empty"),
        ("available-error", {}, "search the web", "error"),
    ]
    for label, overrides, message, mode in cases:
        with monkeypatch.context() as patch:
            _rev6_config(patch, registry, **overrides)
            agent = _rev6_agent(
                registry,
                f"bypass-{label}",
                names=("web_search", "read_file", "kanban_show", "request_toolset"),
                enabled=("web", "file", "kanban", "router_recovery"),
            )
            admission = _rev6_admit(agent, patch, task="worker", owned=True)
            assert admission.status == "READY"
            expected = list(agent.tools)
            agent.tools = [registry.definitions["web_search"]]
            agent.valid_tool_names = {"web_search"}
            if mode == "empty":
                patch.setattr(router, "_get_available_toolsets", lambda: set())
            elif mode == "error":
                def unavailable():
                    raise RuntimeError("available")
                patch.setattr(router, "_get_available_toolsets", unavailable)
            elif mode == "none":
                patch.setattr(router, "_predict_toolsets_via_llm", lambda *args, **kwargs: None)
            elif isinstance(mode, BaseException):
                def classify_error(*args, mode=mode, **kwargs):
                    raise mode
                patch.setattr(router, "_predict_toolsets_via_llm", classify_error)
            elif mode is not None:
                def classify_error(message, available, mode=mode):
                    if isinstance(mode, BaseException):
                        raise mode
                    return None, "bypass"
                patch.setattr(router, "_predict_toolsets_by_rules", classify_error)
                patch.setattr(router, "_is_classifier_enabled", lambda cfg: True)
            else:
                patch.setattr(router, "_predict_toolsets_by_rules", lambda message, available: (set(), "bypass"))
            router.pre_turn_context_build(
                agent=agent,
                session_id=agent.session_id,
                turn_id=f"{label}-turn",
                user_message=message,
            )
            assert agent.tools == expected, label
            assert _rev6_names(agent) == [
                "web_search", "read_file", "kanban_show", "request_toolset"
            ], label
            assert router._get_router_state(agent)._bound_admission_result is admission

def test_ready_candidate_and_partial_assignment_failures_restore_exact_precall_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_token_router_plugin.tools as tool_module

    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    metadata = {"schema_version": 1, "protected_toolsets": ["kanban"], "pinned_tool_names": []}

    def snapshot(agent: Any) -> tuple[Any, ...]:
        state = router._get_router_state(agent)
        return (
            list(agent.tools),
            set(agent.valid_tool_names),
            list(agent.enabled_toolsets),
            state.capability_snapshot(),
            state.admission_slots(),
        )

    candidate_agent = _rev6_agent(
        registry,
        "ready-candidate-failure",
        names=("web_search", "request_toolset"),
        enabled=("web", "router_recovery"),
    )
    _rev6_admit(candidate_agent, monkeypatch, hook_metadata=metadata, owned=False)
    candidate_state_before = snapshot(candidate_agent)
    candidate_admission = router._get_router_state(candidate_agent)._bound_admission_result
    assert isinstance(candidate_admission, EnsureAdmissionResult)
    registry.definitions["read_file"] = {
        "type": "function",
        "function": {"name": "read_file", "description": object()},
    }
    candidate_calls_before = list(registry.definition_calls)
    candidate_result = tool_module.select_and_commit_route(
        candidate_agent,
        candidate_admission,
        predicted_toolsets={"file"},
        floor_toolsets=set(),
        available_toolsets=set(registry.toolsets),
        caller="route",
    )
    assert candidate_result.ok is False
    assert snapshot(candidate_agent) == candidate_state_before
    assert registry.definition_calls != candidate_calls_before

    class SetterAgent:
        def __init__(self, source: Any) -> None:
            object.__setattr__(self, "session_id", source.session_id)
            object.__setattr__(self, "_current_turn_id", "")
            object.__setattr__(self, "tools", list(source.tools))
            object.__setattr__(self, "valid_tool_names", set(source.valid_tool_names))
            object.__setattr__(self, "enabled_toolsets", list(source.enabled_toolsets))
            object.__setattr__(self, "fail_valid_once", True)
            object.__setattr__(self, "tools_assignments", [])

        def __setattr__(self, name: str, value: Any) -> None:
            if name == "tools" and hasattr(self, "tools_assignments"):
                self.tools_assignments.append(value)
            if name == "valid_tool_names" and getattr(self, "fail_valid_once", False):
                object.__setattr__(self, "fail_valid_once", False)
                raise RuntimeError("valid-tool-names-setter")
            object.__setattr__(self, name, value)

    setter_agent = SetterAgent(
        _rev6_agent(
            registry,
            "ready-partial-assignment",
            names=("web_search", "kanban_show", "request_toolset"),
            enabled=("web", "kanban", "router_recovery"),
        )
    )
    admission = _rev6_admit(setter_agent, monkeypatch, hook_metadata=metadata, owned=False)
    setter_before = snapshot(setter_agent)
    setter_state = router._get_router_state(setter_agent)
    registry.definition_calls.clear()
    result = tool_module.select_and_commit_route(
        setter_agent,
        admission,
        predicted_toolsets={"web"},
        floor_toolsets=set(),
        available_toolsets=set(registry.toolsets),
        caller="route",
    )
    assert result.ok is False
    assert result.reason.startswith("COMMIT_FAILED:")
    assert setter_agent.tools_assignments
    assert snapshot(setter_agent) == setter_before
    assert setter_state._bound_admission_result is admission
    assert registry.definition_calls == []
    assert "kanban_sibling" not in _rev6_names(setter_agent)

def test_no_authority_invalid_policy_denies_additions_without_binding_or_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_token_router_plugin.tools as tool_module

    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    valid_metadata = {
        "schema_version": 1,
        "protected_toolsets": [],
        "pinned_tool_names": [],
    }
    invalid_metadata = None
    composition_calls: list[tuple[Any, dict[str, Any]]] = []
    task_reads: list[str] = []
    predicate_calls: list[str] = []
    capture_calls: list[str] = []
    original_compose = router._compose_effective_policy
    original_capture = router._capture_owner_snapshot

    def compose_spy(parsed: Any, owner_snapshot: Any, **kwargs: Any) -> Any:
        composition_calls.append((owner_snapshot, dict(kwargs)))
        return original_compose(parsed, owner_snapshot, **kwargs)

    def task_spy() -> str:
        task_reads.append("read")
        return os.environ.get("HERMES_KANBAN_TASK", "")

    def predicate_spy() -> bool:
        predicate_calls.append("predicate")
        return True

    def capture_spy(agent: Any, surface: Any) -> Any:
        capture_calls.append(str(getattr(agent, "session_id", "")))
        return original_capture(agent, surface)

    monkeypatch.setattr(router, "_compose_effective_policy", compose_spy)
    monkeypatch.setattr(router, "_read_worker_task_id", task_spy)
    monkeypatch.setattr(router, "_is_dispatcher_owned_worker_context", predicate_spy)
    monkeypatch.setattr(router, "_capture_owner_snapshot", capture_spy)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    def make_agent(session_id: str) -> Any:
        return _rev6_agent(
            registry,
            session_id,
            names=("web_search", "request_toolset"),
            enabled=("web", "router_recovery"),
        )

    def state_snapshot(agent: Any) -> tuple[Any, ...]:
        state = router._get_router_state(agent)
        return (
            list(agent.tools),
            set(agent.valid_tool_names),
            list(agent.enabled_toolsets),
            state.capability_snapshot(),
            state.admission_slots(),
        )

    # Every no-envelope caller must use the typed-None neutral composition mode.
    for caller in ("route", "visible_request", "middleware", "post_tool", "fallback"):
        agent = make_agent(f"invalid-clean-{caller}")
        registry.entry_calls.clear()
        registry.definition_calls.clear()
        before = state_snapshot(agent)
        initial = router._ensure_host_admission(
            source=caller,
            agent=agent,
            session_id=agent.session_id,
            untouched_surface=MISSING,
            original_enabled_toolsets=MISSING,
            hook_metadata=invalid_metadata,
            agent_metadata=MISSING,
        )
        assert initial.status == "NO_AUTHORITY_INVALID_POLICY"
        assert initial.owner_snapshot is None
        assert initial.contamination is None
        assert router._get_router_state(agent)._bound_admission_result is None
        after_initial = state_snapshot(agent)
        assert after_initial[:3] == before[:3]
        assert after_initial[4] == (agent.session_id, None, None)
        assert not registry.entry_calls
        assert not registry.definition_calls
        owner_snapshot, compose_kwargs = composition_calls[-1]
        assert owner_snapshot is None
        assert compose_kwargs["has_nonempty_worker_task_id"] is False
        assert compose_kwargs["dispatcher_owned_worker"] is None
        assert compose_kwargs["worker_identity_error"] is None

        if caller == "route":
            result = tool_module.expand_admitted_toolsets(
                agent,
                initial,
                requested_toolsets={"terminal"},
                requested_tool_names=set(),
                caller="route",
                session_id=agent.session_id,
            )
            assert result.ok is False
        elif caller == "visible_request":
            result = json.loads(
                router.request_toolset_handler(
                    {"toolsets": ["terminal"]},
                    agent=agent,
                    session_id=agent.session_id,
                    hermes_token_router_admission=invalid_metadata,
                )
            )
            assert result["ok"] is False
            registry.entry_calls.clear()
            registry.definition_calls.clear()
            explicit_result = json.loads(
                router.request_toolset_handler(
                    {"tool_name": "read_file"},
                    agent=agent,
                    session_id=agent.session_id,
                    hermes_token_router_admission=invalid_metadata,
                )
            )
            assert explicit_result["ok"] is False
            assert explicit_result["reason"] == "INVALID_TRUSTED_POLICY"
            assert not registry.entry_calls
            assert not registry.definition_calls
        elif caller == "middleware":
            assert router.tool_request_middleware(
                agent=agent,
                session_id=agent.session_id,
                tool_name="run_command",
                args={},
                hermes_token_router_admission=invalid_metadata,
            ) is None
        elif caller == "post_tool":
            router.post_tool_call(
                agent=agent,
                session_id=agent.session_id,
                tool_name="run_command",
                hermes_token_router_admission=invalid_metadata,
            )
        else:
            router._handle_full_fallback(agent)
        assert state_snapshot(agent) == after_initial
        assert not registry.entry_calls
        assert not registry.definition_calls

    # A valid empty policy retains ordinary append compatibility, while fallback
    # remains an exact preservation path and creates no marker by itself.
    ordinary = make_agent("valid-empty-ordinary")
    ordinary_initial = router._ensure_host_admission(
        source="visible_request",
        agent=ordinary,
        session_id=ordinary.session_id,
        untouched_surface=MISSING,
        original_enabled_toolsets=MISSING,
        hook_metadata=MISSING,
        agent_metadata=MISSING,
    )
    assert ordinary_initial.status == "NO_AUTHORITY"
    ordinary_result = tool_module.expand_admitted_toolsets(
        ordinary,
        ordinary_initial,
        requested_toolsets={"terminal"},
        requested_tool_names=set(),
        caller="visible_request",
        session_id=ordinary.session_id,
    )
    assert ordinary_result.ok is True
    assert ordinary_result.added_tool_names == ("run_command",)
    assert router._get_router_state(ordinary)._contamination_marker is not None

    fallback_only = make_agent("valid-empty-fallback")
    router._ensure_host_admission(
        source="fallback",
        agent=fallback_only,
        session_id=fallback_only.session_id,
        untouched_surface=MISSING,
        original_enabled_toolsets=MISSING,
        hook_metadata=MISSING,
        agent_metadata=MISSING,
    )
    fallback_before = state_snapshot(fallback_only)
    router._handle_full_fallback(fallback_only)
    assert state_snapshot(fallback_only) == fallback_before
    assert router._get_router_state(fallback_only)._contamination_marker is None

    # A clean invalid call is transient: the next genuinely valid untouched host
    # route can capture exactly once, after one real worker evaluation.
    later = make_agent("invalid-then-capture")
    invalid = router._ensure_host_admission(
        source="visible_request",
        agent=later,
        session_id=later.session_id,
        untouched_surface=MISSING,
        original_enabled_toolsets=MISSING,
        hook_metadata=invalid_metadata,
        agent_metadata=MISSING,
    )
    assert invalid.status == "NO_AUTHORITY_INVALID_POLICY"
    monkeypatch.setenv("HERMES_KANBAN_TASK", "worker-task")
    captured = router._ensure_host_admission(
        source="pre_turn_context_build",
        agent=later,
        session_id=later.session_id,
        untouched_surface=tuple(later.tools),
        original_enabled_toolsets=tuple(later.enabled_toolsets),
        hook_metadata=valid_metadata,
        agent_metadata=MISSING,
    )
    assert captured.status == "READY"
    assert captured.owner_snapshot is not None
    assert capture_calls.count(later.session_id) == 1
    assert task_reads == ["read"]
    assert predicate_calls == ["predicate"]
    reused = router._ensure_host_admission(
        source="pre_llm_call",
        agent=later,
        session_id=later.session_id,
        untouched_surface=(),
        original_enabled_toolsets=(),
        hook_metadata=valid_metadata,
        agent_metadata=MISSING,
    )
    assert reused is captured
    assert capture_calls.count(later.session_id) == 1
    assert task_reads == ["read"]
    assert predicate_calls == ["predicate"]

    # Existing contamination is retained by invalid-policy calls and cannot be
    # replaced by a later valid policy or promoted to an envelope.
    contaminated = make_agent("invalid-contaminated")
    clean = router._ensure_host_admission(
        source="visible_request",
        agent=contaminated,
        session_id=contaminated.session_id,
        untouched_surface=MISSING,
        original_enabled_toolsets=MISSING,
        hook_metadata=MISSING,
        agent_metadata=MISSING,
    )
    first_append = tool_module.expand_admitted_toolsets(
        contaminated,
        clean,
        requested_toolsets={"terminal"},
        requested_tool_names=set(),
        caller="visible_request",
        session_id=contaminated.session_id,
    )
    assert first_append.ok is True
    marker = router._get_router_state(contaminated)._contamination_marker
    assert marker is not None
    before_contaminated = state_snapshot(contaminated)
    invalid_contaminated = router._ensure_host_admission(
        source="visible_request",
        agent=contaminated,
        session_id=contaminated.session_id,
        untouched_surface=MISSING,
        original_enabled_toolsets=MISSING,
        hook_metadata=invalid_metadata,
        agent_metadata=MISSING,
    )
    assert invalid_contaminated.status == "NO_AUTHORITY_INVALID_POLICY"
    assert invalid_contaminated.contamination is marker
    assert state_snapshot(contaminated) == before_contaminated
    assert router._get_router_state(contaminated)._bound_admission_result is None
    valid_contaminated = router._ensure_host_admission(
        source="pre_llm_call",
        agent=contaminated,
        session_id=contaminated.session_id,
        untouched_surface=MISSING,
        original_enabled_toolsets=MISSING,
        hook_metadata=valid_metadata,
        agent_metadata=MISSING,
    )
    assert valid_contaminated.status == "NO_AUTHORITY_CONTAMINATED"
    assert valid_contaminated.contamination is marker
    assert router._get_router_state(contaminated)._bound_admission_result is None

def test_no_authority_marker_transaction_failures_restore_slots_surface_and_capture_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_token_router_plugin.tools as tool_module

    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)

    def make_agent(session_id: str) -> Any:
        return _rev6_agent(
            registry,
            session_id,
            names=("web_search", "request_toolset"),
            enabled=("web", "router_recovery"),
        )

    def snapshot(agent: Any) -> tuple[Any, ...]:
        state = router._get_router_state(agent)
        return (
            list(agent.tools),
            set(agent.valid_tool_names),
            list(agent.enabled_toolsets),
            state.capability_snapshot(),
            state.admission_slots(),
        )

    def no_authority(agent: Any) -> EnsureAdmissionResult:
        result = router._ensure_host_admission(
            source="visible_request",
            agent=agent,
            session_id=agent.session_id,
            untouched_surface=MISSING,
            original_enabled_toolsets=MISSING,
            hook_metadata=MISSING,
            agent_metadata=MISSING,
        )
        assert result.status == "NO_AUTHORITY"
        return result

    # A marker write that raises, and a marker readback mismatch, both happen
    # before candidate assignment and must leave a clean lifecycle slot.
    for mode in ("write", "readback"):
        agent = make_agent(f"marker-{mode}")
        admission = no_authority(agent)
        state = router._get_router_state(agent)
        before = snapshot(agent)
        original_mark = state.mark_no_authority_contaminated
        original_lookup = state.no_authority_contamination
        if mode == "write":
            def mark_then_fail(**kwargs: Any) -> Any:
                original_mark(**kwargs)
                raise RuntimeError("marker-write")
            monkeypatch.setattr(state, "mark_no_authority_contaminated", mark_then_fail)
        else:
            lookup_calls = 0
            def lookup_once(session_id: str) -> Any:
                nonlocal lookup_calls
                lookup_calls += 1
                value = original_lookup(session_id)
                return value if lookup_calls == 1 else None
            monkeypatch.setattr(state, "no_authority_contamination", lookup_once)
        result = tool_module.expand_admitted_toolsets(
            agent,
            admission,
            requested_toolsets={"terminal"},
            requested_tool_names=set(),
            caller="visible_request",
            session_id=agent.session_id,
        )
        assert result.ok is False
        assert result.reason.startswith("TRANSACTION_ROLLED_BACK:")
        assert snapshot(agent) == before
        assert state.admission_slots() == before[4]
        assert state._contamination_marker is None
        assert "run_command" not in _rev6_names(agent)
        monkeypatch.undo()
        _rev6_registry_modules(monkeypatch, registry)
        _rev6_config(monkeypatch, registry)

    class SetterAgent:
        def __init__(self, source: Any) -> None:
            object.__setattr__(self, "session_id", source.session_id)
            object.__setattr__(self, "_current_turn_id", "")
            object.__setattr__(self, "tools", list(source.tools))
            object.__setattr__(self, "valid_tool_names", set(source.valid_tool_names))
            object.__setattr__(self, "enabled_toolsets", list(source.enabled_toolsets))
            object.__setattr__(self, "fail_valid_once", False)
            object.__setattr__(self, "tools_assignments", [])

        def __setattr__(self, name: str, value: Any) -> None:
            if name == "tools" and hasattr(self, "tools_assignments"):
                self.tools_assignments.append(value)
            if name == "valid_tool_names" and getattr(self, "fail_valid_once", False):
                object.__setattr__(self, "fail_valid_once", False)
                raise RuntimeError("coordinated-setter")
            object.__setattr__(self, name, value)

    # Failure after agent.tools accepts the candidate must restore every field
    # and the clean lifecycle slots, not merely return an error.
    partial = SetterAgent(make_agent("marker-partial"))
    partial_admission = no_authority(partial)
    partial_state = router._get_router_state(partial)
    partial_before = snapshot(partial)
    partial.fail_valid_once = True
    partial_result = tool_module.expand_admitted_toolsets(
        partial,
        partial_admission,
        requested_toolsets={"terminal"},
        requested_tool_names=set(),
        caller="visible_request",
        session_id=partial.session_id,
    )
    assert partial_result.ok is False
    assert partial_result.reason.startswith("TRANSACTION_ROLLED_BACK:")
    assert partial.tools_assignments
    assert snapshot(partial) == partial_before
    assert partial_state.admission_slots() == partial_before[4]
    assert partial_state._contamination_marker is None

    # A later append failure retains the original immutable marker by identity.
    contaminated = SetterAgent(make_agent("marker-contaminated"))
    contaminated_admission = no_authority(contaminated)
    first = tool_module.expand_admitted_toolsets(
        contaminated,
        contaminated_admission,
        requested_toolsets={"terminal"},
        requested_tool_names=set(),
        caller="visible_request",
        session_id=contaminated.session_id,
    )
    assert first.ok is True
    contaminated_state = router._get_router_state(contaminated)
    marker = contaminated_state._contamination_marker
    assert marker is not None
    contaminated_before = snapshot(contaminated)
    contaminated.fail_valid_once = True
    later = tool_module.expand_admitted_toolsets(
        contaminated,
        contaminated_admission,
        requested_toolsets={"file"},
        requested_tool_names=set(),
        caller="visible_request",
        session_id=contaminated.session_id,
    )
    assert later.ok is False
    assert later.reason.startswith("TRANSACTION_ROLLED_BACK:")
    assert snapshot(contaminated) == contaminated_before
    assert contaminated_state._contamination_marker is marker
    assert contaminated_state.admission_slots()[2] is marker
    assert "read_file" not in _rev6_names(contaminated)

    # A clean rollback remains eligible for one later authoritative capture.
    fresh = make_agent("marker-capture-after-rollback")
    fresh_admission = no_authority(fresh)
    fresh_state = router._get_router_state(fresh)
    original_mark = fresh_state.mark_no_authority_contaminated
    def fail_once(**kwargs: Any) -> Any:
        original_mark(**kwargs)
        raise RuntimeError("marker-write")
    monkeypatch.setattr(fresh_state, "mark_no_authority_contaminated", fail_once)
    failed = tool_module.expand_admitted_toolsets(
        fresh,
        fresh_admission,
        requested_toolsets={"terminal"},
        requested_tool_names=set(),
        caller="visible_request",
        session_id=fresh.session_id,
    )
    assert failed.ok is False
    monkeypatch.setattr(fresh_state, "mark_no_authority_contaminated", original_mark)
    captured = router._ensure_host_admission(
        source="pre_turn_context_build",
        agent=fresh,
        session_id=fresh.session_id,
        untouched_surface=tuple(fresh.tools),
        original_enabled_toolsets=tuple(fresh.enabled_toolsets),
        hook_metadata={"schema_version": 1, "protected_toolsets": [], "pinned_tool_names": []},
        agent_metadata=MISSING,
    )
    assert captured.status == "READY"
    assert captured.owner_snapshot is not None
    assert fresh_state._contamination_marker is None

def test_empty_message_first_contact_binds_admission_before_early_and_late_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    metadata = {
        "schema_version": 1,
        "protected_toolsets": ["kanban"],
        "pinned_tool_names": [],
    }
    monkeypatch.setenv("HERMES_KANBAN_TASK", "empty-worker")
    monkeypatch.setattr(router, "_is_dispatcher_owned_worker_context", lambda: True)

    for source in ("early", "late"):
        with monkeypatch.context() as patch:
            call_order: list[str] = []
            target = _rev6_agent(
                registry,
                f"empty-first-{source}",
                names=("kanban_complete", "web_search", "kanban_show", "request_toolset"),
                enabled=("kanban", "web", "router_recovery"),
            )
            before_tools = list(target.tools)
            before_valid = set(target.valid_tool_names)
            before_enabled = list(target.enabled_toolsets)
            original_ensure = router._ensure_host_admission
            original_capture = router._capture_owner_snapshot
            original_task = router._read_worker_task_id
            original_predicate = router._is_dispatcher_owned_worker_context

            def ensure_spy(*args: Any, **kwargs: Any) -> Any:
                call_order.append("ensure")
                return original_ensure(*args, **kwargs)

            def capture_spy(*args: Any, **kwargs: Any) -> Any:
                call_order.append("capture")
                return original_capture(*args, **kwargs)

            def task_spy() -> str:
                call_order.append("task")
                return original_task()

            def predicate_spy() -> bool:
                call_order.append("predicate")
                return original_predicate()

            def unavailable() -> set[str]:
                call_order.append("available")
                raise AssertionError("empty-message route must not list available toolsets")

            patch.setattr(router, "_ensure_host_admission", ensure_spy)
            patch.setattr(router, "_capture_owner_snapshot", capture_spy)
            patch.setattr(router, "_read_worker_task_id", task_spy)
            patch.setattr(router, "_is_dispatcher_owned_worker_context", predicate_spy)
            patch.setattr(router, "_get_available_toolsets", unavailable)
            patch.setattr(
                router,
                "_predict_toolsets_by_rules",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("empty-message route must not classify")
                ),
            )

            router._store_agent_ref(target, target.session_id)
            if source == "early":
                router.pre_turn_context_build(
                    agent=target,
                    session_id=target.session_id,
                    turn_id=f"empty-{source}",
                    user_message="",
                    hermes_token_router_admission=metadata,
                )
            else:
                router.pre_llm_call(
                    session_id=target.session_id,
                    turn_id=f"empty-{source}",
                    user_message="",
                    hermes_token_router_admission=metadata,
                )

            state = router._get_router_state(target)
            bound = state._bound_admission_result
            assert bound is not None
            assert bound.status == "READY"
            assert bound.envelope is not None
            assert bound.envelope.pinned_names == frozenset(
                {"kanban_complete", "kanban_show"}
            )
            assert _rev6_names(target) == [
                "kanban_complete", "web_search", "kanban_show", "request_toolset"
            ]
            assert target.tools == before_tools
            assert target.valid_tool_names == before_valid
            assert target.enabled_toolsets == before_enabled
            assert call_order.index("ensure") < call_order.index("capture")
            assert "available" not in call_order
            assert call_order.count("capture") == 1
            assert call_order.count("task") == 1
            assert call_order.count("predicate") == 1
            assert registry.definition_calls == []
            assert state._retry_pending is False
            assert state.last_expansion_result is None

            # Later same-session admission reentry returns the identical stored
            # object before reading new metadata, environment, owners, or tools.
            before_reentry = list(call_order)
            reused = router._ensure_host_admission(
                source="pre_llm_call" if source == "early" else "pre_turn_context_build",
                agent=target,
                session_id=target.session_id,
                untouched_surface=(),
                original_enabled_toolsets=(),
                hook_metadata=None,
                agent_metadata=None,
            )
            assert reused is bound
            assert call_order.count("capture") == before_reentry.count("capture")
            assert call_order.count("task") == before_reentry.count("task")
            assert call_order.count("predicate") == before_reentry.count("predicate")
            assert call_order[-1] == "ensure"
            assert registry.definition_calls == []


def test_incomplete_registration_preserves_admitted_surface_without_narrowing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(
        registry,
        "incomplete-registration",
        names=("web_search", "kanban_show", "request_toolset"),
        enabled=("web", "kanban", "router_recovery"),
    )
    admission = _rev6_admit(agent, monkeypatch, owned=False)
    assert admission.status == "READY"
    before = (list(agent.tools), set(agent.valid_tool_names), list(agent.enabled_toolsets))
    flags = {
        name: getattr(router, name)
        for name in (
            "_registration_checked",
            "_recovery_middleware_registered",
            "_recovery_tool_registered",
        )
    }
    try:
        router._registration_checked = True
        router._recovery_middleware_registered = False
        router._recovery_tool_registered = True
        monkeypatch.setattr(
            router,
            "_predict_toolsets_by_rules",
            lambda message, available: ({"web"}, "web"),
        )
        router.pre_turn_context_build(
            agent=agent,
            session_id=agent.session_id,
            turn_id="registration-route",
            user_message="search the web",
        )
        assert (list(agent.tools), set(agent.valid_tool_names), list(agent.enabled_toolsets)) == before
        state = router._get_router_state(agent)
        assert state.initial_route_applied is False
        assert state._bound_admission_result is admission
    finally:
        router.__dict__.update(flags)



# --- Session capability catalog (request_toolset description) -------------


def _recovery_description(agent: Any) -> str:
    recovery = [item for item in agent.tools if item["function"]["name"] == "request_toolset"]
    assert len(recovery) == 1
    return recovery[0]["function"]["description"]


def _route(agent: Any, turn_id: str, message: str) -> None:
    router.pre_turn_context_build(
        agent=agent,
        session_id=agent.session_id,
        turn_id=turn_id,
        user_message=message,
    )


@pytest.mark.parametrize("host_supplies_recovery", [False, True])
def test_capability_question_sees_admitted_catalog_and_stays_stable(
    monkeypatch: pytest.MonkeyPatch,
    host_supplies_recovery: bool,
) -> None:
    registry = _Rev6Registry()
    shared_before = json.dumps(registry.definitions["request_toolset"], sort_keys=True)
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    names = ("web_search", "read_file", "run_command", "kanban_show")
    if host_supplies_recovery:
        names = names + ("request_toolset",)
    agent = _rev6_agent(
        registry,
        f"catalog-{host_supplies_recovery}",
        names=names,
        enabled=("web", "file", "terminal", "kanban", "router_recovery"),
    )
    assert _rev6_admit(agent, monkeypatch, owned=False).status == "READY"

    _route(agent, "catalog-1", "What can you do?")
    # Option A: capability questions keep the small surface ...
    assert _rev6_names(agent) == ["request_toolset"]
    catalog = _recovery_description(agent)
    # ... and the catalog names every admitted toolset with a description.
    for toolset in ("web", "file", "terminal", "kanban"):
        assert f"{toolset} (" in catalog, toolset
    # Registry toolsets outside the admitted envelope are never advertised.
    for outside in ("memory (", "skills (", "router_recovery"):
        assert outside not in catalog, outside
    assert "what you can do" in catalog
    # The shared registry definition is untouched.
    assert json.dumps(registry.definitions["request_toolset"], sort_keys=True) == shared_before

    # A later capability question in the sticky session still sees the same
    # catalog, byte for byte.
    _route(agent, "catalog-2", "Which tools do you have access to?")
    assert _recovery_description(agent) == catalog

    # Recovery expansion keeps the catalog byte-stable.
    response = json.loads(
        router.request_toolset_handler(
            {"toolsets": ["file"]},
            agent=agent,
            session_id=agent.session_id,
        )
    )
    assert response["ok"] is True
    assert "read_file" in _rev6_names(agent)
    assert _recovery_description(agent) == catalog
    assert json.dumps(registry.definitions["request_toolset"], sort_keys=True) == shared_before


def test_catalog_is_deterministic_across_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    descriptions = []
    for index, names in enumerate((
        ("web_search", "read_file", "kanban_show"),
        ("kanban_show", "read_file", "web_search"),
    )):
        agent = _rev6_agent(registry, f"determinism-{index}", names=names, enabled=("web", "file", "kanban"))
        assert _rev6_admit(agent, monkeypatch, owned=False).status == "READY"
        _route(agent, f"determinism-{index}", "hello")
        descriptions.append(_recovery_description(agent))
    assert descriptions[0] == descriptions[1]
    assert "Capability catalog for this session" in descriptions[0]


def test_host_protected_recovery_control_stays_byte_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(
        registry,
        "protected-recovery",
        names=("web_search", "kanban_show", "request_toolset"),
        enabled=("web", "kanban", "router_recovery"),
    )
    metadata = {
        "schema_version": 1,
        "protected_toolsets": ["kanban"],
        "pinned_tool_names": ["request_toolset"],
    }
    assert _rev6_admit(agent, monkeypatch, hook_metadata=metadata, owned=False).status == "READY"
    _route(agent, "protected-recovery", "What can you do?")
    assert _recovery_description(agent) == "captured-request_toolset"



def test_catalog_describes_partial_admission_by_tool_names(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(
        registry,
        "partial-catalog",
        names=("web_search", "kanban_show"),
        enabled=("web", "kanban"),
    )
    assert _rev6_admit(agent, monkeypatch, owned=False).status == "READY"
    _route(agent, "partial-catalog", "What can you do?")
    catalog = _recovery_description(agent)
    # Only kanban_show was admitted, so the generic "create, assign, move"
    # description must not be used.
    assert "kanban (limited to: kanban_show)" in catalog
    assert "create, assign, move" not in catalog
    # web was admitted in full, so it keeps its generic description.
    assert "web (Search and extract content from web pages)" in catalog


def test_catalog_lists_unusual_identifiers_and_never_drops_overflow() -> None:
    tools_mod = sys.modules["hermes_token_router_plugin.tools"]
    admitted = {"mcp/custom": ["custom_tool"], "_custom": ["other"], "SyntheticCustom": ["x"]}
    catalog = tools_mod.build_capability_catalog_description(admitted)
    for name in ("mcp/custom (", "_custom (", "SyntheticCustom ("):
        assert name in catalog, name
    assert "router_recovery" not in tools_mod.build_capability_catalog_description({"router_recovery": ["request_toolset"]})

    many = {f"toolset_{index:03d}": [f"tool_{index}"] for index in range(tools_mod.CATALOG_MAX_DESCRIBED + 5)}
    catalog = tools_mod.build_capability_catalog_description(many)
    for name in many:
        assert name in catalog, name
    assert "; also: " in catalog


def test_recovery_parameter_description_is_generated_from_registry() -> None:
    tools_mod = sys.modules["hermes_token_router_plugin.tools"]
    schema = tools_mod.build_recovery_tool_schema({"web", "synthetic_custom_toolset"})
    description = schema["parameters"]["properties"]["toolsets"]["description"]
    assert "synthetic_custom_toolset" in description
    assert "video_gen" not in description



def test_host_protected_recovery_owner_stays_byte_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(
        registry,
        "protected-recovery-owner",
        names=("web_search", "kanban_show", "request_toolset"),
        enabled=("web", "kanban", "router_recovery"),
    )
    metadata = {
        "schema_version": 1,
        "protected_toolsets": ["router_recovery"],
        "pinned_tool_names": [],
    }
    assert _rev6_admit(agent, monkeypatch, hook_metadata=metadata, owned=False).status == "READY"
    _route(agent, "protected-recovery-owner", "What can you do?")
    assert "request_toolset" in _rev6_names(agent)
    assert _recovery_description(agent) == "captured-request_toolset"


def test_catalog_copy_parameter_names_the_session_catalog() -> None:
    tools_mod = sys.modules["hermes_token_router_plugin.tools"]
    base = tools_mod.build_recovery_tool_schema({"web", "file", "memory"})
    definition = {"type": "function", "function": {"name": "request_toolset", **base}}
    decorated = tools_mod.with_capability_catalog(definition, {"web": ["web_search"]})
    param = decorated["function"]["parameters"]["properties"]["toolsets"]["description"]
    assert "memory" not in param
    assert "capability catalog" in param
    # The source definition is not mutated.
    assert "memory" in definition["function"]["parameters"]["properties"]["toolsets"]["description"]



def _call_hook_on_worker_thread(agent_holder: Any, hook: Any, **kwargs: Any) -> None:
    """Mimic current Hermes: the caller holds ``agent`` while a timeout worker runs the hook."""
    import threading

    agent = agent_holder  # noqa: F841 - must be a live local in this (caller) frame
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            hook(**kwargs)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(10)
    assert not thread.is_alive()
    assert not errors, errors


def test_pre_llm_call_on_worker_thread_finds_agent_by_session(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(
        registry,
        "worker-thread-route",
        names=("web_search", "read_file", "run_command"),
        enabled=("web", "file", "terminal"),
    )
    assert _rev6_admit(agent, monkeypatch, owned=False).status == "READY"
    router._drop_agent_ref(agent.session_id)
    _call_hook_on_worker_thread(
        agent,
        router.pre_llm_call,
        session_id=agent.session_id,
        turn_id="worker-1",
        user_message="search the web for news",
    )
    assert _rev6_names(agent) == ["web_search", "request_toolset"]


def test_worker_thread_lookup_never_binds_another_sessions_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    other = _rev6_agent(
        registry,
        "worker-thread-other",
        names=("web_search", "read_file", "run_command"),
        enabled=("web", "file", "terminal"),
    )
    before = list(other.tools)
    router._drop_agent_ref("worker-thread-unknown")
    router._drop_agent_ref(other.session_id)
    _call_hook_on_worker_thread(
        other,
        router.pre_llm_call,
        session_id="worker-thread-unknown",
        turn_id="worker-2",
        user_message="search the web for news",
    )
    assert other.tools == before



def test_tool_search_bridge_and_deferred_toolsets_survive_narrowing(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    # Hermes tool search: deferred tools (computer_use, cronjob, a protected
    # deferred toolset) are absent from agent.tools and fronted by the bridge;
    # their toolsets remain in enabled_toolsets.
    agent = _rev6_agent(
        registry,
        "bridge-narrowing",
        names=("web_search", "read_file", "run_command", "tool_search", "tool_describe", "tool_call"),
        enabled=("web", "file", "terminal", "computer_use", "cronjob", "secret_deferred"),
    )
    metadata = {
        "schema_version": 1,
        "protected_toolsets": ["secret_deferred"],
        "pinned_tool_names": [],
    }
    assert _rev6_admit(agent, monkeypatch, hook_metadata=metadata, owned=False).status == "READY"
    _route(agent, "bridge-1", "search the web for news")
    names = _rev6_names(agent)
    assert names[:4] == ["web_search", "tool_search", "tool_describe", "tool_call"]
    assert names[-1] == "request_toolset"
    assert "read_file" not in names and "run_command" not in names
    assert {"computer_use", "cronjob"} <= set(agent.enabled_toolsets)
    assert "file" not in agent.enabled_toolsets
    # A protected toolset is never re-enabled behind the bridge.
    assert "secret_deferred" not in agent.enabled_toolsets
    assert "tool_search" in _recovery_description(agent)

    response = json.loads(
        router.request_toolset_handler({"toolsets": ["file"]}, agent=agent, session_id=agent.session_id)
    )
    assert response["ok"] is True
    assert {"tool_search", "tool_describe", "tool_call", "read_file"} <= set(_rev6_names(agent))
    assert {"computer_use", "cronjob", "file"} <= set(agent.enabled_toolsets)
    assert "secret_deferred" not in agent.enabled_toolsets


def test_restore_returns_exact_enabled_toolsets_with_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    enabled = ("web", "file", "computer_use", "cronjob")
    agent = _rev6_agent(
        registry,
        "bridge-restore",
        names=("web_search", "read_file", "tool_search", "tool_describe", "tool_call"),
        enabled=enabled,
    )
    original_tools = list(agent.tools)
    admission = _rev6_admit(agent, monkeypatch, owned=False)
    assert admission.status == "READY"
    assert router.restore_admitted_envelope(agent, admission) is True
    assert list(agent.tools) == original_tools
    assert list(agent.enabled_toolsets) == list(enabled)



# --- Review round 3 regressions ------------------------------------------


def test_disabled_router_on_worker_thread_never_touches_the_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry, enabled=False)
    agent = _rev6_agent(
        registry,
        "disabled-worker",
        names=("web_search", "read_file"),
        enabled=("web", "file", "cronjob"),
    )
    router._drop_agent_ref(agent.session_id)
    _call_hook_on_worker_thread(
        agent, router.pre_llm_call,
        session_id=agent.session_id, turn_id="d-1", user_message="hello",
    )
    assert list(agent.enabled_toolsets) == ["web", "file", "cronjob"]
    late = {"type": "function", "function": {"name": "late_mcp_tool", "description": "late"}}
    agent.tools = list(agent.tools) + [late]
    agent.valid_tool_names = set(agent.valid_tool_names) | {"late_mcp_tool"}
    agent.enabled_toolsets = list(agent.enabled_toolsets) + ["late_mcp"]
    _call_hook_on_worker_thread(
        agent, router.pre_llm_call,
        session_id=agent.session_id, turn_id="d-2", user_message="hello again",
    )
    assert "late_mcp_tool" in _rev6_names(agent)
    assert "late_mcp_tool" in agent.valid_tool_names
    assert "late_mcp" in agent.enabled_toolsets


def test_disabling_after_narrowing_restores_once_then_stays_hands_off(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    config = {
        "enabled": True, "long_message_decline_chars": 10_000, "short_message_bypass_chars": 0,
        "floor_toolsets": [], "deterministic_rules_enabled": True, "confidence_threshold": 0.0,
        "router_hard_timeout_ms": 1_200, "classifier": {"enabled": False},
    }
    monkeypatch.setattr(router, "_load_config", lambda: {"global": config})
    monkeypatch.setattr(router, "_get_profile_config", lambda *a, **k: config)
    monkeypatch.setattr(router, "_get_available_toolsets", lambda: set(registry.toolsets))
    agent = _rev6_agent(registry, "disable-after-narrow", names=("web_search", "read_file"), enabled=("web", "file"))
    original = list(agent.tools)
    assert _rev6_admit(agent, monkeypatch, owned=False).status == "READY"
    _route(agent, "n-1", "search the web for news")
    assert "read_file" not in _rev6_names(agent)
    config["enabled"] = False
    _route(agent, "n-2", "next")
    assert list(agent.tools) == original
    late = {"type": "function", "function": {"name": "late_mcp_tool", "description": "late"}}
    agent.tools = list(agent.tools) + [late]
    _route(agent, "n-3", "and again")
    assert "late_mcp_tool" in _rev6_names(agent)


def test_route_rejects_stack_agent_with_a_different_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(registry, "actual-session", names=("web_search", "read_file"), enabled=("web", "file"))
    before = list(agent.tools)
    router._drop_agent_ref("different-hook-session")
    router.pre_llm_call(session_id="different-hook-session", turn_id="x", user_message="search the web")
    assert agent.tools == before
    assert router._get_agent_ref("different-hook-session") is not agent


def test_thread_scan_rejects_duplicate_session_ids_and_survives_hostile_objects() -> None:
    import threading

    state_mod = sys.modules["hermes_token_router_plugin.state"]

    class Hostile:
        session_id = "dup-session"

        @property
        def tools(self):  # pragma: no cover - must never be reached first
            raise RuntimeError("hostile property")

    def holder(obj: Any, ready: threading.Event, release: threading.Event) -> None:
        agent = obj  # noqa: F841 - live local for the scan
        ready.set()
        release.wait(5)

    def hostile_holder(ready: threading.Event, release: threading.Event) -> None:
        self = Hostile()  # noqa: F841 - live 'self' local
        ready.set()
        release.wait(5)

    first = SimpleNamespace(session_id="dup-session", tools=[], valid_tool_names=set(), enabled_toolsets=[])
    second = SimpleNamespace(session_id="dup-session", tools=[], valid_tool_names=set(), enabled_toolsets=[])
    release = threading.Event()
    events = [threading.Event() for _ in range(3)]
    threads = [
        threading.Thread(target=holder, args=(first, events[0], release)),
        threading.Thread(target=holder, args=(second, events[1], release)),
        threading.Thread(target=hostile_holder, args=(events[2], release)),
    ]
    for thread in threads:
        thread.start()
    try:
        for event in events:
            assert event.wait(5)
        assert state_mod._find_agent_in_threads("dup-session") is None
        assert state_mod._find_agent_in_threads("no-such-session") is None
    finally:
        release.set()
        for thread in threads:
            thread.join(5)


def test_explicit_agent_beats_a_stale_cached_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    stale = _rev6_agent(registry, "explicit-wins", names=("web_search",), enabled=("web",))
    fresh = _rev6_agent(registry, "explicit-wins", names=("web_search",), enabled=("web",))
    router._store_agent_ref(stale, "explicit-wins")
    assert router._resolve_hook_agent(fresh, "explicit-wins") is fresh


def test_protected_deferred_sibling_is_blocked_behind_the_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(
        registry,
        "protected-bridge",
        names=("web_search", "kanban_show", "tool_search", "tool_describe", "tool_call"),
        enabled=("web", "kanban"),
    )
    metadata = {"schema_version": 1, "protected_toolsets": ["kanban"], "pinned_tool_names": ["kanban_show"]}
    assert _rev6_admit(agent, monkeypatch, hook_metadata=metadata, owned=False).status == "READY"
    _route(agent, "pb-1", "What can you do?")
    assert "kanban" in agent.enabled_toolsets and "tool_call" in _rev6_names(agent)
    blocked = router.pre_tool_call(tool_name="kanban_sibling", session_id=agent.session_id, args={})
    assert isinstance(blocked, dict) and blocked["action"] == "block"
    # Visible/captured tools and ordinary deferred tools are untouched.
    assert router.pre_tool_call(tool_name="kanban_show", session_id=agent.session_id, args={}) is None
    assert router.pre_tool_call(tool_name="memory_search", session_id=agent.session_id, args={}) is None


def test_restore_assigns_the_exact_captured_enabled_list(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    for names, enabled in (
        (("web_search", "tool_search", "tool_describe", "tool_call"), ("cronjob",)),
        (("web_search", "read_file"), ("web", "file", "cronjob")),
    ):
        agent = _rev6_agent(registry, f"exact-{len(names)}", names=names, enabled=enabled)
        admission = _rev6_admit(agent, monkeypatch, owned=False)
        assert admission.status == "READY"
        assert router.restore_admitted_envelope(agent, admission) is True
        assert list(agent.enabled_toolsets) == list(enabled)



# --- Review round 4 regressions ------------------------------------------


def _narrowed_protected_bridge_agent(monkeypatch: pytest.MonkeyPatch, session: str, config: dict) -> Any:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    monkeypatch.setattr(router, "_load_config", lambda: {"global": config})
    monkeypatch.setattr(router, "_get_profile_config", lambda *a, **k: config)
    monkeypatch.setattr(router, "_get_available_toolsets", lambda: set(registry.toolsets))
    agent = _rev6_agent(
        registry,
        session,
        names=("web_search", "kanban_show", "tool_search", "tool_describe", "tool_call"),
        enabled=("web", "kanban"),
    )
    metadata = {"schema_version": 1, "protected_toolsets": ["kanban"], "pinned_tool_names": ["kanban_show"]}
    assert _rev6_admit(agent, monkeypatch, hook_metadata=metadata, owned=False).status == "READY"
    _route(agent, f"{session}-1", "What can you do?")
    return agent


def _base_config() -> dict:
    return {
        "enabled": True, "long_message_decline_chars": 10_000, "short_message_bypass_chars": 0,
        "floor_toolsets": [], "deterministic_rules_enabled": True, "confidence_threshold": 0.0,
        "router_hard_timeout_ms": 1_200, "classifier": {"enabled": False},
    }


def test_failed_disable_restore_keeps_guard_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config()
    agent = _narrowed_protected_bridge_agent(monkeypatch, "failed-restore", config)
    blocked = router.pre_tool_call(tool_name="kanban_sibling", session_id=agent.session_id, args={})
    assert blocked and blocked["action"] == "block"
    config["enabled"] = False
    real_restore = router.restore_admitted_envelope
    monkeypatch.setattr(router, "restore_admitted_envelope", lambda *a, **k: False)
    _route(agent, "failed-restore-2", "next")
    state = router._get_router_state(agent)
    assert state.surface_narrowed is True
    still = router.pre_tool_call(tool_name="kanban_sibling", session_id=agent.session_id, args={})
    assert still and still["action"] == "block"
    monkeypatch.setattr(router, "restore_admitted_envelope", real_restore)
    _route(agent, "failed-restore-3", "again")
    assert state.surface_narrowed is False
    assert "kanban_show" in _rev6_names(agent) and "web_search" in _rev6_names(agent)


def test_explicit_route_agent_from_another_session_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(registry, "actual-explicit", names=("web_search", "read_file"), enabled=("web", "file"))
    before = list(agent.tools)
    router._drop_agent_ref("wrong-explicit")
    router.pre_turn_context_build(
        agent=agent, session_id="wrong-explicit", turn_id="w", user_message="search the web",
    )
    assert agent.tools == before
    assert router._get_agent_ref("wrong-explicit") is not agent


def test_thread_scan_treats_agent_and_self_duplicates_as_ambiguous() -> None:
    import threading

    state_mod = sys.modules["hermes_token_router_plugin.state"]
    first = SimpleNamespace(session_id="tier-dup", tools=[], valid_tool_names=set(), enabled_toolsets=[])
    second = SimpleNamespace(session_id="tier-dup", tools=[], valid_tool_names=set(), enabled_toolsets=[])
    release = threading.Event()
    ready = [threading.Event(), threading.Event()]

    def as_agent() -> None:
        agent = first  # noqa: F841
        ready[0].set()
        release.wait(5)

    def as_self() -> None:
        self = second  # noqa: F841
        ready[1].set()
        release.wait(5)

    threads = [threading.Thread(target=as_agent), threading.Thread(target=as_self)]
    for thread in threads:
        thread.start()
    try:
        assert all(event.wait(5) for event in ready)
        assert state_mod._find_agent_in_threads("tier-dup") is None
    finally:
        release.set()
        for thread in threads:
            thread.join(5)


def test_session_end_restores_and_resets_for_agent_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config()
    agent = _narrowed_protected_bridge_agent(monkeypatch, "session-end", config)
    state = router._get_router_state(agent)
    assert state.surface_narrowed is True
    router.on_session_end(session_id=agent.session_id, agent=agent)
    assert state.surface_narrowed is False
    assert state.initial_route_applied is False
    assert "web_search" in _rev6_names(agent) and "kanban_show" in _rev6_names(agent)
    state.surface_narrowed = True
    state.reset()
    assert state.surface_narrowed is False


def test_own_stack_checks_session_before_touching_foreign_properties() -> None:
    state_mod = sys.modules["hermes_token_router_plugin.state"]
    touched: list[str] = []

    class Foreign:
        session_id = "someone-else"

        @property
        def tools(self):
            touched.append("tools")
            raise RuntimeError("must not be read")

    agent = Foreign()  # noqa: F841 - live local for the own-stack walk
    assert state_mod._get_agent_from_stack("my-session", scan_threads=False) is None
    assert touched == []



# --- Review round 5 regressions ------------------------------------------


def _fail_restore_once(monkeypatch: pytest.MonkeyPatch) -> None:
    real = router.restore_admitted_envelope
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            return False
        return real(*args, **kwargs)

    monkeypatch.setattr(router, "restore_admitted_envelope", flaky)


def test_failed_session_end_restore_keeps_admission_guard_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config()
    agent = _narrowed_protected_bridge_agent(monkeypatch, "failed-end", config)
    state = router._get_router_state(agent)
    _fail_restore_once(monkeypatch)
    router.on_session_end(session_id=agent.session_id, agent=agent)
    assert state.surface_narrowed is True
    assert state.bound_host_admission(agent.session_id) is not None
    blocked = router.pre_tool_call(tool_name="kanban_sibling", session_id=agent.session_id, args={})
    assert blocked and blocked["action"] == "block"
    # Retry: a second session end (or a disabled turn) restores and retires.
    router.on_session_end(session_id=agent.session_id, agent=agent)
    assert state.surface_narrowed is False
    assert state.bound_host_admission(agent.session_id) is None
    assert "web_search" in _rev6_names(agent)


def test_failed_session_end_then_disabled_turn_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config()
    agent = _narrowed_protected_bridge_agent(monkeypatch, "failed-end-disabled", config)
    state = router._get_router_state(agent)
    _fail_restore_once(monkeypatch)
    router.on_session_end(session_id=agent.session_id, agent=agent)
    config["enabled"] = False
    _route(agent, "failed-end-disabled-2", "next")
    assert state.surface_narrowed is False
    assert "web_search" in _rev6_names(agent)


def test_stale_route_after_session_end_and_reuse_never_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(registry, "stale-old", names=("web_search", "read_file"), enabled=("web", "file"))
    assert _rev6_admit(agent, monkeypatch, owned=False).status == "READY"
    entered = threading.Event()
    release = threading.Event()
    real_rules = router._predict_toolsets_by_rules

    def paused_rules(message: str, available: set[str]):
        entered.set()
        release.wait(5)
        return real_rules(message, available)

    monkeypatch.setattr(router, "_predict_toolsets_by_rules", paused_rules)
    worker = threading.Thread(
        target=lambda: router.pre_turn_context_build(
            agent=agent, session_id="stale-old", turn_id="s-1", user_message="search the web for news",
        )
    )
    worker.start()
    assert entered.wait(5)
    # The host gives up on the hook, ends the session and reuses the agent.
    router.on_session_end(session_id="stale-old", agent=agent)
    file_only = [registry.definitions["read_file"]]
    agent.session_id = "stale-new"
    agent.tools = list(file_only)
    agent.valid_tool_names = {"read_file"}
    agent.enabled_toolsets = ["file"]
    release.set()
    worker.join(5)
    assert not worker.is_alive()
    assert agent.tools == file_only
    assert agent.enabled_toolsets == ["file"]
    assert router._get_router_state(agent).surface_narrowed is False


def test_route_commit_deadline_discards_slow_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(registry, "deadline", names=("web_search", "read_file"), enabled=("web", "file"))
    before = list(agent.tools)
    assert _rev6_admit(agent, monkeypatch, owned=False).status == "READY"
    monkeypatch.setattr(router, "_hook_commit_budget_s", lambda: -1.0)
    _route(agent, "deadline-1", "search the web for news")
    assert agent.tools == before



# --- Follow-ups ------------------------------------------------------------


def test_skill_view_stays_visible_after_narrowing(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(
        registry, "skills-floor",
        names=("web_search", "read_file", "skill_view"), enabled=("web", "file", "skills"),
    )
    assert _rev6_admit(agent, monkeypatch, owned=False).status == "READY"
    _route(agent, "skills-floor-1", "search the web for news")
    names = _rev6_names(agent)
    assert "skill_view" in names and "read_file" not in names
    assert "skills" in agent.enabled_toolsets


def test_protected_skills_are_not_kept_unless_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(
        registry, "skills-protected",
        names=("web_search", "skill_view"), enabled=("web", "skills"),
    )
    metadata = {"schema_version": 1, "protected_toolsets": ["skills"], "pinned_tool_names": []}
    assert _rev6_admit(agent, monkeypatch, hook_metadata=metadata, owned=False).status == "READY"
    _route(agent, "skills-protected-1", "search the web for news")
    assert "skill_view" not in _rev6_names(agent)



def test_classifier_rejects_out_of_range_confidence_and_unknown_toolsets() -> None:
    policy = sys.modules["hermes_token_router_plugin.policy"]
    available = {"web", "file"}
    for confidence in (101, "101%", 250, -1, "nan"):
        payload = json.dumps({"toolsets": ["web"], "confidence": confidence})
        assert policy.parse_classifier_payload(payload, available, 0.9) == {"action": "full"}, confidence
    for confidence in (0.95, 95, "95%"):
        payload = json.dumps({"toolsets": ["web"], "confidence": confidence})
        assert policy.parse_classifier_payload(payload, available, 0.9) == {"action": "narrow", "toolsets": ["web"]}
    mixed = json.dumps({"toolsets": ["web", "bogus"], "confidence": 0.99})
    assert policy.parse_classifier_payload(mixed, available, 0.9) == {"action": "full"}


def test_classifier_http_timeout_follows_hard_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = sys.modules["hermes_token_router_plugin.policy"]
    seen: dict[str, float] = {}

    def fake_client(model, provider, base_url, key_env, request_timeout=8.0):
        seen["timeout"] = request_timeout
        return None, None

    monkeypatch.setattr(policy, "_get_router_client", fake_client)
    assert policy._predict_toolsets_via_llm("hi", {"web"}, "m", 0.9, "openrouter", 1.2) is None
    assert seen["timeout"] == pytest.approx(1.2)


def test_floor_defaults_to_none_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    config = _base_config()
    config.pop("floor_toolsets")
    monkeypatch.setattr(router, "_load_config", lambda: {"global": config})
    monkeypatch.setattr(router, "_get_profile_config", lambda *a, **k: config)
    monkeypatch.setattr(router, "_get_available_toolsets", lambda: set(registry.toolsets))
    agent = _rev6_agent(registry, "floor-default", names=("web_search", "read_file", "run_command"), enabled=("web", "file", "terminal"))
    assert _rev6_admit(agent, monkeypatch, owned=False).status == "READY"
    _route(agent, "floor-default-1", "search the web for news")
    assert _rev6_names(agent) == ["web_search", "request_toolset"]


def test_request_toolset_payload_truthfulness(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    registry.toolsets["MixedCase"] = ["mixed_tool"]
    registry.definitions["mixed_tool"] = {
        "type": "function",
        "function": {"name": "mixed_tool", "description": "m", "parameters": {"type": "object", "properties": {}}},
    }
    registry.entries["mixed_tool"] = SimpleNamespace(toolset="MixedCase")
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(
        registry, "payload-truth",
        names=("web_search", "read_file", "mixed_tool"), enabled=("web", "file", "MixedCase"),
    )
    assert _rev6_admit(agent, monkeypatch, owned=False).status == "READY"
    _route(agent, "payload-truth-1", "search the web for news")
    ok = json.loads(router.request_toolset_handler(
        {"toolsets": ["file"], "reason": "need to read notes"}, agent=agent, session_id=agent.session_id,
    ))
    assert ok["ok"] is True and ok["reason"] == "READY" and ok["request_reason"] == "need to read notes"
    mixed = json.loads(router.request_toolset_handler(
        {"toolsets": ["MixedCase"]}, agent=agent, session_id=agent.session_id,
    ))
    assert mixed["ok"] is True and "mixed_tool" in _rev6_names(agent)
    missing = json.loads(router.request_toolset_handler(
        {"tool_name": "no_such_tool"}, agent=agent, session_id=agent.session_id,
    ))
    assert missing["ok"] is False and "no_such_tool" in missing["denied_tool_names"]


def test_retry_latch_resets_only_on_a_new_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(registry, "retry-latch", names=("web_search", "read_file"), enabled=("web", "file"))
    assert _rev6_admit(agent, monkeypatch, owned=False).status == "READY"
    _route(agent, "retry-latch-1", "search the web for news")
    state = router._get_router_state(agent)
    state._retry_pending = True
    router.pre_llm_call(session_id=agent.session_id, turn_id="retry-latch-1", user_message="same turn")
    assert state._retry_pending is True
    router.pre_llm_call(session_id=agent.session_id, turn_id="retry-latch-2", user_message="next")
    assert state._retry_pending is False
    state._retry_pending = True
    router.pre_llm_call(session_id=agent.session_id, turn_id="retry-latch-2", user_message="again")
    assert state._retry_pending is True


# --- Review round 6 regressions ------------------------------------------


_UNSET = object()


def _pause(
    monkeypatch: pytest.MonkeyPatch, module: Any, name: str, result: Any = _UNSET, *, once: bool = False
):
    import threading

    entered = threading.Event()
    release = threading.Event()
    real = getattr(module, name)
    calls = {"n": 0}

    def paused(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if not once or calls["n"] == 1:
            entered.set()
            release.wait(5)
        return real(*args, **kwargs) if result is _UNSET else result

    monkeypatch.setattr(module, name, paused)
    return entered, release


def _end_and_reuse_file_only(agent: Any, registry: Any, old: str, new: str) -> list[dict[str, Any]]:
    router.on_session_end(session_id=old, agent=agent)
    file_only = [registry.definitions["read_file"]]
    agent.session_id = new
    agent.tools = list(file_only)
    agent.valid_tool_names = {"read_file"}
    agent.enabled_toolsets = ["file"]
    return file_only


def _run_paused(target: Any, entered: Any, release: Any, between: Any) -> None:
    import threading

    worker = threading.Thread(target=target)
    worker.start()
    assert entered.wait(5)
    # Session end and reuse must not wait for the paused hook: slow work
    # runs outside the mutation lock.
    host = threading.Thread(target=between)
    host.start()
    host.join(2)
    blocked_on_hook = host.is_alive()
    release.set()
    host.join(5)
    assert not blocked_on_hook
    worker.join(5)
    assert not worker.is_alive()


@pytest.mark.parametrize("pause", ["unresolved_prediction", "registry_selection"])
def test_stale_fallback_or_selection_never_commits_after_reuse(
    monkeypatch: pytest.MonkeyPatch, pause: str
) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(registry, f"r6-{pause}", names=("web_search", "read_file"), enabled=("web", "file"))
    assert _rev6_admit(agent, monkeypatch, owned=False).status == "READY"
    if pause == "unresolved_prediction":
        # Paused, then the prediction is unresolved -> restore fallback path.
        entered, release = _pause(monkeypatch, router, "_predict_toolsets_by_rules", (None, []))
    else:
        # Paused inside selection, after every route-level check.
        entered, release = _pause(monkeypatch, sys.modules["hermes_token_router_plugin.tools"], "_registry_object")
    holder: dict[str, Any] = {}
    _run_paused(
        lambda: router.pre_turn_context_build(
            agent=agent, session_id=f"r6-{pause}", turn_id="r6-1", user_message="search the web for news",
        ),
        entered,
        release,
        lambda: holder.setdefault("file_only", _end_and_reuse_file_only(agent, registry, f"r6-{pause}", "r6-new")),
    )
    state = router._get_router_state(agent)
    assert agent.tools == holder["file_only"]
    assert agent.enabled_toolsets == ["file"]
    assert state.initial_route_applied is False
    assert state.surface_narrowed is False
    assert state.bound_host_admission("r6-new") is None


def test_stale_post_tool_expansion_never_commits_after_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(registry, "r6-post", names=("web_search", "read_file"), enabled=("web", "file"))
    assert _rev6_admit(agent, monkeypatch, owned=False).status == "READY"
    _route(agent, "r6-post-1", "search the web for news")
    entered, release = _pause(monkeypatch, sys.modules["hermes_token_router_plugin.tools"], "_registry_object")
    holder: dict[str, Any] = {}
    _run_paused(
        lambda: router.post_tool_call(agent=agent, session_id="r6-post", tool_name="read_file", args={}),
        entered,
        release,
        lambda: holder.setdefault("file_only", _end_and_reuse_file_only(agent, registry, "r6-post", "r6-post-new")),
    )
    state = router._get_router_state(agent)
    assert agent.tools == holder["file_only"]
    assert state._retry_pending is False


def test_deadline_counts_config_time_and_follows_host_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    import time as time_mod

    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(registry, "r6-deadline", names=("web_search", "read_file"), enabled=("web", "file"))
    before = list(agent.tools)
    assert _rev6_admit(agent, monkeypatch, owned=False).status == "READY"

    def slow_timeout() -> float:
        time_mod.sleep(0.3)
        return 0.4  # budget 0.2 s, already spent reading config

    monkeypatch.setattr(router, "_hook_callback_timeout_s", slow_timeout)
    _route(agent, "r6-deadline-1", "search the web for news")
    assert agent.tools == before
    monkeypatch.setattr(router, "_hook_callback_timeout_s", lambda: 0.0)
    assert router._hook_commit_budget_s() == float("inf")
    monkeypatch.setattr(router, "_hook_callback_timeout_s", lambda: 0.1)
    assert router._hook_commit_budget_s() == pytest.approx(0.05)


def test_local_timeout_normalization_matches_hermes(monkeypatch: pytest.MonkeyPatch) -> None:
    plugins_mod = ModuleType("hermes_cli.plugins")
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", plugins_mod)  # no resolver -> local path
    config_mod = ModuleType("hermes_cli.config")
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config_mod)
    for configured, expected in ((None, 30.0), (0, 0.0), (2000, 600.0), (-5, 30.0), ("x", 30.0), (12, 12.0)):
        plugins_cfg = {} if configured is None else {"hook_callback_timeout": configured}
        config_mod.load_config_readonly = lambda c=plugins_cfg: {"plugins": c}
        assert router._hook_callback_timeout_s() == expected, configured


def test_reuse_after_failed_end_is_quarantined_until_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config()
    agent = _narrowed_protected_bridge_agent(monkeypatch, "r6-quarantine", config)
    state = router._get_router_state(agent)
    real_restore = router.restore_admitted_envelope
    monkeypatch.setattr(router, "restore_admitted_envelope", lambda *a, **k: False)
    router.on_session_end(session_id="r6-quarantine", agent=agent)
    narrowed = list(agent.tools)
    agent.session_id = "r6-quarantine-new"
    blocked = router.pre_tool_call(tool_name="kanban_sibling", session_id="r6-quarantine-new", args={})
    assert blocked and blocked["action"] == "block"
    # Routing a new session while the old surface is unrestored changes nothing.
    _route(agent, "r6-q-1", "search the web for news")
    assert agent.tools == narrowed
    assert state._retired_admission is not None
    # Once the restore works, the agent is released and starts fresh.
    monkeypatch.setattr(router, "restore_admitted_envelope", real_restore)
    _route(agent, "r6-q-2", "search the web for news")
    assert state._retired_admission is None
    assert router.pre_tool_call(tool_name="kanban_sibling", session_id="r6-quarantine-new", args={}) is None or (
        state.bound_host_admission("r6-quarantine-new") is not None
    )


def test_case_insensitive_toolset_match_must_be_unique() -> None:
    assert router._normalize_toolset_name("TEAM", {"Team", "team"}) == "TEAM"
    assert router._normalize_toolset_name("team", {"Team", "team"}) == "team"
    assert router._normalize_toolset_name("mixedcase", {"MixedCase", "web"}) == "MixedCase"
    assert router._normalize_toolset_name("WEB", {"web"}) == "web"
    assert router._normalize_toolset_name("Unknown", {"web"}) == "unknown"


# --- Review round 7 regressions ------------------------------------------

_TOOLS_MOD = "hermes_token_router_plugin.tools"


def _three_tool_agent(monkeypatch: pytest.MonkeyPatch, sid: str, *, admit: bool = True):
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(
        registry, sid, names=("web_search", "read_file", "run_command"), enabled=("web", "file", "terminal"),
    )
    if admit:
        assert _rev6_admit(agent, monkeypatch, owned=False).status == "READY"
    return registry, agent


def test_first_capture_lookup_is_unlocked_and_never_binds_late(monkeypatch: pytest.MonkeyPatch) -> None:
    registry, agent = _three_tool_agent(monkeypatch, "r7-capture", admit=False)
    entered, release = _pause(monkeypatch, router, "_capture_owner_snapshot")
    holder: dict[str, Any] = {}
    _run_paused(
        lambda: router.pre_turn_context_build(
            agent=agent, session_id="r7-capture", turn_id="c-1", user_message="search the web for news",
        ),
        entered,
        release,
        lambda: holder.setdefault("file_only", _end_and_reuse_file_only(agent, registry, "r7-capture", "r7-capture-new")),
    )
    state = router._get_router_state(agent)
    assert agent.tools == holder["file_only"]
    assert state._admission_session_id is None
    assert state.bound_host_admission("r7-capture-new") is None


def test_post_tool_admission_is_fenced(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _Rev6Registry()
    _rev6_registry_modules(monkeypatch, registry)
    _rev6_config(monkeypatch, registry)
    agent = _rev6_agent(registry, "r7-post-adm", names=("web_search", "read_file"), enabled=("web", "file"))
    entered, release = _pause(monkeypatch, router, "_compose_effective_policy")
    holder: dict[str, Any] = {}
    _run_paused(
        lambda: router.post_tool_call(agent=agent, session_id="r7-post-adm", tool_name="run_command", args={}),
        entered,
        release,
        lambda: holder.setdefault("file_only", _end_and_reuse_file_only(agent, registry, "r7-post-adm", "r7-post-new")),
    )
    state = router._get_router_state(agent)
    assert state._admission_session_id is None
    assert agent.tools == holder["file_only"]


@pytest.mark.parametrize("entry", ["middleware", "visible_request"])
def test_recovery_entry_points_are_fenced(monkeypatch: pytest.MonkeyPatch, entry: str) -> None:
    registry, agent = _three_tool_agent(monkeypatch, f"r7-{entry}")
    _route(agent, f"r7-{entry}-1", "search the web for news")
    entered, release = _pause(monkeypatch, sys.modules[_TOOLS_MOD], "_registry_object")
    if entry == "middleware":
        call = lambda: router.tool_request_middleware(
            agent=agent, session_id=f"r7-{entry}", tool_name="run_command", args={},
        )
    else:
        call = lambda: router.request_toolset_handler(
            {"tool_name": "run_command"}, agent=agent, session_id=f"r7-{entry}",
        )
    holder: dict[str, Any] = {}
    _run_paused(
        call,
        entered,
        release,
        lambda: holder.setdefault("file_only", _end_and_reuse_file_only(agent, registry, f"r7-{entry}", "r7-new")),
    )
    assert agent.tools == holder["file_only"]
    assert agent.enabled_toolsets == ["file"]


def test_failed_end_revokes_in_flight_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    _registry, agent = _three_tool_agent(monkeypatch, "r7-revoke")
    _route(agent, "r7-revoke-1", "search the web for news")
    state = router._get_router_state(agent)
    narrowed = list(agent.tools)
    entered, release = _pause(monkeypatch, sys.modules[_TOOLS_MOD], "_registry_object")
    monkeypatch.setattr(router, "restore_admitted_envelope", lambda *a, **k: False)
    _run_paused(
        lambda: router.post_tool_call(agent=agent, session_id="r7-revoke", tool_name="run_command", args={}),
        entered,
        release,
        lambda: router.on_session_end(session_id="r7-revoke", agent=agent),
    )
    assert agent.tools == narrowed
    assert state._retry_pending is False
    assert state._retired_admission is not None


def test_retired_restore_respects_deadline_and_host_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config()
    agent = _narrowed_protected_bridge_agent(monkeypatch, "r7-retired", config)
    state = router._get_router_state(agent)
    real_restore = router.restore_admitted_envelope
    monkeypatch.setattr(router, "restore_admitted_envelope", lambda *a, **k: False)
    router.on_session_end(session_id="r7-retired", agent=agent)
    narrowed = list(agent.tools)
    agent.session_id = "r7-retired-new"
    monkeypatch.setattr(router, "restore_admitted_envelope", real_restore)
    # An expired hook never restores the retired surface.
    monkeypatch.setattr(router, "_hook_commit_budget_s", lambda: -1.0)
    _route(agent, "r7-r-1", "search the web for news")
    assert agent.tools == narrowed and state._retired_admission is not None
    # The host replaced the surface: old-session tools are never written over
    # it, and the ambiguous change does not release the quarantine either.
    monkeypatch.setattr(router, "_hook_commit_budget_s", lambda: 15.0)
    replacement = list(narrowed[:1])
    agent.tools = list(replacement)
    agent.valid_tool_names = {replacement[0]["function"]["name"]}
    router.pre_turn_context_build(agent=agent, session_id="r7-retired-new", turn_id="r7-r-2", user_message="")
    assert state._retired_admission is not None
    assert agent.tools == replacement


def test_overlapping_expansions_do_not_lose_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    _registry, agent = _three_tool_agent(monkeypatch, "r7-lost")
    _route(agent, "r7-lost-1", "search the web for news")
    entered, release = _pause(monkeypatch, sys.modules[_TOOLS_MOD], "_registry_object", once=True)
    _run_paused(
        lambda: router.post_tool_call(agent=agent, session_id="r7-lost", tool_name="run_command", args={}),
        entered,
        release,
        lambda: router.tool_request_middleware(agent=agent, session_id="r7-lost", tool_name="read_file", args={}),
    )
    names = _rev6_names(agent)
    assert "read_file" in names and "run_command" in names and "web_search" in names


def test_explicit_request_does_not_reset_the_retry_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    _registry, agent = _three_tool_agent(monkeypatch, "r7-latch")
    _route(agent, "r7-latch-1", "search the web for news")
    state = router._get_router_state(agent)
    router.post_tool_call(agent=agent, session_id="r7-latch", tool_name="read_file", args={})
    assert state._retry_pending is True
    router.request_toolset_handler({"tool_name": "run_command"}, agent=agent, session_id="r7-latch")
    assert state._retry_pending is True


# --- Review round 8 regressions ------------------------------------------


def _quarantined_agent(monkeypatch: pytest.MonkeyPatch, sid: str):
    config = _base_config()
    agent = _narrowed_protected_bridge_agent(monkeypatch, sid, config)
    state = router._get_router_state(agent)
    real_restore = router.restore_admitted_envelope
    monkeypatch.setattr(router, "restore_admitted_envelope", lambda *a, **k: False)
    router.on_session_end(session_id=sid, agent=agent)
    monkeypatch.setattr(router, "restore_admitted_envelope", real_restore)
    agent.session_id = f"{sid}-new"
    assert state._retired_admission is not None
    return agent, state


def test_same_name_replacement_is_not_overwritten_by_retired_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    agent, state = _quarantined_agent(monkeypatch, "r8-same-name")
    replaced = copy.deepcopy(agent.tools)
    for definition in replaced:
        definition["function"]["description"] = "host replacement"
    agent.tools = replaced
    agent.enabled_toolsets = ["web"]
    _route(agent, "r8-sn-1", "search the web for news")
    assert agent.tools == replaced and agent.enabled_toolsets == ["web"]
    assert state._retired_admission is not None


def test_partial_addition_keeps_quarantine_and_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    agent, state = _quarantined_agent(monkeypatch, "r8-partial")
    extra = {"type": "function", "function": {"name": "extra_tool", "description": "x", "parameters": {"type": "object", "properties": {}}}}
    agent.tools = list(agent.tools) + [extra]
    _route(agent, "r8-p-1", "search the web for news")
    assert state._retired_admission is not None
    blocked = router.pre_tool_call(tool_name="kanban_sibling", session_id=agent.session_id, args={})
    assert blocked and blocked["action"] == "block"


def test_exact_narrowed_surface_is_restored_and_released(monkeypatch: pytest.MonkeyPatch) -> None:
    agent, state = _quarantined_agent(monkeypatch, "r8-exact")
    _route(agent, "r8-e-1", "")
    assert state._retired_admission is None
    assert "kanban_show" in _rev6_names(agent)


def test_abandoned_session_end_never_touches_a_reused_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config()
    agent = _narrowed_protected_bridge_agent(monkeypatch, "r8-end", config)
    entered, release = _pause(monkeypatch, sys.modules[_TOOLS_MOD], "thaw_envelope_definitions")
    monkeypatch.setattr(router, "_hook_commit_budget_s", lambda: 0.5)
    replacement = [copy.deepcopy(agent.tools[0])]

    def reuse() -> None:
        import time as time_mod

        time_mod.sleep(0.6)  # Hermes gives up on the teardown, then reuses
        agent.session_id = "r8-end-new"
        agent.tools = list(replacement)
        agent.enabled_toolsets = ["web"]

    _run_paused(lambda: router.on_session_end(session_id="r8-end", agent=agent), entered, release, reuse)
    assert agent.tools == replacement and agent.enabled_toolsets == ["web"]


def test_overlapping_post_tool_callbacks_recover_once_per_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    _registry, agent = _three_tool_agent(monkeypatch, "r8-latch")
    _route(agent, "r8-latch-1", "search the web for news")
    state = router._get_router_state(agent)
    entered, release = _pause(monkeypatch, sys.modules[_TOOLS_MOD], "_registry_object", once=True)
    _run_paused(
        lambda: router.post_tool_call(agent=agent, session_id="r8-latch", tool_name="run_command", args={}),
        entered,
        release,
        lambda: router.post_tool_call(agent=agent, session_id="r8-latch", tool_name="read_file", args={}),
    )
    names = _rev6_names(agent)
    assert "read_file" in names and "run_command" not in names
    assert state._retry_pending is True


def test_stale_visible_request_reports_stale_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    registry, agent = _three_tool_agent(monkeypatch, "r8-stale", admit=False)
    entered, release = _pause(monkeypatch, router, "_compose_effective_policy")
    holder: dict[str, Any] = {}

    def call() -> None:
        holder["out"] = json.loads(
            router.request_toolset_handler({"tool_name": "run_command"}, agent=agent, session_id="r8-stale")
        )

    _run_paused(
        call,
        entered,
        release,
        lambda: _end_and_reuse_file_only(agent, registry, "r8-stale", "r8-stale-new"),
    )
    assert holder["out"]["ok"] is False and holder["out"]["reason"] == "STALE_LIFECYCLE"
    assert router._get_router_state(agent)._admission_session_id is None
