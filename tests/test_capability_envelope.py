from __future__ import annotations

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
    assert agent.tools[-1] == registry.definitions["request_toolset"]

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
