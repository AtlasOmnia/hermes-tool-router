"""Tool resolution, filtering, expansion, and recovery schema."""

from __future__ import annotations

import contextlib
import contextvars
import copy
import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Optional, Set

try:
    from .capabilities import (
        EnsureAdmissionResult,
        ExpansionResult,
        FrozenToolDefinition,
        HostAdmissionEnvelope,
        freeze_json,
        thaw_envelope_definitions,
        thaw_json,
    )
    from .config import PLUGIN_NAME
    from .policy import TOOLSET_DESCRIPTIONS
    from .state import _get_router_state
except ImportError:  # pragma: no cover - direct loader fallback
    from capabilities import (
        EnsureAdmissionResult,
        ExpansionResult,
        FrozenToolDefinition,
        HostAdmissionEnvelope,
        freeze_json,
        thaw_envelope_definitions,
        thaw_json,
    )
    from config import PLUGIN_NAME
    from policy import TOOLSET_DESCRIPTIONS
    from state import _get_router_state

logger = logging.getLogger(__name__)

RECOVERY_TOOL_NAME = "request_toolset"
RECOVERY_TOOLSET = "router_recovery"
RECOVERY_TOOLSET_CHOICES = sorted(TOOLSET_DESCRIPTIONS)

# Hermes "tool search" (progressive disclosure) replaces deferred tools in the
# model-visible array with these three bridge tools at agent build time.  They
# have no registry toolset, and the deferred tools behind them are scoped by
# the agent's enabled_toolsets.  Narrowing must keep both, or every deferred
# tool (computer_use, cronjob, browser_vault, MCP and plugin tools, ...) becomes
# unreachable for the rest of the session.
BRIDGE_TOOL_NAMES = frozenset({"tool_search", "tool_describe", "tool_call"})

# Hermes renders the skills index (<available_skills>) into the system prompt
# and tells the model to load skills with skill_view.  Pruning the skills
# toolset leaves that instruction pointing at a missing tool, and models then
# report "the skill library is empty".  These two small read-only tools stay
# whenever the host captured them (never a protected, unpinned definition).
ALWAYS_KEEP_TOOL_NAMES = frozenset({"skill_view", "skills_list"})


def _bridge_only_toolsets(envelope: Any, protected_toolsets: Any = ()) -> tuple[str, ...]:
    """Enabled toolsets reachable only through the tool-search bridge.

    These own no definition in the captured surface (their tools were deferred
    behind the bridge), so keeping them enabled costs no schema tokens.
    Protected toolsets are never retained this way: protected capability comes
    only from captured definitions.
    """
    if envelope is None:
        return ()
    names = {item.name for item in envelope.definitions}
    if not names & BRIDGE_TOOL_NAMES:
        return ()
    owners = {item.owner_toolset_at_capture for item in envelope.definitions}
    protected = set(protected_toolsets or ())
    return tuple(
        toolset
        for toolset in (envelope.original_enabled_toolsets or ())
        if isinstance(toolset, str)
        and toolset not in owners
        and toolset != RECOVERY_TOOLSET
        and toolset not in protected
    )


# Bounds for the per-session capability catalog carried in request_toolset's
# description.  Keep it compact and deterministic so the tool prefix stays
# byte-stable (prompt-cache friendly) while the session stays narrowed.
# Entries past CATALOG_MAX_DESCRIBED keep their exact names but drop the
# one-line description, so the catalog never silently omits a toolset.
CATALOG_MAX_DESCRIBED = 60
CATALOG_MAX_DESCRIPTION_CHARS = 80
CATALOG_MAX_PARTIAL_NAMES = 6
# Any admitted owner identifier is listed verbatim, except ones that could
# break the catalog's own punctuation or carry control characters.
_CATALOG_NAME_RE = re.compile(r"^[^\s;(),\x00-\x1f\x7f]{1,64}$")

RECOVERY_BASE_DESCRIPTION = "Load missing Hermes toolsets before continuing the task."


def _catalog_name_ok(name: Any) -> bool:
    return isinstance(name, str) and name != RECOVERY_TOOLSET and bool(_CATALOG_NAME_RE.match(name))


def _catalog_toolset_names(toolsets: Iterable[Any]) -> list[str]:
    return sorted({name for name in toolsets if _catalog_name_ok(name)})


def build_capability_catalog_description(
    admitted: Mapping[str, Iterable[str]] | Iterable[str],
    registry_members: Mapping[str, Iterable[str]] | None = None,
    bridge: bool = False,
) -> str:
    """Return the deterministic request_toolset description for one session.

    ``admitted`` maps each toolset owning definitions in the host-admission
    envelope to its admitted tool names (or is a plain iterable of toolset
    names).  That is what the agent would have with the router off, so the
    catalog never advertises a toolset outside that boundary.  When
    ``registry_members`` shows only part of a toolset was admitted, the entry
    names the admitted tools instead of the toolset's generic description, so
    a read-only subset is not described as read/write.
    """
    if isinstance(admitted, Mapping):
        admitted_map = {
            name: sorted({tool for tool in tools if isinstance(tool, str) and tool})
            for name, tools in admitted.items()
            if _catalog_name_ok(name)
        }
    else:
        admitted_map = {name: [] for name in _catalog_toolset_names(admitted)}
    names = sorted(admitted_map)
    if not names:
        return RECOVERY_BASE_DESCRIPTION
    members = registry_members or {}
    entries = []
    for name in names[:CATALOG_MAX_DESCRIBED]:
        tools = admitted_map[name]
        registry_tools = {tool for tool in members.get(name, ()) if isinstance(tool, str)}
        if tools and registry_tools and not registry_tools <= set(tools):
            listed = ", ".join(tools[:CATALOG_MAX_PARTIAL_NAMES])
            if len(tools) > CATALOG_MAX_PARTIAL_NAMES:
                listed += f", +{len(tools) - CATALOG_MAX_PARTIAL_NAMES} more"
            desc = f"limited to: {listed}"
        else:
            desc = TOOLSET_DESCRIPTIONS.get(name, f"Tools for {name}")
        desc = " ".join(str(desc).split())[:CATALOG_MAX_DESCRIPTION_CHARS]
        entries.append(f"{name} ({desc})")
    catalog = "; ".join(entries)
    overflow = names[CATALOG_MAX_DESCRIBED:]
    if overflow:
        catalog += "; also: " + ", ".join(overflow)
    return (
        "Some of your tools are not loaded yet, to save context. "
        "Capability catalog for this session (toolsets you can load with this tool; "
        "some may already be loaded): " + catalog + ". "
        "When the user asks what you can do or whether you can do something, "
        "answer from this catalog: these capabilities are available on request, "
        "though an account, credential, or approval may still be required. "
        "Call this tool with the exact toolset names to load them before using them. "
        "Never say a listed capability is unavailable without first loading it."
        + (
            " More tools are available on demand through tool_search; they are not "
            "listed here, so search there before saying a capability is unavailable."
            if bridge
            else ""
        )
    )


def with_capability_catalog(
    recovery_definition: dict[str, Any],
    admitted: Mapping[str, Iterable[str]] | Iterable[str],
    registry_members: Mapping[str, Iterable[str]] | None = None,
    bridge: bool = False,
) -> dict[str, Any]:
    """Return an agent-local copy of the recovery definition with the catalog.

    The registry's shared definition is never mutated.
    """
    definition = copy.deepcopy(recovery_definition)
    function = definition.get("function")
    if isinstance(function, dict):
        function["description"] = build_capability_catalog_description(
            admitted, registry_members, bridge
        )
        # Keep one list in the definition: the toolsets parameter names the
        # session catalog rather than the whole registry.
        toolsets_param = (
            function.get("parameters", {}).get("properties", {}).get("toolsets")
            if isinstance(function.get("parameters"), dict)
            else None
        )
        if isinstance(toolsets_param, dict) and "description" in toolsets_param:
            toolsets_param["description"] = (
                "One or more exact toolset names from the capability catalog above."
            )
    return definition


def _recovery_is_host_locked(owner: Any, protected_toolsets: Any, pins: Any) -> bool:
    """True when the host protected or pinned the recovery control itself."""
    return RECOVERY_TOOL_NAME in (pins or ()) or (owner is not None and owner in (protected_toolsets or ()))


def build_recovery_tool_schema(available_toolsets: Set[str]) -> Dict[str, Any]:
    """Build the compact recovery schema from the live registry surface."""
    names = _catalog_toolset_names(available_toolsets) or list(RECOVERY_TOOLSET_CHOICES)
    shown = names[:CATALOG_MAX_DESCRIBED]
    listed = ", ".join(shown) + (f", plus {len(names) - len(shown)} more" if len(names) > len(shown) else "")
    return {
        "description": RECOVERY_BASE_DESCRIPTION,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "toolsets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "uniqueItems": True,
                    "description": (
                        "One or more Hermes toolset names to add for this session. "
                        f"Registered toolsets: {listed}. Names are validated against "
                        "the live registry when called."
                    ),
                },
                "tool_name": {
                    "type": "string",
                    "description": "Optional registered tool whose owner toolset should be loaded.",
                },
                "reason": {"type": "string", "maxLength": 200},
            },
        },
    }


# Import-time fallback; register() rebuilds this from the live registry.
RECOVERY_TOOL_SCHEMA: Dict[str, Any] = build_recovery_tool_schema(set(RECOVERY_TOOLSET_CHOICES))


def _get_agent_local_tool_names(agent: Any, toolset_name: str) -> Set[str]:
    """Return agent-scoped tools that intentionally bypass the global registry."""
    if agent is None or toolset_name != "memory":
        return set()
    manager = getattr(agent, "_memory_manager", None)
    get_names = getattr(manager, "get_all_tool_names", None)
    if not callable(get_names):
        return set()
    try:
        raw_names = get_names()
        if not isinstance(raw_names, (set, list, tuple)):
            return set()
        names = {str(name) for name in raw_names if str(name)}
    except Exception:
        return set()
    state = _get_router_state(agent)
    return names & state._full_tool_names if state._full_tool_names else names


def _resolve_toolset_to_tool_names(toolsets: Set[str], agent: Any = None) -> Set[str]:
    """Resolve a set of toolset names to tool names via the registry."""
    tool_names: Set[str] = set()
    try:
        from tools.registry import registry
    except Exception as exc:
        logger.warning(
            "%s: failed to import tool registry for resolution: %s",
            PLUGIN_NAME, exc,
        )
        return tool_names

    for ts in sorted(toolsets):
        try:
            ts_tools = registry.get_tool_names_for_toolset(ts)
            if not ts_tools:
                logger.warning("%s: toolset '%s' resolved to no tools", PLUGIN_NAME, ts)
                continue
            logger.debug(
                "%s: toolset '%s' resolved to %d tools",
                PLUGIN_NAME,
                ts,
                len(ts_tools),
            )
            tool_names.update(ts_tools)
        except Exception as exc:
            logger.warning(
                "%s: failed to resolve toolset '%s': %s",
                PLUGIN_NAME, ts, exc,
            )
        tool_names.update(_get_agent_local_tool_names(agent, ts))

    if not tool_names:
        return tool_names

    try:
        definitions = registry.get_definitions(tool_names, quiet=True) or []
        materialized_names = {
            definition.get("function", {}).get("name", "")
            for definition in definitions
        }
        materialized_names.discard("")
        dropped_names = tool_names - materialized_names
        if dropped_names:
            logger.debug(
                "%s: dropped %d unmaterialized tool name(s): %s",
                PLUGIN_NAME,
                len(dropped_names),
                sorted(dropped_names)[:10],
            )
        return materialized_names
    except Exception as exc:
        logger.debug("%s: materialization filter skipped: %s", PLUGIN_NAME, exc)
        return tool_names
def _get_recovery_tool_definition() -> Optional[Dict[str, Any]]:
    """Return the OpenAI-format recovery tool definition, if registered."""
    try:
        from tools.registry import registry
        defs = registry.get_definitions({RECOVERY_TOOL_NAME}, quiet=True)
        return defs[0] if defs else None
    except Exception as exc:
        logger.debug("%s: recovery tool definition unavailable: %s", PLUGIN_NAME, exc)
        return None
def _infer_toolset_from_tool(tool_name: str, registry, agent: Any = None) -> Optional[str]:
    """Determine which toolset a tool belongs to."""
    # First try registry lookup.
    try:
        entry = registry.get_entry(tool_name)
        if entry and hasattr(entry, "toolset"):
            return entry.toolset
    except Exception:
        pass
    manager = getattr(agent, "_memory_manager", None) if agent is not None else None
    has_tool = getattr(manager, "has_tool", None)
    try:
        if callable(has_tool) and has_tool(tool_name):
            return "memory"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Admission-aware selection and transaction primitives.  Protected definitions
# never come from the live registry.


class StaleCommitError(RuntimeError):
    """The hook's lifecycle ended (or its deadline passed) before it committed."""


class SurfaceChangedError(RuntimeError):
    """Another commit changed the surface after this candidate was prepared."""


_EXPANSION_ATTEMPTS = 3


# A hook installs a fence (lock, still_valid) for its whole body.  Every
# coordinated assignment checks it under the lock immediately before
# assigning, so slow preparation runs unlocked and an abandoned callback can
# never commit into a later session.  No fence (direct callers, tests) means
# unfenced, as before.
_COMMIT_FENCE: contextvars.ContextVar = contextvars.ContextVar(
    "hermes_token_router_commit_fence", default=None
)


@contextlib.contextmanager
def commit_fence(lock: Any, still_valid: Any):
    token = _COMMIT_FENCE.set((lock, still_valid))
    try:
        yield
    finally:
        _COMMIT_FENCE.reset(token)


@contextlib.contextmanager
def fenced_section():
    """Hold the active fence's lock; yield whether the hook may still mutate."""
    fence = _COMMIT_FENCE.get()
    if fence is None:
        yield True
        return
    lock, still_valid = fence
    with lock:
        try:
            ok = bool(still_valid())
        except Exception:  # noqa: BLE001 - an unknown lifecycle is a stale one
            ok = False
        yield ok


def _tool_name(definition: Any) -> str:
    if not isinstance(definition, dict):
        return ""
    function = definition.get("function")
    if not isinstance(function, dict):
        return ""
    name = function.get("name")
    return name if isinstance(name, str) else ""


def _registry_object() -> Any:
    try:
        from tools.registry import registry
        return registry
    except Exception:
        return None


def _registry_tool_names(registry: Any, toolset: str) -> tuple[str, ...]:
    if registry is None:
        return ()
    try:
        names = registry.get_tool_names_for_toolset(toolset) or []
    except Exception:
        return ()
    return tuple(sorted({name for name in names if isinstance(name, str) and name}))


def _envelope_owner_map(envelope: HostAdmissionEnvelope | None) -> dict[str, str | None]:
    if envelope is None:
        return {}
    return {
        item.name: item.owner_toolset_at_capture
        for item in envelope.definitions
    }


def _envelope_definition_map(
    envelope: HostAdmissionEnvelope | None,
) -> dict[str, FrozenToolDefinition]:
    if envelope is None:
        return {}
    return {item.name: item for item in envelope.definitions}


def _current_tool_names(agent: Any) -> set[str]:
    return {
        _tool_name(definition)
        for definition in (getattr(agent, "tools", None) or [])
        if _tool_name(definition)
    }


def _owner_for_name(name: str, agent: Any, registry: Any) -> str | None:
    try:
        owner = _infer_toolset_from_tool(name, registry, agent) if registry is not None else None
    except Exception:
        owner = None
    return owner


def _is_protected_name(
    name: str,
    owner: str | None,
    admission: EnsureAdmissionResult,
) -> bool:
    policy = admission.effective_policy
    if policy is None:
        return False
    return owner in policy.protected_toolsets or name in policy.pinned_tool_names


def _copy_agent_fields(agent: Any) -> dict[str, tuple[bool, Any]]:
    result: dict[str, tuple[bool, Any]] = {}
    for field_name in ("tools", "valid_tool_names", "enabled_toolsets"):
        try:
            result[field_name] = (hasattr(agent, field_name), getattr(agent, field_name, None))
        except Exception:
            result[field_name] = (False, None)
    return result


def _validate_candidate_definitions(candidate: list[dict[str, Any]]) -> None:
    """Validate every candidate mapping before the first coordinated setter."""
    seen: set[str] = set()
    for index, definition in enumerate(candidate):
        if type(definition) is not dict:
            raise ValueError(f"MALFORMED_CANDIDATE_DEFINITION:{index}")
        function = definition.get("function")
        if type(function) is not dict:
            raise ValueError(f"MALFORMED_CANDIDATE_FUNCTION:{index}")
        name = function.get("name")
        if type(name) is not str or not name or name != name.strip():
            raise ValueError(f"MALFORMED_CANDIDATE_NAME:{index}")
        if name in seen:
            raise ValueError(f"DUPLICATE_CANDIDATE_NAME:{name}")
        seen.add(name)
        try:
            freeze_json(definition)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"MALFORMED_CANDIDATE_JSON:{index}") from exc


def _restore_agent_fields(agent: Any, snapshot: dict[str, tuple[bool, Any]]) -> None:
    for field_name, (present, value) in snapshot.items():
        if present:
            setattr(agent, field_name, value)
        else:
            try:
                delattr(agent, field_name)
            except AttributeError:
                pass


def names_include_bridge(names: Any) -> bool:
    return bool(set(names) & BRIDGE_TOOL_NAMES)


def _commit_candidate(
    agent: Any,
    state: Any,
    candidate: list[dict[str, Any]],
    *,
    owner_by_name: dict[str, str | None],
    base_enabled_toolsets: Any,
    added_tool_names: tuple[str, ...],
    admitted_toolsets: tuple[str, ...],
    denied_toolsets: tuple[str, ...],
    denied_tool_names: tuple[str, ...],
    predicted_toolsets: set[str] | None = None,
    initial: bool = False,
    retained_toolsets: tuple[str, ...] = (),
    exact_enabled_toolsets: tuple[str, ...] | None = None,
    surface_narrowed: bool | None = None,
    expected_revision: int | None = None,
) -> None:
    """Apply all coordinated fields, allowing the caller to roll back.

    ``retained_toolsets`` stay in ``enabled_toolsets`` without being part of
    the admitted surface (bridge-only toolsets; see _bridge_only_toolsets).
    """
    _validate_candidate_definitions(candidate)
    names = tuple(_tool_name(definition) for definition in candidate)
    represented = {
        owner_by_name[name]
        for name in names
        if owner_by_name.get(name) and owner_by_name[name] != RECOVERY_TOOLSET
    }
    base = list(base_enabled_toolsets or ()) if isinstance(base_enabled_toolsets, (list, tuple, set)) else []
    retained = set(retained_toolsets) if names_include_bridge(names) else set()
    enabled: list[str] = [
        toolset
        for toolset in base
        if toolset in represented
        or toolset in retained
        or (toolset == RECOVERY_TOOLSET and RECOVERY_TOOL_NAME in names)
    ]
    enabled.extend(sorted(represented - set(enabled)))
    if exact_enabled_toolsets is not None:
        # Exact restore: the captured list as-is, no owner reconstruction.
        enabled = list(exact_enabled_toolsets)
    with fenced_section() as may_commit:
        if not may_commit:
            raise StaleCommitError("stale router lifecycle; commit discarded")
        if expected_revision is not None and getattr(state, "surface_revision", 0) != expected_revision:
            raise SurfaceChangedError("surface changed while the candidate was prepared")
        _assign_candidate(
            agent,
            state,
            candidate,
            names=names,
            enabled=enabled,
            represented=represented,
            added_tool_names=added_tool_names,
            denied_toolsets=denied_toolsets,
            denied_tool_names=denied_tool_names,
            predicted_toolsets=predicted_toolsets,
            initial=initial,
            surface_narrowed=surface_narrowed,
        )


def _assign_candidate(
    agent: Any,
    state: Any,
    candidate: list[dict[str, Any]],
    *,
    names: tuple[str, ...],
    enabled: list[str],
    represented: set[Any],
    added_tool_names: tuple[str, ...],
    denied_toolsets: tuple[str, ...],
    denied_tool_names: tuple[str, ...],
    predicted_toolsets: set[str] | None,
    initial: bool,
    surface_narrowed: bool | None,
) -> None:
    state_snapshot = state.capability_snapshot()
    agent_snapshot = _copy_agent_fields(agent)
    try:
        agent.tools = candidate
        if hasattr(agent, "valid_tool_names"):
            agent.valid_tool_names = set(names)
        if hasattr(agent, "enabled_toolsets"):
            agent.enabled_toolsets = enabled
        state.installed_tool_names = set(names)
        state.admitted_toolsets = set(represented)
        state.denied_toolsets = set(denied_toolsets)
        state.denied_tool_names = set(denied_tool_names)
        state.active = True
        if predicted_toolsets is not None:
            state.predicted_toolsets = set(predicted_toolsets)
        if initial:
            state.initial_route_applied = True
            state.initial_toolsets = set(represented)
            state.active_toolsets = set(represented)
        else:
            new_active = set(state.active_toolsets) | set(represented)
            if added_tool_names:
                state.expansion_count += 1
            state.active_toolsets = new_active
        state.last_expansion_result = None
        if surface_narrowed is not None:
            state.surface_narrowed = surface_narrowed
    except Exception:
        _restore_agent_fields(agent, agent_snapshot)
        state.restore_capability_snapshot(state_snapshot)
        raise
    state.surface_revision = getattr(state, "surface_revision", 0) + 1
    state.committed_surface = _surface_fingerprint(agent)


def _surface_fingerprint(agent: Any) -> Any:
    """Exact, comparable snapshot of the coordinated host fields."""
    try:
        return (
            freeze_json(list(getattr(agent, "tools", None) or [])),
            tuple(getattr(agent, "enabled_toolsets", None) or ()),
        )
    except Exception:  # noqa: BLE001 - unfreezable surfaces never match
        return object()


def surface_is_router_committed(agent: Any, state: Any) -> bool:
    committed = getattr(state, "committed_surface", None)
    return committed is not None and _surface_fingerprint(agent) == committed


def _failure_result(
    admission: EnsureAdmissionResult,
    *,
    requested_toolsets: tuple[str, ...],
    denied_toolsets: tuple[str, ...],
    denied_tool_names: tuple[str, ...],
    installed_names: set[str],
    reason: str,
) -> ExpansionResult:
    requested = tuple(sorted(dict.fromkeys(requested_toolsets)))
    denied = tuple(sorted(dict.fromkeys(denied_toolsets)))
    denied_names = tuple(sorted(dict.fromkeys(denied_tool_names)))
    expanded = tuple(sorted(set(requested) - set(denied)))
    return ExpansionResult(
        ok=False,
        requested_toolsets=requested,
        expanded_toolsets=expanded,
        denied_toolsets=denied,
        denied_tool_names=denied_names,
        added_tool_names=(),
        installed_tool_names=tuple(sorted(installed_names)),
        reason=reason,
        retry_allowed=False,
    )


def _ordinary_definition_candidates(
    agent: Any,
    toolset: str,
    requested_names: set[str],
    envelope: HostAdmissionEnvelope | None,
    registry: Any,
) -> tuple[list[tuple[str, dict[str, Any], str | None]], set[str]]:
    """Return ordinary candidates and the names that were actually resolvable."""
    envelope_map = _envelope_definition_map(envelope)
    names = set(requested_names) if requested_names else set(_registry_tool_names(registry, toolset))
    names.update(_get_agent_local_tool_names(agent, toolset))
    current = _current_tool_names(agent)
    candidates: list[tuple[str, dict[str, Any], str | None]] = []
    available: set[str] = set()
    for name in sorted(names):
        if not name:
            continue
        if name in current:
            available.add(name)
            continue
        if name in envelope_map and envelope_map[name].owner_toolset_at_capture == toolset:
            candidates.append((name, thaw_json(envelope_map[name].payload), toolset))
            available.add(name)
            continue
        if toolset == "memory" and name in _get_agent_local_tool_names(agent, toolset):
            # Agent-local memory definitions may be supplied by the caller's
            # current surface; no global protected lookup is needed here.
            continue
        if registry is None:
            continue
        try:
            definitions = registry.get_definitions({name}, quiet=True) or []
        except Exception:
            definitions = []
        for definition in definitions:
            if _tool_name(definition) == name:
                candidates.append((name, definition, toolset))
                available.add(name)
                break
    return candidates, available


def select_and_commit_route(
    agent: Any,
    admission: EnsureAdmissionResult,
    *,
    predicted_toolsets: set[str],
    floor_toolsets: set[str],
    available_toolsets: set[str],
    caller: str = "route",
) -> ExpansionResult:
    """Select a READY envelope plus ordinary compatibility definitions."""
    state = _get_router_state(agent)
    prepared_revision = getattr(state, "surface_revision", 0)
    envelope = admission.envelope
    current = _current_tool_names(agent)
    requested = set(predicted_toolsets) | set(floor_toolsets)
    if admission.status != "READY" or envelope is None:
        if admission.status == "SAFE_NO_PRUNE" and envelope is not None:
            restored = thaw_envelope_definitions(envelope)
            owner_map = _envelope_owner_map(envelope)
            try:
                _commit_candidate(
                    agent,
                    state,
                    restored,
                    owner_by_name=owner_map,
                    base_enabled_toolsets=envelope.original_enabled_toolsets,
                    added_tool_names=(),
                    admitted_toolsets=(),
                    denied_toolsets=(),
                    denied_tool_names=(),
                    predicted_toolsets=set(),
                    initial=True,
                    exact_enabled_toolsets=tuple(envelope.original_enabled_toolsets or ()),
                    surface_narrowed=False,
                )
            except Exception as exc:
                return _failure_result(
                    admission,
                    requested_toolsets=tuple(requested),
                    denied_toolsets=tuple(requested),
                    denied_tool_names=(),
                    installed_names=current,
                    reason=f"RESTORE_FAILED:{type(exc).__name__}",
                )
        return ExpansionResult(
            ok=False,
            requested_toolsets=tuple(sorted(requested)),
            expanded_toolsets=(),
            denied_toolsets=tuple(sorted(requested)),
            denied_tool_names=(),
            added_tool_names=(),
            installed_tool_names=tuple(sorted(_current_tool_names(agent))),
            reason=admission.status,
            retry_allowed=False,
        )

    registry = _registry_object()
    owner_map = _envelope_owner_map(envelope)
    definition_map = _envelope_definition_map(envelope)
    protected_toolsets = admission.effective_policy.protected_toolsets if admission.effective_policy else frozenset()
    pins = admission.effective_policy.pinned_tool_names if admission.effective_policy else frozenset()
    selected: dict[str, dict[str, Any]] = {}
    selected_owner: dict[str, str | None] = {}
    denied_toolsets: set[str] = set()
    denied_names: set[str] = set()
    ordinary_candidates: list[tuple[str, dict[str, Any], str | None]] = []

    # Pinned definitions and the tool-search bridge are always selected from
    # the immutable envelope (the bridge fronts deferred tools; see
    # BRIDGE_TOOL_NAMES).
    for item in envelope.definitions:
        always_keep = item.name in ALWAYS_KEEP_TOOL_NAMES and (
            item.owner_toolset_at_capture not in protected_toolsets
        )
        if item.name in pins or item.name in BRIDGE_TOOL_NAMES or always_keep:
            selected[item.name] = thaw_json(item.payload)
            selected_owner[item.name] = item.owner_toolset_at_capture

    for toolset in sorted(requested):
        if toolset not in available_toolsets and toolset not in owner_map.values():
            denied_toolsets.add(toolset)
            continue
        if toolset in protected_toolsets:
            admitted = [
                item for item in envelope.definitions
                if item.owner_toolset_at_capture == toolset
            ]
            if not admitted:
                denied_toolsets.add(toolset)
                denied_names.update(_registry_tool_names(registry, toolset))
                continue
            for item in admitted:
                selected[item.name] = thaw_json(item.payload)
                selected_owner[item.name] = item.owner_toolset_at_capture
            continue
        candidates, available = _ordinary_definition_candidates(
            agent, toolset, set(), envelope, registry
        )
        ordinary_candidates.extend(candidates)
        if not available:
            denied_toolsets.add(toolset)

    if denied_toolsets and not (set(requested) - denied_toolsets):
        return _failure_result(
            admission,
            requested_toolsets=tuple(requested),
            denied_toolsets=tuple(denied_toolsets),
            denied_tool_names=tuple(denied_names),
            installed_names=current,
            reason="PROTECTED_DENIED",
        )

    # Toolsets owning definitions in the admitted envelope, with their
    # admitted tool names: what this agent would see with the router off.
    # This is the session capability catalog.
    catalog_toolsets: dict[str, set[str]] = {}
    for item in envelope.definitions:
        if item.owner_toolset_at_capture and item.name != RECOVERY_TOOL_NAME:
            catalog_toolsets.setdefault(item.owner_toolset_at_capture, set()).add(item.name)
    catalog_members = {
        toolset: _registry_tool_names(registry, toolset) for toolset in catalog_toolsets
    }
    catalog_bridge = any(item.name in BRIDGE_TOOL_NAMES for item in envelope.definitions)

    # A request_toolset supplied in the host surface stays in envelope order.
    # It is this plugin's own control, so it carries the session catalog
    # unless the host protected or pinned it (then it stays byte-exact).
    if RECOVERY_TOOL_NAME in definition_map:
        envelope_recovery = definition_map[RECOVERY_TOOL_NAME]
        recovery_payload = thaw_json(envelope_recovery.payload)
        if not _recovery_is_host_locked(envelope_recovery.owner_toolset_at_capture, protected_toolsets, pins):
            recovery_payload = with_capability_catalog(
                recovery_payload, catalog_toolsets, catalog_members, catalog_bridge
            )
        selected[RECOVERY_TOOL_NAME] = recovery_payload
        selected_owner[RECOVERY_TOOL_NAME] = envelope_recovery.owner_toolset_at_capture

    for name, definition, owner in ordinary_candidates:
        selected.setdefault(name, definition)
        selected_owner.setdefault(name, owner)

    # Keep already-installed requested ordinary definitions in the candidate;
    # they are not additions, but omission would falsely narrow the route.
    for item in envelope.definitions:
        if item.name in current and (
            item.owner_toolset_at_capture in requested or item.name in pins
        ):
            selected.setdefault(item.name, thaw_json(item.payload))
            selected_owner.setdefault(item.name, item.owner_toolset_at_capture)

    # A recovery control absent from the host envelope is an ordinary control
    # permitted only after a successful READY route, and always comes last.
    if RECOVERY_TOOL_NAME not in selected:
        recovery = _get_recovery_tool_definition()
        if recovery is not None and _tool_name(recovery) == RECOVERY_TOOL_NAME:
            # Agent-local copy carrying the session capability catalog.  Built
            # once here and carried unchanged through later expansions.
            recovery = with_capability_catalog(
                recovery, catalog_toolsets, catalog_members, catalog_bridge
            )
            selected[RECOVERY_TOOL_NAME] = recovery
            selected_owner[RECOVERY_TOOL_NAME] = RECOVERY_TOOLSET

    envelope_order = [
        item.name for item in envelope.definitions if item.name in selected
    ]
    ordinary_order = sorted(
        (name for name in selected if name not in envelope_order and name != RECOVERY_TOOL_NAME),
        key=lambda name: (selected_owner.get(name) or "", name),
    )
    ordered_names = envelope_order + ordinary_order
    if RECOVERY_TOOL_NAME in selected and RECOVERY_TOOL_NAME not in ordered_names:
        ordered_names.append(RECOVERY_TOOL_NAME)
    candidate = [selected[name] for name in ordered_names]
    added = tuple(name for name in ordered_names if name not in current)
    installed = set(ordered_names)
    admitted = tuple(sorted({selected_owner[name] for name in ordered_names if selected_owner.get(name)}))
    if not candidate and envelope.definitions:
        candidate = thaw_envelope_definitions(envelope)
        ordered_names = [_tool_name(item) for item in candidate]
        installed = set(ordered_names)
    try:
        _commit_candidate(
            agent,
            state,
            candidate,
            owner_by_name={**owner_map, **selected_owner},
            base_enabled_toolsets=envelope.original_enabled_toolsets,
            added_tool_names=added,
            admitted_toolsets=admitted,
            denied_toolsets=tuple(denied_toolsets),
            denied_tool_names=tuple(denied_names),
            predicted_toolsets=set(predicted_toolsets),
            initial=True,
            retained_toolsets=_bridge_only_toolsets(envelope, protected_toolsets),
            surface_narrowed=True,
            expected_revision=prepared_revision,
        )
    except Exception as exc:
        return _failure_result(
            admission,
            requested_toolsets=tuple(requested),
            denied_toolsets=tuple(denied_toolsets) or tuple(requested),
            denied_tool_names=tuple(denied_names),
            installed_names=current,
            reason=f"COMMIT_FAILED:{type(exc).__name__}",
        )
    return ExpansionResult(
        ok=not denied_toolsets and not denied_names,
        requested_toolsets=tuple(sorted(requested)),
        expanded_toolsets=tuple(sorted(set(requested) - denied_toolsets)),
        denied_toolsets=tuple(sorted(denied_toolsets)),
        denied_tool_names=tuple(sorted(denied_names)),
        added_tool_names=added,
        installed_tool_names=tuple(sorted(installed)),
        reason="READY",
        retry_allowed=bool(added),
    )


def expand_admitted_toolsets(
    agent: Any,
    admission: EnsureAdmissionResult,
    *,
    requested_toolsets: set[str],
    requested_tool_names: set[str],
    caller: str,
    session_id: str,
    claim_retry_for: str | None = None,
) -> ExpansionResult:
    """Apply one truthful expansion transaction for every caller.

    The candidate is prepared unlocked against a surface revision; if another
    commit landed meanwhile it is recomputed against the new surface (a
    bounded number of times) instead of overwriting it.
    """
    result: ExpansionResult | None = None
    for _attempt in range(_EXPANSION_ATTEMPTS):
        result = _expand_once(
            agent,
            admission,
            requested_toolsets=requested_toolsets,
            requested_tool_names=requested_tool_names,
            caller=caller,
            session_id=session_id,
            claim_retry_for=claim_retry_for,
        )
        if result.reason != "SURFACE_CHANGED":
            return result
    assert result is not None
    return result


def _expand_once(
    agent: Any,
    admission: EnsureAdmissionResult,
    *,
    requested_toolsets: set[str],
    requested_tool_names: set[str],
    caller: str,
    session_id: str,
    claim_retry_for: str | None = None,
) -> ExpansionResult:
    state = _get_router_state(agent)
    prepared_revision = getattr(state, "surface_revision", 0)
    current = _current_tool_names(agent)
    envelope = admission.envelope
    owner_map = _envelope_owner_map(envelope)
    definition_map = _envelope_definition_map(envelope)
    policy = admission.effective_policy
    protected_toolsets = policy.protected_toolsets if policy else frozenset()
    pinned_names = policy.pinned_tool_names if policy else frozenset()
    requested = set(requested_toolsets)
    requested_names = set(requested_tool_names)

    # Fail-closed statuses must not perform any owner, registry, alias, or
    # definition lookup.  Already-installed names remain reportable reality;
    # only additions are denied.
    if admission.status in {
        "CAPTURE_INVALID_NO_MUTATION",
        "SAFE_NO_PRUNE",
        "SESSION_MISMATCH",
        "NO_AUTHORITY_UNATTACHABLE",
        "NO_AUTHORITY_INVALID_POLICY",
    }:
        denied_names = {name for name in requested_names if name not in current}
        reason = admission.status
        if admission.status == "NO_AUTHORITY_UNATTACHABLE":
            reason = "NO_PERSISTENT_CONTAMINATION_GUARD"
        elif admission.status == "NO_AUTHORITY_INVALID_POLICY":
            reason = "INVALID_TRUSTED_POLICY"
        return _failure_result(
            admission,
            requested_toolsets=tuple(requested),
            denied_toolsets=tuple(requested),
            denied_tool_names=tuple(denied_names),
            installed_names=current,
            reason=reason,
        )

    registry = _registry_object()

    # Resolve explicit names only for classification.  Retrieval below is
    # permitted solely for ordinary names or exact envelope definitions.
    name_owner: dict[str, str | None] = dict(owner_map)
    for name in sorted(requested_names):
        name_owner.setdefault(name, _owner_for_name(name, agent, registry))
        owner = name_owner.get(name)
        if owner:
            requested.add(owner)

    denied_toolsets: set[str] = set()
    denied_names: set[str] = set()
    selected: dict[str, dict[str, Any]] = {}
    selected_owner: dict[str, str | None] = {}
    candidate_rows: list[tuple[str, dict[str, Any], str | None]] = []

    for toolset in sorted(requested):
        names_for_toolset = {
            name for name in requested_names if name_owner.get(name) == toolset
        }
        is_protected = toolset in protected_toolsets or bool(
            names_for_toolset & set(pinned_names)
        )
        if is_protected:
            admitted_items = [
                item for item in (envelope.definitions if envelope else ())
                if item.owner_toolset_at_capture == toolset
                and (not names_for_toolset or item.name in names_for_toolset)
            ]
            if not admitted_items and envelope is not None:
                admitted_items = [
                    item
                    for item in envelope.definitions
                    if item.name in requested_names and item.name in pinned_names
                ]
            if not admitted_items:
                denied_toolsets.add(toolset)
                denied_names.update(names_for_toolset or _registry_tool_names(registry, toolset))
                continue
            for item in admitted_items:
                selected[item.name] = thaw_json(item.payload)
                selected_owner[item.name] = item.owner_toolset_at_capture
            continue
        candidates, available = _ordinary_definition_candidates(
            agent, toolset, names_for_toolset, envelope, registry
        )
        candidate_rows.extend(candidates)
        if not available:
            denied_toolsets.add(toolset)
            denied_names.update(names_for_toolset)

    for name in requested_names:
        if name in current:
            continue
        owner = name_owner.get(name)
        if _is_protected_name(name, owner, admission):
            if envelope and name in definition_map:
                selected[name] = thaw_json(definition_map[name].payload)
                selected_owner[name] = definition_map[name].owner_toolset_at_capture
            else:
                denied_names.add(name)
                if owner:
                    denied_toolsets.add(owner)

    for name, definition, owner in candidate_rows:
        selected.setdefault(name, definition)
        selected_owner.setdefault(name, owner)

    # Never manufacture an item for a protected request.  Existing ordinary
    # definitions remain part of the candidate and are reported as installed.
    for name in current:
        if (
            name == RECOVERY_TOOL_NAME
            and name in definition_map
            and not _recovery_is_host_locked(
                definition_map[name].owner_toolset_at_capture, protected_toolsets, pinned_names
            )
        ):
            # Keep the session's catalog-bearing copy byte-stable.
            definition = next(
                (item for item in (getattr(agent, "tools", None) or []) if _tool_name(item) == name),
                None,
            )
            selected.setdefault(name, definition if definition is not None else thaw_json(definition_map[name].payload))
            selected_owner.setdefault(name, owner_map.get(name))
        elif name in definition_map:
            selected.setdefault(name, thaw_json(definition_map[name].payload))
            selected_owner.setdefault(name, owner_map.get(name))
        else:
            definition = next(
                (item for item in (getattr(agent, "tools", None) or []) if _tool_name(item) == name),
                None,
            )
            if definition is not None:
                selected.setdefault(name, definition)
                selected_owner.setdefault(name, name_owner.get(name))

    envelope_order = [item.name for item in (envelope.definitions if envelope else ()) if item.name in selected]
    ordinary_order = sorted(
        (name for name in selected if name not in envelope_order),
        key=lambda name: (selected_owner.get(name) or "", name),
    )
    ordered_names = envelope_order + ordinary_order
    candidate = [selected[name] for name in ordered_names]
    added = tuple(name for name in ordered_names if name not in current)
    if not added:
        ok = not denied_toolsets and not denied_names
        result = ExpansionResult(
            ok=ok,
            requested_toolsets=tuple(sorted(requested)),
            expanded_toolsets=tuple(sorted(set(requested) - denied_toolsets)),
            denied_toolsets=tuple(sorted(denied_toolsets)),
            denied_tool_names=tuple(sorted(denied_names)),
            added_tool_names=(),
            installed_tool_names=tuple(sorted(current)),
            reason="NO_MUTATION" if ok else ("PROTECTED_DENIED" if denied_toolsets else "UNRESOLVED"),
            retry_allowed=False,
        )
        return result

    # Snapshot, contamination marker, commit and rollback all happen under the
    # hook's fence: a stale callback neither commits nor rolls a later
    # session back to its own snapshot.
    with fenced_section() as may_commit:
        surface_changed = getattr(state, "surface_revision", 0) != prepared_revision
        if not may_commit or surface_changed:
            return ExpansionResult(
                ok=False,
                requested_toolsets=tuple(sorted(requested)),
                expanded_toolsets=(),
                denied_toolsets=tuple(sorted(requested)),
                denied_tool_names=tuple(sorted(denied_names)),
                added_tool_names=(),
                installed_tool_names=tuple(sorted(current)),
                reason="SURFACE_CHANGED" if may_commit else "STALE_LIFECYCLE",
                retry_allowed=False,
            )
        # A no-authority append must persist the immutable contamination marker
        # before the first assignment.  Capture the complete pre-call state and
        # exact lifecycle slots for rollback.
        original_state = state.capability_snapshot()
        original_slots = state.admission_slots()
        agent_snapshot = _copy_agent_fields(agent)
        marker = state.no_authority_contamination(session_id)
        try:
            if admission.status in {"NO_AUTHORITY", "NO_AUTHORITY_CONTAMINATED"}:
                if marker is None:
                    first_toolsets = tuple(
                        sorted(
                            {
                                selected_owner[name]
                                for name in added
                                if selected_owner.get(name)
                            }
                        )
                    )
                    marker = state.mark_no_authority_contaminated(
                        session_id=session_id,
                        first_commit_caller=caller,
                        first_added_tool_names=tuple(sorted(added)),
                        first_added_toolsets=first_toolsets,
                    )
                    if state.no_authority_contamination(session_id) is not marker:
                        raise RuntimeError("CONTAMINATION_READBACK_FAILED")
                elif state.no_authority_contamination(session_id) is not marker:
                    raise RuntimeError("CONTAMINATION_IDENTITY_FAILED")
            _commit_candidate(
                agent,
                state,
                candidate,
                owner_by_name={**owner_map, **selected_owner},
                base_enabled_toolsets=getattr(agent, "enabled_toolsets", ()),
                added_tool_names=added,
                admitted_toolsets=tuple(sorted({selected_owner[name] for name in ordered_names if selected_owner.get(name)})),
                denied_toolsets=tuple(sorted(denied_toolsets)),
                denied_tool_names=tuple(sorted(denied_names)),
                predicted_toolsets=set(state.active_toolsets) | {
                    selected_owner[name] for name in added if selected_owner.get(name)
                },
                initial=False,
                retained_toolsets=_bridge_only_toolsets(envelope, protected_toolsets),
            )
        except Exception as exc:
            _restore_agent_fields(agent, agent_snapshot)
            state.restore_capability_snapshot(original_state)
            state.restore_admission_slots(original_slots)
            if state.admission_slots() != original_slots:
                raise RuntimeError("TRANSACTION_ROLLBACK_FAILED") from exc
            return ExpansionResult(
                ok=False,
                requested_toolsets=tuple(sorted(requested)),
                expanded_toolsets=(),
                denied_toolsets=tuple(sorted(requested)),
                denied_tool_names=tuple(sorted(denied_names)),
                added_tool_names=(),
                installed_tool_names=tuple(sorted(current)),
                reason=f"TRANSACTION_ROLLED_BACK:{type(exc).__name__}",
                retry_allowed=False,
            )
        if claim_retry_for and claim_retry_for in (getattr(agent, "valid_tool_names", None) or set()):
            # Claimed in the same critical section as the commit, so two
            # overlapping automatic recoveries cannot both succeed.
            state._retry_pending = True

    return ExpansionResult(
        ok=not denied_toolsets and not denied_names,
        requested_toolsets=tuple(sorted(requested)),
        expanded_toolsets=tuple(sorted(set(requested) - denied_toolsets)),
        denied_toolsets=tuple(sorted(denied_toolsets)),
        denied_tool_names=tuple(sorted(denied_names)),
        added_tool_names=added,
        installed_tool_names=tuple(sorted(set(ordered_names))),
        reason=(
            admission.status
            if admission.status in {"READY", "NO_AUTHORITY", "NO_AUTHORITY_CONTAMINATED"}
            else "EXPANDED"
        ),
        retry_allowed=True,
    )


def restore_admitted_envelope(
    agent: Any,
    admission: EnsureAdmissionResult,
) -> bool:
    """Restore only the exact captured envelope, without registry retrieval."""
    envelope = admission.envelope
    if envelope is None:
        return False
    state = _get_router_state(agent)
    definitions = thaw_envelope_definitions(envelope)
    names = tuple(_tool_name(definition) for definition in definitions)
    with fenced_section() as may_commit:
        if not may_commit:
            return False
        return _restore_locked(agent, state, envelope, definitions, names)


def _restore_locked(agent: Any, state: Any, envelope: Any, definitions: Any, names: tuple[str, ...]) -> bool:
    try:
        _commit_candidate(
            agent,
            state,
            definitions,
            owner_by_name=_envelope_owner_map(envelope),
            base_enabled_toolsets=envelope.original_enabled_toolsets,
            added_tool_names=(),
            admitted_toolsets=tuple(
                sorted({item.owner_toolset_at_capture for item in envelope.definitions if item.owner_toolset_at_capture})
            ),
            denied_toolsets=(),
            denied_tool_names=(),
            predicted_toolsets=set(),
            initial=True,
            exact_enabled_toolsets=tuple(envelope.original_enabled_toolsets or ()),
            surface_narrowed=False,
        )
    except Exception:
        return False
    state.installed_tool_names = set(names)
    state._retry_pending = False
    state._fallback_triggered = False
    return True
