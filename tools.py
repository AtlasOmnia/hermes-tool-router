"""Tool resolution, filtering, expansion, and recovery schema."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

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
    from .config import PLUGIN_NAME, _get_profile_config, _load_config
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
    from config import PLUGIN_NAME, _get_profile_config, _load_config
    from policy import TOOLSET_DESCRIPTIONS
    from state import _get_router_state

logger = logging.getLogger(__name__)

RECOVERY_TOOL_NAME = "request_toolset"
RECOVERY_TOOLSET = "router_recovery"
RECOVERY_TOOLSET_CHOICES = sorted(TOOLSET_DESCRIPTIONS)


def build_recovery_tool_schema(available_toolsets: Set[str]) -> Dict[str, Any]:
    """Build the compact recovery schema from the live registry surface."""
    return {
        "description": "Load missing Hermes toolsets before continuing the task.",
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
                        "One or more Hermes toolset names to add for this session, such as "
                        "file, terminal, web, browser, computer_use, vision, image_gen, "
                        "video, video_gen, memory, skills, session_search, cronjob, "
                        "delegation, tts, or code_execution. Names are validated against "
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
def _filter_tool_definitions(
    tool_defs: List[Dict[str, Any]],
    allowed_tool_names: Set[str],
) -> List[Dict[str, Any]]:
    """Filter a list of tool definitions to only those in allowed_tool_names."""
    return [
        td for td in tool_defs
        if td.get("function", {}).get("name") in allowed_tool_names
    ]
def _get_tool_definitions_for_names(tool_names: Set[str]) -> List[Dict[str, Any]]:
    """Load tool definitions directly from the registry."""
    if not tool_names:
        return []
    try:
        from tools.registry import registry
        return registry.get_definitions(set(tool_names), quiet=True)
    except Exception as exc:
        logger.warning("%s: failed to load tool definitions from registry: %s", PLUGIN_NAME, exc)
        return []
def _get_recovery_tool_definition() -> Optional[Dict[str, Any]]:
    """Return the OpenAI-format recovery tool definition, if registered."""
    try:
        from tools.registry import registry
        defs = registry.get_definitions({RECOVERY_TOOL_NAME}, quiet=True)
        return defs[0] if defs else None
    except Exception as exc:
        logger.debug("%s: recovery tool definition unavailable: %s", PLUGIN_NAME, exc)
        return None
def _ensure_recovery_tool(agent: Any) -> None:
    """Keep a tiny request_toolset escape hatch available after narrowing."""
    recovery_def = _get_recovery_tool_definition()
    if recovery_def is None:
        return

    tools = list(agent.tools or [])
    if not any(t.get("function", {}).get("name") == RECOVERY_TOOL_NAME for t in tools):
        tools.append(recovery_def)
        agent.tools = tools

    if hasattr(agent, "valid_tool_names"):
        agent.valid_tool_names = set(getattr(agent, "valid_tool_names", set()) or set())
        agent.valid_tool_names.add(RECOVERY_TOOL_NAME)

    if hasattr(agent, "enabled_toolsets") and agent.enabled_toolsets is not None:
        if RECOVERY_TOOLSET not in agent.enabled_toolsets:
            agent.enabled_toolsets = list(agent.enabled_toolsets) + [RECOVERY_TOOLSET]
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


def _detect_missing_toolset(
    tool_name: str,
    agent: Any = None,
) -> Optional[str]:
    """Check if a tool was called that's outside the current predicted set.

    Returns the toolset name if the tool is registered but not in our
    predicted set, or None if it's already covered.
    """
    try:
        from tools.registry import registry
        return _infer_toolset_from_tool(tool_name, registry, agent)
    except Exception:
        return None
def _get_all_tool_names() -> Set[str]:
    """Get all registered tool names through public toolset APIs."""
    try:
        from tools.registry import registry
        names: Set[str] = set()
        for toolset in registry.get_registered_toolset_names():
            names.update(registry.get_tool_names_for_toolset(toolset) or [])
        return names
    except Exception:
        return set()
def _cache_full_toolset(agent: Any) -> None:
    """Cache the full tool definitions from the agent."""
    if not (hasattr(agent, "tools") and agent.tools):
        return

    current_defs = list(agent.tools)
    current_names = {
        t.get("function", {}).get("name", "")
        for t in current_defs
    }

    # Do not replace a previously captured full surface with a narrowed
    # per-turn surface. This matters after a no-tool turn where only
    # request_toolset remains available.
    if current_names and current_names <= {RECOVERY_TOOL_NAME}:
        return
    state = _get_router_state(agent)
    if (
        state._full_tool_defs is not None
        and len(current_defs) < len(state._full_tool_defs)
    ):
        return

    state._full_tool_defs = current_defs
    state._full_tool_names = current_names
def _apply_predicted_tools(
    agent: Any,
    predicted_toolsets: Set[str],
    available_toolsets: Set[str],
) -> None:
    """Apply the predicted toolset filter to agent.tools.

    Mutates agent.tools and agent.enabled_toolsets in-place.
    """
    # Use configured floor toolsets plus prediction; never re-add the full tool list.
    cfg = _get_profile_config(_load_config())
    floor_toolsets = set(cfg.get("floor_toolsets", ["terminal", "file", "web"]))
    allowed_toolsets = (set(predicted_toolsets) | floor_toolsets) & set(available_toolsets)

    if not allowed_toolsets:
        if not predicted_toolsets and not floor_toolsets:
            agent.tools = []
            if hasattr(agent, "enabled_toolsets") and agent.enabled_toolsets is not None:
                agent.enabled_toolsets = []
            if hasattr(agent, "valid_tool_names"):
                agent.valid_tool_names = set()
            _ensure_recovery_tool(agent)
            logger.debug("%s: applied no-tool route", PLUGIN_NAME)
            return
        raise RuntimeError("no allowed toolsets after applying prediction and floor")

    # Record original tool count for logging
    original_count = len(agent.tools) if hasattr(agent, "tools") and agent.tools else 0

    # Resolve allowed toolsets to tool names
    allowed_tool_names = _resolve_toolset_to_tool_names(allowed_toolsets, agent)
    if not allowed_tool_names:
        raise RuntimeError(
            f"no tool names resolved for allowed_toolsets={sorted(allowed_toolsets)}"
        )

    # Filter from cached full definitions if available, otherwise current definitions
    state = _get_router_state(agent)
    source_tool_defs = (
        state._full_tool_defs
        if state._full_tool_defs is not None
        else list(agent.tools or [])
    )
    agent.tools = _filter_tool_definitions(source_tool_defs, allowed_tool_names)

    new_count = len(agent.tools) if hasattr(agent, "tools") and agent.tools else 0
    if new_count == 0:
        agent.tools = _get_tool_definitions_for_names(allowed_tool_names)
        new_count = len(agent.tools) if hasattr(agent, "tools") and agent.tools else 0

    if new_count == 0:
        raise RuntimeError(
            f"filter removed all tools for allowed_toolsets={sorted(allowed_toolsets)}"
        )

    if len(allowed_tool_names) > new_count:
        missing_names = allowed_tool_names - {
            t.get("function", {}).get("name", "") for t in agent.tools
        }
        logger.warning(
            "%s: %d resolved tools were not present in cached definitions: %s",
            PLUGIN_NAME,
            len(missing_names),
            sorted(missing_names)[:10],
        )

    # Update agent.enabled_toolsets so codec/transparency is preserved
    if hasattr(agent, "enabled_toolsets") and agent.enabled_toolsets is not None:
        filtered = [ts for ts in agent.enabled_toolsets if ts in allowed_toolsets]
        for ts in sorted(allowed_toolsets):
            if ts not in filtered:
                filtered.append(ts)
        agent.enabled_toolsets = filtered

    # Update valid_tool_names for agent consistency
    if hasattr(agent, "valid_tool_names"):
        agent.valid_tool_names = {
            t.get("function", {}).get("name", "") for t in agent.tools
        }

    _ensure_recovery_tool(agent)
    new_count = len(agent.tools) if hasattr(agent, "tools") and agent.tools else 0

    # Log the trim result
    logger.debug(
        "%s: trimmed tools from %d to %d (allowed_toolsets=%s, model=%s)",
        PLUGIN_NAME,
        original_count,
        new_count,
        sorted(allowed_toolsets),
        _get_router_state(agent).router_model,
    )
def _restore_full_tools(agent: Any) -> None:
    """Restore the full tool definitions cached before reduction."""
    state = _get_router_state(agent)
    if state._full_tool_defs is not None:
        agent.tools = list(state._full_tool_defs)
        if hasattr(agent, "valid_tool_names"):
            agent.valid_tool_names = set(state._full_tool_names)
def _expand_toolset(agent: Any, toolset_name: str) -> None:
    """Expand the agent's tool set to include a specific toolset."""
    try:
        from tools.registry import registry

        # Get current predicted toolsets
        state = _get_router_state(agent)
        state.expand_surface({toolset_name})
        if state.predicted_toolsets is not None:
            state.predicted_toolsets.add(toolset_name)
        else:
            state.predicted_toolsets = {toolset_name}

        # Get tool names for this toolset
        ts_tools = set(registry.get_tool_names_for_toolset(toolset_name))
        ts_tools.update(_get_agent_local_tool_names(agent, toolset_name))

        # If we have the full cached tool defs, filter from there
        if state._full_tool_defs is not None:
            # Get currently loaded tool names
            current_names = set()
            if hasattr(agent, "tools") and agent.tools:
                current_names = {
                    t.get("function", {}).get("name", "")
                    for t in agent.tools
                }

            # Add the missing tools
            expanded_defs = list(agent.tools) if hasattr(agent, "tools") and agent.tools else []
            for td in state._full_tool_defs:
                tn = td.get("function", {}).get("name", "")
                if tn in ts_tools and tn not in current_names:
                    expanded_defs.append(td)
                    current_names.add(tn)

            missing = ts_tools - current_names
            for td in _get_tool_definitions_for_names(missing):
                tn = td.get("function", {}).get("name", "")
                if tn and tn not in current_names:
                    expanded_defs.append(td)
                    current_names.add(tn)

            agent.tools = expanded_defs
            agent.valid_tool_names = current_names
        else:
            # No cache — fall back to rebuilding from registry
            all_tools = _get_all_tool_names()
            if hasattr(agent, "valid_tool_names"):
                agent.valid_tool_names.update(ts_tools & all_tools)

        # Update enabled_toolsets
        if hasattr(agent, "enabled_toolsets") and agent.enabled_toolsets is not None:
            if toolset_name not in agent.enabled_toolsets:
                agent.enabled_toolsets = list(agent.enabled_toolsets) + [toolset_name]

    except Exception as exc:
        logger.warning(
            "%s: failed to expand toolset '%s': %s",
            PLUGIN_NAME, toolset_name, exc,
        )
        _handle_full_fallback(agent)
def _handle_full_fallback(agent: Any) -> None:
    """Handle a full fallback — restore all tools from cache."""
    _restore_full_tools(agent)
    state = _get_router_state(agent)
    state.active = False
    state.predicted_toolsets = None
    state._fallback_triggered = True
    state._retry_pending = True
    total = len(agent.tools) if hasattr(agent, "tools") and agent.tools else 0
    logger.info(
        "%s: full fallback — restored all %d tools",
        PLUGIN_NAME, total,
    )

# ---------------------------------------------------------------------------
# Admission-aware selection and transaction primitives.  The legacy helpers
# above remain available to older Hermes versions; all new callers use these
# functions so that protected definitions never come from the live registry.


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
) -> None:
    """Apply all coordinated fields, allowing the caller to roll back."""
    _validate_candidate_definitions(candidate)
    names = tuple(_tool_name(definition) for definition in candidate)
    represented = {
        owner_by_name[name]
        for name in names
        if owner_by_name.get(name) and owner_by_name[name] != RECOVERY_TOOLSET
    }
    base = list(base_enabled_toolsets or ()) if isinstance(base_enabled_toolsets, (list, tuple, set)) else []
    enabled: list[str] = [
        toolset
        for toolset in base
        if toolset in represented
        or (toolset == RECOVERY_TOOLSET and RECOVERY_TOOL_NAME in names)
    ]
    enabled.extend(sorted(represented - set(enabled)))
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
    except Exception:
        _restore_agent_fields(agent, agent_snapshot)
        state.restore_capability_snapshot(state_snapshot)
        raise


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

    # Pinned definitions are always selected from the immutable envelope.
    for item in envelope.definitions:
        if item.name in pins:
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

    # A request_toolset supplied in the host surface stays in envelope order.
    if RECOVERY_TOOL_NAME in definition_map:
        selected[RECOVERY_TOOL_NAME] = thaw_json(definition_map[RECOVERY_TOOL_NAME].payload)
        selected_owner[RECOVERY_TOOL_NAME] = definition_map[RECOVERY_TOOL_NAME].owner_toolset_at_capture

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
) -> ExpansionResult:
    """Apply one truthful expansion transaction for every caller."""
    state = _get_router_state(agent)
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
        if name in definition_map:
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

    return ExpansionResult(
        ok=not denied_toolsets and not denied_names,
        requested_toolsets=tuple(sorted(requested)),
        expanded_toolsets=tuple(sorted(set(requested) - denied_toolsets)),
        denied_toolsets=tuple(sorted(denied_toolsets)),
        denied_tool_names=tuple(sorted(denied_names)),
        added_tool_names=added,
        installed_tool_names=tuple(sorted(set(ordered_names))),
        reason="NO_AUTHORITY" if admission.status == "NO_AUTHORITY" else "NO_AUTHORITY_CONTAMINATED",
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
        )
    except Exception:
        return False
    state.installed_tool_names = set(names)
    state._retry_pending = False
    state._fallback_triggered = False
    return True
