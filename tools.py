"""Tool resolution, filtering, expansion, and recovery schema."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

try:
    from .config import PLUGIN_NAME, _get_profile_config, _load_config
    from .policy import TOOLSET_DESCRIPTIONS
    from .state import _get_router_state
except ImportError:  # pragma: no cover - direct loader fallback
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
def _resolve_toolset_to_tool_names(toolsets: Set[str]) -> Set[str]:
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
def _infer_toolset_from_tool(tool_name: str, registry) -> Optional[str]:
    """Determine which toolset a tool belongs to."""
    # First try registry lookup
    try:
        entry = registry.get_entry(tool_name)
        if entry and hasattr(entry, "toolset"):
            return entry.toolset
    except Exception:
        pass
    return None
def _detect_missing_toolset(
    tool_name: str,
) -> Optional[str]:
    """Check if a tool was called that's outside the current predicted set.

    Returns the toolset name if the tool is registered but not in our
    predicted set, or None if it's already covered.
    """
    try:
        from tools.registry import registry
        return _infer_toolset_from_tool(tool_name, registry)
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
    allowed_tool_names = _resolve_toolset_to_tool_names(allowed_toolsets)
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
