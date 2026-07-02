"""Hermes Tool Router Plugin.

Thin plugin entrypoint: registration, hook handlers, and recovery tool handler.
The implementation lives in focused sibling modules.
"""

from __future__ import annotations

import difflib
import json
import logging
from typing import Any, Dict, Optional, Set

try:
    from .config import (
        CONFIG_FILE,
        DEFAULT_ROUTER_MODEL,
        DEFAULT_ROUTER_PROVIDER,
        PLUGIN_NAME,
        _get_profile_config,
        _get_router_model,
        _get_router_provider,
        _is_router_active,
        _load_config,
    )
    from .policy import (
        ACTION_HINT_RE,
        PLAIN_ANSWER_RE,
        ROUTER_SYSTEM_PROMPT,
        TOOLSET_DESCRIPTIONS,
        TOOLSET_INTENT_RULES,
        _build_toolset_description_block,
        _extract_confidence,
        _get_available_toolsets,
        _get_router_client,
        _predict_toolsets_by_rules,
        _predict_toolsets_via_llm,
    )
    from .state import (
        ROUTER_STATE_ATTR,
        RouterState,
        _get_agent_from_stack,
        _get_agent_ref,
        _get_router_state,
        _store_agent_ref,
    )
    from .tools import (
        RECOVERY_TOOL_NAME,
        RECOVERY_TOOL_SCHEMA,
        RECOVERY_TOOLSET,
        RECOVERY_TOOLSET_CHOICES,
        _apply_predicted_tools,
        _cache_full_toolset,
        _detect_missing_toolset,
        _ensure_recovery_tool,
        _expand_toolset,
        _filter_tool_definitions,
        _get_all_tool_names,
        _get_recovery_tool_definition,
        _get_tool_definitions_for_names,
        _handle_full_fallback,
        _infer_toolset_from_tool,
        _resolve_toolset_to_tool_names,
        _restore_full_tools,
    )
except ImportError:  # pragma: no cover - direct loader fallback
    from config import (
        CONFIG_FILE,
        DEFAULT_ROUTER_MODEL,
        DEFAULT_ROUTER_PROVIDER,
        PLUGIN_NAME,
        _get_profile_config,
        _get_router_model,
        _get_router_provider,
        _is_router_active,
        _load_config,
    )
    from policy import (
        ACTION_HINT_RE,
        PLAIN_ANSWER_RE,
        ROUTER_SYSTEM_PROMPT,
        TOOLSET_DESCRIPTIONS,
        TOOLSET_INTENT_RULES,
        _build_toolset_description_block,
        _extract_confidence,
        _get_available_toolsets,
        _get_router_client,
        _predict_toolsets_by_rules,
        _predict_toolsets_via_llm,
    )
    from state import (
        ROUTER_STATE_ATTR,
        RouterState,
        _get_agent_from_stack,
        _get_agent_ref,
        _get_router_state,
        _store_agent_ref,
    )
    from tools import (
        RECOVERY_TOOL_NAME,
        RECOVERY_TOOL_SCHEMA,
        RECOVERY_TOOLSET,
        RECOVERY_TOOLSET_CHOICES,
        _apply_predicted_tools,
        _cache_full_toolset,
        _detect_missing_toolset,
        _ensure_recovery_tool,
        _expand_toolset,
        _filter_tool_definitions,
        _get_all_tool_names,
        _get_recovery_tool_definition,
        _get_tool_definitions_for_names,
        _handle_full_fallback,
        _infer_toolset_from_tool,
        _resolve_toolset_to_tool_names,
        _restore_full_tools,
    )

logger = logging.getLogger(__name__)


def _mark_turn_routed(agent: Any, state: RouterState, turn_id: str, source: str) -> None:
    """Record that this agent/turn already passed through the router."""
    if not turn_id:
        return
    state.routed_turn_id = turn_id
    state.routed_source = source
    try:
        setattr(agent, "_token_router_early_hook_turn_id", turn_id)
    except Exception:
        logger.debug("%s: failed to set compatibility turn marker", PLUGIN_NAME, exc_info=True)


def _was_turn_routed(agent: Any, state: RouterState, turn_id: str) -> bool:
    """Return True when the early hook already handled this turn."""
    if not turn_id:
        return False
    if getattr(state, "routed_turn_id", None) == turn_id:
        return True
    return getattr(agent, "_token_router_early_hook_turn_id", None) == turn_id


def _route_tool_surface(source: str, agent: Any = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Predict relevant toolsets and narrow the live agent tool surface."""
    cfg = _load_config()
    if not _is_router_active(cfg):
        if agent is None:
            agent = _get_agent_ref()
        if agent is not None:
            _get_router_state(agent).reset()
        return None

    user_message = kwargs.get("user_message", "")
    if not user_message:
        return None

    if agent is not None:
        _store_agent_ref(agent)
    else:
        agent = _get_agent_ref()
        if agent is None:
            agent = _get_agent_from_stack()
            if agent is not None:
                _store_agent_ref(agent)
                logger.info("%s: acquired agent reference via stack introspection", PLUGIN_NAME)
            else:
                logger.warning("%s: no agent reference; full set fallback", PLUGIN_NAME)
                return None

    turn_id = kwargs.get("turn_id") or getattr(agent, "_current_turn_id", "") or ""
    state = _get_router_state(agent)

    if source != "pre_turn_context_build" and _was_turn_routed(agent, state, turn_id):
        logger.debug("%s: skipping duplicate late pre_llm_call for early-routed turn", PLUGIN_NAME)
        return None

    def complete() -> None:
        if source == "pre_turn_context_build":
            _mark_turn_routed(agent, state, turn_id, source)
        return None

    # Get profile config
    profile_cfg = _get_profile_config(cfg)
    if not profile_cfg.get("enabled", False):
        return complete()

    decline_chars = profile_cfg.get("long_message_decline_chars", 2000)
    short_bypass_chars = profile_cfg.get("short_message_bypass_chars", 0)
    floor_toolsets = set(profile_cfg.get("floor_toolsets", ["terminal", "file", "web"]))
    confidence_threshold = float(profile_cfg.get("confidence_threshold", 0.0))
    router_model = _get_router_model(profile_cfg)
    router_provider = _get_router_provider(profile_cfg)
    state.router_model = router_model

    profile_name = profile_cfg.get("_profile_name", "unknown")
    logger.debug(
        "%s: predicting for profile=%s using model=%s, message_len=%d",
        PLUGIN_NAME,
        profile_name,
        router_model,
        len(user_message),
    )

    # Cache the current full tool definitions before we modify anything
    _cache_full_toolset(agent)

    # Bypass for long/complex messages; don't risk missing a tool
    if len(user_message) > decline_chars:
        logger.debug(
            "%s: bypass reduction (message too long: %d chars > %d)",
            PLUGIN_NAME, len(user_message), decline_chars,
        )
        _restore_full_tools(agent)
        state.active = False
        state.predicted_toolsets = None
        return complete()

    # Bypass for very short or ambiguous messages; only if threshold > 0
    if short_bypass_chars > 0 and len(user_message.strip()) < short_bypass_chars:
        logger.debug(
            "%s: bypass reduction (message too short: %d chars < %d)",
            PLUGIN_NAME, len(user_message.strip()), short_bypass_chars,
        )
        _restore_full_tools(agent)
        state.active = False
        state.predicted_toolsets = None
        return complete()

    # Get available toolsets
    try:
        available = _get_available_toolsets()
    except Exception:
        logger.warning(
            "%s: could not list toolsets; full set fallback",
            PLUGIN_NAME,
        )
        _restore_full_tools(agent)
        state.active = False
        state.predicted_toolsets = None
        return complete()

    if not available:
        return complete()

    predicted: Optional[Set[str]]
    deterministic_enabled = bool(profile_cfg.get("deterministic_rules_enabled", True))
    rule_reason = "disabled"
    if deterministic_enabled:
        predicted, rule_reason = _predict_toolsets_by_rules(user_message, available)
    else:
        predicted = None

    if predicted is not None:
        logger.info(
            "%s: deterministic route reason=%s predicted_toolsets=%s",
            PLUGIN_NAME,
            rule_reason,
            sorted(predicted),
        )
    else:
        # Predict toolsets via router model
        try:
            predicted = _predict_toolsets_via_llm(
                user_message,
                available,
                router_model,
                confidence_threshold,
                router_provider,
            )
        except Exception as exc:
            logger.warning(
                "%s: prediction failed: %s; full set fallback",
                PLUGIN_NAME, exc,
            )
            _restore_full_tools(agent)
            state.active = False
            state.predicted_toolsets = None
            return complete()

    if predicted is None:
        # Bypass; keep full toolsets
        _restore_full_tools(agent)
        state.active = False
        state.predicted_toolsets = None
        logger.debug("%s: bypass; keeping full toolset", PLUGIN_NAME)
        logger.debug(
            "%s: predicted toolsets bypassed for profile=%s",
            PLUGIN_NAME,
            profile_name,
        )
        return complete()

    # Log prediction
    logger.debug(
        "%s: predicted_toolsets=%s",
        PLUGIN_NAME,
        sorted(predicted),
    )

    # Filter agent.tools to only the predicted toolsets
    try:
        _apply_predicted_tools(agent, predicted, available)
    except Exception as exc:
        logger.warning(
            "%s: failed to apply predicted tools: %s; full set fallback",
            PLUGIN_NAME, exc,
        )
        _restore_full_tools(agent)
        state.active = False
        state.predicted_toolsets = None
        return complete()

    # Store agent-scoped state for post_tool_call/request_toolset
    state.active = True
    state.predicted_toolsets = predicted
    state._fallback_triggered = False
    state._retry_pending = False

    total_tools = len(agent.tools) if hasattr(agent, "tools") and agent.tools else 0
    logger.info(
        "%s: narrowed to %d toolsets: %s (%d tools)",
        PLUGIN_NAME,
        len(predicted),
        sorted(predicted),
        total_tools,
    )
    return complete()


def request_toolset_handler(args: Dict[str, Any], **kwargs: Any) -> str:
    """Expand the live agent with one requested toolset."""
    requested_toolset = str(args.get("toolset") or args.get("toolset_name") or "").strip().lower()
    requested_tool = str(args.get("tool_name") or "").strip()
    reason = str(args.get("reason") or "").strip()[:200]

    try:
        from tools.registry import registry
        available = set(registry.get_registered_toolset_names())
        if requested_tool and not requested_toolset:
            requested_toolset = _infer_toolset_from_tool(requested_tool, registry) or ""
        alias_target = registry.get_toolset_alias_target(requested_toolset) if requested_toolset else None
        if alias_target:
            requested_toolset = alias_target
    except Exception:
        available = set()

    if not requested_toolset:
        return json.dumps({
            "ok": False,
            "error": "toolset or resolvable tool_name is required",
            "requested_toolset": requested_toolset,
            "requested_tool": requested_tool,
        })

    if available and requested_toolset not in available:
        suggestions = difflib.get_close_matches(requested_toolset, sorted(available), n=5)
        return json.dumps({
            "ok": False,
            "error": f"unknown toolset: {requested_toolset}",
            "requested_toolset": requested_toolset,
            "requested_tool": requested_tool,
            "suggestions": suggestions,
            "available_toolsets": sorted(available),
        })

    agent = _get_agent_ref() or _get_agent_from_stack()
    if agent is None:
        return json.dumps({
            "ok": False,
            "error": "no live agent reference",
            "requested_toolset": requested_toolset,
            "requested_tool": requested_tool,
        })

    state = _get_router_state(agent)
    if state._full_tool_defs is None:
        _cache_full_toolset(agent)
    _expand_toolset(agent, requested_toolset)
    _ensure_recovery_tool(agent)
    state.active = True
    state._retry_pending = False

    enabled_tools = sorted(getattr(agent, "valid_tool_names", set()) or set())
    return json.dumps({
        "ok": True,
        "toolset": requested_toolset,
        "requested_tool": requested_tool,
        "reason": reason,
        "enabled_tools": enabled_tools,
    })


def register(ctx) -> None:
    """Register the hermes-token-router plugin.

    Hooks registered:
      - pre_turn_context_build : primary route before prompt/tool assembly
      - pre_llm_call           : late fallback hook for older Hermes builds
      - post_tool_call         : best-effort expansion after executed tool calls
    """

    try:
        ctx.register_hook("pre_turn_context_build", pre_turn_context_build)
    except Exception as exc:
        logger.warning(
            "%s: pre_turn_context_build hook unavailable; using pre_llm_call fallback: %s",
            PLUGIN_NAME,
            exc,
        )

    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("post_tool_call", post_tool_call)

    try:
        from tools.registry import registry
        registry.register(
            name=RECOVERY_TOOL_NAME,
            toolset=RECOVERY_TOOLSET,
            schema=RECOVERY_TOOL_SCHEMA,
            handler=request_toolset_handler,
            description=RECOVERY_TOOL_SCHEMA["description"],
            emoji="",
        )
    except Exception as exc:
        logger.warning("%s: failed to register recovery tool: %s", PLUGIN_NAME, exc)

    logger.info(
        "%s plugin registered (hooks: pre_turn_context_build, pre_llm_call, post_tool_call; tool: request_toolset)",
        PLUGIN_NAME,
    )


def pre_turn_context_build(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Early hook: route before prompt, skills, preflight, and tool schema assembly."""
    route_kwargs = dict(kwargs)
    agent = route_kwargs.pop("agent", None)
    return _route_tool_surface("pre_turn_context_build", agent=agent, **route_kwargs)


def pre_llm_call(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Late fallback hook for older Hermes builds or missed early hooks."""
    return _route_tool_surface("pre_llm_call", **kwargs)

def post_tool_call(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Post-tool hook: detect missing tools and expand agent.tools dynamically.

    If a tool was called that isn't in our predicted set, we:
    1. Find the toolset it belongs to
    2. Add that toolset's tools to agent.tools
    3. Update agent.enabled_toolsets
    4. Set _retry_pending flag so the conversation loop retries this turn

    Returns None (observer hook, no return value consumed by caller).
    """

    agent = _get_agent_ref() or _get_agent_from_stack()
    if agent is None:
        return None
    state = _get_router_state(agent)

    if not state.active:
        return None

    # Check if we already expanded — skip repeated expansions for the same turn
    if state._retry_pending:
        return None

    tool_name = kwargs.get("tool_name", "")
    if not tool_name:
        return None

    try:
        # Check if this tool is already in the current tool set
        if hasattr(agent, "valid_tool_names"):
            if tool_name in agent.valid_tool_names:
                return None  # Tool is already loaded

        # Check if the tool is not available at all
        all_tool_names = _get_all_tool_names()
        if tool_name not in all_tool_names:
            logger.debug(
                "%s: tool '%s' not found in registry at all — skipping",
                PLUGIN_NAME, tool_name,
            )
            return None

        # Find which toolset this tool belongs to
        from tools.registry import registry
        missing_toolset = _infer_toolset_from_tool(tool_name, registry)

        if missing_toolset is None:
            logger.debug(
                "%s: could not determine toolset for '%s' — checking by name prefix",
                PLUGIN_NAME, tool_name,
            )
            # Fall back: add all tools for the inferred toolset
            try:
                from tools.registry import registry
                for entry in registry._snapshot_entries():
                    if entry.name == tool_name:
                        missing_toolset = entry.toolset
                        break
            except Exception:
                pass

        if missing_toolset is None:
            logger.debug(
                "%s: could not determine toolset for '%s' — full fallback",
                PLUGIN_NAME, tool_name,
            )
            _handle_full_fallback(agent)
            return None

        # If this toolset is already in the predicted set, something else
        # is wrong — don't expand.
        if (state.predicted_toolsets is not None
                and missing_toolset in state.predicted_toolsets):
            logger.debug(
                "%s: tool '%s' in toolset '%s' but not in valid_tool_names — "
                "likely a check_fn issue, not router",
                PLUGIN_NAME, tool_name, missing_toolset,
            )
            # Still expand since the tool isn't available
            _expand_toolset(agent, missing_toolset)
        else:
            # Expand the predicted set with the missing toolset
            _expand_toolset(agent, missing_toolset)

        # Signal retry by setting the flag — the agent loop will re-read
        # agent.tools before the next API call
        state._retry_pending = True

        logger.info(
            "%s: recall — added toolset '%s' (tool '%s' not in predicted set %s). "
            "%d tools now available.",
            PLUGIN_NAME,
            missing_toolset,
            tool_name,
            sorted(state.predicted_toolsets) if state.predicted_toolsets else "N/A",
            len(agent.tools) if hasattr(agent, "tools") and agent.tools else 0,
        )

    except Exception as exc:
        logger.warning(
            "%s: post_tool_call handler failed: %s — full fallback",
            PLUGIN_NAME, exc,
        )
        _handle_full_fallback(agent)

    return None


_cfg = _load_config() if CONFIG_FILE.exists() else {}
_prof_cfg = _get_profile_config(_cfg)
logger.info("%s: plugin loaded profile=%s enabled=%s", PLUGIN_NAME, _prof_cfg.get("_profile_name"), _prof_cfg.get("enabled"))
