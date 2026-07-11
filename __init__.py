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
        _get_classifier_connection,
        _get_router_model,
        _get_router_provider,
        _is_classifier_enabled,
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
        _drop_agent_ref,
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
        build_recovery_tool_schema,
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
        _get_classifier_connection,
        _get_router_model,
        _get_router_provider,
        _is_classifier_enabled,
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
        _drop_agent_ref,
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
        build_recovery_tool_schema,
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

    session_id = str(kwargs.get("session_id") or getattr(agent, "session_id", "") or "")
    if agent is not None:
        _store_agent_ref(agent, session_id)
    else:
        # The current hook stack is authoritative; a cached session mapping can
        # be stale after resets or test/profile reloads.
        agent = _get_agent_from_stack() or _get_agent_ref(session_id)
        if agent is not None:
            session_id = session_id or str(getattr(agent, "session_id", "") or "")
            _store_agent_ref(agent, session_id)
            logger.info("%s: acquired agent reference for compatibility hook", PLUGIN_NAME)
        else:
            logger.warning("%s: no agent reference; full set fallback", PLUGIN_NAME)
            return None

    turn_id = kwargs.get("turn_id") or getattr(agent, "_current_turn_id", "") or ""
    state = _get_router_state(agent)

    # Production policy: classify only the initial tool surface. Later turns
    # reuse the session-sticky surface so tool schemas remain cache-stable.
    if state.initial_route_applied and source == "pre_turn_context_build":
        if turn_id:
            _mark_turn_routed(agent, state, turn_id, "sticky_surface")
        return None

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
    classifier_base_url, classifier_api_key_env = _get_classifier_connection(profile_cfg)
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
        # The external classifier is opt-in. An unresolved deterministic route
        # fails open immediately when it is disabled—zero network latency.
        if not _is_classifier_enabled(profile_cfg):
            predicted = None
        else:
            try:
                predicted = _predict_toolsets_via_llm(
                    user_message,
                    available,
                    router_model,
                    confidence_threshold,
                    router_provider,
                    max(0.05, float(profile_cfg.get("router_hard_timeout_ms", 1200)) / 1000.0),
                    classifier_base_url,
                    classifier_api_key_env,
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

    # Store agent-scoped state for post_tool_call/request_toolset.
    state.active = True
    state.predicted_toolsets = predicted
    state.set_initial_surface(set(predicted) | floor_toolsets)
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
    raw_toolsets = args.get("toolsets") or []
    requested_toolsets = {
        str(name).strip().lower() for name in raw_toolsets if str(name).strip()
    } if isinstance(raw_toolsets, list) else set()
    if requested_toolset:
        requested_toolsets.add(requested_toolset)
    requested_tool = str(args.get("tool_name") or "").strip()
    reason = str(args.get("reason") or "").strip()[:200]

    try:
        from tools.registry import registry
        available = set(registry.get_registered_toolset_names())
        if requested_tool:
            owner = _infer_toolset_from_tool(requested_tool, registry)
            if owner:
                requested_toolsets.add(owner)
        requested_toolsets = {
            registry.get_toolset_alias_target(name) or name for name in requested_toolsets
        }
    except Exception:
        available = set()

    if not requested_toolsets:
        return json.dumps({
            "ok": False,
            "error": "toolsets or resolvable tool_name is required",
            "requested_toolsets": [],
            "requested_tool": requested_tool,
        })

    unknown = requested_toolsets - available if available else set()
    if unknown:
        bad = sorted(unknown)[0]
        suggestions = difflib.get_close_matches(bad, sorted(available), n=5)
        return json.dumps({
            "ok": False,
            "error": f"unknown toolset: {bad}",
            "requested_toolsets": sorted(requested_toolsets),
            "requested_tool": requested_tool,
            "suggestions": suggestions,
            "available_toolsets": sorted(available),
        })

    session_id = str(kwargs.get("session_id") or "")
    agent = _get_agent_ref(session_id) or _get_agent_from_stack()
    if agent is None:
        return json.dumps({
            "ok": False,
            "error": "no live agent reference",
            "requested_toolsets": sorted(requested_toolsets),
            "requested_tool": requested_tool,
        })

    state = _get_router_state(agent)
    if state._full_tool_defs is None:
        _cache_full_toolset(agent)
    for toolset_name in sorted(requested_toolsets):
        _expand_toolset(agent, toolset_name)
    _ensure_recovery_tool(agent)
    state.active = True
    state._retry_pending = False

    enabled_tools = sorted(getattr(agent, "valid_tool_names", set()) or set())
    response = {
        "ok": True,
        "toolsets": sorted(requested_toolsets),
        "requested_tool": requested_tool,
        "reason": reason,
        "enabled_tools": enabled_tools,
    }
    if len(requested_toolsets) == 1:
        response["toolset"] = next(iter(requested_toolsets))
    return json.dumps(response)


def tool_request_middleware(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Expand a registry-known pruned tool before Hermes validates/dispatches it.

    Hermes applies ``tool_request`` middleware before it passes the current
    ``valid_tool_names`` set into normal dispatch. Updating the agent here lets
    the original call continue through ordinary check_fn, approvals, and
    execution without an invalid-tool round trip.
    """
    session_id = str(kwargs.get("session_id") or "")
    tool_name = str(kwargs.get("tool_name") or "").strip()
    args = kwargs.get("args")
    if not tool_name or not isinstance(args, dict):
        return None
    agent = _get_agent_ref(session_id)
    if agent is None:
        return None
    state = _get_router_state(agent)
    if not state.active or tool_name in (getattr(agent, "valid_tool_names", set()) or set()):
        return None
    try:
        from tools.registry import registry
        toolset_name = _infer_toolset_from_tool(tool_name, registry)
    except Exception:
        toolset_name = None
    if not toolset_name:
        return None
    _expand_toolset(agent, toolset_name)
    if tool_name not in (getattr(agent, "valid_tool_names", set()) or set()):
        return None
    logger.info(
        "%s: middleware recovery added toolset=%s for tool=%s session=%s",
        PLUGIN_NAME,
        toolset_name,
        tool_name,
        session_id,
    )
    return {"args": dict(args), "router_recovered": toolset_name}


def on_session_end(**kwargs: Any) -> None:
    session_id = str(kwargs.get("session_id") or "")
    if session_id:
        _drop_agent_ref(session_id)


def register(ctx) -> None:
    """Register the hermes-token-router plugin.

    Hooks registered:
      - pre_turn_context_build : primary route before prompt/tool assembly
      - pre_llm_call           : late fallback hook for older Hermes builds
      - post_tool_call         : best-effort expansion after executed tool calls
    """

    try:
        from hermes_cli.plugins import VALID_HOOKS
        if "pre_turn_context_build" in VALID_HOOKS:
            ctx.register_hook("pre_turn_context_build", pre_turn_context_build)
    except Exception as exc:
        logger.debug("%s: early routing hook unavailable: %s", PLUGIN_NAME, exc)

    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("post_tool_call", post_tool_call)
    ctx.register_hook("on_session_end", on_session_end)
    try:
        ctx.register_middleware("tool_request", tool_request_middleware)
    except Exception as exc:
        logger.warning("%s: tool_request middleware unavailable: %s", PLUGIN_NAME, exc)

    try:
        from tools.registry import registry
        recovery_schema = build_recovery_tool_schema(set(registry.get_registered_toolset_names()))
        registry.register(
            name=RECOVERY_TOOL_NAME,
            toolset=RECOVERY_TOOLSET,
            schema=recovery_schema,
            handler=request_toolset_handler,
            description=recovery_schema["description"],
            emoji="",
        )
    except Exception as exc:
        logger.warning("%s: failed to register recovery tool: %s", PLUGIN_NAME, exc)

    logger.info(
        "%s plugin registered (routing: pre_llm_call compatibility; middleware: tool_request; tool: request_toolset)",
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

    session_id = str(kwargs.get("session_id") or "")
    agent = _get_agent_ref(session_id) or _get_agent_from_stack()
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
