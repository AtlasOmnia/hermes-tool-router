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
    from .capabilities import (
        MISSING,
        EnsureAdmissionResult,
        OwnerSnapshot,
        TrustedHostPolicy,
        compose_effective_policy,
        read_trusted_host_policy,
        result_for_status,
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
        expand_admitted_toolsets,
        _filter_tool_definitions,
        _get_agent_local_tool_names,
        _get_all_tool_names,
        _get_recovery_tool_definition,
        _get_tool_definitions_for_names,
        _handle_full_fallback,
        _infer_toolset_from_tool,
        _resolve_toolset_to_tool_names,
        _restore_full_tools,
        restore_admitted_envelope,
        select_and_commit_route,
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
    from capabilities import (
        MISSING,
        EnsureAdmissionResult,
        OwnerSnapshot,
        TrustedHostPolicy,
        compose_effective_policy,
        read_trusted_host_policy,
        result_for_status,
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
        expand_admitted_toolsets,
        _filter_tool_definitions,
        _get_agent_local_tool_names,
        _get_all_tool_names,
        _get_recovery_tool_definition,
        _get_tool_definitions_for_names,
        _handle_full_fallback,
        _infer_toolset_from_tool,
        _resolve_toolset_to_tool_names,
        _restore_full_tools,
        restore_admitted_envelope,
        select_and_commit_route,
    )

logger = logging.getLogger(__name__)

# Registration is checked at plugin startup. If a Hermes update removes either
# recovery mechanism, routing must fail open rather than leave a narrowed
# surface that cannot reliably add a missed capability.
_registration_checked = False
_recovery_middleware_registered = False
_recovery_tool_registered = False


def _recovery_is_ready() -> bool:
    """Return whether a registered runtime can safely narrow tool schemas."""
    # Direct-import test harnesses do not call register(); preserve their
    # isolated behavior. A real plugin runtime always marks registration.
    if not _registration_checked:
        return True
    return _recovery_middleware_registered and _recovery_tool_registered


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

    if not _recovery_is_ready():
        _restore_full_tools(agent)
        state.active = False
        state.predicted_toolsets = None
        logger.warning(
            "%s: recovery registration incomplete; keeping full tool surface",
            PLUGIN_NAME,
        )
        return None

    # Production policy: classify only the initial tool surface. Later turns
    # reuse the session-sticky surface so tool schemas remain cache-stable.
    if state.initial_route_applied:
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
    # Empty and period/ellipsis-only probes carry no actionable intent. Route
    # them deterministically even on classifier-first profiles: asking a
    # stochastic model to interpret "." can yield a low-confidence "all"
    # response, and the safe fail-open then defeats the smallest-surface smoke
    # test. Other symbols and emoji remain ambiguous and keep the normal
    # classifier/fail-open path.
    probe_chars = {char for char in user_message if not char.isspace()}
    if not probe_chars or probe_chars <= {".", "…"}:
        predicted = set()
        rule_reason = "empty_or_period_probe"
    elif deterministic_enabled:
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


def _legacy_request_toolset_handler(args: Dict[str, Any], **kwargs: Any) -> str:
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


def _legacy_tool_request_middleware(**kwargs: Any) -> Optional[Dict[str, Any]]:
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
    agent = kwargs.get("agent")
    if agent is None and session_id:
        agent = _get_agent_ref(session_id)
    if session_id and agent is not None:
        state = _attached_capability_state(agent)
        if state is not None:
            state.end_admission(session_id)
    if session_id:
        _drop_agent_ref(session_id)


def register(ctx) -> None:
    """Register the hermes-token-router plugin.

    Hooks registered:
      - pre_turn_context_build : primary route before prompt/tool assembly
      - pre_llm_call           : late fallback hook for older Hermes builds
      - post_tool_call         : best-effort expansion after executed tool calls
    """

    global _registration_checked, _recovery_middleware_registered, _recovery_tool_registered
    _registration_checked = True
    _recovery_middleware_registered = False
    _recovery_tool_registered = False

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
        _recovery_middleware_registered = True
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
        _recovery_tool_registered = True
    except Exception as exc:
        logger.warning("%s: failed to register recovery tool: %s", PLUGIN_NAME, exc)

    logger.info(
        "%s plugin registered (routing: pre_llm_call compatibility; middleware: %s; tool: %s)",
        PLUGIN_NAME,
        _recovery_middleware_registered,
        _recovery_tool_registered,
    )


def pre_turn_context_build(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Early hook: route before prompt, skills, preflight, and tool schema assembly."""
    route_kwargs = dict(kwargs)
    agent = route_kwargs.pop("agent", None)
    return _route_tool_surface("pre_turn_context_build", agent=agent, **route_kwargs)


def pre_llm_call(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Late fallback hook for older Hermes builds or missed early hooks."""
    return _route_tool_surface("pre_llm_call", **kwargs)

def _legacy_post_tool_call(**kwargs: Any) -> Optional[Dict[str, Any]]:
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

def _read_worker_task_id() -> str:
    """Read the host worker signal at the adapter boundary only."""
    import os

    return os.environ.get("HERMES_KANBAN_TASK", "")


def _is_dispatcher_owned_worker_context() -> bool:
    """Ask Hermes once whether the current context owns a worker task."""
    try:
        from agent.delegation_context import is_dispatcher_owned_worker_context
    except ImportError:
        # Standalone plugin tests have no Hermes delegation module.  An absent
        # optional adapter is a known non-worker context; a present callable
        # that raises is handled as uncertainty by _ensure_host_admission.
        return False
    return bool(is_dispatcher_owned_worker_context())


def _capture_owner_snapshot(agent: Any, untouched_surface: Any) -> OwnerSnapshot:
    """Build one ordered owner snapshot without retrieving definitions."""
    from collections.abc import Mapping

    incoming_names: list[str] = []
    malformed: list[int] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for index, definition in enumerate(untouched_surface):
        name: Any = None
        if isinstance(definition, Mapping):
            function = definition.get("function")
            if isinstance(function, Mapping):
                name = function.get("name")
        if type(name) is not str or not name or name != name.strip():
            malformed.append(index)
            continue
        incoming_names.append(name)
        if name in seen:
            duplicates.append(name)
        seen.add(name)

    if malformed or duplicates:
        errors = [f"MALFORMED_DEFINITION_INDEX:{index}" for index in malformed]
        errors.extend(f"DUPLICATE_DEFINITION_NAME:{name}" for name in duplicates)
        return OwnerSnapshot(
            incoming_names=tuple(incoming_names),
            owner_by_name=(),
            registry_import_ok=False,
            registry_lookup_error_names=(),
            agent_local_lookup_error_names=(),
            malformed_definition_indexes=tuple(malformed),
            duplicate_names=tuple(dict.fromkeys(duplicates)),
            errors=tuple(dict.fromkeys(errors)),
        )

    owners: list[tuple[str, str | None]] = []
    registry_lookup_errors: list[str] = []
    local_lookup_errors: list[str] = []
    try:
        from tools.registry import registry
    except Exception:
        return OwnerSnapshot(
            incoming_names=tuple(incoming_names),
            owner_by_name=tuple((name, None) for name in incoming_names),
            registry_import_ok=False,
            registry_lookup_error_names=(),
            agent_local_lookup_error_names=(),
            malformed_definition_indexes=(),
            duplicate_names=(),
            errors=("REGISTRY_IMPORT_UNAVAILABLE",),
        )

    manager = getattr(agent, "_memory_manager", None)
    has_local = getattr(manager, "has_tool", None)
    for name in incoming_names:
        owner: str | None = None
        try:
            entry = registry.get_entry(name)
            candidate = getattr(entry, "toolset", None) if entry is not None else None
            if type(candidate) is str and candidate:
                owner = candidate
        except Exception:
            registry_lookup_errors.append(name)
        if owner is None and callable(has_local):
            try:
                if has_local(name):
                    owner = "memory"
            except Exception:
                local_lookup_errors.append(name)
        owners.append((name, owner))

    errors = []
    if registry_lookup_errors:
        errors.append("REGISTRY_OWNER_LOOKUP_UNAVAILABLE")
    if local_lookup_errors:
        errors.append("AGENT_LOCAL_OWNER_LOOKUP_UNAVAILABLE")
    return OwnerSnapshot(
        incoming_names=tuple(incoming_names),
        owner_by_name=tuple(owners),
        registry_import_ok=True,
        registry_lookup_error_names=tuple(dict.fromkeys(registry_lookup_errors)),
        agent_local_lookup_error_names=tuple(dict.fromkeys(local_lookup_errors)),
        malformed_definition_indexes=(),
        duplicate_names=(),
        errors=tuple(errors),
    )


def _compose_effective_policy(
    parsed_policy: TrustedHostPolicy,
    owner_snapshot: OwnerSnapshot | None,
    *,
    adapter_protected_toolsets: frozenset[str],
    worker_toolset: str,
    has_nonempty_worker_task_id: bool,
    dispatcher_owned_worker: bool | None,
    worker_identity_error: str | None,
) -> TrustedHostPolicy:
    """Compose the sole typed host/no-envelope adapter seam."""
    return compose_effective_policy(
        parsed_policy,
        owner_snapshot,
        adapter_protected_toolsets=adapter_protected_toolsets,
        worker_toolset=worker_toolset,
        has_nonempty_worker_task_id=has_nonempty_worker_task_id,
        dispatcher_owned_worker=dispatcher_owned_worker,
        worker_identity_error=worker_identity_error,
    )


def _attached_capability_state(agent: Any) -> RouterState | None:
    """Attach RouterState and verify identity; never substitute global state."""
    if agent is None:
        return None
    try:
        state = getattr(agent, ROUTER_STATE_ATTR, None)
    except Exception:
        state = None
    if isinstance(state, RouterState):
        return state
    candidate = RouterState()
    try:
        setattr(agent, ROUTER_STATE_ATTR, candidate)
        readback = getattr(agent, ROUTER_STATE_ATTR, None)
    except Exception:
        return None
    return candidate if readback is candidate else None


# Rev 6 admission adapter.  Every live caller uses this authoritative seam.


def _ensure_host_admission(
    *,
    source: str,
    agent: Any = None,
    session_id: str = "",
    untouched_surface: Any = MISSING,
    original_enabled_toolsets: Any = MISSING,
    hook_metadata: Any = MISSING,
    agent_metadata: Any = MISSING,
) -> EnsureAdmissionResult:
    """Resolve one attached lifecycle and perform exactly one capture, if able."""
    state = _attached_capability_state(agent)
    if state is None:
        return result_for_status(
            "NO_AUTHORITY_UNATTACHABLE",
            session_id,
            diagnostics=("NO_PERSISTENT_CONTAMINATION_GUARD",),
        )
    prior = state.bound_host_admission(session_id)
    if prior is not None:
        return prior
    marker = state.no_authority_contamination(session_id)
    authoritative = (
        untouched_surface is not MISSING
        and original_enabled_toolsets is not MISSING
        and marker is None
    )
    if authoritative:
        parsed_policy = read_trusted_host_policy(hook_metadata, agent_metadata)
        try:
            owner_snapshot = _capture_owner_snapshot(agent, untouched_surface)
        except Exception as exc:
            owner_snapshot = OwnerSnapshot(
                incoming_names=(),
                owner_by_name=(),
                registry_import_ok=False,
                registry_lookup_error_names=(),
                agent_local_lookup_error_names=(),
                malformed_definition_indexes=(),
                duplicate_names=(),
                errors=(f"OWNER_SNAPSHOT_FAILED:{type(exc).__name__}",),
            )
        task_id = _read_worker_task_id()
        has_task = type(task_id) is str and bool(task_id)
        dispatcher_owned: bool | None = None
        worker_error: str | None = None
        if has_task:
            try:
                dispatcher_owned = _is_dispatcher_owned_worker_context()
            except Exception:
                worker_error = "WORKER_PREDICATE_UNAVAILABLE"
        effective_policy = _compose_effective_policy(
            parsed_policy,
            owner_snapshot,
            adapter_protected_toolsets=frozenset({"kanban"}),
            worker_toolset="kanban",
            has_nonempty_worker_task_id=has_task,
            dispatcher_owned_worker=dispatcher_owned,
            worker_identity_error=worker_error,
        )
        return state.ensure_host_admission(
            session_id=session_id,
            untouched_surface=untouched_surface,
            original_enabled_toolsets=original_enabled_toolsets,
            effective_policy=effective_policy,
            owner_snapshot=owner_snapshot,
        )

    parsed_policy = read_trusted_host_policy(hook_metadata, agent_metadata)
    effective_policy = _compose_effective_policy(
        parsed_policy,
        None,
        adapter_protected_toolsets=frozenset({"kanban"}),
        worker_toolset="kanban",
        has_nonempty_worker_task_id=False,
        dispatcher_owned_worker=None,
        worker_identity_error=None,
    )
    return state.ensure_host_admission(
        session_id=session_id,
        untouched_surface=MISSING,
        original_enabled_toolsets=MISSING,
        effective_policy=effective_policy,
        owner_snapshot=MISSING,
    )


def _route_tool_surface(
    source: str,
    agent: Any = None,
    runtime_profile_name: Optional[str] = None,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Capture admission before empty-message exits, then route one surface."""
    cfg = _load_config()
    if agent is None:
        agent = _get_agent_from_stack() or _get_agent_ref(
            str(kwargs.get("session_id") or "") or None
        )
    session_id = str(kwargs.get("session_id") or getattr(agent, "session_id", "") or "")
    if agent is not None:
        _store_agent_ref(agent, session_id)
    if agent is None or not session_id:
        return None

    state = _get_router_state(agent)
    marker = state.no_authority_contamination(session_id)
    first_contact = not state.initial_route_applied and marker is None
    hook_metadata = kwargs.get("hermes_token_router_admission", MISSING)
    try:
        agent_metadata = getattr(agent, "_hermes_token_router_admission", MISSING)
    except Exception:
        agent_metadata = MISSING
    if first_contact:
        untouched_surface = tuple(getattr(agent, "tools", ()) or ())
        original_enabled = tuple(getattr(agent, "enabled_toolsets", ()) or ())
    else:
        untouched_surface = MISSING
        original_enabled = MISSING
    admission = _ensure_host_admission(
        source=source,
        agent=agent,
        session_id=session_id,
        untouched_surface=untouched_surface,
        original_enabled_toolsets=original_enabled,
        hook_metadata=hook_metadata,
        agent_metadata=agent_metadata,
    )

    user_message = kwargs.get("user_message", "")
    turn_id = kwargs.get("turn_id") or getattr(agent, "_current_turn_id", "") or ""
    current_profile_cfg = _get_profile_config(cfg)
    if not _is_router_active(cfg) or not current_profile_cfg.get("enabled", False):
        if admission.status in {"READY", "SAFE_NO_PRUNE"} and admission.envelope is not None:
            restore_admitted_envelope(agent, admission)
        state.active = False
        state.predicted_toolsets = None
        if source == "pre_turn_context_build":
            _mark_turn_routed(agent, state, turn_id, source)
        return None
    if not user_message:
        # Admission is intentionally bound before this return.  Do not run
        # classifier, available-toolset, retrieval, recovery, or retry work.
        return None
    if not _recovery_is_ready():
        # The target runtime must retain its current surface until both
        # recovery mechanisms are registered successfully.
        state.active = False
        state.predicted_toolsets = None
        if source == "pre_turn_context_build":
            _mark_turn_routed(agent, state, turn_id, source)
        return None
    if state.initial_route_applied:
        turn_id = kwargs.get("turn_id") or getattr(agent, "_current_turn_id", "") or ""
        if turn_id:
            _mark_turn_routed(agent, state, turn_id, "sticky_surface")
        return None
    turn_id = kwargs.get("turn_id") or getattr(agent, "_current_turn_id", "") or ""
    if source != "pre_turn_context_build" and _was_turn_routed(agent, state, turn_id):
        return None

    profile_cfg = _get_profile_config(cfg)
    if not profile_cfg.get("enabled", False):
        if admission.status in {"READY", "SAFE_NO_PRUNE"} and admission.envelope is not None:
            restore_admitted_envelope(agent, admission)
        state.active = False
        state.predicted_toolsets = None
        if source == "pre_turn_context_build":
            _mark_turn_routed(agent, state, turn_id, source)
        return None
    if admission.status != "READY":
        # SAFE_NO_PRUNE restores only its already captured envelope.  All
        # other statuses preserve the current surface and deny additions.
        if admission.status == "SAFE_NO_PRUNE" and admission.envelope is not None:
            restore_admitted_envelope(agent, admission)
        state.active = False
        state.predicted_toolsets = None
        if source == "pre_turn_context_build":
            _mark_turn_routed(agent, state, turn_id, source)
        return None

    decline_chars = profile_cfg.get("long_message_decline_chars", 2000)
    short_bypass_chars = profile_cfg.get("short_message_bypass_chars", 0)
    floor_toolsets = set(profile_cfg.get("floor_toolsets", ["terminal", "file", "web"]))
    try:
        available = _get_available_toolsets()
    except Exception:
        available = set()
    if not available:
        restore_admitted_envelope(agent, admission)
        if source == "pre_turn_context_build":
            _mark_turn_routed(agent, state, turn_id, source)
        return None
    if len(user_message) > decline_chars:
        restore_admitted_envelope(agent, admission)
        if source == "pre_turn_context_build":
            _mark_turn_routed(agent, state, turn_id, source)
        return None
    if short_bypass_chars > 0 and len(user_message.strip()) < short_bypass_chars:
        restore_admitted_envelope(agent, admission)
        if source == "pre_turn_context_build":
            _mark_turn_routed(agent, state, turn_id, source)
        return None
    if not available:
        restore_admitted_envelope(agent, admission)
        if source == "pre_turn_context_build":
            _mark_turn_routed(agent, state, turn_id, source)
        return None

    deterministic_enabled = bool(profile_cfg.get("deterministic_rules_enabled", True))
    probe_chars = {char for char in user_message if not char.isspace()}
    if not probe_chars or probe_chars <= {".", "…"}:
        predicted: Optional[Set[str]] = set()
    elif deterministic_enabled:
        predicted, _ = _predict_toolsets_by_rules(user_message, available)
    elif not _is_classifier_enabled(profile_cfg):
        predicted = None
    else:
        try:
            predicted = _predict_toolsets_via_llm(
                user_message,
                available,
                _get_router_model(profile_cfg),
                float(profile_cfg.get("confidence_threshold", 0.0)),
                _get_router_provider(profile_cfg),
                max(0.05, float(profile_cfg.get("router_hard_timeout_ms", 1200)) / 1000.0),
                *_get_classifier_connection(profile_cfg),
            )
        except Exception:
            predicted = None
    if predicted is None:
        restore_admitted_envelope(agent, admission)
        if source == "pre_turn_context_build":
            _mark_turn_routed(agent, state, turn_id, source)
        return None

    result = select_and_commit_route(
        agent,
        admission,
        predicted_toolsets=set(predicted),
        floor_toolsets=floor_toolsets,
        available_toolsets=available,
        caller="route",
    )
    state.last_expansion_result = result
    if source == "pre_turn_context_build":
        _mark_turn_routed(agent, state, turn_id, source)
    return None


def _call_admission_for_current(
    source: str,
    agent: Any,
    session_id: str,
    kwargs: dict[str, Any],
) -> EnsureAdmissionResult:
    try:
        agent_metadata = getattr(agent, "_hermes_token_router_admission", MISSING)
    except Exception:
        agent_metadata = MISSING
    return _ensure_host_admission(
        source=source,
        agent=agent,
        session_id=session_id,
        untouched_surface=MISSING,
        original_enabled_toolsets=MISSING,
        hook_metadata=kwargs.get("hermes_token_router_admission", MISSING),
        agent_metadata=agent_metadata,
    )


def request_toolset_handler(args: Dict[str, Any], **kwargs: Any) -> str:
    """Return a truthful result for one guarded expansion request."""
    args = args if isinstance(args, dict) else {}
    requested_toolset = str(args.get("toolset") or args.get("toolset_name") or "").strip().lower()
    raw_toolsets = args.get("toolsets") or []
    requested_toolsets = {
        str(name).strip().lower()
        for name in raw_toolsets
        if isinstance(raw_toolsets, list) and str(name).strip()
    }
    if requested_toolset:
        requested_toolsets.add(requested_toolset)
    requested_tool = str(args.get("tool_name") or "").strip()
    reason = str(args.get("reason") or "").strip()[:200]
    session_id = str(kwargs.get("session_id") or "")
    agent = _get_agent_ref(session_id) or kwargs.get("agent") or _get_agent_from_stack()
    admission: EnsureAdmissionResult | None = None
    if agent is not None:
        session_id = session_id or str(getattr(agent, "session_id", "") or "")
        if session_id:
            _store_agent_ref(agent, session_id)
        admission = _call_admission_for_current(
            "visible_request", agent, session_id, kwargs
        )
    available: set[str] = set()
    try:
        from tools.registry import registry as validation_registry
        available = set(validation_registry.get_registered_toolset_names())
        if requested_tool and (
            admission is None
            or admission.status
            not in {
                "CAPTURE_INVALID_NO_MUTATION",
                "SAFE_NO_PRUNE",
                "SESSION_MISMATCH",
                "NO_AUTHORITY_UNATTACHABLE",
                "NO_AUTHORITY_INVALID_POLICY",
            }
        ):
            owner = _infer_toolset_from_tool(requested_tool, validation_registry, None)
            if owner:
                requested_toolsets.add(owner)
        requested_toolsets = {
            validation_registry.get_toolset_alias_target(name) or name
            for name in requested_toolsets
        }
    except Exception:
        available = set()
    local_memory_available = bool(_get_agent_local_tool_names(agent, "memory")) if agent is not None else False
    if agent is not None and requested_tool:
        bound = getattr(_get_router_state(agent), "_bound_admission_result", None)
        envelope = getattr(bound, "envelope", None) if bound is not None else None
        if envelope is not None:
            captured = next((item for item in envelope.definitions if item.name == requested_tool), None)
            if captured is not None:
                if captured.owner_toolset_at_capture:
                    requested_toolsets.add(captured.owner_toolset_at_capture)
                elif (
                    bound is not None
                    and bound.effective_policy is not None
                    and requested_tool in bound.effective_policy.pinned_tool_names
                ):
                    requested_toolsets.update(bound.effective_policy.protected_toolsets)
    if not requested_toolsets and not (requested_tool and admission is not None):
        return json.dumps({
            "ok": False,
            "error": "toolsets or resolvable tool_name is required",
            "requested_toolsets": [],
            "requested_tool": requested_tool,
        })
    unknown = requested_toolsets - available if available else set()
    if local_memory_available:
        unknown.discard("memory")
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
    if agent is None:
        return json.dumps({
            "ok": False,
            "requested_toolsets": sorted(requested_toolsets),
            "expanded_toolsets": [],
            "denied_toolsets": sorted(requested_toolsets),
            "denied_tool_names": [requested_tool] if requested_tool else [],
            "added_tool_names": [],
            "installed_tool_names": [],
            "requested_tool": requested_tool,
            "reason": "NO_PERSISTENT_CONTAMINATION_GUARD",
            "retry_allowed": False,
        })
    try:
        from tools.registry import registry as live_registry
        requested_toolsets = {
            str(live_registry.get_toolset_alias_target(name) or name)
            for name in requested_toolsets
        }
    except Exception:
        pass
    if admission is None:
        admission = _call_admission_for_current(
            "visible_request", agent, session_id, kwargs
        )
    result = expand_admitted_toolsets(
        agent,
        admission,
        requested_toolsets=requested_toolsets,
        requested_tool_names={requested_tool} if requested_tool else set(),
        caller="visible_request",
        session_id=session_id,
    )
    if result.retry_allowed:
        _get_router_state(agent)._retry_pending = False
    payload = {
        "ok": result.ok,
        "toolsets": list(result.requested_toolsets),
        "requested_toolsets": list(result.requested_toolsets),
        "expanded_toolsets": list(result.expanded_toolsets),
        "denied_toolsets": list(result.denied_toolsets),
        "denied_tool_names": list(result.denied_tool_names),
        "added_tool_names": list(result.added_tool_names),
        "installed_tool_names": list(result.installed_tool_names),
        "requested_tool": requested_tool,
        "reason": reason or result.reason,
        "retry_allowed": result.retry_allowed,
        "enabled_tools": sorted(getattr(agent, "valid_tool_names", set()) or set()),
    }
    if len(result.requested_toolsets) == 1:
        payload["toolset"] = result.requested_toolsets[0]
    return json.dumps(payload, sort_keys=True)


def tool_request_middleware(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Admit a missing exact tool before normal Hermes dispatch validation."""
    session_id = str(kwargs.get("session_id") or "")
    agent = _get_agent_ref(session_id) or kwargs.get("agent") or _get_agent_from_stack()
    tool_name = str(kwargs.get("tool_name") or "").strip()
    args = kwargs.get("args")
    if agent is None or not tool_name or not isinstance(args, dict):
        return None
    session_id = session_id or str(getattr(agent, "session_id", "") or "")
    if tool_name in (getattr(agent, "valid_tool_names", set()) or set()):
        return None
    admission = _call_admission_for_current(
        "middleware", agent, session_id, kwargs
    )
    expand_admitted_toolsets(
        agent,
        admission,
        requested_toolsets=set(),
        requested_tool_names={tool_name},
        caller="middleware",
        session_id=session_id,
    )
    if tool_name not in (getattr(agent, "valid_tool_names", set()) or set()):
        return None
    return {"args": dict(args), "router_recovered": _toolset_for_installed_name(tool_name, admission)}


def _toolset_for_installed_name(name: str, admission: EnsureAdmissionResult) -> str:
    if admission.envelope is not None:
        for item in admission.envelope.definitions:
            if item.name == name and item.owner_toolset_at_capture:
                return item.owner_toolset_at_capture
    registry = _registry_for_handler()
    owner = _owner_for_name_handler(name, registry)
    return owner or "unknown"


def _registry_for_handler() -> Any:
    try:
        from tools.registry import registry
        return registry
    except Exception:
        return None


def _owner_for_name_handler(name: str, registry: Any) -> str | None:
    if registry is None:
        return None
    try:
        entry = registry.get_entry(name)
        return getattr(entry, "toolset", None) if entry is not None else None
    except Exception:
        return None


def post_tool_call(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Retry only after a guarded exact installation succeeded."""
    session_id = str(kwargs.get("session_id") or "")
    agent = _get_agent_ref(session_id) or kwargs.get("agent") or _get_agent_from_stack()
    tool_name = str(kwargs.get("tool_name") or "").strip()
    if agent is None or not tool_name:
        return None
    session_id = session_id or str(getattr(agent, "session_id", "") or "")
    state = _get_router_state(agent)
    if state._retry_pending or tool_name in (getattr(agent, "valid_tool_names", set()) or set()):
        return None
    admission = _call_admission_for_current(
        "post_tool", agent, session_id, kwargs
    )
    result = expand_admitted_toolsets(
        agent,
        admission,
        requested_toolsets=set(),
        requested_tool_names={tool_name},
        caller="post_tool",
        session_id=session_id,
    )
    if tool_name in (getattr(agent, "valid_tool_names", set()) or set()) and result.retry_allowed:
        state._retry_pending = True
    return None


def _handle_full_fallback(agent: Any) -> None:
    """Fail closed: restore a valid envelope, otherwise preserve current state."""
    state = _get_router_state(agent)
    bound = getattr(state, "_bound_admission_result", None)
    if isinstance(bound, EnsureAdmissionResult):
        if bound.status in {"READY", "SAFE_NO_PRUNE"} and bound.envelope is not None:
            if restore_admitted_envelope(agent, bound):
                state.active = False
                state.predicted_toolsets = None
                state._fallback_triggered = True
                state._retry_pending = False
        return
    # No authority, capture-invalid, contamination, and mismatch paths are
    # preservation paths.  They must not use the mutable cache or claim retry.
    if getattr(state, "_admission_session_id", None) is not None:
        return
    if state._full_tool_defs is not None:
        agent.tools = list(state._full_tool_defs)
        if hasattr(agent, "valid_tool_names"):
            agent.valid_tool_names = set(state._full_tool_names)
