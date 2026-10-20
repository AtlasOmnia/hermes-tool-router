"""Hermes Tool Router Plugin.

Thin plugin entrypoint: registration, hook handlers, and recovery tool handler.
The implementation lives in focused sibling modules.
"""

from __future__ import annotations

import contextlib
import difflib
import json
import logging
import math
import time
from typing import Any, Dict, Optional, Set

try:
    from .config import (
        _agent_profile_name,
        _env_override,
        _get_profile_settings,
        CONFIG_FILE,
        DEFAULT_ROUTER_MODEL,
        DEFAULT_ROUTER_PROVIDER,
        PLUGIN_NAME,
        _get_profile_config,
        _get_classifier_connection,
        _get_classifier_extra_body,
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
    from .intent import CHINESE_ABSTENTION_REASON, chinese_abstention_reason
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
        _find_agent_in_threads,
        _get_validated_agent_ref,
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
        expand_admitted_toolsets,
        _get_agent_local_tool_names,
        _get_recovery_tool_definition,
        _infer_toolset_from_tool,
        _resolve_toolset_to_tool_names,
        commit_fence,
        fenced_section,
        restore_admitted_envelope,
        select_and_commit_route,
        surface_is_router_committed,
    )
except ImportError:  # pragma: no cover - direct loader fallback
    from config import (
        _agent_profile_name,
        _env_override,
        _get_profile_settings,
        CONFIG_FILE,
        DEFAULT_ROUTER_MODEL,
        DEFAULT_ROUTER_PROVIDER,
        PLUGIN_NAME,
        _get_profile_config,
        _get_classifier_connection,
        _get_classifier_extra_body,
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
    from intent import CHINESE_ABSTENTION_REASON, chinese_abstention_reason
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
        _find_agent_in_threads,
        _get_validated_agent_ref,
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
        expand_admitted_toolsets,
        _get_agent_local_tool_names,
        _get_recovery_tool_definition,
        _infer_toolset_from_tool,
        _resolve_toolset_to_tool_names,
        commit_fence,
        fenced_section,
        restore_admitted_envelope,
        select_and_commit_route,
        surface_is_router_committed,
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
    with fenced_section() as may_commit:
        if may_commit:
            _mark_turn_routed_locked(agent, state, turn_id, source)


class _HookFence:
    """Lifecycle token for one hook or tool-handler invocation.

    ``still_valid`` holds while the lifecycle generation is unchanged, the
    agent still carries the hook's session id, the bound admission slot is the
    one this hook saw (once it pinned it) and the deadline has not passed.
    """

    def __init__(
        self,
        agent: Any,
        state: RouterState,
        session_id: str,
        deadline: float,
        require_retry_available: bool = False,
    ) -> None:
        self.require_retry_available = require_retry_available
        self.agent = agent
        self.state = state
        self.session_id = session_id
        self.deadline = deadline
        self.generation = state.lifecycle_generation
        self.slot: tuple[Any, int] | None = None

    def pin_admission(self) -> None:
        self.slot = _admission_slot(self.state)

    def rebase(self) -> None:
        """Adopt the lifecycle this hook itself just started (retirement release)."""
        self.generation = self.state.lifecycle_generation
        self.slot = None

    def still_valid(self) -> bool:
        return (
            self.state.lifecycle_generation == self.generation
            and _agent_session_matches_id(self.agent, self.session_id)
            and (self.slot is None or _admission_slot(self.state) == self.slot)
            and time.monotonic() <= self.deadline
            and not (self.require_retry_available and self.state._retry_pending)
        )


@contextlib.contextmanager
def _hook_fence(
    agent: Any, session_id: str, *, deadline: float = math.inf, require_retry_available: bool = False
):
    state = _get_router_state(agent)
    fence = _HookFence(agent, state, session_id, deadline, require_retry_available)
    with commit_fence(state.mutation_lock, fence.still_valid):
        yield fence


def _admission_slot(state: RouterState) -> tuple[Any, int]:
    """Identity of the bound admission (session id + result object)."""
    return (
        getattr(state, "_admission_session_id", None),
        id(getattr(state, "_bound_admission_result", None)),
    )


def _set_inactive(state: RouterState) -> None:
    with fenced_section() as may_commit:
        if may_commit:
            state.active = False
            state.predicted_toolsets = None


def _mark_turn_routed_locked(agent: Any, state: RouterState, turn_id: str, source: str) -> None:
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


def on_session_end(**kwargs: Any) -> None:
    session_id = str(kwargs.get("session_id") or "")
    agent = kwargs.get("agent")
    if agent is None and session_id:
        agent = _get_agent_ref(session_id)
    if session_id and agent is not None and not _agent_session_matches_id(agent, session_id):
        # Never tear down another session's agent.
        agent = None
    restored = True
    if session_id and agent is not None:
        state = _attached_capability_state(agent)
        if state is not None:
            with _hook_fence(
                agent, session_id, deadline=time.monotonic() + _hook_commit_budget_s()
            ) as fence:
                restored = _end_session_fenced(agent, state, session_id, fence)
    if session_id and restored:
        _drop_agent_ref(session_id)


def _end_session_fenced(
    agent: Any, state: RouterState, session_id: str, fence: _HookFence | None = None
) -> bool:
    """Session-end teardown; an abandoned teardown never touches a new lifecycle."""
    if not _retry_retired_restore(agent, state, session_id, fence):
        # An older session on this agent is still unrestored: this session
        # never routed (quarantine), so there is nothing to undo.
        return True
    restored = True
    if getattr(state, "surface_narrowed", False):
        bound = state.bound_host_admission(session_id)
        restored = bool(
            bound is not None
            and bound.envelope is not None
            and bound.status in {"READY", "SAFE_NO_PRUNE"}
            and restore_admitted_envelope(agent, bound)
        )
    with fenced_section() as may_commit:
        if not may_commit:
            # Abandoned or superseded teardown: leave whatever lifecycle is
            # current untouched (a later session end retries).
            return False
        if restored:
            # Retire the lifecycle; a reused agent object starts its next
            # session fresh, and in-flight hooks from this session can no
            # longer commit (generation bump).
            state.end_admission(session_id)
            state.reset()
            return True
        bound = state.bound_host_admission(session_id)
        # Keep the admission, the narrowed markers and the agent reference:
        # the protected-tool guard stays engaged and a later disabled turn or
        # session end can retry the restore.  A reused agent stays
        # quarantined under this admission (_retry_retired_restore).
        state._retired_admission = (session_id, bound) if bound is not None else None
        # Revoke this session's in-flight hooks even though the admission is
        # kept for enforcement and retry.
        state.lifecycle_generation += 1
        logger.warning(
            "%s: could not restore the narrowed surface at session end for %s; "
            "keeping its admission so enforcement and retry remain possible",
            PLUGIN_NAME,
            session_id,
        )
        return False


# Process-wide switch set by /router.  None = follow config.yaml (and the
# HERMES_TOKEN_ROUTER_ENABLED environment override).
_RUNTIME_OVERRIDE: bool | None = None


def _profile_name_for(agent: Any = None) -> str | None:
    """Owning profile of the AGENT, not of this process.

    Under a multiplexed gateway the process env holds the launch profile
    (default); the agent carries its own. Mirrors the host convention in
    run_agent.py (`agent._profile_name` first); placeholders such as "?"
    are ignored (#26).
    """
    return _agent_profile_name(agent)


def _config_enabled(cfg: Any, agent: Any = None) -> bool:
    name = _profile_name_for(agent)
    return bool(_is_router_active(cfg, name) and _get_profile_config(cfg, name).get("enabled", False))


def _router_enabled(cfg: Any, agent: Any = None) -> bool:
    """Precedence: HERMES_TOKEN_ROUTER_ENABLED=off (kill switch) > /router > config."""
    if _env_override() is False:
        return False
    runtime = _RUNTIME_OVERRIDE
    if runtime is not None:
        return runtime
    return _config_enabled(cfg, agent)


def _routing_settings(cfg: Any, agent: Any = None) -> dict[str, Any]:
    """Profile routing settings, independent of whether config enables routing."""
    return _get_profile_settings(cfg, _profile_name_for(agent))


_ROUTER_HELP = (
    "Usage: /router on | off | status | reset\n"
    "  off    stop routing now; a narrowed session gets its full tool list back on the next turn\n"
    "  on     route new sessions (and sessions not yet routed) from the next turn\n"
    "  reset  follow the plugin's config.yaml again\n"
    "  status show the current state\n"
    "Applies to this Hermes process until it restarts; config.yaml is not changed."
)


def router_command(raw_args: str = "") -> str:
    """Slash command handler: no model call, no core change."""
    global _RUNTIME_OVERRIDE
    arg = (raw_args or "").strip().lower()
    if arg in {"", "status"}:
        return _router_status_text()
    if arg in {"help", "-h", "--help"}:
        return _ROUTER_HELP
    if arg in {"off", "disable", "0", "false"}:
        _RUNTIME_OVERRIDE = False
        return (
            "Tool router OFF for this Hermes process. Sessions it narrowed get their full "
            "tool list back on the next turn. config.yaml is unchanged; /router reset to follow it."
        )
    if arg in {"on", "enable", "1", "true"}:
        if _env_override() is False:
            return (
                "Not turned on: HERMES_TOKEN_ROUTER_ENABLED is set to off for this process, "
                "and that kill switch takes precedence over /router."
            )
        _RUNTIME_OVERRIDE = True
        return (
            "Tool router ON for this Hermes process from the next turn (a session that was "
            "already restored to the full list stays full). config.yaml is unchanged."
        )
    if arg in {"reset", "default", "config"}:
        _RUNTIME_OVERRIDE = None
        return "Tool router follows config.yaml again. " + _router_status_text()
    return f"Unknown option: {arg}\n\n{_ROUTER_HELP}"


def _router_status_text() -> str:
    runtime = _RUNTIME_OVERRIDE  # one snapshot for the whole message
    try:
        cfg = _load_config()
        settings = _get_profile_settings(cfg)
        configured = bool(settings.get("enabled", False))
        profile = settings.get("_profile_name", "unknown")
        effective = _router_enabled(cfg) if runtime is _RUNTIME_OVERRIDE else bool(runtime)
    except Exception:  # noqa: BLE001 - status must never fail
        configured, profile, effective = False, "unknown", False
    env = _env_override()
    if env is False:
        source = "HERMES_TOKEN_ROUTER_ENABLED=off (kill switch; /router on cannot override it)"
    elif runtime is not None:
        source = f"/router {'on' if runtime else 'off'}"
    elif env is True:
        source = "HERMES_TOKEN_ROUTER_ENABLED=on together with config.yaml"
    else:
        source = "config.yaml"
    return (
        f"Tool router is {'ON' if effective else 'OFF'} for this process (decided by {source}). "
        f"config.yaml: {'on' if configured else 'off'} for profile {profile}; "
        f"environment: {'unset' if env is None else ('on' if env else 'off')}; "
        f"/router: {'not set' if runtime is None else ('on' if runtime else 'off')}."
    )


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
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("on_session_end", on_session_end)
    try:
        # /router on|off|status: handled by Hermes's slash-command dispatch,
        # never sent to the model.
        ctx.register_command(
            "router",
            handler=router_command,
            description="Turn the tool router on or off for this Hermes process (no model call).",
            args_hint="on|off|status",
        )
    except Exception as exc:  # noqa: BLE001 - older hosts lack register_command
        logger.debug("%s: /router command unavailable: %s", PLUGIN_NAME, exc)
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

_cfg = _load_config() if CONFIG_FILE.exists() else {}
_prof_cfg = _get_profile_config(_cfg)
logger.info("%s: plugin loaded profile=%s enabled=%s", PLUGIN_NAME, _prof_cfg.get("_profile_name"), _prof_cfg.get("enabled"))

# global.debug_logging: true raises THIS logger to DEBUG (every bail-out
# reason logs) and writes router-debug.log in the plugin directory without
# touching host logging, whose file handlers cap at INFO.
try:
    if isinstance(_cfg.get("global"), dict) and _cfg["global"].get("debug_logging"):
        import logging as _logging
        from logging.handlers import RotatingFileHandler as _RFH
        from pathlib import Path as _P
        logger.setLevel(_logging.DEBUG)
        if not any(getattr(h, "_htr_debug", False) for h in logger.handlers):
            _dbg_h = _RFH(_P(__file__).resolve().parent / "router-debug.log",
                          maxBytes=2_000_000, backupCount=2, encoding="utf-8")
            _dbg_h.setLevel(_logging.DEBUG)
            _dbg_h.setFormatter(_logging.Formatter(
                "%(asctime)s %(levelname)s [%(session_id)s] %(name)s: %(message)s",
                defaults={"session_id": ""}))
            _dbg_h._htr_debug = True
            logger.addHandler(_dbg_h)
except Exception as _dbg_exc:  # noqa: BLE001 - optional diagnostics never block plugin load
    logger.warning("%s: debug_logging unavailable: %s", PLUGIN_NAME, _dbg_exc)

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
        return _bind_admission(
            state,
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
    return _bind_admission(
        state,
        session_id=session_id,
        untouched_surface=MISSING,
        original_enabled_toolsets=MISSING,
        effective_policy=effective_policy,
        owner_snapshot=MISSING,
    )


def _bind_admission(state: RouterState, *, session_id: str, **prepared: Any) -> EnsureAdmissionResult:
    """Bind prepared admission atomically; a stale hook never reserves a slot.

    Owner snapshot and policy composition (registry work) are prepared by the
    caller outside the lock; only this binding step is serialized.
    """
    with fenced_section() as may_commit:
        if not may_commit:
            return result_for_status(
                "SESSION_MISMATCH", session_id, diagnostics=("STALE_LIFECYCLE",)
            )
        return state.ensure_host_admission(session_id=session_id, **prepared)


def _route_tool_surface(
    source: str,
    agent: Any = None,
    runtime_profile_name: Optional[str] = None,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Capture admission before empty-message exits, then route one surface."""
    # Hermes's abandonment clock is already running; everything below counts.
    route_started = time.monotonic()
    cfg = _load_config()
    if agent is None:
        # Same order as before for inline hooks (own stack, then the stored
        # session ref); current Hermes runs pre_llm_call on a timeout worker
        # thread, so finally search the other threads by exact session id.
        # Every candidate must carry the hook's exact session id.
        hook_session_id = str(kwargs.get("session_id") or "") or None
        if hook_session_id:
            agent = (
                _get_agent_from_stack(hook_session_id, scan_threads=False)
                or _get_validated_agent_ref(hook_session_id)
                or _find_agent_in_threads(hook_session_id)
            )
        else:
            agent = _get_agent_from_stack() or _get_agent_ref(None)
    elif kwargs.get("session_id") and not _agent_session_matches_id(
        agent, str(kwargs.get("session_id"))
    ):
        # An explicit agent from another session is never routed or cached.
        return None
    session_id = str(kwargs.get("session_id") or getattr(agent, "session_id", "") or "")
    if agent is not None:
        _store_agent_ref(agent, session_id)
    if agent is None or not session_id:
        logger.debug(
            "%s: %s source=%s session=%s (hook fired, no usable agent)",
            PLUGIN_NAME, "AGENT_LOOKUP_FAILED" if agent is None else "NO_SESSION_ID",
            source, session_id or "?")
        return None

    state = _get_router_state(agent)
    with _hook_fence(agent, session_id, deadline=route_started + _hook_commit_budget_s()) as fence:
        if not _retry_retired_restore(agent, state, session_id, fence):
            # A previous session on this agent could not be restored; keep the
            # surface and the guard as they are until that restore succeeds.
            return None
        return _route_fenced(source, agent, cfg, state, session_id, fence, kwargs)


def _route_fenced(
    source: str,
    agent: Any,
    cfg: Any,
    state: RouterState,
    session_id: str,
    fence: _HookFence,
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Route body; every mutation below is checked against the hook's fence."""
    if not _router_enabled(cfg, agent):
        # A disabled router must not touch a surface it never narrowed:
        # no capture, no restore (hosts add tools mid-session).  If it did
        # narrow earlier in this session, restore the exact envelope once.
        # The flag is cleared only by a successful restore (inside
        # restore_admitted_envelope), so a failed or impossible restore keeps
        # the protected-tool guard engaged and is retried next turn.
        if getattr(state, "surface_narrowed", False):
            bound = state.bound_host_admission(session_id)
            restored = False
            if (
                bound is not None
                and bound.status in {"READY", "SAFE_NO_PRUNE"}
                and bound.envelope is not None
            ):
                restored = bool(restore_admitted_envelope(agent, bound))
            with fenced_section() as may_commit:
                if (
                    may_commit
                    and not restored
                    and getattr(state, "_restore_warned_session", None) != session_id
                ):
                    state._restore_warned_session = session_id
                    logger.warning(
                        "%s: router disabled but the narrowed surface could not be restored "
                        "for session %s; will retry every turn",
                        PLUGIN_NAME,
                        session_id,
                    )
        _set_inactive(state)
        return None
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
    # Preparation (registry ownership, policy) runs unlocked; only the
    # binding is fenced, so a stale hook never reserves a session slot.
    admission = _ensure_host_admission(
        source=source,
        agent=agent,
        session_id=session_id,
        untouched_surface=untouched_surface,
        original_enabled_toolsets=original_enabled,
        hook_metadata=hook_metadata,
        agent_metadata=agent_metadata,
    )
    if "STALE_LIFECYCLE" in tuple(getattr(admission, "diagnostics", ()) or ()):
        return None
    fence.pin_admission()

    user_message = kwargs.get("user_message", "")
    turn_id = kwargs.get("turn_id") or getattr(agent, "_current_turn_id", "") or ""
    if not user_message:
        # Admission is intentionally bound before this return.  Do not run
        # classifier, available-toolset, retrieval, recovery, or retry work.
        return None
    if not _recovery_is_ready():
        # The target runtime must retain its current surface until both
        # recovery mechanisms are registered successfully.
        _set_inactive(state)
        if source == "pre_turn_context_build":
            _mark_turn_routed(agent, state, turn_id, source)
        return None
    if state.initial_route_applied:
        turn_id = kwargs.get("turn_id") or getattr(agent, "_current_turn_id", "") or ""
        with fenced_section() as may_commit:
            if not may_commit:
                return None
            # post_tool_call recovery may fire once per turn: reset the latch
            # only on a genuine turn transition (or when turns are unknown).
            if not turn_id or turn_id != getattr(state, "_retry_turn_id", None):
                state._retry_pending = False
                state._retry_turn_id = turn_id or None
            if turn_id:
                _mark_turn_routed(agent, state, turn_id, "sticky_surface")
        return None
    turn_id = kwargs.get("turn_id") or getattr(agent, "_current_turn_id", "") or ""
    if source != "pre_turn_context_build" and _was_turn_routed(agent, state, turn_id):
        return None

    if chinese_abstention_reason(user_message) == CHINESE_ABSTENTION_REASON:
        restore_admitted_envelope(agent, admission)
        if source == "pre_turn_context_build":
            _mark_turn_routed(agent, state, turn_id, source)
        return None

    # Enablement (config, environment, /router) was decided at the top of
    # this route; settings apply whether or not config.yaml enables routing.
    profile_cfg = _routing_settings(cfg, agent)
    if admission.status != "READY":
        # SAFE_NO_PRUNE restores only its already captured envelope.  All
        # other statuses preserve the current surface and deny additions.
        if admission.status == "SAFE_NO_PRUNE" and admission.envelope is not None:
            restore_admitted_envelope(agent, admission)
        _set_inactive(state)
        if source == "pre_turn_context_build":
            _mark_turn_routed(agent, state, turn_id, source)
        return None

    decline_chars = profile_cfg.get("long_message_decline_chars", 2000)
    short_bypass_chars = profile_cfg.get("short_message_bypass_chars", 0)
    # Same default as the shipped config.yaml: no floor unless configured.
    floor_toolsets = set(profile_cfg.get("floor_toolsets", []) or [])
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
                extra_body=_get_classifier_extra_body(profile_cfg),
            )
        except Exception:
            predicted = None
    if predicted is None:
        restore_admitted_envelope(agent, admission)
        if source == "pre_turn_context_build":
            _mark_turn_routed(agent, state, turn_id, source)
        return None

    # Selection reads the registry unlocked; the fence is re-checked under
    # the lock immediately before the coordinated assignment.
    result = select_and_commit_route(
        agent,
        admission,
        predicted_toolsets=set(predicted),
        floor_toolsets=floor_toolsets,
        available_toolsets=available,
        caller="route",
    )
    with fenced_section() as may_commit:
        if not may_commit:
            # Late or stale hook (abandoned by a Hermes hook timeout, session
            # ended or agent reused meanwhile): nothing was committed.
            logger.warning(
                "%s: discarded a stale route for session %s", PLUGIN_NAME, session_id
            )
            return None
        state.last_expansion_result = result
        state._retry_turn_id = turn_id or None
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


def pre_tool_call(**kwargs: Any) -> dict[str, Any] | None:
    """Keep protected capability inside the envelope behind Hermes tool search.

    Hermes unwraps ``tool_call`` before this hook, so ``tool_name`` is the
    real deferred tool.  The bridge authorizes any deferrable tool whose
    toolset is enabled; a protected toolset can stay enabled because some of
    its tools were captured (and are visible), which would expose uncaptured
    siblings.  While this router has the surface narrowed, block a tool that
    is not visible, not a captured definition, and owned by a protected
    toolset.  Everything else proceeds untouched.
    """
    tool_name = str(kwargs.get("tool_name") or "")
    session_id = str(kwargs.get("session_id") or "")
    if not tool_name or not session_id:
        return None
    try:
        agent = _resolve_hook_agent(kwargs.get("agent"), session_id)
        if agent is None:
            return None
        state = _get_router_state(agent)
        if not getattr(state, "surface_narrowed", False):
            return None
        if tool_name in set(getattr(agent, "valid_tool_names", ()) or ()):
            return None
        admission = state.bound_host_admission(session_id)
        if admission is None or admission.envelope is None or admission.effective_policy is None:
            # Agent reused after a failed or expired teardown: enforce the
            # retired (or still-bound, orphaned) policy.
            retired = getattr(state, "_retired_admission", None) or _orphaned_lifecycle(
                state, session_id
            )
            admission = retired[1] if retired is not None else None
        if admission is None or admission.envelope is None or admission.effective_policy is None:
            return None
        protected = admission.effective_policy.protected_toolsets
        if not protected:
            return None
        if any(item.name == tool_name for item in admission.envelope.definitions):
            return None
        from tools.registry import registry
        entry = registry.get_entry(tool_name)
        owner = getattr(entry, "toolset", None) if entry is not None else None
    except Exception as exc:  # noqa: BLE001 - never break unrelated tool calls
        logger.debug("%s: pre_tool_call guard skipped: %s", PLUGIN_NAME, exc)
        return None
    if owner in protected:
        return {
            "action": "block",
            "message": (
                f"'{tool_name}' belongs to the protected toolset '{owner}' and was not "
                "admitted for this session."
            ),
        }
    return None


# Hermes abandons (but does not cancel) timeout-bounded hook callbacks after
# plugins.hook_callback_timeout (default 30 s, <= 0 disables the timeout, max
# 600 s).  A hook that has not committed within half of that never commits.
_DEFAULT_HOOK_CALLBACK_TIMEOUT_S = 30.0
_MAX_HOOK_CALLBACK_TIMEOUT_S = 600.0


def _hook_callback_timeout_s() -> float:
    """Hermes's own effective timeout; the same normalization as a fallback."""
    try:
        from hermes_cli.plugins import _resolve_hook_callback_timeout

        return float(_resolve_hook_callback_timeout())
    except Exception as exc:  # noqa: BLE001 - standalone/older hosts normalize locally
        logger.debug("%s: Hermes timeout resolver unavailable (%s)", PLUGIN_NAME, exc)
    timeout = _DEFAULT_HOOK_CALLBACK_TIMEOUT_S
    try:
        from hermes_cli.config import load_config_readonly

        plugins_cfg = (load_config_readonly() or {}).get("plugins")
        if isinstance(plugins_cfg, dict) and plugins_cfg.get("hook_callback_timeout") is not None:
            configured = float(plugins_cfg["hook_callback_timeout"])
            if configured >= 0:
                timeout = min(configured, _MAX_HOOK_CALLBACK_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s: hook_callback_timeout unavailable (%s); using %ss", PLUGIN_NAME, exc, timeout)
    return timeout


def _hook_commit_budget_s() -> float:
    """Seconds after hook start during which a commit is still allowed."""
    timeout = _hook_callback_timeout_s()
    if not math.isfinite(timeout) or timeout <= 0:
        # Hermes runs the hook synchronously and never abandons it.
        return math.inf
    return timeout * 0.5


def _orphaned_lifecycle(state: RouterState, session_id: str) -> tuple[str, Any] | None:
    """A narrowed surface still bound to another session's admission."""
    bound_session = getattr(state, "_admission_session_id", None)
    bound = getattr(state, "_bound_admission_result", None)
    if (
        getattr(state, "surface_narrowed", False)
        and bound_session is not None
        and bound_session != session_id
        and bound is not None
    ):
        return (bound_session, bound)
    return None


def _retry_retired_restore(
    agent: Any, state: RouterState, session_id: str, fence: _HookFence | None = None
) -> bool:
    """Release an agent left narrowed by a failed session-end restore.

    Returns True when the agent is free for ``session_id``.  Until then it is
    quarantined: no capture, no routing, and pre_tool_call keeps the retired
    policy.  Runs under the caller's fence (lifecycle and deadline).

    Handoff rule (fresh-agent recovery): the quarantine is released only by
    a successful restore of the retired envelope, and that restore is only
    attempted while the agent shows exactly the surface this router last
    committed (every definition and enabled_toolsets).  Any other change is
    ambiguous (a partial addition is not a handoff), so the agent stays
    quarantined; recovery is to discard that agent, a fresh one is unaffected.
    """
    if getattr(state, "_retired_admission", None) is None:
        orphan = _orphaned_lifecycle(state, session_id)
        if orphan is None:
            return True
        with fenced_section() as may_commit:
            if not may_commit:
                return False
            if getattr(state, "_retired_admission", None) is None and _orphaned_lifecycle(
                state, session_id
            ) == orphan:
                # A teardown expired before recording retirement: adopt the
                # still-narrowed lifecycle as quarantine.
                state._retired_admission = orphan
    retired = getattr(state, "_retired_admission", None)
    if retired is None:
        return True
    retired_session, retired_admission = retired
    if retired_session == session_id:
        return True
    with fenced_section() as may_commit:
        if not may_commit:
            return False
        if getattr(state, "_retired_admission", None) is not retired:
            return True
        if surface_is_router_committed(agent, state) and restore_admitted_envelope(
            agent, retired_admission
        ):
            state.end_admission(retired_session)
            state.reset()
            if fence is not None:
                fence.rebase()
            return True
        if getattr(state, "_quarantine_warned_session", None) != session_id:
            state._quarantine_warned_session = session_id
            logger.warning(
                "%s: agent reused for session %s while session %s's narrowed surface is "
                "still unrestored; routing is paused and protected tools stay guarded. "
                "Recovery: discard this agent (a fresh agent is unaffected).",
                PLUGIN_NAME,
                session_id,
                retired_session,
            )
    return False


def _resolve_hook_agent(explicit: Any, session_id: Any) -> Any:
    """Explicit agent first, then validated stored ref, then stack/threads.

    Every candidate must carry the call's session id when one is given.
    """
    sid = str(session_id or "")
    if explicit is not None and (not sid or _agent_session_matches_id(explicit, sid)):
        return explicit
    if sid:
        return _get_validated_agent_ref(sid) or _get_agent_from_stack(sid)
    return _get_agent_ref(None) or _get_agent_from_stack()


def _agent_session_matches_id(agent: Any, session_id: str) -> bool:
    try:
        return str(getattr(agent, "session_id", "") or "") == session_id
    except Exception:  # noqa: BLE001 - foreign objects
        return False


def _normalize_toolset_name(name: str, available: set[str]) -> str:
    """Exact name, else the unique case-insensitive match, else lower-case.

    Ambiguous case-insensitive matches (``Team`` and ``team``) are returned
    unchanged so validation rejects them instead of picking one at random.
    """
    if name in available:
        return name
    matches = sorted(candidate for candidate in available if candidate.lower() == name.lower())
    if len(matches) == 1:
        return matches[0]
    if matches:
        return name
    return name.lower()


def request_toolset_handler(args: Dict[str, Any], **kwargs: Any) -> str:
    """Return a truthful result for one guarded expansion request."""
    session_id = str(kwargs.get("session_id") or "")
    agent = _resolve_hook_agent(kwargs.get("agent"), session_id)
    if agent is None:
        return _request_toolset(args, **kwargs)
    sid = session_id or str(getattr(agent, "session_id", "") or "")
    with _hook_fence(agent, sid):
        return _request_toolset(args, **kwargs)


def _request_toolset(args: dict[str, Any], **kwargs: Any) -> str:
    args = args if isinstance(args, dict) else {}
    requested_toolset = str(args.get("toolset") or args.get("toolset_name") or "").strip()
    raw_toolsets = args.get("toolsets") or []
    requested_toolsets = {
        str(name).strip()
        for name in raw_toolsets
        if isinstance(raw_toolsets, list) and str(name).strip()
    }
    if requested_toolset:
        requested_toolsets.add(requested_toolset)
    requested_tool = str(args.get("tool_name") or "").strip()
    reason = str(args.get("reason") or "").strip()[:200]
    session_id = str(kwargs.get("session_id") or "")
    agent = _resolve_hook_agent(kwargs.get("agent"), session_id)
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
        # Exact registered names first (mixed-case custom/MCP toolsets), then
        # a case-insensitive match, then the lower-cased name for aliases.
        requested_toolsets = {_normalize_toolset_name(name, available) for name in requested_toolsets}
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
    # The automatic post-tool retry latch is per turn; an explicit request
    # does not reset it.
    ok = result.ok
    outcome = result.reason
    if "STALE_LIFECYCLE" in tuple(getattr(admission, "diagnostics", ()) or ()):
        outcome = "STALE_LIFECYCLE"
    denied_tool_names = list(result.denied_tool_names)
    if requested_tool and requested_tool not in (getattr(agent, "valid_tool_names", set()) or set()):
        # An exact tool that is still not installed is never a success.
        ok = False
        if requested_tool not in denied_tool_names:
            denied_tool_names.append(requested_tool)
        if outcome in {"NO_MUTATION", "READY", "EXPANDED"}:
            outcome = "UNRESOLVED_TOOL"
    payload = {
        "ok": ok,
        "toolsets": list(result.requested_toolsets),
        "requested_toolsets": list(result.requested_toolsets),
        "expanded_toolsets": list(result.expanded_toolsets),
        "denied_toolsets": list(result.denied_toolsets),
        "denied_tool_names": sorted(denied_tool_names),
        "added_tool_names": list(result.added_tool_names),
        "installed_tool_names": list(result.installed_tool_names),
        "requested_tool": requested_tool,
        # Machine outcome; the caller's own justification is kept separately.
        "reason": outcome,
        "request_reason": reason,
        "retry_allowed": result.retry_allowed,
        "enabled_tools": sorted(getattr(agent, "valid_tool_names", set()) or set()),
    }
    if len(result.requested_toolsets) == 1:
        payload["toolset"] = result.requested_toolsets[0]
    return json.dumps(payload, sort_keys=True)


def _disabled_router_must_not_touch(agent: Any) -> bool:
    """A disabled router never mutates a surface it did not narrow.

    Recovery hooks (tool_request middleware, post_tool_call) only exist to undo
    this router's own narrowing; on an unrouted session they would rebuild the
    host's tool surface from the router's partial owner map.
    """
    try:
        if _router_enabled(_load_config(), agent):
            return False
    except Exception:  # noqa: BLE001 - fail closed: an unreadable config never mutates
        return True
    return not getattr(_get_router_state(agent), "active", False)


def tool_request_middleware(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Admit a missing exact tool before normal Hermes dispatch validation."""
    session_id = str(kwargs.get("session_id") or "")
    agent = _resolve_hook_agent(kwargs.get("agent"), session_id)
    tool_name = str(kwargs.get("tool_name") or "").strip()
    args = kwargs.get("args")
    if agent is None or not tool_name or not isinstance(args, dict):
        return None
    session_id = session_id or str(getattr(agent, "session_id", "") or "")
    if tool_name in (getattr(agent, "valid_tool_names", set()) or set()):
        return None
    if _disabled_router_must_not_touch(agent):
        return None
    with _hook_fence(agent, session_id) as fence:
        admission = _call_admission_for_current(
            "middleware", agent, session_id, kwargs
        )
        fence.pin_admission()
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
    agent = _resolve_hook_agent(kwargs.get("agent"), session_id)
    tool_name = str(kwargs.get("tool_name") or "").strip()
    if agent is None or not tool_name:
        return None
    session_id = session_id or str(getattr(agent, "session_id", "") or "")
    state = _get_router_state(agent)
    if state._retry_pending or tool_name in (getattr(agent, "valid_tool_names", set()) or set()):
        return None
    if _disabled_router_must_not_touch(agent):
        return None
    started = time.monotonic()
    # One automatic recovery per turn: an overlapping callback that finds the
    # latch taken at its commit boundary does not commit.
    with _hook_fence(
        agent, session_id, deadline=started + _hook_commit_budget_s(), require_retry_available=True
    ) as fence:
        admission = _call_admission_for_current(
            "post_tool", agent, session_id, kwargs
        )
        fence.pin_admission()
        result = expand_admitted_toolsets(
            agent,
            admission,
            requested_toolsets=set(),
            requested_tool_names={tool_name},
            caller="post_tool",
            session_id=session_id,
            claim_retry_for=tool_name,
        )
    del result
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
