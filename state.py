"""Agent-scoped router state for hermes-token-router."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

try:
    from .capabilities import (
        MISSING,
        EnsureAdmissionResult,
        NoAuthorityContamination,
        OwnerSnapshot,
        TrustedHostPolicy,
        build_host_admission_envelope,
        result_for_status,
    )
except ImportError:  # pragma: no cover - direct loader fallback
    from capabilities import (
        MISSING,
        EnsureAdmissionResult,
        NoAuthorityContamination,
        OwnerSnapshot,
        TrustedHostPolicy,
        build_host_admission_envelope,
        result_for_status,
    )

logger = logging.getLogger(__name__)

_agent_ref: Any = None
_agent_refs: Dict[str, Any] = {}

def _get_agent_from_stack() -> Any:
    """Walk the call stack to find the agent object.
    
    The pre_llm_call hook is invoked from build_turn_context() in agent/turn_context.py,
    which receives 'agent' as its first parameter. We walk up the stack frames to find it.
    
    This avoids any core Hermes modifications — pure Python introspection, update-safe.
    """
    import sys
    frame = sys._getframe()
    try:
        # Walk up the stack looking for a frame with an 'agent' local
        while frame is not None:
            agent = frame.f_locals.get("agent")
            if agent is not None and hasattr(agent, "tools"):
                return agent
            frame = frame.f_back
    finally:
        del frame  # prevent reference cycles
    return None

class RouterState:
    """Per-conversation state for the tool router."""

    def __init__(self):
        self.active: bool = False
        self.predicted_toolsets: Optional[Set[str]] = None
        self.router_model: Optional[str] = None
        self.full_toolsets: Set[str] = set()
        self.config: dict = {}
        self._fallback_triggered: bool = False
        self._full_tool_defs: Optional[List[Dict[str, Any]]] = None
        self._full_tool_names: Set[str] = set()
        self._retry_pending: bool = False
        self.routed_turn_id: Optional[str] = None
        self.routed_source: Optional[str] = None
        self.initial_route_applied: bool = False
        self.initial_toolsets: Set[str] = set()
        self.active_toolsets: Set[str] = set()
        self.registry_generation: int = 0
        self.expansion_count: int = 0
        self.installed_tool_names: Set[str] = set()
        self.admitted_toolsets: Set[str] = set()
        self.denied_toolsets: Set[str] = set()
        self.denied_tool_names: Set[str] = set()
        self.last_expansion_result: Any = None
        # Capability lifecycle is intentionally attached to this RouterState.
        self._admission_session_id: Optional[str] = None
        self._bound_admission_result: Optional[EnsureAdmissionResult] = None
        self._contamination_marker: Optional[NoAuthorityContamination] = None

    def bound_host_admission(self, session_id: str) -> Optional[EnsureAdmissionResult]:
        """Return the one bound result, or a session-mismatch result."""
        if self._admission_session_id is not None and self._admission_session_id != session_id:
            return result_for_status("SESSION_MISMATCH", session_id)
        return self._bound_admission_result

    def no_authority_contamination(
        self, session_id: str
    ) -> Optional[NoAuthorityContamination]:
        """Return the immutable first-append marker for this session."""
        if self._admission_session_id != session_id:
            return None
        return self._contamination_marker

    def mark_no_authority_contaminated(
        self,
        *,
        session_id: str,
        first_commit_caller: str,
        first_added_tool_names: tuple[str, ...],
        first_added_toolsets: tuple[str, ...],
    ) -> NoAuthorityContamination:
        """Persist and return the first ordinary-append marker before mutation."""
        if self._admission_session_id not in (None, session_id):
            raise RuntimeError("SESSION_MISMATCH")
        if self._bound_admission_result is not None:
            raise RuntimeError("ADMISSION_ALREADY_BOUND")
        if self._contamination_marker is not None:
            return self._contamination_marker
        if first_commit_caller not in {"route", "visible_request", "middleware", "post_tool"}:
            raise ValueError("INVALID_CONTAMINATION_CALLER")
        if not first_added_tool_names or not first_added_toolsets:
            raise ValueError("EMPTY_CONTAMINATION_MARKER")
        marker = NoAuthorityContamination(
            session_id=session_id,
            first_commit_caller=first_commit_caller,  # type: ignore[arg-type]
            first_added_tool_names=tuple(first_added_tool_names),
            first_added_toolsets=tuple(first_added_toolsets),
        )
        self._admission_session_id = session_id
        self._contamination_marker = marker
        return marker

    def ensure_host_admission(
        self,
        *,
        session_id: str,
        untouched_surface: Any = MISSING,
        original_enabled_toolsets: Any = MISSING,
        effective_policy: Optional[TrustedHostPolicy] = None,
        owner_snapshot: Any = MISSING,
    ) -> EnsureAdmissionResult:
        """Bind one authoritative result or return transient no-authority state."""
        if self._admission_session_id is not None and self._admission_session_id != session_id:
            return result_for_status("SESSION_MISMATCH", session_id)
        if self._bound_admission_result is not None:
            return self._bound_admission_result
        self._admission_session_id = session_id

        contamination = self._contamination_marker
        if effective_policy is None:
            effective_policy = TrustedHostPolicy(frozenset(), frozenset(), (), False, ("MISSING_POLICY",))

        no_envelope = (
            untouched_surface is MISSING
            or original_enabled_toolsets is MISSING
            or owner_snapshot is MISSING
        )
        if no_envelope:
            if not effective_policy.valid:
                return result_for_status(
                    "NO_AUTHORITY_INVALID_POLICY",
                    session_id,
                    effective_policy=effective_policy,
                    contamination=contamination,
                    diagnostics=effective_policy.errors,
                )
            if contamination is not None:
                return result_for_status(
                    "NO_AUTHORITY_CONTAMINATED",
                    session_id,
                    effective_policy=effective_policy,
                    contamination=contamination,
                    diagnostics=("ORDINARY_APPEND_BEFORE_HOST_CAPTURE",),
                )
            return result_for_status(
                "NO_AUTHORITY",
                session_id,
                effective_policy=effective_policy,
            )

        if not isinstance(owner_snapshot, OwnerSnapshot):
            result = result_for_status(
                "CAPTURE_INVALID_NO_MUTATION",
                session_id,
                diagnostics=("MALFORMED_OWNER_SNAPSHOT",),
            )
            self._bound_admission_result = result
            return result

        envelope, diagnostics = build_host_admission_envelope(
            session_id=session_id,
            untouched_surface=untouched_surface,
            original_enabled_toolsets=original_enabled_toolsets,
            effective_policy=effective_policy,
            owner_snapshot=owner_snapshot,
        )
        if envelope is None:
            result = result_for_status(
                "CAPTURE_INVALID_NO_MUTATION",
                session_id,
                diagnostics=tuple(diagnostics),
            )
            self._bound_admission_result = result
            return result

        status = "READY"
        if not effective_policy.valid or effective_policy.preserve_input_surface:
            status = "SAFE_NO_PRUNE"
        result = result_for_status(
            status,  # type: ignore[arg-type]
            session_id,
            envelope=envelope,
            effective_policy=effective_policy,
            owner_snapshot=owner_snapshot,
            diagnostics=tuple(diagnostics) + effective_policy.errors,
        )
        self._bound_admission_result = result
        return result

    def end_admission(self, session_id: str) -> bool:
        """Clear only the matching session's capability lifecycle slot."""
        if self._admission_session_id != session_id:
            return False
        self._admission_session_id = None
        self._bound_admission_result = None
        self._contamination_marker = None
        return True

    def set_initial_surface(self, toolsets: Set[str]) -> bool:
        """Set the routed surface once; later calls cannot shrink or replace it."""
        if self.initial_route_applied:
            return False
        selected = set(toolsets)
        self.initial_route_applied = True
        self.initial_toolsets = selected
        self.active_toolsets = set(selected)
        return True

    def expand_surface(self, toolsets: Set[str]) -> Set[str]:
        """Monotonically add toolsets and return the names newly activated."""
        additions = set(toolsets) - self.active_toolsets
        if additions:
            self.active_toolsets.update(additions)
            self.expansion_count += 1
        return additions

    def reset(self):
        self.active = False
        self.predicted_toolsets = None
        self.router_model = None
        self.full_toolsets = set()
        self.config = {}
        self._fallback_triggered = False
        self._full_tool_defs = None
        self._full_tool_names = set()
        self._retry_pending = False
        self.routed_turn_id = None
        self.routed_source = None
        self.initial_route_applied = False
        self.initial_toolsets = set()
        self.active_toolsets = set()
        self.registry_generation = 0
        self.expansion_count = 0
        self.installed_tool_names = set()
        self.admitted_toolsets = set()
        self.denied_toolsets = set()
        self.denied_tool_names = set()
        self.last_expansion_result = None

    def capability_snapshot(self) -> dict[str, Any]:
        """Return a shallow snapshot of coordinated router fields."""
        return {
            name: getattr(self, name)
            for name in (
                "active",
                "predicted_toolsets",
                "router_model",
                "full_toolsets",
                "config",
                "_fallback_triggered",
                "_full_tool_defs",
                "_full_tool_names",
                "_retry_pending",
                "routed_turn_id",
                "routed_source",
                "initial_route_applied",
                "initial_toolsets",
                "active_toolsets",
                "registry_generation",
                "expansion_count",
                "installed_tool_names",
                "admitted_toolsets",
                "denied_toolsets",
                "denied_tool_names",
                "last_expansion_result",
            )
        }

    def restore_capability_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore a snapshot without recalculating or clearing admission slots."""
        for name, value in snapshot.items():
            setattr(self, name, value)

    def admission_slots(self) -> tuple[object, object, object]:
        """Expose lifecycle slots for transaction readback tests."""
        return (
            self._admission_session_id,
            self._bound_admission_result,
            self._contamination_marker,
        )

    def restore_admission_slots(self, slots: tuple[object, object, object]) -> None:
        """Restore lifecycle slots exactly after a failed transaction."""
        self._admission_session_id, self._bound_admission_result, self._contamination_marker = slots


_router_state = RouterState()

ROUTER_STATE_ATTR = "_hermes_token_router_state"

def _get_router_state(agent: Any = None) -> RouterState:
    """Return router state scoped to the live Hermes agent when possible."""
    if agent is not None:
        state = getattr(agent, ROUTER_STATE_ATTR, None)
        if not isinstance(state, RouterState):
            state = RouterState()
            try:
                setattr(agent, ROUTER_STATE_ATTR, state)
            except Exception:
                # A non-attachable agent must never borrow process-global
                # capability state.  The transient object is intentionally
                # unusable as a persistent lifecycle guard.
                return RouterState()
        return state
    if _agent_ref is not None:
        return _get_router_state(_agent_ref)
    return _router_state
def _store_agent_ref(agent: Any, session_id: Optional[str] = None) -> None:
    """Store a live agent under its session identity for concurrent safety."""
    global _agent_ref
    _agent_ref = agent
    key = session_id or getattr(agent, "session_id", None)
    if key:
        _agent_refs[str(key)] = agent


def _drop_agent_ref(session_id: str) -> None:
    """Release a finished session's cached agent reference."""
    global _agent_ref
    removed = _agent_refs.pop(str(session_id), None)
    if removed is _agent_ref:
        _agent_ref = None


def _get_agent_ref(session_id: Optional[str] = None) -> Any:
    """Resolve by session; ambiguous global lookup deliberately returns None."""
    if session_id:
        return _agent_refs.get(str(session_id))
    unique = {id(agent): agent for agent in _agent_refs.values()}
    if len(unique) == 1:
        return next(iter(unique.values()))
    if not unique:
        return _agent_ref
    return None
