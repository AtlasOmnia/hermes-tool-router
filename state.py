"""Agent-scoped router state for hermes-token-router."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

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
                return _router_state
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
