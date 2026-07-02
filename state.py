"""Agent-scoped router state for hermes-token-router."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

try:
    from .config import PLUGIN_NAME
except ImportError:  # pragma: no cover - direct loader fallback
    from config import PLUGIN_NAME

logger = logging.getLogger(__name__)

_agent_ref: Any = None

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

def _store_agent_ref(agent: Any) -> None:
    """Store the agent reference globally for subsequent hook calls."""
    global _agent_ref
    _agent_ref = agent


def _get_agent_ref() -> Any:
    """Return the best-effort cached live agent reference."""
    return _agent_ref
