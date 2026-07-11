from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "hermes_tool_router_state_test"
spec = importlib.util.spec_from_file_location(PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[PKG] = module
spec.loader.exec_module(module)

from hermes_tool_router_state_test.state import RouterState


def test_session_surface_can_only_expand():
    state = RouterState()
    state.set_initial_surface({"web"})
    state.expand_surface({"browser"})
    state.expand_surface({"web"})
    assert state.active_toolsets == {"web", "browser"}
    assert state.expansion_count == 1


def test_initial_surface_is_applied_only_once():
    state = RouterState()
    assert state.set_initial_surface({"web"}) is True
    assert state.set_initial_surface({"file"}) is False
    assert state.active_toolsets == {"web"}


def test_state_is_agent_scoped():
    class Agent:
        pass

    from hermes_tool_router_state_test.state import _get_router_state

    first, second = Agent(), Agent()
    _get_router_state(first).set_initial_surface({"web"})
    assert _get_router_state(second).active_toolsets == set()


def test_agent_lookup_requires_session_when_multiple_agents_are_live():
    class Agent:
        pass

    from hermes_tool_router_state_test.state import _drop_agent_ref, _get_agent_ref, _store_agent_ref

    first, second = Agent(), Agent()
    _store_agent_ref(first, "session-a")
    _store_agent_ref(second, "session-b")
    assert _get_agent_ref("session-a") is first
    assert _get_agent_ref("session-b") is second
    assert _get_agent_ref() is None
    _drop_agent_ref("session-a")
    assert _get_agent_ref("session-a") is None
