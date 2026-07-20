from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
PKG = "hermes_tool_router_tools_test"
spec = importlib.util.spec_from_file_location(
    PKG,
    ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[PKG] = module
spec.loader.exec_module(module)

from hermes_tool_router_tools_test.tools import _resolve_toolset_to_tool_names


class Registry:
    def get_tool_names_for_toolset(self, name):
        assert name == "terminal"
        return ["close_terminal", "process", "read_terminal", "terminal"]

    def get_definitions(self, names, quiet=True):
        assert quiet is True
        return [
            {"type": "function", "function": {"name": name}}
            for name in sorted(names & {"process", "terminal"})
        ]


def test_toolset_resolution_drops_names_without_definitions(monkeypatch):
    registry_module = ModuleType("tools.registry")
    registry_module.registry = Registry()
    tools_package = ModuleType("tools")
    tools_package.registry = registry_module
    monkeypatch.setitem(sys.modules, "tools", tools_package)
    monkeypatch.setitem(sys.modules, "tools.registry", registry_module)

    assert _resolve_toolset_to_tool_names({"terminal"}) == {"process", "terminal"}
