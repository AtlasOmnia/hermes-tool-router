from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "hermes_tool_router_registry_test"
spec = importlib.util.spec_from_file_location(PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[PKG] = module
spec.loader.exec_module(module)

from hermes_tool_router_registry_test.registry_adapter import RegistryAdapter


class Entry:
    def __init__(self, toolset):
        self.toolset = toolset


class Registry:
    def get_registered_toolset_names(self):
        return ["web", "synthetic"]

    def get_tool_names_for_toolset(self, name):
        return {"web": ["web_search"], "synthetic": ["synthetic_run"]}.get(name, [])

    def get_entry(self, name):
        return Entry("synthetic") if name == "synthetic_run" else None

    def get_definitions(self, names, quiet=True):
        return [{"type": "function", "function": {"name": name}} for name in sorted(names)]

    def get_toolset_alias_target(self, name):
        return "web" if name == "search" else None


def test_registry_adapter_discovers_dynamic_toolsets_without_private_apis():
    adapter = RegistryAdapter(Registry())
    assert adapter.available_toolsets() == {"web", "synthetic"}
    assert adapter.tools_for_toolset("synthetic") == {"synthetic_run"}
    assert adapter.toolset_for_tool("synthetic_run") == "synthetic"
    assert adapter.resolve_alias("search") == "web"
