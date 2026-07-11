from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "hermes_tool_router_compat_test"
spec = importlib.util.spec_from_file_location(PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[PKG] = module
spec.loader.exec_module(module)

from hermes_tool_router_compat_test.compat import CompatibilityMode, detect_compatibility


def test_compatibility_modes_are_explicit():
    assert detect_compatibility({"pre_tool_surface_build", "on_unavailable_tool_call"}) is CompatibilityMode.PRODUCTION
    assert detect_compatibility({"pre_turn_context_build"}) is CompatibilityMode.REDUCED_SAFETY
    assert detect_compatibility({"pre_llm_call"}) is CompatibilityMode.LATE_COMPATIBILITY
    assert detect_compatibility(set()) is CompatibilityMode.UNSUPPORTED_FOR_TOKEN_SAVINGS
