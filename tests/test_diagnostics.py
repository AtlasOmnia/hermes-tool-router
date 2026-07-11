from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "hermes_tool_router_diagnostics_test"
spec = importlib.util.spec_from_file_location(PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[PKG] = module
spec.loader.exec_module(module)

from hermes_tool_router_diagnostics_test.compat import CompatibilityMode
from hermes_tool_router_diagnostics_test.diagnostics import build_report


def test_diagnostic_report_reports_late_hook_savings_with_middleware_recovery():
    report = build_report(
        hooks={"pre_llm_call"},
        middleware={"tool_request"},
        toolsets={"web", "file"},
        profile_name="default",
        enabled=True,
    )
    assert report["compatibility_mode"] == CompatibilityMode.LATE_COMPATIBILITY.value
    assert report["first_turn_savings_available"] is True
    assert report["preflight_routing_available"] is False
    assert report["automatic_recovery_available"] is True
