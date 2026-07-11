from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "hermes_tool_router_recovery_schema_test"
spec = importlib.util.spec_from_file_location(PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[PKG] = module
spec.loader.exec_module(module)

from hermes_tool_router_recovery_schema_test.tools import build_recovery_tool_schema


def test_recovery_schema_accepts_multiple_live_validated_requests_without_stale_enum():
    schema = build_recovery_tool_schema({"web", "synthetic"})
    props = schema["parameters"]["properties"]
    assert props["toolsets"]["items"] == {"type": "string"}
    assert props["toolsets"]["minItems"] == 1
    assert "toolset" not in props
