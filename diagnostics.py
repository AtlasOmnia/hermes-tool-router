"""Local-only compatibility and readiness diagnostics."""

from __future__ import annotations

import json
from typing import Iterable

try:
    from .compat import CompatibilityMode, detect_compatibility
except ImportError:  # pragma: no cover
    from compat import CompatibilityMode, detect_compatibility


def build_report(
    *,
    hooks: Iterable[str],
    middleware: Iterable[str] = (),
    toolsets: Iterable[str],
    profile_name: str,
    enabled: bool,
) -> dict:
    hook_set = set(hooks)
    middleware_set = set(middleware)
    mode = detect_compatibility(hook_set)
    early = mode in {CompatibilityMode.PRODUCTION, CompatibilityMode.REDUCED_SAFETY}
    routes_before_request = early or mode is CompatibilityMode.LATE_COMPATIBILITY
    return {
        "profile": profile_name,
        "enabled": bool(enabled),
        "compatibility_mode": mode.value,
        "first_turn_savings_available": bool(enabled and routes_before_request),
        "preflight_routing_available": bool(enabled and early),
        "automatic_recovery_available": (
            "on_unavailable_tool_call" in hook_set or "tool_request" in middleware_set
        ),
        "registered_toolset_count": len(set(toolsets)),
        "hooks": sorted(hook_set),
        "middleware": sorted(middleware_set),
    }


def inspect_live_runtime(profile_name: str = "default", enabled: bool = True) -> dict:
    try:
        from hermes_cli.plugins import VALID_HOOKS
    except Exception:
        VALID_HOOKS = set()
    try:
        from hermes_cli.middleware import VALID_MIDDLEWARE
    except Exception:
        VALID_MIDDLEWARE = set()
    try:
        import model_tools  # noqa: F401 - triggers built-in tool registration
        from tools.registry import registry
        toolsets = set(registry.get_registered_toolset_names())
    except Exception:
        toolsets = set()
    return build_report(
        hooks=VALID_HOOKS,
        middleware=VALID_MIDDLEWARE,
        toolsets=toolsets,
        profile_name=profile_name,
        enabled=enabled,
    )


def main() -> int:
    report = inspect_live_runtime()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["first_turn_savings_available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
