"""User-owned Hermes toolset routing policy.

This file is loaded by ``agent.toolset_policy`` when present.  Keeping the
policy here means Hermes updates only need to preserve the small core seam,
while the routing rules, floors, and thresholds remain in ~/.hermes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("run_agent")


def _dedupe(names: List[str]) -> List[str]:
    result: List[str] = []
    for name in names:
        if name and name not in result:
            result.append(name)
    return result


def resolve_toolsets(
    *,
    user_message: Optional[str],
    router_config: Dict[str, Any],
    available_toolsets: Dict[str, str],
    platform_toolsets: Optional[List[str]],
    enabled_toolsets_explicit: bool,
    disabled_toolsets: Optional[List[str]],
    safe_core_toolsets: List[str],
    skill_toolsets: List[str],
    platform: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve effective toolsets before Hermes assembles tool schemas."""
    disabled = set(disabled_toolsets or [])
    # Keep the default reduced surface as small as the classifier allows.
    # Core Hermes still passes a safe floor for conservative built-in behavior,
    # but this user-owned policy intentionally avoids adding web to every
    # terminal/file-only first turn.
    safe_core: List[str] = []
    skills = [
        name for name in skill_toolsets
        if name in available_toolsets and name not in disabled
    ]

    effective = list(platform_toolsets) if platform_toolsets is not None else None
    router_predicted: Optional[List[str]] = None

    def filter_platform_unsupported(names: List[str]) -> List[str]:
        if platform == "api_server":
            return [name for name in names if name != "clarify"]
        return names

    if router_config.get("enabled") and user_message:
        try:
            from agent.toolset_router import predict_toolsets

            predicted, should_reduce = predict_toolsets(
                user_message,
                router_config,
                available_toolsets,
            )
        except Exception as exc:
            logger.warning("hermes-token-router: classifier failed: %s", exc)
            predicted, should_reduce = [], False

        if should_reduce:
            wanted = _dedupe(safe_core + list(predicted or []) + skills)
            if enabled_toolsets_explicit and platform_toolsets is not None:
                allowed = set(platform_toolsets) | set(skills)
                wanted = [name for name in wanted if name in allowed]
            effective = filter_platform_unsupported(wanted)
            router_predicted = list(effective)
        elif not enabled_toolsets_explicit:
            # Router declined because confidence was low or unavailable.  Full
            # surface is safer than retaining an implicit platform subset.
            effective = None

    if router_predicted is None and effective is not None:
        effective = _dedupe(list(effective) + skills)
    if effective is not None:
        effective = filter_platform_unsupported(effective)

    return {
        "enabled_toolsets": effective,
        "router_predicted_toolsets": router_predicted,
        "memory_tier": None,
        "merge_skill_required": router_predicted is None,
        "source": "hermes-token-router",
        "cache_fragment": {
            "source": "hermes-token-router",
            "router_predicted": router_predicted or [],
            "explicit": bool(enabled_toolsets_explicit),
        },
    }
