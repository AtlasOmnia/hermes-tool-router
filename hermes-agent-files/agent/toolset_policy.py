"""Toolset policy seam for router-first schema reduction.

The default policy is deliberately small and conservative.  A user-owned
policy can live outside the repo at:

    ~/.hermes/plugins/hermes-token-router/router_policy.py

If that file defines ``resolve_toolsets(**kwargs)``, it owns the routing
policy while core Hermes keeps only the stable integration seam.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger("run_agent")

SAFE_CORE_TOOLSETS = ("web",)


@dataclass
class ToolsetPolicyResult:
    enabled_toolsets: Optional[List[str]]
    router_predicted_toolsets: Optional[List[str]] = None
    memory_tier: Optional[str] = None
    merge_skill_required: bool = True
    source: str = "builtin"
    cache_fragment: Dict[str, Any] = field(default_factory=dict)


def _dedupe(names: List[str]) -> List[str]:
    result: List[str] = []
    for name in names:
        if name and name not in result:
            result.append(name)
    return result


def _active_skill_toolsets(disabled_toolsets: Optional[List[str]]) -> List[str]:
    try:
        from agent.skill_utils import get_skill_required_toolsets
        required = sorted(get_skill_required_toolsets())
    except Exception:
        return []

    disabled = set(disabled_toolsets or [])
    return [name for name in required if name not in disabled]


def _filter_platform_unsupported(names: List[str], platform: Optional[str]) -> List[str]:
    if platform == "api_server":
        return [name for name in names if name != "clarify"]
    return names


def _external_policy_path(router_config: Dict[str, Any]) -> Optional[Path]:
    path = (router_config or {}).get("policy_path")
    if path:
        return Path(str(path)).expanduser()

    default_path = get_hermes_home() / "plugins" / "hermes-token-router" / "router_policy.py"
    if default_path.exists():
        return default_path
    return None


def _run_external_policy(path: Path, **kwargs: Any) -> Optional[ToolsetPolicyResult]:
    try:
        spec = importlib.util.spec_from_file_location("hermes_external_toolset_policy", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        resolver = getattr(module, "resolve_toolsets", None)
        if resolver is None:
            return None
        raw = resolver(**kwargs)
        if raw is None:
            return None
        if isinstance(raw, ToolsetPolicyResult):
            raw.source = raw.source or str(path)
            return raw
        if isinstance(raw, dict):
            return ToolsetPolicyResult(
                enabled_toolsets=raw.get("enabled_toolsets"),
                router_predicted_toolsets=raw.get("router_predicted_toolsets"),
                memory_tier=raw.get("memory_tier"),
                merge_skill_required=bool(raw.get("merge_skill_required", True)),
                source=str(raw.get("source") or path),
                cache_fragment=dict(raw.get("cache_fragment") or {}),
            )
    except Exception as exc:
        logger.warning("external toolset policy failed (%s): %s", path, exc)
    return None


def resolve_effective_toolsets(
    *,
    user_message: Optional[str],
    router_config: Optional[Dict[str, Any]],
    available_toolsets: Dict[str, str],
    platform_toolsets: Optional[List[str]],
    enabled_toolsets_explicit: bool,
    disabled_toolsets: Optional[List[str]] = None,
    platform: Optional[str] = None,
) -> ToolsetPolicyResult:
    """Resolve the effective toolset list before schema assembly.

    ``enabled_toolsets=None`` is the legacy/full-surface sentinel.  A concrete
    list narrows schema assembly to those toolsets.
    """
    router_config = router_config or {}
    platform_toolsets = list(platform_toolsets) if platform_toolsets is not None else None
    skill_toolsets = _active_skill_toolsets(disabled_toolsets)

    kwargs = {
        "user_message": user_message,
        "router_config": router_config,
        "available_toolsets": available_toolsets,
        "platform_toolsets": platform_toolsets,
        "enabled_toolsets_explicit": enabled_toolsets_explicit,
        "disabled_toolsets": disabled_toolsets,
        "safe_core_toolsets": list(SAFE_CORE_TOOLSETS),
        "skill_toolsets": list(skill_toolsets),
        "platform": platform,
    }
    policy_path = _external_policy_path(router_config)
    if policy_path is not None:
        external = _run_external_policy(policy_path, **kwargs)
        if external is not None:
            return external

    effective = platform_toolsets
    router_predicted: Optional[List[str]] = None

    if router_config.get("enabled") and user_message:
        try:
            from agent.toolset_router import predict_toolsets

            predicted, should_reduce = predict_toolsets(
                user_message,
                router_config,
                available_toolsets,
            )
            if should_reduce:
                floor = [name for name in SAFE_CORE_TOOLSETS if name in available_toolsets]
                wanted = _dedupe(floor + list(predicted or []) + skill_toolsets)
                if enabled_toolsets_explicit and platform_toolsets is not None:
                    allowed = set(platform_toolsets) | set(skill_toolsets)
                    wanted = [name for name in wanted if name in allowed]
                effective = _filter_platform_unsupported(wanted, platform)
                router_predicted = list(effective)
            elif not enabled_toolsets_explicit:
                effective = None
        except Exception as exc:
            logger.warning("toolset policy router failed, falling back to full tools: %s", exc)
            if not enabled_toolsets_explicit:
                effective = None

    if router_predicted is None and skill_toolsets and effective is not None:
        effective = _dedupe(list(effective) + skill_toolsets)
    if effective is not None:
        effective = _filter_platform_unsupported(effective, platform)

    return ToolsetPolicyResult(
        enabled_toolsets=effective,
        router_predicted_toolsets=router_predicted,
        merge_skill_required=router_predicted is None,
        source="builtin",
        cache_fragment={
            "source": "builtin",
            "router_predicted": router_predicted or [],
            "explicit": bool(enabled_toolsets_explicit),
        },
    )
