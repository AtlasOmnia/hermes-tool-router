"""Hermes hook compatibility detection."""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class CompatibilityMode(str, Enum):
    PRODUCTION = "production"
    REDUCED_SAFETY = "reduced_safety"
    LATE_COMPATIBILITY = "late_compatibility"
    UNSUPPORTED_FOR_TOKEN_SAVINGS = "unsupported_for_token_savings"


def detect_compatibility(hooks: Iterable[str], middleware: Iterable[str] = ()) -> CompatibilityMode:
    available = set(hooks)
    early = "pre_tool_surface_build" in available or "pre_turn_context_build" in available
    # Recovery is the legacy hook or Hermes's tool_request middleware (the
    # mechanism this plugin actually registers).
    recovery = "on_unavailable_tool_call" in available or "tool_request" in set(middleware)
    if early and recovery:
        return CompatibilityMode.PRODUCTION
    if early:
        return CompatibilityMode.REDUCED_SAFETY
    if "pre_llm_call" in available:
        return CompatibilityMode.LATE_COMPATIBILITY
    return CompatibilityMode.UNSUPPORTED_FOR_TOKEN_SAVINGS
