"""Typed routing decisions shared by policy, application, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Iterable, Optional


class RouteAction(str, Enum):
    """How the router should shape the tool surface."""

    NARROW = "narrow"
    FULL = "full"
    NO_TOOLS = "no_tools"


@dataclass(frozen=True)
class RouteDecision:
    """An explicit, auditable routing result.

    ``FULL`` represents uncertainty/fail-open. ``NO_TOOLS`` is an intentional
    answer-only route and is never overloaded as an error signal.
    """

    action: RouteAction
    toolsets: FrozenSet[str] = field(default_factory=frozenset)
    confidence: Optional[float] = None
    reason_code: str = ""
    source: str = "deterministic"

    def __post_init__(self) -> None:
        if self.action is RouteAction.NARROW and not self.toolsets:
            raise ValueError("a narrow route requires at least one toolset")
        if self.action is not RouteAction.NARROW and self.toolsets:
            raise ValueError("only narrow routes may contain toolsets")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    @classmethod
    def narrow(
        cls,
        toolsets: Iterable[str],
        *,
        confidence: Optional[float],
        reason_code: str,
        source: str = "deterministic",
    ) -> "RouteDecision":
        return cls(
            RouteAction.NARROW,
            frozenset(str(name).strip() for name in toolsets if str(name).strip()),
            confidence,
            reason_code,
            source,
        )

    @classmethod
    def full(
        cls,
        *,
        reason_code: str,
        confidence: Optional[float] = None,
        source: str = "policy",
    ) -> "RouteDecision":
        return cls(RouteAction.FULL, confidence=confidence, reason_code=reason_code, source=source)

    @classmethod
    def no_tools(
        cls,
        *,
        reason_code: str,
        confidence: Optional[float] = 1.0,
        source: str = "deterministic",
    ) -> "RouteDecision":
        return cls(RouteAction.NO_TOOLS, confidence=confidence, reason_code=reason_code, source=source)
