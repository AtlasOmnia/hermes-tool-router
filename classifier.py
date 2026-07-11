"""Bounded, structured optional classifier support."""

from __future__ import annotations

import json
import math
import queue
import threading
from typing import Callable, Iterable, TypeVar

try:
    from .models import RouteAction, RouteDecision
except ImportError:  # pragma: no cover
    from models import RouteAction, RouteDecision


T = TypeVar("T")


def call_with_hard_timeout(call: Callable[[], T], timeout_seconds: float) -> tuple[bool, T | None]:
    """Run a call on a daemon thread and return at the deadline without joining it."""
    results: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            results.put((True, call()), block=False)
        except BaseException as exc:  # propagated on the caller thread
            results.put((False, exc), block=False)

    threading.Thread(target=worker, name="hermes-tool-router", daemon=True).start()
    try:
        succeeded, value = results.get(timeout=max(0.001, timeout_seconds))
    except queue.Empty:
        return False, None
    if not succeeded:
        raise value  # type: ignore[misc]
    return True, value  # type: ignore[return-value]


def _strip_fence(content: str) -> str:
    text = (content or "").strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline >= 0 else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def parse_classifier_output(
    content: str,
    available_toolsets: Iterable[str],
    confidence_threshold: float,
) -> RouteDecision:
    """Validate classifier JSON and fail open on every ambiguous condition."""

    try:
        result = json.loads(_strip_fence(content))
    except (TypeError, ValueError, json.JSONDecodeError):
        return RouteDecision.full(reason_code="invalid_json", source="classifier")
    if not isinstance(result, dict):
        return RouteDecision.full(reason_code="invalid_shape", source="classifier")

    raw_confidence = result.get("confidence")
    if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
        return RouteDecision.full(reason_code="missing_confidence", source="classifier")
    confidence = float(raw_confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return RouteDecision.full(reason_code="invalid_confidence", source="classifier")
    if confidence < confidence_threshold:
        return RouteDecision.full(
            reason_code="low_confidence", confidence=confidence, source="classifier"
        )

    action = str(result.get("action") or "").strip().lower()
    reason = str(result.get("reason_code") or "classifier").strip()[:80]
    if action == RouteAction.FULL.value:
        return RouteDecision.full(reason_code=reason, confidence=confidence, source="classifier")
    if action == RouteAction.NO_TOOLS.value:
        return RouteDecision.no_tools(reason_code=reason, confidence=confidence, source="classifier")
    if action != RouteAction.NARROW.value:
        return RouteDecision.full(reason_code="unknown_action", confidence=confidence, source="classifier")

    raw_toolsets = result.get("toolsets")
    if not isinstance(raw_toolsets, list) or not raw_toolsets:
        return RouteDecision.full(reason_code="invalid_toolsets", confidence=confidence, source="classifier")
    selected = {str(name).strip() for name in raw_toolsets if str(name).strip()}
    available = set(available_toolsets)
    if not selected or not selected <= available:
        return RouteDecision.full(reason_code="unknown_toolset", confidence=confidence, source="classifier")
    return RouteDecision.narrow(
        selected,
        confidence=confidence,
        reason_code=reason,
        source="classifier",
    )
