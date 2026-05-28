"""Persistent feedback records for router/toolset maturation.

The router should not silently rewrite itself at runtime.  This module records
privacy-conscious routing outcomes so a human or offline job can tune policy
and smoke cases from evidence.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from hermes_constants import get_hermes_home

logger = logging.getLogger("run_agent")

FEEDBACK_VERSION = 1
MAX_RECORD_BYTES = 32_768


def feedback_path() -> Path:
    return get_hermes_home() / "router_feedback" / "events.jsonl"


def _json_default(value: Any) -> str:
    return str(value)


def _append(record: Dict[str, Any]) -> None:
    try:
        path = feedback_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, default=_json_default)
        if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
            record = {
                "type": record.get("type", "unknown"),
                "version": FEEDBACK_VERSION,
                "ts": record.get("ts", time.time()),
                "truncated": True,
                "message_sha256": record.get("message_sha256", ""),
                "session_id": record.get("session_id", ""),
            }
            line = json.dumps(record, sort_keys=True, default=_json_default)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:
        logger.debug("router feedback write failed: %s", exc)


def start_turn(agent: Any, *, user_message: str | None = None) -> None:
    telemetry = dict(getattr(agent, "_router_toolset_telemetry", {}) or {})
    if not telemetry:
        return
    record = {
        "version": FEEDBACK_VERSION,
        "type": "turn_start",
        "ts": time.time(),
        "session_id": getattr(agent, "session_id", "") or "",
        "platform": getattr(agent, "platform", "") or telemetry.get("platform", ""),
        "message_sha256": telemetry.get("message_sha256", ""),
        "message_chars": len(user_message or ""),
        "router": telemetry,
    }
    agent._router_feedback_record = record
    agent._router_feedback_events = []
    _append(record)


def record_event(agent: Any, event_type: str, **fields: Any) -> None:
    telemetry = dict(getattr(agent, "_router_toolset_telemetry", {}) or {})
    if not telemetry:
        return
    event = {
        "version": FEEDBACK_VERSION,
        "type": event_type,
        "ts": time.time(),
        "session_id": getattr(agent, "session_id", "") or "",
        "message_sha256": telemetry.get("message_sha256", ""),
        **fields,
    }
    try:
        events = getattr(agent, "_router_feedback_events", None)
        if isinstance(events, list):
            events.append(event)
    except Exception:
        pass
    _append(event)


def finish_turn(agent: Any, result: Dict[str, Any]) -> None:
    telemetry = dict(getattr(agent, "_router_toolset_telemetry", {}) or {})
    if not telemetry:
        return
    events = getattr(agent, "_router_feedback_events", []) or []
    record = {
        "version": FEEDBACK_VERSION,
        "type": "turn_finish",
        "ts": time.time(),
        "session_id": getattr(agent, "session_id", "") or "",
        "message_sha256": telemetry.get("message_sha256", ""),
        "completed": bool(result.get("completed")),
        "failed": bool(result.get("failed")),
        "partial": bool(result.get("partial")),
        "api_calls": result.get("api_calls", 0),
        "turn_exit_reason": result.get("turn_exit_reason", ""),
        "first_prompt_tokens": result.get("first_prompt_tokens", 0),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "recalled_toolsets": sorted(getattr(agent, "_router_recalled_toolsets", set()) or []),
        "event_types": [event.get("type") for event in events],
        "router": telemetry,
    }
    _append(record)


def iter_feedback_records(path: Path | None = None) -> Iterable[Dict[str, Any]]:
    path = path or feedback_path()
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    yield parsed
    except FileNotFoundError:
        return


def summarize_feedback(path: Path | None = None, *, limit: int = 5000) -> Dict[str, Any]:
    records = list(iter_feedback_records(path))
    if limit > 0:
        records = records[-limit:]

    finishes = [r for r in records if r.get("type") == "turn_finish"]
    events = [r for r in records if r.get("type") not in {"turn_start", "turn_finish"}]

    by_effective: Counter[str] = Counter()
    recalls: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    first_prompt_by_effective: Dict[str, List[int]] = defaultdict(list)

    for record in finishes:
        router = record.get("router") or {}
        effective = router.get("effective_toolsets")
        effective_key = "full" if effective is None else ",".join(effective)
        by_effective[effective_key] += 1
        first_prompt = record.get("first_prompt_tokens")
        if isinstance(first_prompt, int) and first_prompt > 0:
            first_prompt_by_effective[effective_key].append(first_prompt)
        for toolset in record.get("recalled_toolsets") or []:
            recalls[toolset] += 1
        if record.get("failed") or record.get("partial"):
            failures[effective_key] += 1

    for event in events:
        if event.get("type") == "router_recall":
            recalls[str(event.get("toolset") or "unknown")] += 1

    avg_first_prompt = {
        key: int(sum(values) / len(values))
        for key, values in first_prompt_by_effective.items()
        if values
    }

    recommendations = []
    if recalls:
        recommendations.append(
            "Review recalled toolsets; repeated recalls suggest router policy or skill requires_toolsets is too narrow."
        )
    if failures:
        recommendations.append(
            "Review failed/partial turns by effective toolset; quality loss may come from over-narrowing or prompt guidance."
        )

    return {
        "path": str(path or feedback_path()),
        "records": len(records),
        "turns": len(finishes),
        "by_effective_toolsets": dict(by_effective),
        "avg_first_prompt_tokens": avg_first_prompt,
        "recalls_by_toolset": dict(recalls),
        "failures_by_effective_toolsets": dict(failures),
        "recommendations": recommendations,
    }
