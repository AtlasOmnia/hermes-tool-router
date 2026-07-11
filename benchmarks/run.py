#!/usr/bin/env python3
"""Evaluate deterministic routing against the versioned prompt corpus."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "hermes_tool_router_benchmark"
spec = importlib.util.spec_from_file_location(PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
assert spec is not None and spec.loader is not None
plugin = importlib.util.module_from_spec(spec)
sys.modules[PKG] = plugin
spec.loader.exec_module(plugin)

from hermes_tool_router_benchmark.policy import _predict_toolsets_by_rules
from scoring import score_routes

AVAILABLE = {
    "web", "browser", "file", "terminal", "git", "vision", "image_gen",
    "video", "video_gen", "memory", "session_search", "cronjob", "delegation",
    "clarify", "skills",
}


def main() -> int:
    corpus = Path(__file__).with_name("prompts.jsonl")
    pairs: list[tuple[set[str], set[str]]] = []
    critical_misses: list[str] = []
    full_fallbacks = 0
    for line in corpus.read_text().splitlines():
        row = json.loads(line)
        required = set(row["required_toolsets"])
        predicted, _reason = _predict_toolsets_by_rules(row["prompt"], AVAILABLE)
        if predicted is None:
            full_fallbacks += 1
            predicted = set(AVAILABLE)
        pairs.append((required, set(predicted)))
        if row.get("critical") and not required <= set(predicted):
            critical_misses.append(row["id"])

    result = score_routes(pairs)
    result.update({
        "records": len(pairs),
        "full_fallback_rate": full_fallbacks / len(pairs),
        "critical_misses": critical_misses,
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if critical_misses else 0


if __name__ == "__main__":
    raise SystemExit(main())
