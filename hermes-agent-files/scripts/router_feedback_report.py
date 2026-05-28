#!/usr/bin/env python3
"""Summarize persisted Hermes router feedback JSONL records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.router_feedback import feedback_path, summarize_feedback  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=feedback_path())
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = summarize_feedback(args.path, limit=args.limit)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print(f"Router feedback: {summary['path']}")
    print(f"Records: {summary['records']}  Turns: {summary['turns']}")
    print()
    print("Effective toolsets:")
    for key, count in sorted(summary["by_effective_toolsets"].items()):
        avg = summary["avg_first_prompt_tokens"].get(key, "-")
        print(f"  {key or '<none>'}: {count} turns, avg first_prompt={avg}")
    print()
    print("Recalls:")
    recalls = summary["recalls_by_toolset"]
    if recalls:
        for key, count in sorted(recalls.items()):
            print(f"  {key}: {count}")
    else:
        print("  none")
    print()
    print("Failures/partials:")
    failures = summary["failures_by_effective_toolsets"]
    if failures:
        for key, count in sorted(failures.items()):
            print(f"  {key or '<none>'}: {count}")
    else:
        print("  none")
    if summary["recommendations"]:
        print()
        print("Recommendations:")
        for item in summary["recommendations"]:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
