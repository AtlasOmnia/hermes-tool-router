#!/usr/bin/env python3
"""Paired live test-profile evaluator for hermes-token-router."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

SESSION_RE = re.compile(r"Session:\s+([A-Za-z0-9_]+)")
ROUTE_RE = re.compile(r"predicted_toolsets=\[([^]]*)\]")
TOOL_RE = re.compile(r"tool ([A-Za-z0-9_]+) completed", re.IGNORECASE)
FIRST_INPUT_RE = re.compile(r"API call #1:.*?\bin=(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="router-test")
    parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[1] / "benchmarks/live_cases.json")
    parser.add_argument("--state-db", type=Path, default=Path.home() / ".hermes/profiles/router-test/state.db")
    parser.add_argument("--agent-log", type=Path, default=Path.home() / ".hermes/profiles/router-test/logs/agent.log")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "runs/live_router_eval.json")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--baseline-cases", default="answer_only,terminal,web")
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def run_case(profile: str, case: dict[str, Any], routed: bool, timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["HERMES_TOKEN_ROUTER_ENABLED"] = "true" if routed else "false"
    prompt = case["prompt"].replace(
        "{{HERMES_PYTHON}}",
        str(Path.home() / ".hermes/hermes-agent/.venv/bin/python"),
    )
    started = time.monotonic()
    proc = subprocess.run(
        ["hermes", "--profile", profile, "chat", "-q", prompt],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        timeout=timeout,
        check=False,
    )
    elapsed = time.monotonic() - started
    match = SESSION_RE.search(proc.stdout)
    return {
        "case_id": case["id"],
        "mode": "routed" if routed else "baseline",
        "returncode": proc.returncode,
        "session_id": match.group(1) if match else "",
        "elapsed_seconds": round(elapsed, 3),
        "stdout_tail": proc.stdout[-2000:],
    }


def session_metrics(db_path: Path, session_id: str) -> dict[str, Any]:
    if not session_id:
        return {}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                      tool_call_count, api_call_count, estimated_cost_usd
               FROM sessions WHERE id = ?""",
            (session_id,),
        ).fetchone()
        answer = conn.execute(
            "SELECT content FROM messages WHERE session_id = ? AND role = 'assistant' ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    result = dict(row) if row else {}
    result["answer"] = answer[0] if answer else ""
    return result


def log_metrics(log_text: str, session_id: str) -> dict[str, Any]:
    lines = [line for line in log_text.splitlines() if session_id and session_id in line]
    joined = "\n".join(lines)
    route_match = ROUTE_RE.search(joined)
    route = []
    if route_match:
        route = re.findall(r"['\"]([^'\"]+)['\"]", route_match.group(1))
    tools = sorted(set(TOOL_RE.findall(joined)))
    errors = [line for line in lines if " ERROR " in line or "returned error" in line]
    input_match = FIRST_INPUT_RE.search(joined)
    return {
        "predicted_toolsets": route,
        "completed_tools": tools,
        "errors": errors,
        "first_request_input_tokens": int(input_match.group(1)) if input_match else None,
    }


def evaluate_case(case: dict[str, Any], run: dict[str, Any]) -> None:
    expected_sets = set(case.get("expected_toolsets", []))
    predicted_sets = set(run.get("predicted_toolsets", []))
    expected_tools = set(case.get("expected_tools", []))
    completed_tools = set(run.get("completed_tools", []))
    answer = run.get("answer", "")
    run["route_recall_ok"] = expected_sets.issubset(predicted_sets)
    run["route_exact_ok"] = expected_sets == predicted_sets
    run["tool_success_ok"] = not expected_tools or bool(expected_tools & completed_tools)
    run["answer_check_ok"] = all(text.lower() in answer.lower() for text in case.get("must_contain", []))
    run["case_passed"] = bool(
        run.get("returncode") == 0
        and run.get("session_id")
        and run["route_recall_ok"]
        and run["tool_success_ok"]
        and run["answer_check_ok"]
        and not run.get("errors")
    )


def main() -> int:
    args = parse_args()
    cases = json.loads(args.cases.read_text())
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    baseline_ids = {item.strip() for item in args.baseline_cases.split(",") if item.strip()}
    Path("/tmp/router-test-probe.txt").write_text("ROUTER_TEST_PROBE_OK\n")

    results: list[dict[str, Any]] = []
    for case in cases:
        routed_run = run_case(args.profile, case, True, args.timeout)
        time.sleep(0.2)
        routed_run.update(session_metrics(args.state_db, routed_run["session_id"]))
        routed_run.update(log_metrics(args.agent_log.read_text(errors="replace"), routed_run["session_id"]))
        evaluate_case(case, routed_run)
        results.append(routed_run)

        if case["id"] in baseline_ids:
            baseline_run = run_case(args.profile, case, False, args.timeout)
            time.sleep(0.2)
            baseline_run.update(session_metrics(args.state_db, baseline_run["session_id"]))
            baseline_run.update(log_metrics(args.agent_log.read_text(errors="replace"), baseline_run["session_id"]))
            results.append(baseline_run)

    routed = [row for row in results if row["mode"] == "routed"]
    pairs: list[dict[str, Any]] = []
    for case_id in baseline_ids:
        route_row = next((row for row in results if row["case_id"] == case_id and row["mode"] == "routed"), None)
        base_row = next((row for row in results if row["case_id"] == case_id and row["mode"] == "baseline"), None)
        base_tokens = base_row.get("first_request_input_tokens") if base_row else None
        route_tokens = route_row.get("first_request_input_tokens") if route_row else None
        if route_row and base_row and base_tokens and route_tokens:
            reduction = 1.0 - route_tokens / base_tokens
            pairs.append(
                {
                    "case_id": case_id,
                    "routed_first_request_tokens": route_tokens,
                    "baseline_first_request_tokens": base_tokens,
                    "token_reduction": reduction,
                }
            )

    count = max(1, len(routed))
    route_recall = sum(bool(row.get("route_recall_ok")) for row in routed) / count
    route_exact = sum(bool(row.get("route_exact_ok")) for row in routed) / count
    tool_success = sum(bool(row.get("tool_success_ok")) for row in routed) / count
    answer_accuracy = sum(bool(row.get("answer_check_ok")) for row in routed) / count
    pass_rate = sum(bool(row.get("case_passed")) for row in routed) / count
    avg_reduction = sum(pair["token_reduction"] for pair in pairs) / len(pairs) if pairs else 0.0
    score = 40 * route_recall + 20 * tool_success + 15 * answer_accuracy + 10 * route_exact + 15 * max(0.0, avg_reduction)
    metrics = {
        "cases": len(routed),
        "route_recall": route_recall,
        "route_exact_accuracy": route_exact,
        "tool_success_rate": tool_success,
        "answer_accuracy": answer_accuracy,
        "case_pass_rate": pass_rate,
        "paired_average_token_reduction": avg_reduction,
        "paired_cases": pairs,
        "total_elapsed_seconds": sum(row["elapsed_seconds"] for row in results),
    }
    payload = {
        "score": round(score, 6),
        "accepted": route_recall == 1.0 and tool_success >= 0.9 and answer_accuracy >= 0.9,
        "reason": f"live test-profile router evaluation: {sum(bool(r.get('case_passed')) for r in routed)}/{len(routed)} cases passed",
        "metrics": metrics,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
