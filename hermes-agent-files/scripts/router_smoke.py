#!/usr/bin/env python3
"""Run lightweight router/gateway smoke prompts and report token usage.

This intentionally uses only the standard library so it can run in a freshly
updated Hermes checkout without installing test extras.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_HERMES_REPO = Path.home() / ".hermes" / "hermes-agent"


def _repo_path(*parts: str) -> str:
    return str(DEFAULT_HERMES_REPO.joinpath(*parts))


COMMON_CASES = [
    {
        "name": "trivial",
        "route": "minimal",
        "prompt": "Reply with exactly: final-pong",
        "max_tokens": 64,
        "expect": r"\bfinal-pong\b",
        "prompt_ceiling": 4_000,
        "expect_effective": [],
        "expect_full_surface": False,
    },
    {
        "name": "greeting",
        "route": "minimal",
        "prompt": "hi",
        "max_tokens": 64,
        "expect": r".+",
        "prompt_ceiling": 4_000,
        "expect_effective": [],
        "expect_full_surface": False,
    },
    {
        "name": "date",
        "route": "terminal",
        "prompt": "What is today's date? Use the local system, then answer in one short sentence.",
        "max_tokens": 128,
        "expect": r"\b((Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b.*\b20\d{2}|20\d{2}-\d{2}-\d{2})\b",
        "prompt_ceiling": 22_000,
        "expect_effective": ["terminal"],
        "expect_full_surface": False,
    },
    {
        "name": "file",
        "route": "file",
        "prompt": (
            f"Use file search under {_repo_path()} to check whether "
            "agent/toolset_policy.py exists. Answer yes or no."
        ),
        "max_tokens": 128,
        "expect": r"\byes\b",
        "prompt_ceiling": 22_000,
        "expect_effective": ["file"],
        "expect_full_surface": False,
    },
]

MATRIX_ONLY_CASES = [
    {
        "name": "vague_repo_action",
        "route": "file_or_full",
        "prompt": (
            f"Can you help me check this repo? Look at {_repo_path()} "
            "and tell me the most relevant top-level project file you find."
        ),
        "max_tokens": 192,
        "expect": r"\b(AGENTS\.md|pyproject\.toml|README|setup\.py|requirements|hermes)\b",
        "prompt_ceiling": 90_000,
        "expect_effective_any": [["file"], None],
    },
    {
        "name": "casual_state",
        "route": "terminal",
        "prompt": "Quick check: what OS is this Hermes gateway running on?",
        "max_tokens": 128,
        "expect": r"\b(Darwin|macOS|Linux|Windows|OS)\b",
        "prompt_ceiling": 22_000,
        "expect_effective": ["terminal"],
        "expect_full_surface": False,
    },
    {
        "name": "file_search",
        "route": "file",
        "prompt": (
            f"Use file search under {_repo_path()} for a file named toolset_router.py. "
            "Answer with the matching relative path only."
        ),
        "max_tokens": 128,
        "expect": r"agent/toolset_router\.py",
        "prompt_ceiling": 13_000,
        "expect_effective": ["file"],
        "expect_full_surface": False,
    },
    {
        "name": "git_status",
        "route": "terminal",
        "prompt": (
            f"Check the git status for {_repo_path()}. "
            "Answer with either 'clean' or the changed file names."
        ),
        "max_tokens": 192,
        "expect": r"\b(clean|modified|changed|scripts/router_smoke\.py|tools/terminal_tool\.py)\b",
        "prompt_ceiling": 22_000,
        "expect_effective": ["terminal"],
        "expect_full_surface": False,
    },
    {
        "name": "system",
        "route": "terminal",
        "prompt": "Check the local OS/kernel with a system command and answer in one short sentence.",
        "max_tokens": 128,
        "expect": r"\b(Darwin|macOS|Linux|Windows|kernel|OS)\b",
        "prompt_ceiling": 22_000,
        "expect_effective": ["terminal"],
        "expect_full_surface": False,
    },
    {
        "name": "long_task_decline",
        "route": "router_decline_full_surface",
        "prompt": (
            "This is a deliberately long first-turn prompt used to verify that the router declines "
            "to reduce the tool surface instead of falling into a clarify-only floor. "
            "Read these constraints carefully: do not ask clarifying questions and do not summarize "
            "the constraints. The actual task requires local system access, and the prompt is long "
            "enough to cross the router's length threshold, so a clarify-only tool floor would fail. "
            "Repeat this sentence for length padding: Hermes router threshold regression smoke. "
            "Repeat this sentence for length padding: Hermes router threshold regression smoke. "
            "Repeat this sentence for length padding: Hermes router threshold regression smoke. "
            "Repeat this sentence for length padding: Hermes router threshold regression smoke. "
            "Repeat this sentence for length padding: Hermes router threshold regression smoke. "
            "Final task: use a shell command like `test -d` to check whether "
            f"{_repo_path('agent')} exists as a directory, "
            "then answer exactly long-decline-yes or long-decline-no."
        ),
        "max_tokens": 64,
        "expect": r"\blong-decline-yes\b",
        "prompt_ceiling": 90_000,
        "expect_effective": None,
        "expect_full_surface": True,
    },
]

WEB_CASES = [
    {
        "name": "web_current",
        "route": "web",
        "prompt": (
            "Use web search to find the current Python 3 stable release from python.org. "
            "Answer with the version and source domain only."
        ),
        "max_tokens": 192,
        "expect": r"\bpython\.org\b",
        "prompt_ceiling": 18_000,
        "expect_effective": ["web"],
        "expect_full_surface": False,
    },
]

DEFAULT_CASES = COMMON_CASES
PROFILE_CASES = {
    "quick": COMMON_CASES,
    "matrix": COMMON_CASES + MATRIX_ONLY_CASES,
    "web": WEB_CASES,
    "all": COMMON_CASES + MATRIX_ONLY_CASES + WEB_CASES,
}

FAILURE_RE = re.compile(
    r"(traceback|tool error|tool failure|returned error|illegal time format|command failed)",
    re.IGNORECASE,
)
DECISION_MARKER = "router_toolset_decision "


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_api_config(config_path: Path) -> tuple[str, str | None]:
    host = "127.0.0.1"
    port = 8642
    key = None

    data = _load_yaml(config_path)
    platforms = data.get("platforms") if isinstance(data, dict) else None
    api_server = platforms.get("api_server") if isinstance(platforms, dict) else None
    extra = api_server.get("extra") if isinstance(api_server, dict) else None
    if isinstance(extra, dict):
        host = str(extra.get("host") or host)
        port = int(extra.get("port") or port)
        key = str(extra.get("key") or "") or None

    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"

    return f"http://{host}:{port}/v1/chat/completions", key


def _read_log_tail(path: Path, start_size: int) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(start_size)
            return fh.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ""


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _post_json(url: str, api_key: str | None, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _message_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"].get("content") or ""
    except Exception:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, ensure_ascii=False)


def _usage(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage")
    return usage if isinstance(usage, dict) else {}


def _extract_decisions(log_chunk: str) -> list[dict[str, Any]]:
    decisions = []
    for line in log_chunk.splitlines():
        if DECISION_MARKER not in line:
            continue
        raw = line.split(DECISION_MARKER, 1)[1].strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            decisions.append(parsed)
    return decisions


def _format_decision(decision: dict[str, Any]) -> str:
    predicted = decision.get("predicted_toolsets")
    effective = decision.get("effective_toolsets")
    predicted_text = "none" if not predicted else ",".join(map(str, predicted))
    effective_text = "full" if effective is None else ",".join(map(str, effective))
    return (
        f"decision=predicted[{predicted_text}] "
        f"effective[{effective_text}] "
        f"full={bool(decision.get('full_surface'))} "
        f"tools={decision.get('loaded_tool_count', '?')} "
        f"source={decision.get('policy_source') or '?'}"
    )


def _check_decision(case: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    failures = []
    expected_full = case.get("expect_full_surface")
    if expected_full is not None and bool(decision.get("full_surface")) is not bool(expected_full):
        failures.append(
            f"expected full_surface={bool(expected_full)} but got {bool(decision.get('full_surface'))}"
        )
    if "expect_effective" in case and not bool(decision.get("full_surface")):
        expected_effective = case.get("expect_effective")
        actual_effective = decision.get("effective_toolsets")
        if actual_effective != expected_effective:
            failures.append(
                f"expected effective_toolsets={expected_effective!r} but got {actual_effective!r}"
            )
    if "expect_effective_any" in case:
        expected_any = case.get("expect_effective_any") or []
        actual_effective = None if bool(decision.get("full_surface")) else decision.get("effective_toolsets")
        if actual_effective not in expected_any:
            failures.append(
                f"expected effective_toolsets/full_surface in {expected_any!r} but got {actual_effective!r}"
            )
    return failures


def _all_cases() -> list[dict[str, Any]]:
    seen = set()
    cases = []
    for group in (COMMON_CASES, MATRIX_ONLY_CASES, WEB_CASES):
        for case in group:
            if case["name"] in seen:
                continue
            seen.add(case["name"])
            cases.append(case)
    return cases


def main() -> int:
    home = Path.home()
    default_config = home / ".hermes" / "config.yaml"
    default_log = home / ".hermes" / "logs" / "gateway.log"
    default_error_log = home / ".hermes" / "logs" / "gateway.error.log"
    default_url, config_key = _read_api_config(default_config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=default_url)
    parser.add_argument("--api-key", default=os.getenv("API_SERVER_KEY") or config_key)
    parser.add_argument("--model", default="hermes")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--log-file", type=Path, default=default_log)
    parser.add_argument("--error-log-file", type=Path, default=default_error_log)
    parser.add_argument("--no-log-scan", action="store_true")
    parser.add_argument("--strict-failures", action="store_true")
    parser.add_argument(
        "--strict-ceilings",
        action="store_true",
        help="Treat prompt token ceiling warnings as failures.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_CASES),
        default="quick",
        help="Case profile to run. 'matrix' skips live web; 'all' includes it.",
    )
    parser.add_argument(
        "--case",
        choices=[case["name"] for case in _all_cases()],
        action="append",
        help="Run only the named case. May be passed more than once.",
    )
    args = parser.parse_args()

    selected = PROFILE_CASES[args.profile]
    if args.case:
        wanted = set(args.case)
        selected = [case for case in _all_cases() if case["name"] in wanted]

    any_failure = False
    print(
        f"router_smoke url={args.url} profile={args.profile} "
        f"cases={','.join(case['name'] for case in selected)}"
    )

    for case in selected:
        log_starts = {
            args.log_file: _file_size(args.log_file),
            args.error_log_file: _file_size(args.error_log_file),
        }
        started = time.time()
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": case["prompt"]}],
            "max_tokens": case["max_tokens"],
            "temperature": 0,
        }

        try:
            response = _post_json(args.url, args.api_key, payload, args.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            print(f"{case['name']}: HTTP {exc.code}: {body}", file=sys.stderr)
            any_failure = True
            continue
        except Exception as exc:
            print(f"{case['name']}: request failed: {exc}", file=sys.stderr)
            any_failure = True
            continue

        elapsed = time.time() - started
        usage = _usage(response)
        text = _message_text(response).replace("\n", " ")
        prompt_tokens = usage.get("prompt_tokens")
        first_prompt_tokens = usage.get("first_prompt_tokens")
        ceiling_tokens = first_prompt_tokens if isinstance(first_prompt_tokens, int) and first_prompt_tokens > 0 else prompt_tokens
        print(
            f"{case['name']}: ok {elapsed:.1f}s "
            f"route={case.get('route', '?')} "
            f"prompt={prompt_tokens if prompt_tokens is not None else '?'} "
            f"first_prompt={first_prompt_tokens if first_prompt_tokens is not None else '?'} "
            f"completion={usage.get('completion_tokens', '?')} "
            f"total={usage.get('total_tokens', '?')} "
            f"text={text[:180]!r}"
        )

        expect = case.get("expect")
        if expect and not re.search(str(expect), text, re.IGNORECASE):
            any_failure = True
            print(
                f"{case['name']}: expected response to match /{expect}/",
                file=sys.stderr,
            )

        ceiling = case.get("prompt_ceiling")
        if isinstance(ceiling_tokens, int) and isinstance(ceiling, int) and ceiling_tokens > ceiling:
            msg = (
                f"{case['name']}: prompt token ceiling exceeded: "
                f"{ceiling_tokens} > {ceiling}"
            )
            print(msg, file=sys.stderr)
            if args.strict_ceilings:
                any_failure = True

        if not args.no_log_scan:
            chunk = "".join(
                _read_log_tail(path, start_size)
                for path, start_size in log_starts.items()
            )
            decisions = _extract_decisions(chunk)
            if decisions:
                decision = decisions[-1]
                print(f"{case['name']}: {_format_decision(decision)}")
                for failure in _check_decision(case, decision):
                    any_failure = True
                    print(f"{case['name']}: decision mismatch: {failure}", file=sys.stderr)
            else:
                print(f"{case['name']}: decision=unavailable", file=sys.stderr)
            failures = [line for line in chunk.splitlines() if FAILURE_RE.search(line)]
            if failures:
                any_failure = True
                print(f"{case['name']}: log failure markers={len(failures)}", file=sys.stderr)
                for line in failures[-5:]:
                    print(f"  {line[-240:]}", file=sys.stderr)

    return 1 if any_failure and args.strict_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
