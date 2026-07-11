#!/usr/bin/env python3
"""Measure live Hermes tool-schema request estimates with provider-aware code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes-source", type=Path, required=True)
    parser.add_argument("--toolsets", default="web")
    args = parser.parse_args()
    sys.path.insert(0, str(args.hermes_source.resolve()))

    import model_tools
    from agent.model_metadata import estimate_request_tokens_rough

    selected = [name.strip() for name in args.toolsets.split(",") if name.strip()]
    full_defs = model_tools.get_tool_definitions(quiet_mode=True)
    routed_defs = model_tools.get_tool_definitions(enabled_toolsets=selected, quiet_mode=True)
    messages = [{"role": "user", "content": "schema measurement"}]
    full_tokens = estimate_request_tokens_rough(messages, tools=full_defs or None)
    routed_tokens = estimate_request_tokens_rough(messages, tools=routed_defs or None)
    reduction = 1.0 - (routed_tokens / full_tokens) if full_tokens else 0.0
    print(json.dumps({
        "full_tools": len(full_defs),
        "routed_tools": len(routed_defs),
        "toolsets": selected,
        "full_request_tokens": full_tokens,
        "routed_request_tokens": routed_tokens,
        "estimated_reduction": reduction,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
