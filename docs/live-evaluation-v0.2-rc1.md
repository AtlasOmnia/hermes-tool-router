# Live evaluation: v0.2 RC1

Date: 2026-07-11

Profile: `edith` (isolated test profile)

Model: `openai/gpt-5.4-mini` through OpenRouter

Hermes compatibility mode: `pre_llm_call` late compatibility

Router classifier: disabled; deterministic policy only

## Runtime architecture

The tested router requires **no Hermes Agent core modifications**. It uses stock extension points:

- `pre_llm_call` narrows the live tool surface before the provider request.
- `tool_request` middleware expands a registry-known pruned tool before normal validation and execution.
- `request_toolset` provides explicit, model-visible recovery and validates names against the live registry at call time.

An optional earlier hook could reduce internal preflight work, but it is not required for provider-request token savings or tool execution. No router patch is present in `apply-patches.sh`.

## Paired methodology

The evaluator launches fresh Edith sessions with the same profile and prompt in two modes:

1. Router enabled.
2. Router bypassed for that subprocess with `HERMES_TOKEN_ROUTER_ENABLED=false`.

The primary efficiency metric is the actual `in=` token count from Hermes's first provider request log entry. SQLite session `input_tokens` is not used for schema-footprint comparisons because it excludes cache-read tokens and aggregates billing across calls.

The fixed safe-core corpus covers:

- no-tool answer
- terminal
- file
- web
- skills
- cronjob
- session search
- computer use
- sandboxed code execution
- todo

## Results

- Cases passed: **10/10**
- Required-toolset recall: **100%**
- Exact route accuracy: **100%**
- Tool-call success: **100%**
- Answer accuracy: **100%**
- Autoresearch score: **94.239983**
- Evaluator wall time: **118.505 seconds**

### First-provider-request tokens

| Case | Routed | Full tools | Reduction |
|---|---:|---:|---:|
| no-tool answer | 7,911 | 22,125 | 64.24% |
| terminal | 9,305 | 22,154 | 58.00% |
| web | 8,288 | 22,135 | 62.56% |
| **Average** | — | — | **61.60%** |

## Reproduction

Direct evaluator:

```bash
.venv/bin/python scripts/live_router_evaluator.py \
  --output runs/live_router_eval.json
```

Through `hermes-autoresearch`:

```bash
PYTHONPATH=~/Desktop/hermes-autoresearch/src \
  .venv/bin/python -m hermes_autoresearch.cli \
  --config autoresearch/router_live_eval.json
```

The autoresearch run is bounded to one evaluator trial and a 30-minute hard ceiling. The current corpus normally finishes in about two minutes.

## Scope and caveats

This report validates the safe representative core, not every requirement-gated or externally billable integration. Image generation, video generation/analysis, TTS, browser automation, vision, memory writes, delegation, and platform-specific integrations require a separate controlled corpus.

The live average is below the current aspirational 70% stable-release token-reduction gate. RC1 should therefore remain a release candidate even though routing and tool execution passed this corpus without regression.
