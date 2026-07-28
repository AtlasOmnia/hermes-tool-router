# Hermes Tool Router

Experimental standalone Hermes Agent plugin for reducing first-turn tool-schema overhead without repeatedly changing the tool prefix later in the conversation.

> Test on a separate Hermes profile first. Do not install an experimental router directly on a primary profile.

## How it works

<p align="center">
  <img src="docs/tool-router-flow.svg" alt="Hermes Tool Router flow: classify the first turn, narrow the tool schema when confidence is high, fail open when uncertain, and add missing toolsets permanently for the session." width="100%">
</p>

The router classifies only the first turn. High-confidence routes send a smaller tool schema to the main model; uncertainty keeps the full surface. If a pruned registered tool is needed later, Hermes adds its owning toolset without re-pruning the session.

## v0.2 design

- **First-turn routing:** deterministic intent classification narrows the live tool surface before the first provider request through stock Hermes hooks.
- **Session-sticky surface:** later turns reuse the initial surface instead of reclassifying and shrinking it.
- **Monotonic recovery:** requested toolsets are added permanently for the session.
- **Fail open:** uncertainty, invalid classifier output, missing confidence, timeout, registry mismatch, or unsupported runtime keeps the full tool surface.
- **Optional classifier:** external model routing is disabled by default. Deterministic misses fall back immediately with no network call.
- **Dynamic recovery:** `request_toolset` is generated from the live toolset registry and can request multiple toolsets.

## Critical compatibility fact

First-turn token savings work through either an early Hermes surface hook or the stock `pre_llm_call` compatibility path. In current Hermes, `pre_llm_call` runs after the initial preflight estimate but before the actual provider request is assembled and before the loop's request-pressure estimate; mutating `agent.tools` there still reduces the transmitted tool schemas. The tradeoff is that the plugin must recover the live agent through compatibility logic unless Hermes passes it explicitly.

Run diagnostics:

```bash
python diagnostics.py
```

Exit code `0` means a routing path is available before the provider request. Exit code `2` means the current runtime must not claim first-turn savings. The report separately states whether routing happens before the initial preflight estimate.

Automatic execution recovery uses Hermes's generic `tool_request` middleware. When the model emits a registry-known tool that was pruned, the middleware expands its owning toolset before normal validation and dispatch, then the original call continues through ordinary requirement checks and approvals. `request_toolset` remains the visible fallback.

### No Hermes core patch required

The tested implementation is plugin-only. Current Hermes already provides both extension points needed for provider-request reduction and automatic tool recovery. There is no router edit in Hermes core and no router entry in `apply-patches.sh`. An optional earlier hook could reduce internal preflight work, but it is not required for the measured request savings or successful tool execution.

See [docs/compatibility.md](docs/compatibility.md).

## Install on a test profile

```bash
hermes profile create router-test --clone
mkdir -p ~/.hermes/profiles/router-test/plugins
cp -R . ~/.hermes/profiles/router-test/plugins/hermes-token-router
```

Enable the plugin in the test profile, then set `profiles.router-test.enabled: true` in the plugin's `config.yaml`. Start a fresh session after configuration changes.

## Configuration

The safe default is disabled. Important v2 settings:

```yaml
global:
  enabled: false
  routing_scope: first_turn
  expansion_mode: monotonic
  shrink_mid_session: false
  floor_toolsets: []
  deterministic_rules_enabled: true
  confidence_threshold: 0.90
  fail_open: true
  classifier:
    enabled: false
```

The classifier remains opt-in because adding a network request before every uncertain main-model call can erase latency gains. When disabled, unresolved deterministic requests keep all tools. Direct DeepSeek is the default hosted classifier; OpenRouter is used only when explicitly selected. A local OpenAI-compatible endpoint can be configured as:

```yaml
classifier:
  enabled: true
  provider: custom
  model: router-local
  base_url: http://127.0.0.1:1234/v1
  api_key_env: null  # or an environment-variable name when authentication is required
```

## Development

```bash
python3 -m py_compile *.py tests/*.py benchmarks/*.py
python3 -m pytest
python3 benchmarks/run.py
python3 -m build
```

The committed 500-record corpus is a synthetic regression suite. It validates deterministic contracts; it does not replace live full-tools-versus-routed E2E testing.

## Current measured schema reduction

Against the live 39-tool Hermes registry, Hermes's own rough request estimator measured:

- `web`: 18,627 → 490 tokens (**97.37% reduction**)
- `file,terminal`: 18,627 → 3,207 tokens (**82.78% reduction**)
- `browser,web`: 18,627 → 3,364 tokens (**81.94% reduction**)

See [`docs/baselines/v0.2-rc1.md`](docs/baselines/v0.2-rc1.md) for commands and caveats. These are estimator results, not provider billing receipts.

## Live test-profile A/B validation

A paired full-tools-versus-routed evaluation using `openai/gpt-5.4-mini` produced:

- **10/10** safe-core cases passed;
- **100%** required-toolset recall;
- **100%** exact route accuracy;
- **100%** tool-call success and answer accuracy;
- **61.60%** average reduction in actual first-provider-request tokens across the no-tool, terminal, and web pairs;
- **94.239983** autoresearch score.

The run used stock Hermes hooks and no core patch. See [`docs/live-evaluation-v0.2-rc1.md`](docs/live-evaluation-v0.2-rc1.md) for the corpus, methodology, commands, raw token counts, and scope limitations.

## Current release gates

A stable release must demonstrate:

- at least 70% median first-turn schema-token reduction;
- at least 99.5% required-toolset recall and 100% critical-class recall;
- no unrecovered registered-tool failures in E2E testing;
- no task-success regression against the full-tool baseline;
- cache-stable serialized tool schemas after routing or the last expansion.

No quantitative production claim should be made until a versioned live validation report satisfies those gates.

## License

MIT.
