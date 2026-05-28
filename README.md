# Hermes Token Router

A public-ready packaging folder for the Hermes router-first tool-loading work.

This project is for people who already run Hermes and want to reduce first-turn token bloat by loading only the tool groups that are likely needed for the user's request. If the router is unsure, it declines reduction and Hermes loads the full tool surface.

## What this includes

- A portable external policy plugin: `plugin/hermes-token-router/`
- Public source overlay/reference files under `hermes-agent-files/`
- Patch status note: `patches/PATCH_NOT_INCLUDED.md`
- Smoke and feedback scripts for measuring first-prompt token savings
- Focused tests that cover routing, fallback behavior, compact identity, and feedback privacy

## Design summary

The router runs before the main model call. It predicts a small set of toolsets such as `file`, `terminal`, or `web`. Hermes then assembles only those tool schemas for the first turn.

Important safety rule: when prediction returns `([], False)`, the router has declined to reduce. That means Hermes should load the full tool surface, not a tiny clarify-only floor.

The external policy plugin lives outside the Hermes core checkout so Hermes updates are less likely to overwrite local routing rules.

## Conservative routing

For conservative deployments, set a confidence threshold and keep the long-message decline threshold visible in router config:

```yaml
router_config:
  enabled: true
  confidence_threshold: 0.7
  long_message_decline_chars: 600
```

Without a confidence threshold, a classifier response with low confidence can still reduce the tool surface. A higher threshold preserves quality by forcing uncertain prompts back to the full tool surface.

`long_message_decline_chars` defaults to `600`. That is intentionally conservative: longer, multi-step prompts skip router reduction and load the full tool surface. If your prompts are often long and you want more savings, you can raise this value, but doing so sends more complex prompts through the classifier and can increase misrouting risk.

## Local router model

The router classifier defaults to a hosted auxiliary model, but it can point at any OpenAI-compatible local endpoint. For LM Studio or a local vLLM/OpenAI-compatible server:

```yaml
router_config:
  enabled: true
  provider: custom
  model: local-router-model
  base_url: http://localhost:1234/v1
  api_key: local
  confidence_threshold: 0.7
  long_message_decline_chars: 600
```

For Ollama's OpenAI-compatible endpoint, use the Ollama model name and base URL for your setup, commonly:

```yaml
router_config:
  enabled: true
  provider: custom
  model: qwen2.5:7b
  base_url: http://localhost:11434/v1
  api_key: local
  confidence_threshold: 0.7
```

Use a small, reliable instruction-following model for the router. If the classifier is weak, keep `confidence_threshold` high or disable reduction for safety.

## Measuring savings

Do not rely on static benchmark numbers from someone else's machine. Token savings depend on the Hermes version, enabled toolsets, provider, model, and local policy.

Run the smoke script in your Hermes checkout:

```bash
python scripts/router_smoke.py --profile all --strict-failures
```

Read these fields in the output:

- `first_prompt`: the first prompt payload after routing; this is the main number to compare.
- `prompt`: provider-reported prompt tokens for the request.
- `effective`: the toolsets actually loaded for that case.
- `full_surface`: whether the router declined reduction and loaded the full tool surface.

For an apples-to-apples comparison, run once with router reduction enabled and once with it disabled, using the same model and Hermes config.

## Cache tradeoff

The gateway cache key includes a hash of the current user message so each unique prompt can get the right tool surface. This improves per-turn tool accuracy and is especially useful for single-turn API use.

For longer conversational sessions, that choice can reduce reuse of cached agent instances and tool-schema prompt prefixes when the needed tool list changes between turns. In plain terms: this design favors per-turn correctness over maximum multi-turn prompt-cache reuse.

## Three memory/context tiers

1. Slim startup identity: compact turns use `SOUL-compact.md`, a marked compact section in `SOUL.md`, or a short truncated SOUL fallback. If no SOUL file exists, Hermes uses the generic default identity.
2. Skill-aware context: skills declare the tools they require, so choosing a skill also brings the right toolsets.
3. Feedback telemetry: local aggregate records track route, effective toolsets, first-prompt tokens, and failures without storing raw prompts.

## Compact SOUL options

Preferred dedicated file:

```md
~/.hermes/SOUL-compact.md
```

Or a curated section inside `SOUL.md`:

```md
<!-- compact-start -->
Short portable identity here.
<!-- compact-end -->
```

If `<!-- compact-start -->` is present without `<!-- compact-end -->`, compact identity is read from the start marker to the end of the file. If neither marker nor dedicated file exists, compact turns use the start of `SOUL.md`, capped by `HERMES_COMPACT_SOUL_MAX_CHARS` or the built-in default.

## Install sketch

This folder is not published and does not install itself yet.

For a manual test install:

1. Back up your Hermes checkout.
2. Compare/copy the files in `hermes-agent-files/` into the matching paths in `~/.hermes/hermes-agent`.
3. Copy `plugin/hermes-token-router/` to `~/.hermes/plugins/hermes-token-router/`.
4. Restart Hermes.
5. Run `python scripts/router_smoke.py --profile matrix --strict-failures` from the Hermes checkout.

No install script is included yet. Version-aware install tooling should wait until Hermes has a stable public patch/API surface; a validator that cannot reliably detect compatibility would create false confidence.

On Windows, `~/.hermes` means `%USERPROFILE%\.hermes`.

## Patch note

A generated patch file is intentionally not included because stale local patches can drift from the sanitized source overlay. Regenerate patches only from a clean public baseline after final review.

## Privacy

Do not publish your real `.hermes` folder, config files, `.env`, auth files, session databases, logs, memories, skills, router feedback events, or provider profiles. This public package intentionally excludes those.