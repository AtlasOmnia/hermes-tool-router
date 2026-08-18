---
name: hermes-tool-router-onboarding
description: First-run onboarding for the hermes-tool-router plugin — explain what the router does, offer the optional classifier API key exactly once, and verify the router is live. Use when a fresh Hermes profile has hermes-tool-router installed and was never onboarded, or when the user asks about router setup, the classifier, or its API key.
triggers:
  - First interaction on a fresh Hermes profile that has the hermes-tool-router plugin installed
  - User asks what hermes-tool-router does, how to enable it, or how to configure the classifier/API key
  - User asks to verify that tool routing is active on their profile
---

# Hermes Tool Router — First-Run Onboarding

This skill is the onboarding surface for the **hermes-tool-router** plugin (the standalone
pre-turn tool router). Use it once per fresh profile: explain what the router does, offer the
optional classifier API key exactly once, and verify the router is live. It is written against
the plugin's real behavior — read `config.yaml` and `README.md` in the plugin checkout before
changing anything.

> The router is disabled by default and experimental. Test it on a separate Hermes profile
> first; do not install it directly on a primary profile.

## 1. What the router does

Before the first provider request of a session, the router predicts which toolsets the request
needs and narrows the live tool surface to only those toolsets, cutting first-turn token and
tool-schema overhead. Key behaviors (all in this plugin, v0.2+):

- **Pre-turn toolset prediction:** deterministic intent rules classify the user's first turn
  and load only the toolsets that request needs (hook: `pre_llm_call`). Measured schema
  reductions in the README baselines are large, but they are estimator results, not billing
  receipts — treat them as indicative, never as a guarantee.
- **Session-sticky surface:** later turns reuse the initial routed surface instead of
  reclassifying and shrinking it every turn (`routing_scope: first_turn`,
  `shrink_mid_session: false`).
- **Monotonic expansion:** if a pruned registered tool is requested later, its owning toolset
  is added permanently for the session and is never re-pruned (`expansion_mode: monotonic`,
  plus the `tool_request` middleware and `request_toolset` fallback).
- **Fail open:** uncertainty, invalid classifier output, missing confidence, timeout, registry
  mismatch, or unsupported runtime keeps the full tool surface. Nothing is ever pruned
  unless the router is confident (`fail_open: true`, `confidence_threshold: 0.90`).

The router is **disabled by default** (`global.enabled: false`). It must be enabled explicitly
(globally or per profile) before any of the above runs.

## 2. Why an API key is valuable — and why it is optional

The router is **fully functional with no API key at all**: deterministic rules handle the
common first-turn cases and fail-open handles everything else, with zero network calls.

The optional **classifier** is a separate, opt-in feature: when deterministic rules cannot
resolve a first-turn request, a small external model (direct DeepSeek by default, or OpenRouter
when explicitly selected) routes it instead of falling straight back to the full catalog. This
is typically more accurate for ambiguous requests and can avoid a full-catalog first turn.

The classifier is:

- **OPTIONAL** — keyless deterministic routing + fail-open is a complete, supported mode.
- **OPT-IN** — disabled by default (`classifier.enabled: false`); never auto-enabled.
- **CONFIDENCE-GATED** — any output below `confidence_threshold: 0.90` (or with missing/
  invalid confidence) fails open to the full surface.
- **FAIL-OPEN** — invalid JSON, unknown actions or toolsets, timeout, or a missing API key
  all fall back to the full tool surface, never a crash and never a wrong prune.

It is **never a universal speed guarantee**: adding a network request before uncertain calls can
erase latency gains, which is exactly why it stays opt-in. And it **never replaces the user's
main Hermes model** — it only classifies the first turn's toolset choice; the main model still
does all real work.

## 3. Decision gate — ask once

On a fresh profile's first interaction, ask the user a clear yes/no question, e.g.:

> The tool router works fine with no API key (deterministic rules + fail-open). Optionally, you
> can enable its classifier — a small external model that routes ambiguous first-turn requests
> more accurately. Do you want to add an API key for the classifier? (Yes / No)

Do **not** ask on any later interaction (see §5).

### Yes — enable the classifier

Walk the user through these steps (all config lives in the plugin's `config.yaml`; the README's
"Configuration" section has the exact YAML):

1. **Enable the router itself** — the classifier does nothing unless the router is on. Set
   `global.enabled: true` (or `profiles.<profile-name>.enabled: true` for a single profile).
2. **Enable the classifier** — set `classifier.enabled: true` (under `global:` or under the
   profile block).
3. **Pick a provider and expose the key:**
   - Direct DeepSeek (default — leave `classifier.provider` as `null` or unset): export
     `DEEPSEEK_API_KEY` in the profile's environment. Model defaults to `deepseek-chat`.
   - OpenRouter: set `classifier.provider: openrouter` and export `OPENROUTER_API_KEY`.
   - Custom OpenAI-compatible endpoint: set `provider`, `model`, `base_url`, and — only when
     authentication is required — `api_key_env` naming the env var that holds the key.
4. **Start a fresh session** after config changes — the router reads config at session start.

### No — nothing more to do

Affirm that the profile is fully covered: deterministic rules handle the common cases and
fail-open guarantees the full tool surface whenever they cannot. No key, no config change, no
further action needed. Record the decision (see §5) so future sessions skip the gate.

## 4. Verification — confirm the router is live

After enabling (or to confirm an existing install):

1. **Diagnostics:** run `python diagnostics.py` from the plugin checkout. Exit code `0` means a
   routing path is available before the provider request; exit code `2` means the current
   runtime must not claim first-turn savings. The JSON report also shows
   `first_turn_savings_available`, `automatic_recovery_available`, and the compatibility mode.
2. **Logs:** the plugin logs its activation as
   `hermes-token-router: profile=<name> enabled=True` (debug level). A fresh session's logs
   must show the profile name and `enabled=True`.
3. **Real tool-call smoke:** in a fresh session, make a request that actually needs a tool
   (e.g. web, file, or terminal work). Confirm the tool executes normally. If you want to
   exercise recovery, request something that was pruned from the routed surface and confirm its
   toolset is added for the rest of the session (monotonic expansion).

## 5. Offer once, no nagging

- Offer the classifier/API key **once**, on a fresh profile's first interaction.
- **Skip silently** when the profile is already configured: router already enabled, classifier
  already enabled, or the user already declined.
- **Never nag** on later interactions, regardless of the answer.
- Record the outcome (e.g. in profile memory/notes: "tool-router onboarding: classifier
  declined" or "classifier enabled via DeepSeek") so subsequent sessions can detect it and stay
  quiet.

## 6. Pitfalls

- **Keyless works fine.** The classifier is not required; deterministic rules + fail-open are a
  complete, supported configuration. Do not imply the router needs a key.
- **The classifier only classifies the first turn** (`routing_scope: first_turn`). It never
  re-routes mid-session; mid-session behavior is session-sticky with monotonic expansion.
- **Never attribute deterministic savings to the classifier.** The baseline savings come from
  deterministic rules + fail-open and exist with the classifier disabled (its default). The
  classifier is an accuracy aid for ambiguous requests, not the source of the savings.
- **Artifact caveats.** README token-reduction numbers are estimator results, not provider
  billing receipts, and release-gate claims require a versioned live validation report. Do not
  cite them as universal guarantees.
- **No key / wrong env var = silent fail-open.** If `DEEPSEEK_API_KEY`/`OPENROUTER_API_KEY` is
  missing, the classifier client is skipped and routing fails open (safe, but the classifier is
  not actually running). Verify the exact env var name if the user expected classification.
- **Latency trade-off.** A network call before uncertain requests can erase gains — another
  reason the classifier stays opt-in.
- **Fresh session required.** Config is read at session start; changing `config.yaml` mid-session
  has no effect until a new session.
- **Disabled-by-default.** Enabling the classifier while the router itself is disabled
  (`enabled: false`) does nothing.
