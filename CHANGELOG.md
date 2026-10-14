# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [SemVer](https://semver.org/) with pre-release tags (e.g. `0.2.0rc1`).

## [Unreleased]

### Fixed

- Current Hermes runs `pre_llm_call`/`post_tool_call` on a timeout worker thread, where the stack walk could not see the agent, so the router silently did nothing. Agent lookup now also searches other live threads for the agent with the exact hook `session_id` (never another session's agent).
- Review round 9 (dev-critic2: ship, medium confidence): no code change; three residual non-blocking issues documented as known limits in the README (expired teardown + reuse guard gap, commit-to-latch overlap, teardown reference cleanup after releasing an older quarantine).
- Review round 8: retirement handoff is strict — the retired envelope is restored only while the agent shows exactly the router's last committed surface (definitions and enabled toolsets); any other change keeps the quarantine (no release on partial or same-name changes); recovery is discarding the agent. Session-end teardown is fenced with its own deadline. Overlapping post-tool callbacks commit at most one automatic recovery per turn (latch revalidated at the commit boundary). A stale visible request reports `STALE_LIFECYCLE` instead of `SESSION_MISMATCH`.
- Review round 7: every live entry point (route, post-tool, `tool_request` middleware, visible `request_toolset`) runs under one lifecycle fence. Admission preparation (registry ownership, policy) runs unlocked and only the binding is fenced, so a stale hook never reserves a session slot. A failed session-end restore now revokes in-flight hooks (generation bump) while keeping the admission. Releasing a quarantined agent runs under the route's fence and deadline and never writes old-session tools over a surface the host has already replaced; a persistent quarantine logs one warning per session with the recovery step. Expansions are prepared against a surface revision and recomputed if another commit landed first (no lost updates). An explicit `request_toolset` no longer resets the automatic retry latch.
- Review round 6: every surface or lifecycle mutation (route, fallback restores, admission binding, post-tool expansion, turn bookkeeping, retry latch) is checked against the hook's fence (lifecycle generation, exact session, bound admission, deadline) under the per-agent lock immediately before assignment; slow preparation (config, registry, catalog) runs unlocked, so session end never waits on a stalled hook. The deadline is fixed at hook start from Hermes's own normalized `plugins.hook_callback_timeout` (half of it; none when the timeout is disabled). An agent reused after a failed session-end restore is quarantined: no capture or routing, the retired policy keeps guarding protected siblings, and the restore is retried until it succeeds. The post-tool retry latch resets only on a new turn. Case-insensitive toolset matches must be unique.
- Follow-ups: `skill_view`/`skills_list` stay visible after narrowing when captured (Hermes puts the skills index in the system prompt; pruning them made models report "no skills"). Removed the shadowed legacy routing path and its helpers (~700 lines). Classifier confidence must land in [0,1] after the one tolerant rescale (101/"250%" fail open) and any unknown toolset name fails open; the classifier HTTP timeout now equals the routing hard timeout. `floor_toolsets` defaults to none, matching the shipped config. The compatibility report counts `tool_request` middleware as recovery. `request_toolset` resolves exact (mixed-case) toolset names before lower-casing, never reports success for a tool that is still not installed, reports the real admission outcome in `reason` and the caller's text in `request_reason`. The post-tool retry latch resets every turn.
- Review round 5: a failed session-end restore keeps the session's admission (guard engaged, retry possible) instead of discarding it; route and post-tool commits are fenced by a lifecycle generation, session match, admission identity, lock and deadline so a hook abandoned by a Hermes timeout can never overwrite a later session's surface; "Tell me about your tools" is recognized again.
- Review round 4: a failed disable-time restore no longer clears the narrowed state (the protected-tool guard stays engaged and the restore is retried); an explicit route agent from another session is ignored; cross-thread lookup treats `agent`/`self` duplicates as ambiguous; the own-stack walk checks the session before any other attribute; session end restores a narrowed surface and resets routing state; capability questions are recognized only when every word is inventory vocabulary or filler (fixes "Show my cron jobs using your tools").
- A disabled router is now a true no-op on a session it never narrowed (it previously re-applied its first-turn snapshot every turn, removing tools added mid-session). Agent lookup validates the session id at every step, prefers an explicit agent, rejects ambiguous cross-thread matches, and contains exceptions from foreign objects. Envelope restore assigns the captured `enabled_toolsets` exactly. A `pre_tool_call` guard keeps uncaptured tools of protected toolsets unreachable through Hermes `tool_call` while narrowed. Capability questions must be the whole message ("list your tools and schedule a reminder" keeps its cronjob route).
- Narrowing no longer breaks Hermes tool search: the `tool_search`/`tool_describe`/`tool_call` bridge is always kept, and toolsets reachable only through the bridge stay in `enabled_toolsets` (protected toolsets excepted), so deferred tools such as `computer_use`, `cronjob_manage`, `browser_vault`, MCP and plugin tools remain usable. Envelope restore now restores `enabled_toolsets` exactly when the bridge is present. The capability catalog points the model to `tool_search` for deferred tools.

- The model can now say what it can do after first-turn narrowing: `request_toolset` carries a per-session capability catalog (admitted toolsets with one-line descriptions) instead of a generic one-line description, so capability questions no longer depend on the model first attempting a pruned tool. The `toolsets` parameter description is now generated from the registry, as documented.
- Standalone capability questions ("what can you do?", "what tools do you have access to?") are classified as `capability_inquiry` and route to the small surface consistently; previously some wordings fell back to the full surface and others to answer-only. Messages that also carry a task keep their normal routing.
- Partially admitted toolsets are listed with their admitted tool names rather than the toolset's generic description.

- OpenRouter classifier requests now attribute `HTTP-Referer` to this repository (`AtlasOmnia/hermes-tool-router`) instead of the upstream `hermes-agent` org.
- Classifier response parsing is consolidated onto one strict fail-open contract (`parse_classifier_payload`), shared with `classifier.parse_classifier_output`. Documented string/percent confidence tolerance ("95%", `"0.93"`) is retained; non-finite values (NaN/inf) now provably fail open.
- Replaced a lambda assignment in the smoke test (ruff E731); lint is clean.
- Protected tool definitions now remain bounded by immutable per-agent/session host admission, with exact pinned floors, append-only ordinary compatibility, and transactional rollback across recovery paths.
- Invalid or missing admission authority fails open without broadening protected capabilities; session contamination is recorded before ordinary compatibility appends.

### Added

- Contract tests covering `parse_classifier_payload`: missing/malformed/nonfinite/below-threshold confidence, "all" signal, empty toolset lists, unknown toolsets, and invalid JSON all fail open.
- Added focused Rev 6 contract coverage for host admission, protected-envelope recovery, lifecycle boundaries, truthful retry, and atomic rollback.

## [0.2.0rc1] - 2026-08-06

### Added

- First-turn, session-sticky tool-surface routing via stock Hermes hooks (`pre_turn_context_build` primary, `pre_llm_call` compatibility).
- Monotonic per-session recovery through `tool_request` middleware plus a model-visible `request_toolset` fallback generated from the live registry.
- Optional external classifier (DeepSeek default, OpenRouter supported, custom OpenAI-compatible endpoints) with hard deadline daemon-thread transport, confidence gating, and fail-open behavior on every ambiguous condition.
- Deterministic intent policy with inert-probe no-tool routes (whitespace/period-only prompts).
- Benchmarks corpus and Hermes-native schema-token measurement scripts; live evaluation docs.
- GitHub Actions CI matrix for Python 3.11 / 3.12 / 3.13 (compile, pytest, benchmarks, build).
