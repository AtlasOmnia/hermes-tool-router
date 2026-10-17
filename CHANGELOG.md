# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [SemVer](https://semver.org/) with pre-release tags (e.g. `0.2.0rc1`).

## [Unreleased]

### Added

- `classifier.extra_body` (forwarded to the classifier `chat.completions` call) and `classifier.disable_thinking: true` sugar for reasoning-template local models (#20, #21).
- One INFO `surface committed` line per committed surface; opt-in `global.debug_logging` writes a rotating `router-debug.log` (#20, #21).
- `/router on|off|status|reset` slash command: switches routing for the running Hermes process without a model call or config change (CLI, desktop/TUI, gateways; not one-shot `-z`). `HERMES_TOKEN_ROUTER_ENABLED=0` stays a kill switch it cannot override; turning a config-disabled profile on keeps that profile's configured settings; status reports config, environment and override separately.
- Capability awareness after first-turn narrowing: `request_toolset` carries a per-session capability catalog (admitted toolsets with one-line descriptions; partially admitted toolsets list their admitted tools), so "what can you do?" is answered from the catalog instead of from the pruned surface. The `toolsets` parameter description is generated from the registry.
- Standalone capability questions ("what can you do?", "list your tools", "tell me about your tools") are classified as `capability_inquiry` and keep the small surface; messages that also carry a task keep their normal routing.
- A `pre_tool_call` guard keeps uncaptured tools of protected toolsets unreachable through Hermes `tool_call` while the surface is narrowed (and, after a failed session-end restore, for a reused agent).
- `skill_view` and `skills_list` stay visible after narrowing when the host captured them (not for a protected, unpinned skills toolset). Hermes lists skills in the system prompt; pruning them made models report "no skills".
- `request_toolset` returns the machine outcome in `reason` (`READY`, `NO_AUTHORITY`, `UNRESOLVED_TOOL`, `STALE_LIFECYCLE`, ...) and echoes the caller's justification as `request_reason`.
- Contract tests for `parse_classifier_payload`, Rev 6 host admission / protected-envelope recovery / lifecycle / rollback, and regression tests for every review finding below (146 tests).

### Changed

- Lifecycle fencing: every live entry point (first-turn route, `post_tool_call`, `tool_request` middleware, visible `request_toolset`, `on_session_end`) runs under one fence — lifecycle generation, exact session id, bound admission, and a deadline of half of Hermes's normalized `plugins.hook_callback_timeout` (none when the timeout is disabled). Slow preparation (config, registry, catalog, admission ownership/policy) runs outside the per-agent lock; every coordinated assignment is revalidated under the lock immediately before it is applied, so a hook Hermes abandoned after a timeout never commits into a later session and session end never waits on a stalled hook.
- Residual limits from review round 9 are fixed: a session end that expires before recording retirement no longer leaves a reused agent unguarded (the still-bound admission is adopted as quarantine); automatic post-tool recovery claims its once-per-turn latch inside the same transaction as its commit; a session end that releases an older quarantine completes its own teardown.
- Failed session-end restore: the admission and guard are kept, in-flight hooks of that session are revoked, and the restore is retried at the next turn or session end. A reused agent is quarantined (no capture or routing, retired policy still guards protected tools); it is released only by restoring the retired envelope while the agent shows exactly the router's last committed surface. Any other change is treated as ambiguous; recovery is to discard the agent.
- Expansions are prepared against a surface revision and recomputed (up to 3 times) if another commit landed first, so overlapping recoveries never drop each other's tools. Automatic post-tool recovery is limited to once per turn (latch revalidated at commit; an explicit `request_toolset` does not reset it).
- Classifier: confidence must be in [0, 1] after the one tolerant rescale ("95%", `"0.93"`; 101 or "250%" fail open), any unknown toolset name fails open, and the HTTP timeout equals the routing hard timeout. Parsing is one strict fail-open contract (`parse_classifier_payload`) shared with `classifier.parse_classifier_output`.
- `floor_toolsets` defaults to none, matching the shipped config. The compatibility report counts `tool_request` middleware as recovery.
- `request_toolset` matches exact registered toolset names first (mixed-case custom/MCP toolsets), then a unique case-insensitive match; ambiguous matches are rejected.
- Protected tool definitions stay bounded by immutable per-agent/session host admission, with exact pinned floors, append-only ordinary compatibility and transactional rollback; invalid or missing admission authority fails open without broadening protected capabilities.
- OpenRouter classifier requests attribute `HTTP-Referer` to this repository (`AtlasOmnia/hermes-tool-router`).

### Fixed

- Profile resolution under a multiplexed gateway: turn-time routing binds to the agent's own profile (`agent._profile_name`) before the process environment, so served sessions no longer silently apply the launch profile's policy (#20, #21).
- The router silently did nothing on current Hermes: `pre_llm_call`/`post_tool_call` run on a timeout worker thread where the stack walk could not see the agent. Agent lookup now also searches other live threads for the agent with the exact hook `session_id` (never another session's agent; ambiguous matches rejected; foreign-object exceptions contained).
- Narrowing broke Hermes tool search: the `tool_search`/`tool_describe`/`tool_call` bridge is now always kept and toolsets reachable only through it stay in `enabled_toolsets` (protected toolsets excepted), so deferred tools such as `computer_use`, `cronjob_manage`, `browser_vault`, MCP and plugin tools remain usable.
- A disabled router re-applied its first-turn snapshot every turn, removing tools added mid-session; it is now a no-op on a session it never narrowed, and a failed disable-time restore keeps the guard engaged and is retried.
- `request_toolset` reported success for a tool that was still not installed, and labelled successful expansions `NO_AUTHORITY_CONTAMINATED`.
- The plugin manifest declared `pre_turn_context_build`, which current Hermes does not know (`hermes plugins doctor` reported an error); the hook is still registered when the host supports it.
- Smoke-test lambda assignment (ruff E731).

### Removed

- The shadowed legacy routing path (`_route_tool_surface` duplicate) and its legacy-only helpers (~700 lines).

### Known limits

- Disabling after narrowing restores the captured surface once and drops tools added while narrowed; re-enabling in the same session does not narrow again. The guard lets connector (`connectors__*`) and unregistered names through.

## [0.2.0rc1] - 2026-08-06

### Added

- First-turn, session-sticky tool-surface routing via stock Hermes hooks (`pre_turn_context_build` primary, `pre_llm_call` compatibility).
- Monotonic per-session recovery through `tool_request` middleware plus a model-visible `request_toolset` fallback generated from the live registry.
- Optional external classifier (DeepSeek default, OpenRouter supported, custom OpenAI-compatible endpoints) with hard deadline daemon-thread transport, confidence gating, and fail-open behavior on every ambiguous condition.
- Deterministic intent policy with inert-probe no-tool routes (whitespace/period-only prompts).
- Benchmarks corpus and Hermes-native schema-token measurement scripts; live evaluation docs.
- GitHub Actions CI matrix for Python 3.11 / 3.12 / 3.13 (compile, pytest, benchmarks, build).
