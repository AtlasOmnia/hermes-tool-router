# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [SemVer](https://semver.org/) with pre-release tags (e.g. `0.2.0rc1`).

## [Unreleased]

### Fixed

- OpenRouter classifier requests now attribute `HTTP-Referer` to this repository (`AtlasOmnia/hermes-tool-router`) instead of the upstream `hermes-agent` org.
- Classifier response parsing is consolidated onto one strict fail-open contract (`parse_classifier_payload`), shared with `classifier.parse_classifier_output`. Documented string/percent confidence tolerance ("95%", `"0.93"`) is retained; non-finite values (NaN/inf) now provably fail open.
- Replaced a lambda assignment in the smoke test (ruff E731); lint is clean.

### Added

- Contract tests covering `parse_classifier_payload`: missing/malformed/nonfinite/below-threshold confidence, "all" signal, empty toolset lists, unknown toolsets, and invalid JSON all fail open.

## [0.2.0rc1] - 2026-08-06

### Added

- First-turn, session-sticky tool-surface routing via stock Hermes hooks (`pre_turn_context_build` primary, `pre_llm_call` compatibility).
- Monotonic per-session recovery through `tool_request` middleware plus a model-visible `request_toolset` fallback generated from the live registry.
- Optional external classifier (DeepSeek default, OpenRouter supported, custom OpenAI-compatible endpoints) with hard deadline daemon-thread transport, confidence gating, and fail-open behavior on every ambiguous condition.
- Deterministic intent policy with inert-probe no-tool routes (whitespace/period-only prompts).
- Benchmarks corpus and Hermes-native schema-token measurement scripts; live evaluation docs.
- GitHub Actions CI matrix for Python 3.11 / 3.12 / 3.13 (compile, pytest, benchmarks, build).
