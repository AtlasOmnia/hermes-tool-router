# Router on/off smoke test — 2026-10-14

Paired run on a cloned test profile after #16 (capability catalog, current-Hermes compatibility). Each prompt was sent once with the router forced off (`HERMES_TOKEN_ROUTER_ENABLED=0`) and once forced on (`HERMES_TOKEN_ROUTER_ENABLED=1`), as a fresh one-shot session (`hermes -z`). Deterministic rules only (no classifier). Main model: Claude Opus via the Anthropic API.

Token figures are the provider-reported input tokens for the whole task (fresh input + cache read + cache write), summed over every main-model call in the session, from Hermes's `session_model_usage` table. Title-generation calls are excluded.

## No-tool prompts (one model call = the starting prompt)

| Prompt | Router off | Router on | Change |
|---|---:|---:|---:|
| What can you do? | 29,030 | 17,954 | −38% |
| Explain the difference between TCP and UDP in two sentences. | 29,490 | 18,414 | −38% |
| Write a haiku about autumn. | 29,036 | 19,958 | −31% |
| **Average** | **29,185** | **18,775** | **−36%** |

## Tool tasks (whole task, all calls)

| Prompt | Off: calls / tokens | On: calls / tokens | Change | Tools called (on) |
|---|---:|---:|---:|---|
| Read the file /etc/hosts and tell me how many lines it has. | 2 / 58,318 | 2 / 40,162 | −31% | `read_file` |
| Run the shell command uname -s and tell me the result. | 2 / 59,379 | 2 / 40,123 | −32% | `terminal` |
| Search the web for the latest stable Python release version. | 2 / 62,286 | 3 / 62,268 | 0% | `web_search`, `web_extract` |
| List my scheduled jobs. | 3 / 96,786 | 2 / 42,324 | −56% | `cronjob_manage` |

With the router off, the scheduled-jobs task needed `tool_describe` + `tool_call` (Hermes tool search) to reach the deferred cron tool; routed, the tool was already in the narrowed surface. In the web task the routed run chose one extra fetch, which cancelled its savings.

**All seven tasks:** 364,325 → 241,203 input tokens (−34%).

All 14 runs exited 0, answers were equivalent in content and correctness, and the router logged no warnings or errors.

## Larger profile

The same "What can you do?" prompt on a larger production-style profile (more toolsets and integrations, OpenRouter classifier enabled): 58,608 → 24,813 input tokens (−58%).

## Caveats

- One run per prompt and mode, one main model; not a statistical benchmark.
- Token counts, not cost: cache reads are billed far below fresh input, so dollar savings differ.
- Savings scale with the size of the full tool catalog; small profiles save less.
- One-shot sessions measure the first turn; later turns reuse the session-sticky surface.
