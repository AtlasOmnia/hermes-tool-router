# Proposal: `pre_turn_context_build` Hook for Hermes

## Status: Version-dependent

The hook existed in a July 1, 2026 Hermes build but is not present in every later/current hook registry. The plugin therefore supports stock `pre_llm_call` as a compatibility route and uses live diagnostics rather than version strings. This document remains the proposal for a cleaner explicit-agent, pre-preflight surface.

Minimum Hermes requirement for the clean pre-preflight path: a build containing `pre_turn_context_build` (or an equivalent early surface hook) and its valid-hook registry entry. Builds exposing only `pre_llm_call` use the stock late compatibility path, which still narrows the actual provider request but cannot change work already performed by the initial preflight stage.

## Summary

Add a generic Hermes core hook named `pre_turn_context_build` that fires once per turn before Hermes assembles the system prompt, skill prompt, preflight token estimate, and final tool schema payload.

This hook would let plugins such as `hermes-token-router` adjust the active tool surface before prompt assembly without monkey-patching `agent.turn_context.build_turn_context` at runtime.

## Motivation

The test-profile token-router currently needs to reduce tool schema bloat before the main LLM request is built. Today it does that with a plugin-side runtime wrapper around `build_turn_context`. The wrapper is intentionally fail-open and guarded by `inspect.signature`, but it still depends on Hermes internals:

- positional argument order in `build_turn_context`;
- `summarize_user_message_for_log` being passed as a keyword argument;
- the turn-context function remaining the single prompt/tool assembly entrypoint.

A first-class hook would replace that brittle shim with a stable plugin contract.

## Proposed Hook

Hook name:

```text
pre_turn_context_build
```

Alternative acceptable name:

```text
pre_prompt_assembly
```

The hook should fire after Hermes has initialized per-turn metadata and refreshed available tools, but before:

- system prompt restore/build;
- skill prompt injection/filtering;
- preflight token estimate/compression;
- final tool schema assembly for the provider request.

## Suggested Signature

Hermes core would invoke the hook with a dictionary payload similar to:

```python
invoke_hook(
    "pre_turn_context_build",
    agent=agent,
    session_id=agent.session_id,
    task_id=effective_task_id,
    turn_id=turn_id,
    user_message=original_user_message,
    conversation_history=list(messages),
    is_first_turn=(not bool(conversation_history)),
    model=agent.model,
    provider=getattr(agent, "provider", "") or "",
    platform=getattr(agent, "platform", None) or "",
    sender_id=getattr(agent, "_user_id", None) or "",
    available_toolsets=list(getattr(agent, "enabled_toolsets", []) or []),
    available_tool_names=list(getattr(agent, "valid_tool_names", set()) or []),
)
```

The important part is that the live `agent` object is included. Some plugins need to mutate `agent.tools`, `agent.valid_tool_names`, or `agent.enabled_toolsets` before prompt/tool assembly.

## Return Contract

The hook can support either observer-style mutation or a structured return. The safest initial contract is:

```python
None
```

Meaning: no-op, or plugin has already mutated the live agent directly.

A future structured return could be supported:

```python
{
    "toolsets": ["web", "browser"],
    "tool_names": ["web_search", "browser_open"],
    "context": "optional user-context injection"
}
```

For the first implementation, `None` is enough. It matches current hook style and keeps backward compatibility simple.

## Current Wrapper vs. Clean Hook

Current experimental test-profile shim:

```text
plugin register()
  -> import agent.turn_context
  -> inspect build_turn_context signature
  -> monkey-patch build_turn_context
  -> wrapper waits for summarize_user_message_for_log
  -> wrapper calls token-router before prompt assembly
  -> original build_turn_context continues
```

Clean hook registration:

```text
plugin register()
  -> ctx.register_hook("pre_turn_context_build", pre_turn_context_build)

Hermes build_turn_context()
  -> invoke_hook("pre_turn_context_build", agent=agent, ...)
  -> system prompt assembly
  -> skill prompt assembly
  -> preflight estimate/compression
  -> provider tool schema assembly
```

The token-router plugin would delete `integration.py`'s monkey patch and move the early-route bridge into a normal hook handler.

## Token-Router Usage

The token-router handler would look like:

```python
def pre_turn_context_build(**kwargs):
    agent = kwargs["agent"]
    user_message = kwargs.get("user_message", "")
    # Predict toolsets, mutate agent.tools / valid_tool_names / enabled_toolsets.
    # Keep request_toolset available as the recovery tool.
    return None
```

The existing late `pre_llm_call` hook could remain as a fallback for older Hermes versions or be removed once the new hook is available everywhere.

## Backward Compatibility

Plugins that do not register `pre_turn_context_build` are unaffected.

If no plugin handles the hook, Hermes continues exactly as it does today.

Existing hooks such as `pre_llm_call` and `post_tool_call` do not need to change. This proposal adds one earlier extension point; it does not remove or reinterpret existing hooks.

## Concurrency

The hook is invoked once per turn with the live per-turn `agent` reference and turn identifiers. Plugins should store mutable state on the agent or key it by `session_id` + `turn_id`.

Hermes core does not need shared global state for this hook. The call is local to the current turn-context build, which makes it safer for gateway sessions and concurrent profiles.

## Failure Handling

Hermes should treat hook exceptions like other plugin hook failures:

- log the plugin error;
- continue with the original full tool surface;
- do not fail the user turn because an optimization hook failed.

This preserves the token-router's fail-open safety model.

## Non-Goals

This proposal does not implement automatic invalid-tool recovery in `conversation_loop.py`.

That is a separate seam. A future hook such as `on_invalid_tool_call` or `on_pruned_tool_call` could let plugins expand tools when a model calls a registry-known tool that was pruned from the active request. This proposal only replaces the runtime wrapper needed before prompt/tool assembly.

## Migration Plan

1. Add `pre_turn_context_build` invocation to Hermes core before prompt/tool assembly. Done 2026-07-01.
2. Update `hermes-token-router` to register the new hook. Done 2026-07-01.
3. Remove the runtime wrapper from the plugin. Done 2026-07-01.
4. Keep `pre_llm_call` registered as a backward-compatible late fallback for older Hermes builds or missed early-hook calls.
