# Compatibility

| Runtime hooks | Mode | First-turn savings | Recovery |
|---|---|---:|---|
| Early or late routing hook + `tool_request` middleware | production recovery | Yes | Automatic plus `request_toolset` |
| Routing hook without `tool_request` middleware | reduced safety | Yes | `request_toolset` only |
| No routing hook | unsupported for token savings | No | None |

The repository originally documented `pre_turn_context_build` as implemented in Hermes on July 1, 2026. Current Hermes source must still be checked because hook availability can change between releases. `diagnostics.py` reads the live hook registry rather than trusting version strings.

The plugin fails open when no routing hook is available. The stock `pre_llm_call` path can reduce the actual provider payload because it runs before request assembly, though it does not affect the earlier preflight estimate and currently relies on compatibility agent lookup.

## Current tested requirement

Current Hermes needs no source modification for the tested late-compatibility mode. The plugin uses stock `pre_llm_call` and `tool_request` middleware. An early `pre_turn_context_build`-style hook remains an optional upstream optimization for reducing preflight work, not an installation requirement.
