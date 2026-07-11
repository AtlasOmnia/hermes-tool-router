# Architecture

The v2 router classifies the initial user turn, applies the smallest confidently sufficient toolset, and keeps that serialized schema surface stable. State is stored on the live agent. Recovery only adds capabilities; it never removes them.

## Invariants

1. Route before schema assembly or do not claim savings.
2. Classify once per session.
3. `active_toolsets` is monotonic.
4. Missing/invalid confidence fails open.
5. Dynamic registry names drive recovery schemas.
6. Unknown profiles never inherit the first enabled profile by insertion order.
7. External classification is opt-in.

## Required upstream seams

- Early surface hook: `pre_tool_surface_build` or `pre_turn_context_build`, with the live agent passed explicitly.
- Tool-request middleware: Hermes's existing `tool_request` middleware runs before validation/dispatch. The plugin uses it to expand a registered pruned tool's owner toolset; normal check functions, approvals, and execution remain authoritative.
