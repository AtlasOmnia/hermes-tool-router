# Safety model

The router is an optimization, never an authorization boundary. It may remove schemas from a model request, but it must not bypass Hermes tool validation, terminal approvals, credential requirements, or platform restrictions.

## Fail-open conditions

- Unknown profile identity
- Missing early hook
- Registry lookup or generation mismatch
- Classifier disabled, unavailable, timed out, malformed, low-confidence, or returns unknown toolsets
- Partial/failed expansion

## Recovery

`request_toolset` accepts one or more live registry toolset names or a specific registered tool name. Successful expansion is permanent for the session. Automatic recovery uses Hermes's `tool_request` middleware before dispatch and never bypasses normal validation or approvals.

## Privacy

Deterministic routing is local. External classification is opt-in. Local metrics must not store raw prompts by default, and no telemetry is sent externally by this plugin.
