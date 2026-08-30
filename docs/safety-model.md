# Safety model

The router is an optimization, never an authorization boundary. It may remove schemas from a model request, but it must not bypass Hermes tool validation, terminal approvals, credential requirements, or platform restrictions. For protected definitions it also enforces a no-broadening transformation: a protected definition may be exposed only when it is in that agent/session's immutable host-admission envelope. Hermes remains the final execution authority.

## Fail-open conditions

- Unknown profile identity
- Missing early hook
- Registry lookup or generation mismatch
- Classifier disabled, unavailable, timed out, malformed, low-confidence, or returns unknown toolsets
- Partial/failed expansion
- Worker identity or owner mapping uncertainty (preserve the valid envelope; do not prune)
- Invalid or conflicting trusted policy (preserve the current surface and deny new additions)
- No trusted host input (preserve the current surface and use append-only ordinary compatibility only)

## Recovery

`request_toolset` accepts one or more live registry toolset names or a specific registered tool name. Successful expansion is permanent for the session. Automatic recovery uses Hermes's `tool_request` middleware before dispatch and never bypasses normal validation or approvals. Live registry definitions may support ordinary compatibility, but a protected definition is retrieved only from the immutable host envelope. The global recovery schema/catalog is discoverability only; the per-agent admission result is authoritative.

## Rev 6 host-admission contract

The standalone router separates two immutable concepts:

- **Protected ceiling:** the protected toolsets and exact definitions that the host admitted for this agent and session. Protected definitions never come from registry reconstruction.
- **Pinned floor:** admitted exact definitions that must remain visible. Every pin is protected, but an admitted protected definition may remain optional for a direct orchestrator.

The host supplies the policy through the trusted `hermes_token_router_admission` hook keyword and/or the agent's `_hermes_token_router_admission` attribute. A policy is exactly a version-1 mapping with `schema_version`, `protected_toolsets`, and `pinned_tool_names`. `MISSING/MISSING` means no policy was supplied and is a valid empty policy; explicit `None`, malformed values, duplicate/empty/whitespace-padded identifiers, and conflicting channels are invalid. Identifiers are never normalized.

Before the router's first mutation, the adapter resolves the live agent/session, reads the untouched definitions and enabled-toolset order, builds one ordered owner snapshot, evaluates worker identity once, composes an immutable effective policy, and binds one result to the agent-attached `RouterState`. The generic capability module is pure and has no Hermes, environment, registry, or protected-tool authority. Subsequent same-session callers reuse the identical bound result. Session end clears the binding; disable/re-enable does not recapture it.

A dispatcher-owned worker requires both a nonempty `HERMES_KANBAN_TASK` and the dispatcher-owned predicate. Complete owner mapping pins the exact incoming worker definitions. Predicate, registry, or owner uncertainty is `SAFE_NO_PRUNE`. Direct orchestrators may route only their admitted protected subset, with pins optional unless metadata pins an exact present name. Specialists, delegated children, inherited non-dispatcher contexts, and cron do not gain worker authority from inherited environment. An explicit present owner-unmapped pin remains protected by exact name for a known non-worker; absent siblings remain denied.

For an attachable agent without trustworthy untouched input, `NO_AUTHORITY` preserves the current surface, denies protected additions, and permits append-only ordinary compatibility. The first ordinary append persists an immutable `NoAuthorityContamination` marker before any surface assignment; later same-session calls remain append-only for ordinary tools but can never bind an envelope or promote the appended definition. Invalid policy returns transient `NO_AUTHORITY_INVALID_POLICY`, preserves the current surface, denies every new addition, creates no clean marker, and retains an existing marker. An unattachable agent is installed-only for ordinary tools and cannot add definitions because no session-local guard can be persisted. Capture-invalid input is stricter: it binds `CAPTURE_INVALID_NO_MUTATION` and permits no expansion.

All expansion callers—route, visible `request_toolset`, `tool_request` middleware, post-tool recovery, and fallback—return deterministic admitted/denied/installed fields. Mixed requests may append independently admitted ordinary definitions while explicitly denying protected items. Total protected denial and capture-invalid handling are side-effect-free. Candidate construction and coordinated assignment are transactional; failures restore the exact pre-call surface, effective state, counters, fallback/retry fields, and lifecycle-slot identity. Fallback restores a fresh thaw of the exact envelope or preserves the current no-authority surface; it never fills protected gaps from the live registry.

The empty-message branch runs only after agent/session resolution and admission binding. It performs no classifier, available-toolset, definition-retrieval, recovery-control, retry, or recapture work. The early hook is preferred; `pre_llm_call` remains a late compatibility path for older Hermes builds and cannot claim pre-preflight savings. These local tests are synthetic contract evidence, not production runtime acceptance or rollout approval.

## Privacy

Deterministic routing is local. External classification is opt-in. Local metrics must not store raw prompts by default, and no telemetry is sent externally by this plugin.
