# Architecture

The v2 router classifies the initial user turn, applies the smallest confidently sufficient toolset, and keeps that serialized schema surface stable. State is stored on the live agent. Recovery may append only ordinary compatibility definitions; protected definitions are selected only from the agent/session's immutable host-admission envelope and are never reconstructed from the global registry.

## Invariants

1. Route before schema assembly or do not claim savings.
2. Classify once per session.
3. `active_toolsets` is monotonic.
4. Missing/invalid confidence fails open.
5. The global recovery schema/catalog is discoverability only; protected definitions come only from a valid immutable envelope.
6. Unknown profiles never inherit the first enabled profile by insertion order.
7. External classification is opt-in.

## Required upstream seams

- Early surface hook: `pre_tool_surface_build` or `pre_turn_context_build`, with the live agent passed explicitly.
- Tool-request middleware: Hermes's existing `tool_request` middleware runs before validation/dispatch. The plugin uses it to expand a registered pruned tool's owner toolset; normal check functions, approvals, and execution remain authoritative.

## Rev 6 admission data flow

The live composition root is deliberately split:

```text
trusted hook kwargs / agent attribute
        -> read_trusted_host_policy()
untouched agent.tools + enabled_toolsets
        -> one ordered OwnerSnapshot
HERMES_KANBAN_TASK + dispatcher-owned predicate (adapter only)
        -> _compose_effective_policy(OwnerSnapshot | None, ...)
        -> agent-attached RouterState.ensure_host_admission()
        -> tools.py candidate selection / transactional commit
```

`capabilities.py` is pure standard-library code. It owns the type-tagged JSON freeze/thaw codec, immutable `TrustedHostPolicy`, `OwnerSnapshot`, `HostAdmissionEnvelope`, contamination/result contracts, and pure candidate-facing decisions. It never imports Hermes, reads environment, calls worker predicates, consults the live registry, or names Kanban. The adapter in `__init__.py` is the only Hermes-aware layer: it supplies one real snapshot for authoritative untouched capture or typed `None` plus literal neutral worker values for no-envelope calls. `MISSING` is restricted to the generic state provenance seam and is never passed to policy composition.

An envelope freezes complete OpenAI tool mappings in original order and records the owner observed at capture. Protected names are the union of captured protected owners and valid present pins. A pinned floor is mandatory visibility; a protected ceiling is the maximum admitted protected surface. Direct-orchestrator protected tools are optional unless pinned. Worker auto-pinning requires both a nonempty task identifier and a true dispatcher-owned predicate; any worker/owner uncertainty produces `SAFE_NO_PRUNE` and restores the complete envelope. A valid present owner-unmapped explicit pin remains protected by exact name, while absent siblings are never fetched.

`RouterState` is the sole capability authority and is attached to the exact agent with identity readback. Its mutually exclusive lifecycle slots hold either one immutable bound admission result or one `NoAuthorityContamination` marker. Same-session bound results are returned by identity before new reads; a different session receives `SESSION_MISMATCH`. Disable/re-enable retains the slot, while exact session end clears it before the agent reference is released. The process-global `_router_state` is only a legacy non-authority fallback and never stores capability lifecycle state.

No-authority first contact is intentionally conservative. An attachable clean state preserves the current surface and may append independently admitted ordinary definitions. Before the first append it persists and reads back the immutable contamination marker; every coordinated setter is then transactional. A contaminated state remains append-only for ordinary definitions but can never capture or promote an envelope. Invalid policy is transient `NO_AUTHORITY_INVALID_POLICY`: it preserves all fields, denies every new ordinary/protected addition, performs no owner or definition lookup, creates no clean marker, and retains an existing marker. An unattachable agent is installed-only. Capture-invalid input binds `CAPTURE_INVALID_NO_MUTATION` and permits no expansion.

The route, visible request, middleware, post-tool recovery, and fallback callers all consume the same admission result and return deterministic fields. A mixed ordinary/protected request may report an ordinary append while explicitly reporting protected denial; total protected denial is an atomic no-op. Candidate validation happens before the first setter. If a setter fails after a partial assignment, the exact pre-call definitions, schemas, order, valid/enabled names, effective state, counters, fallback/retry markers, routed-turn fields, and lifecycle-slot identity are restored. Recovery retry is allowed only after exact admitted installation and a result marked `retry_allowed`.

The empty-message path binds admission before returning and therefore preserves worker pins even when no classifier work is needed. It must not list available toolsets, classify, retrieve definitions, create recovery controls, schedule retry, or recapture. The early `pre_turn_context_build` hook is preferred; `pre_llm_call` is a compatibility path for older Hermes and cannot claim pre-preflight savings. The global recovery schema/catalog advertises discoverability only; it does not grant per-agent authority. Hermes registry checks, approvals, credentials, and dispatcher validation remain the final execution layer. The repository tests provide synthetic local contract evidence, not live runtime acceptance.
