"""Immutable capability-envelope primitives for the token router."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

MISSING = object()

FrozenScalar: TypeAlias = (
    tuple[Literal["null"], None]
    | tuple[Literal["bool"], bool]
    | tuple[Literal["int"], int]
    | tuple[Literal["float"], str]
    | tuple[Literal["string"], str]
)
FrozenJSON: TypeAlias = (
    FrozenScalar
    | tuple[Literal["object"], tuple[tuple[str, "FrozenJSON"], ...]]
    | tuple[Literal["array"], tuple["FrozenJSON", ...]]
)


@dataclass(frozen=True)
class TrustedHostPolicy:
    """Parsed immutable policy supplied by a trusted host channel."""

    protected_toolsets: frozenset[str]
    pinned_tool_names: frozenset[str]
    source_channels: tuple[Literal["hook", "agent", "hermes_adapter"], ...]
    valid: bool
    errors: tuple[str, ...] = ()
    preserve_input_surface: bool = False


@dataclass(frozen=True)
class OwnerSnapshot:
    incoming_names: tuple[str, ...]
    owner_by_name: tuple[tuple[str, str | None], ...]
    registry_import_ok: bool
    registry_lookup_error_names: tuple[str, ...]
    agent_local_lookup_error_names: tuple[str, ...]
    malformed_definition_indexes: tuple[int, ...]
    duplicate_names: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class FrozenToolDefinition:
    name: str
    payload: FrozenJSON
    owner_toolset_at_capture: str | None


@dataclass(frozen=True)
class NoAuthorityContamination:
    session_id: str
    first_commit_caller: Literal["route", "visible_request", "middleware", "post_tool"]
    first_added_tool_names: tuple[str, ...]
    first_added_toolsets: tuple[str, ...]
    reason: Literal["ordinary_append_before_host_capture"] = "ordinary_append_before_host_capture"


@dataclass(frozen=True)
class HostAdmissionEnvelope:
    session_id: str
    definitions: tuple[FrozenToolDefinition, ...]
    original_enabled_toolsets: tuple[str, ...]
    owner_snapshot: OwnerSnapshot
    effective_policy: TrustedHostPolicy
    protected_names: frozenset[str]
    pinned_names: frozenset[str]
    preserve_input_surface: bool
    preserve_reason: str


AdmissionStatus = Literal[
    "READY",
    "SAFE_NO_PRUNE",
    "CAPTURE_INVALID_NO_MUTATION",
    "NO_AUTHORITY",
    "NO_AUTHORITY_INVALID_POLICY",
    "NO_AUTHORITY_CONTAMINATED",
    "NO_AUTHORITY_UNATTACHABLE",
    "SESSION_MISMATCH",
]


@dataclass(frozen=True)
class EnsureAdmissionResult:
    status: AdmissionStatus
    session_id: str
    envelope: HostAdmissionEnvelope | None
    effective_policy: TrustedHostPolicy | None
    owner_snapshot: OwnerSnapshot | None
    contamination: NoAuthorityContamination | None
    diagnostics: tuple[str, ...]
    surface_rule: Literal["ENVELOPE_SELECTION", "RESTORE_ENVELOPE", "PRESERVE_CURRENT"]
    ordinary_rule: Literal["APPEND_ONLY_COMPATIBILITY", "DENY"]
    protected_rule: Literal["ENVELOPE_ONLY", "DENY"]


@dataclass(frozen=True)
class AdmissionDecision:
    requested_toolsets: tuple[str, ...]
    requested_tool_names: tuple[str, ...]
    admitted_toolsets: tuple[str, ...]
    denied_toolsets: tuple[str, ...]
    denied_tool_names: tuple[str, ...]
    selected_envelope_names: tuple[str, ...]
    ordinary_registry_names: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ExpansionResult:
    ok: bool
    requested_toolsets: tuple[str, ...]
    expanded_toolsets: tuple[str, ...]
    denied_toolsets: tuple[str, ...]
    denied_tool_names: tuple[str, ...]
    added_tool_names: tuple[str, ...]
    installed_tool_names: tuple[str, ...]
    reason: str
    retry_allowed: bool = False


def _policy_with(
    policy: TrustedHostPolicy,
    *,
    protected_toolsets: frozenset[str] | None = None,
    pinned_tool_names: frozenset[str] | None = None,
    source_channels: tuple[Literal["hook", "agent", "hermes_adapter"], ...] | None = None,
    valid: bool | None = None,
    errors: tuple[str, ...] | None = None,
    preserve_input_surface: bool | None = None,
) -> TrustedHostPolicy:
    return TrustedHostPolicy(
        protected_toolsets=(
            policy.protected_toolsets if protected_toolsets is None else protected_toolsets
        ),
        pinned_tool_names=(
            policy.pinned_tool_names if pinned_tool_names is None else pinned_tool_names
        ),
        source_channels=(
            policy.source_channels if source_channels is None else source_channels
        ),
        valid=policy.valid if valid is None else valid,
        errors=policy.errors if errors is None else errors,
        preserve_input_surface=(
            policy.preserve_input_surface
            if preserve_input_surface is None
            else preserve_input_surface
        ),
    )


def _definition_name(definition: Any) -> str | None:
    if not isinstance(definition, Mapping):
        return None
    function = definition.get("function")
    if not isinstance(function, Mapping):
        return None
    name = function.get("name")
    return name if type(name) is str and bool(name) and name == name.strip() else None


def build_host_admission_envelope(
    *,
    session_id: str,
    untouched_surface: Any,
    original_enabled_toolsets: Any,
    effective_policy: TrustedHostPolicy,
    owner_snapshot: OwnerSnapshot,
) -> tuple[HostAdmissionEnvelope | None, tuple[str, ...]]:
    """Freeze the complete untouched host surface exactly once."""
    if not isinstance(session_id, str) or not session_id:
        return None, ("MISSING_SESSION_ID",)
    if not isinstance(untouched_surface, (list, tuple)):
        return None, ("MALFORMED_DEFINITION_SURFACE",)
    if not isinstance(original_enabled_toolsets, (list, tuple)):
        return None, ("MALFORMED_ENABLED_TOOLSETS",)
    if any(type(name) is not str for name in original_enabled_toolsets):
        return None, ("MALFORMED_ENABLED_TOOLSETS",)

    names: list[str] = []
    malformed: list[int] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    frozen: list[FrozenToolDefinition] = []
    owner_lookup = dict(owner_snapshot.owner_by_name)
    for index, definition in enumerate(untouched_surface):
        name = _definition_name(definition)
        if name is None:
            malformed.append(index)
            continue
        names.append(name)
        if name in seen:
            duplicates.append(name)
        seen.add(name)
        try:
            frozen.append(
                FrozenToolDefinition(
                    name=name,
                    payload=freeze_json(definition),
                    owner_toolset_at_capture=owner_lookup.get(name),
                )
            )
        except (TypeError, ValueError):
            malformed.append(index)
    if malformed or duplicates:
        diagnostics = [f"MALFORMED_DEFINITION_INDEX:{index}" for index in malformed]
        diagnostics.extend(f"DUPLICATE_DEFINITION_NAME:{name}" for name in duplicates)
        return None, tuple(dict.fromkeys(diagnostics))
    if tuple(names) != owner_snapshot.incoming_names:
        return None, ("OWNER_SNAPSHOT_SURFACE_MISMATCH",)
    if len(owner_snapshot.owner_by_name) != len(names):
        return None, ("OWNER_SNAPSHOT_CARDINALITY_MISMATCH",)
    missing_pins = effective_policy.pinned_tool_names - set(names)
    if missing_pins:
        return None, tuple(
            ["PIN_ABSENT_FROM_INCOMING_SURFACE"]
            + [f"MISSING_PIN:{name}" for name in sorted(missing_pins)]
        )

    protected_names = {
        item.name
        for item in frozen
        if item.owner_toolset_at_capture in effective_policy.protected_toolsets
    }
    protected_names.update(effective_policy.pinned_tool_names)
    envelope = HostAdmissionEnvelope(
        session_id=session_id,
        definitions=tuple(frozen),
        original_enabled_toolsets=tuple(original_enabled_toolsets),
        owner_snapshot=owner_snapshot,
        effective_policy=effective_policy,
        protected_names=frozenset(protected_names),
        pinned_names=frozenset(effective_policy.pinned_tool_names),
        preserve_input_surface=effective_policy.preserve_input_surface,
        preserve_reason=(effective_policy.errors[0] if effective_policy.errors else ""),
    )
    return envelope, ()


def thaw_envelope_definitions(envelope: HostAdmissionEnvelope) -> list[dict[str, Any]]:
    """Return fresh OpenAI mappings from an immutable envelope."""
    return [thaw_json(item.payload) for item in envelope.definitions]


def envelope_names(envelope: HostAdmissionEnvelope) -> tuple[str, ...]:
    return tuple(item.name for item in envelope.definitions)


def result_for_status(
    status: AdmissionStatus,
    session_id: str,
    *,
    envelope: HostAdmissionEnvelope | None = None,
    effective_policy: TrustedHostPolicy | None = None,
    owner_snapshot: OwnerSnapshot | None = None,
    contamination: NoAuthorityContamination | None = None,
    diagnostics: tuple[str, ...] = (),
) -> EnsureAdmissionResult:
    if status == "READY":
        rules = ("ENVELOPE_SELECTION", "APPEND_ONLY_COMPATIBILITY", "ENVELOPE_ONLY")
    elif status == "SAFE_NO_PRUNE":
        rules = ("RESTORE_ENVELOPE", "DENY", "DENY")
    elif status == "CAPTURE_INVALID_NO_MUTATION":
        rules = ("PRESERVE_CURRENT", "DENY", "DENY")
    elif status == "NO_AUTHORITY":
        rules = ("PRESERVE_CURRENT", "APPEND_ONLY_COMPATIBILITY", "DENY")
    elif status == "NO_AUTHORITY_CONTAMINATED":
        rules = ("PRESERVE_CURRENT", "APPEND_ONLY_COMPATIBILITY", "DENY")
    else:
        rules = ("PRESERVE_CURRENT", "DENY", "DENY")
    return EnsureAdmissionResult(
        status=status,
        session_id=session_id,
        envelope=envelope,
        effective_policy=effective_policy,
        owner_snapshot=owner_snapshot,
        contamination=contamination,
        diagnostics=tuple(diagnostics),
        surface_rule=rules[0],
        ordinary_rule=rules[1],
        protected_rule=rules[2],
    )


def _invalid_policy(
    source_channels: tuple[Literal["hook", "agent"], ...],
    errors: list[str],
) -> TrustedHostPolicy:
    return TrustedHostPolicy(
        protected_toolsets=frozenset(),
        pinned_tool_names=frozenset(),
        source_channels=source_channels,
        valid=False,
        errors=tuple(dict.fromkeys(errors)),
    )


def _parse_policy_channel(value: Any, channel: Literal["hook", "agent"]):
    if value is MISSING:
        return None, []
    if not isinstance(value, Mapping) or type(value) is not dict:
        return None, [f"{channel}:INVALID_MAPPING"]
    expected = {"schema_version", "protected_toolsets", "pinned_tool_names"}
    if set(value) != expected:
        return None, [f"{channel}:UNKNOWN_OR_MISSING_KEY"]
    version = value.get("schema_version")
    if type(version) is not int or version != 1:
        return None, [f"{channel}:INVALID_SCHEMA_VERSION"]
    parsed: dict[str, frozenset[str]] = {}
    errors: list[str] = []
    for field_name in ("protected_toolsets", "pinned_tool_names"):
        raw = value.get(field_name)
        if type(raw) is not list:
            errors.append(f"{channel}:INVALID_{field_name.upper()}")
            continue
        identifiers: list[str] = []
        for identifier in raw:
            if type(identifier) is not str or not identifier or identifier != identifier.strip():
                errors.append(f"{channel}:INVALID_{field_name.upper()}_IDENTIFIER")
                continue
            identifiers.append(identifier)
        if len(identifiers) != len(set(identifiers)):
            errors.append(f"{channel}:DUPLICATE_{field_name.upper()}")
        parsed[field_name] = frozenset(identifiers)
    if errors:
        return None, errors
    return parsed, []


def read_trusted_host_policy(
    hook_metadata: Any = MISSING,
    agent_metadata: Any = MISSING,
) -> TrustedHostPolicy:
    """Parse the two trusted metadata channels without conflating absence/null."""
    parsed_channels: list[tuple[str, dict[str, frozenset[str]]]] = []
    errors: list[str] = []
    for channel, value in (("hook", hook_metadata), ("agent", agent_metadata)):
        parsed, channel_errors = _parse_policy_channel(value, channel)  # type: ignore[arg-type]
        errors.extend(channel_errors)
        if parsed is not None:
            parsed_channels.append((channel, parsed))
    if errors:
        return _invalid_policy(
            tuple(channel for channel, _ in parsed_channels),
            errors,
        )
    if not parsed_channels:
        return TrustedHostPolicy(frozenset(), frozenset(), (), True)
    first = parsed_channels[0][1]
    if len(parsed_channels) == 2 and parsed_channels[1][1] != first:
        return _invalid_policy(("hook", "agent"), ["CONFLICTING_CHANNELS"])
    channels = cast(
        tuple[Literal["hook", "agent", "hermes_adapter"], ...],
        tuple(channel for channel, _ in parsed_channels),
    )
    return TrustedHostPolicy(
        protected_toolsets=first["protected_toolsets"],
        pinned_tool_names=first["pinned_tool_names"],
        source_channels=channels,  # type: ignore[arg-type]
        valid=True,
    )


def freeze_json(value: Any) -> FrozenJSON:
    """Freeze an exact JSON-compatible object graph without coercion."""
    return _freeze_json(value, set())


def _freeze_json(value: Any, active: set[int]) -> FrozenJSON:
    value_type = type(value)
    if value_type is type(None):
        return ("null", None)
    if value_type is bool:
        return ("bool", value)
    if value_type is int:
        return ("int", value)
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("non-finite float is not valid JSON")
        return ("float", value.hex())
    if value_type is str:
        return ("string", value)

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError("cyclic JSON mapping")
        active.add(identity)
        try:
            items: list[tuple[str, FrozenJSON]] = []
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError("JSON object keys must be exact strings")
                items.append((key, _freeze_json(item, active)))
            return ("object", tuple(items))
        finally:
            active.remove(identity)

    if value_type is list or value_type is tuple:
        identity = id(value)
        if identity in active:
            raise ValueError("cyclic JSON array")
        active.add(identity)
        try:
            return ("array", tuple(_freeze_json(item, active) for item in value))
        finally:
            active.remove(identity)

    raise TypeError(f"unsupported JSON value: {value_type.__name__}")


def thaw_json(value: FrozenJSON) -> Any:
    """Thaw a validated frozen graph into a fresh mutable JSON graph."""
    return _thaw_json(value)


def _thaw_json(value: Any) -> Any:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("malformed frozen JSON node")
    tag, payload = value
    if type(tag) is not str:
        raise TypeError("frozen JSON tag must be an exact string")

    if tag == "null":
        if payload is not None:
            raise ValueError("null payload must be None")
        return None
    if tag == "bool":
        if type(payload) is not bool:
            raise TypeError("bool payload must be an exact bool")
        return payload
    if tag == "int":
        if type(payload) is not int:
            raise TypeError("int payload must be an exact int")
        return payload
    if tag == "float":
        if type(payload) is not str:
            raise TypeError("float payload must be an exact string")
        try:
            parsed = float.fromhex(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid canonical float payload") from exc
        if not math.isfinite(parsed) or parsed.hex() != payload:
            raise ValueError("noncanonical float payload")
        return parsed
    if tag == "string":
        if type(payload) is not str:
            raise TypeError("string payload must be an exact string")
        return payload
    if tag == "object":
        if type(payload) is not tuple:
            raise TypeError("object payload must be an exact tuple")
        result: dict[str, Any] = {}
        for pair in payload:
            if type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str:
                raise ValueError("malformed object member")
            key = pair[0]
            if key in result:
                raise ValueError("duplicate frozen object key")
            result[key] = _thaw_json(pair[1])
        return result
    if tag == "array":
        if type(payload) is not tuple:
            raise TypeError("array payload must be an exact tuple")
        return [_thaw_json(item) for item in payload]
    raise ValueError(f"unknown frozen JSON tag: {tag!r}")


def canonical_json(value: FrozenJSON) -> str:
    """Serialize thawed JSON using the envelope's canonical OpenAI form."""
    return json.dumps(
        thaw_json(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def compose_effective_policy(
    parsed_policy: TrustedHostPolicy,
    owner_snapshot: OwnerSnapshot | None,
    *,
    adapter_protected_toolsets: frozenset[str],
    worker_toolset: str,
    has_nonempty_worker_task_id: bool,
    dispatcher_owned_worker: bool | None,
    worker_identity_error: str | None,
) -> TrustedHostPolicy:
    """Compose trusted policy from immutable inputs without host lookups.

    The Hermes adapter supplies either one real owner snapshot or typed ``None``.
    The generic core deliberately has no environment, registry, or Hermes policy
    knowledge; it only combines the immutable values it was given.
    """
    if owner_snapshot is None:
        if (
            has_nonempty_worker_task_id
            or dispatcher_owned_worker is not None
            or worker_identity_error is not None
        ):
            raise ValueError("no-envelope composition requires neutral worker values")
        return TrustedHostPolicy(
            protected_toolsets=frozenset(
                set(parsed_policy.protected_toolsets) | set(adapter_protected_toolsets)
            ),
            pinned_tool_names=frozenset(parsed_policy.pinned_tool_names),
            source_channels=_append_adapter_channel(parsed_policy.source_channels),
            valid=parsed_policy.valid,
            errors=parsed_policy.errors,
            preserve_input_surface=parsed_policy.preserve_input_surface,
        )

    protected = frozenset(
        set(parsed_policy.protected_toolsets) | set(adapter_protected_toolsets)
    )
    pins = set(parsed_policy.pinned_tool_names)
    errors = list(parsed_policy.errors)
    preserve = parsed_policy.preserve_input_surface
    names = set(owner_snapshot.incoming_names)
    missing_pins = sorted(pins - names)
    if missing_pins:
        errors.append("PIN_ABSENT_FROM_INCOMING_SURFACE")
        errors.extend(f"MISSING_PIN:{name}" for name in missing_pins)

    uncertainty = False
    if has_nonempty_worker_task_id and dispatcher_owned_worker:
        if worker_identity_error:
            uncertainty = True
            errors.append(worker_identity_error)
        if not owner_snapshot.registry_import_ok:
            uncertainty = True
            errors.append("WORKER_REGISTRY_IMPORT_UNAVAILABLE")
        if owner_snapshot.registry_lookup_error_names:
            uncertainty = True
            errors.append("WORKER_REGISTRY_OWNER_LOOKUP_UNAVAILABLE")
        if owner_snapshot.agent_local_lookup_error_names:
            uncertainty = True
            errors.append("WORKER_AGENT_LOCAL_OWNER_LOOKUP_UNAVAILABLE")
        if any(owner is None for _, owner in owner_snapshot.owner_by_name):
            uncertainty = True
            errors.append("WORKER_OWNER_UNMAPPED")
        if not uncertainty:
            pins.update(
                name
                for name, owner in owner_snapshot.owner_by_name
                if owner == worker_toolset
            )
    elif worker_identity_error and has_nonempty_worker_task_id:
        uncertainty = True
        errors.append(worker_identity_error)

    if uncertainty:
        preserve = True

    channels = _append_adapter_channel(parsed_policy.source_channels)
    valid = parsed_policy.valid and not missing_pins
    return TrustedHostPolicy(
        protected_toolsets=protected,
        pinned_tool_names=frozenset(pins),
        source_channels=channels,
        valid=valid,
        errors=tuple(dict.fromkeys(errors)),
        preserve_input_surface=preserve,
    )


def _append_adapter_channel(
    channels: tuple[Literal["hook", "agent", "hermes_adapter"], ...],
) -> tuple[Literal["hook", "agent", "hermes_adapter"], ...]:
    if "hermes_adapter" in channels:
        return channels
    return cast(
        tuple[Literal["hook", "agent", "hermes_adapter"], ...],
        channels + ("hermes_adapter",),
    )


# Small explicit aliases keep the primitive convenient for older callers.
freeze = freeze_json
thaw = thaw_json
serialize_canonical_json = canonical_json
