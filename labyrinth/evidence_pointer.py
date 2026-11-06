"""Canonical, inert Labyrinth five-arm *source* evidence pointers.

This parser compares claims to separately supplied expected pins. It does not
authenticate the producer of those expectations, execute a model, attest
checkpoint bytes, manage consent, or grant any runtime authority. The caller
must obtain expected pins from an independently governed source; reconstructing
them from the pointer defeats the intended substitution check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import re
from typing import Any


SCHEMA = "labyrinth.source-evidence-pointer.v1"
EXPERIMENT = "labyrinth.minilm-five-arm-independent-evidence-source.v1"
QUALIFICATION = "SOURCE_ONLY_UNATTESTED"

_MAX_BYTES = 4096
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_GIT_FIELDS = ("source_head", "source_tree")
_DIGEST_FIELDS = (
    "plan_sha256", "train_exposure_sha256", "control_matrix_sha256",
    "development_event_sha256", "checkpoint_manifest_sha256", "scope_sha256",
)
_REZON_FIELDS = ("rezon_source_head", "rezon_receipt_bundle_sha256")


class SourceEvidenceError(ValueError):
    """A source claim or its external expected pins are malformed or mismatched."""


@dataclass(frozen=True, slots=True)
class ExpectedSourcePinsV1:
    """Inputs that a consumer MUST acquire independently of the submitted pointer."""

    run_id: str
    source_head: str
    source_tree: str
    plan_sha256: str
    train_exposure_sha256: str
    control_matrix_sha256: str
    development_event_sha256: str
    checkpoint_manifest_sha256: str
    scope_sha256: str
    rezon_source_head: str | None = None
    rezon_receipt_bundle_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class SourceEvidencePointerV1:
    schema: str
    experiment: str
    qualification_state: str
    run_id: str
    source_head: str
    source_tree: str
    plan_sha256: str
    train_exposure_sha256: str
    control_matrix_sha256: str
    development_event_sha256: str
    checkpoint_manifest_sha256: str
    scope_sha256: str
    rezon_source_head: str | None = None
    rezon_receipt_bundle_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class SourceEvidenceMatchV1:
    """Source pin match only; every operational and governance grant is false."""

    pointer: SourceEvidencePointerV1
    qualification_state: str = field(default=QUALIFICATION, init=False)
    host_execution_verified: bool = field(default=False, init=False)
    independent_checkpoint_verified: bool = field(default=False, init=False)
    model_identity_verified: bool = field(default=False, init=False)
    consent_verified: bool = field(default=False, init=False)
    permissions_verified: bool = field(default=False, init=False)
    can_authorize_training: bool = field(default=False, init=False)
    can_authorize_graduation: bool = field(default=False, init=False)
    can_authorize_merge: bool = field(default=False, init=False)
    can_authorize_deployment: bool = field(default=False, init=False)


def _validate_pins(value: Any) -> None:
    if type(value) not in (ExpectedSourcePinsV1, SourceEvidencePointerV1):
        raise SourceEvidenceError("exact typed source pins required")
    run_id = value.run_id
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise SourceEvidenceError("invalid bounded run identity")
    for name in _GIT_FIELDS:
        pin = getattr(value, name)
        if type(pin) is not str or _GIT_SHA.fullmatch(pin) is None:
            raise SourceEvidenceError(f"invalid exact Git {name}")
    for name in _DIGEST_FIELDS:
        pin = getattr(value, name)
        if type(pin) is not str or _DIGEST.fullmatch(pin) is None:
            raise SourceEvidenceError(f"invalid SHA-256 {name}")
    rezon_source = value.rezon_source_head
    rezon_receipt = value.rezon_receipt_bundle_sha256
    if (rezon_source is None) != (rezon_receipt is None):
        raise SourceEvidenceError("optional Rezon references must be jointly absent or present")
    if rezon_source is not None:
        if (type(rezon_source) is not str or _GIT_SHA.fullmatch(rezon_source) is None
                or type(rezon_receipt) is not str
                or _DIGEST.fullmatch(rezon_receipt) is None):
            raise SourceEvidenceError("invalid opaque Rezon source/bundle references")


def _validate_pointer(pointer: Any) -> None:
    if type(pointer) is not SourceEvidencePointerV1:
        raise SourceEvidenceError("exact source pointer type required")
    if (type(pointer.schema) is not str or pointer.schema != SCHEMA
            or type(pointer.experiment) is not str or pointer.experiment != EXPERIMENT
            or type(pointer.qualification_state) is not str
            or pointer.qualification_state != QUALIFICATION):
        raise SourceEvidenceError("wrong schema, control experiment, or qualification status")
    _validate_pins(pointer)


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def encode_source_evidence_pointer(pointer: SourceEvidencePointerV1) -> bytes:
    """Serialize only the frozen schema; never sign or authorize its claims."""
    _validate_pointer(pointer)
    encoded = _canonical(asdict(pointer))
    if len(encoded) > _MAX_BYTES:
        raise SourceEvidenceError("oversized source pointer")
    return encoded


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise SourceEvidenceError("duplicate source pointer key")
        out[key] = value
    return out


def _reject_non_json_constant(token: str) -> None:
    raise SourceEvidenceError(f"non-JSON constant {token} prohibited")


def verify_source_evidence_pointer(
    payload: bytes, *, expected: ExpectedSourcePinsV1,
) -> SourceEvidenceMatchV1:
    """Compare a canonical inert pointer with independently obtained exact pins.

    A match attests *no* model execution, host, signer, checkpoint bytes,
    replay exclusion, identity, consent, permission, graduation or deployment.
    """
    if type(expected) is not ExpectedSourcePinsV1:
        raise SourceEvidenceError("independently supplied exact expected pins required")
    _validate_pins(expected)
    if type(payload) is not bytes or not 2 <= len(payload) <= _MAX_BYTES:
        raise SourceEvidenceError("bounded canonical pointer bytes required")
    try:
        parsed = json.loads(payload.decode("ascii"), object_pairs_hook=_unique_pairs,
                            parse_constant=_reject_non_json_constant)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise SourceEvidenceError("invalid pointer encoding") from exc
    if type(parsed) is not dict or set(parsed) != {f.name for f in fields(SourceEvidencePointerV1)}:
        raise SourceEvidenceError("pointer has missing, extra or privileged fields")
    if payload != _canonical(parsed):
        raise SourceEvidenceError("noncanonical source pointer encoding")
    try:
        pointer = SourceEvidencePointerV1(**parsed)
    except TypeError as exc:
        raise SourceEvidenceError("invalid source pointer fields") from exc
    _validate_pointer(pointer)
    for item in fields(ExpectedSourcePinsV1):
        name = item.name
        if getattr(pointer, name) != getattr(expected, name):
            raise SourceEvidenceError(f"unmatched independently expected {name}")
    return SourceEvidenceMatchV1(pointer=pointer)
