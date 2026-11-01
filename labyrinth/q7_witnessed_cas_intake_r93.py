"""Strict *source-only* intake for a purported externally witnessed Q7 chain.

The wire contract binds the already signed public precommit/anchor/training
receipts to an independently pinned Ed25519 witness and a two-step hash-head
transition. This code **does not** operate a host CAS, know wall-clock order,
protect signing keys, or consume a physical event. Callers must acquire the
expected before/after heads from a genuinely independent, rollback-resistant
host owner and confirm post-commit readback *outside* this verifier. A copied
ledger and self-issued pins can still validate identical source bytes.

Never include sealed questions, labels, private bank material or a private
training/evaluation sample in this envelope. The result cannot grant Q7 PASS,
graduation, merge or deployment authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import NamedTuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .final_exam_precommit import SignedFinalExamPrecommit
from .q7_temporal_custody_source import (
    Q7SourceOrderError, SignedQ7SourcePhase, verify_q7_source_phase_order,
)


SCHEMA = "labyrinth.q7.witnessed-source-chain.v1"
_SIGN_DOMAIN = b"labyrinth.q7.witnessed-source-chain.v1\0"
_HEAD_DOMAIN = b"labyrinth.q7.witnessed-chain-head.v1\0"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_HEX128 = re.compile(r"[0-9a-f]{128}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_MAX_BYTES = 8192
_MAX_INDEX = 2**63 - 1
_WIRE_FIELDS = frozenset({
    "schema", "host_id", "exam_id", "event_id", "precommit_sha256",
    "anchor_receipt_sha256", "training_start_receipt_sha256",
    "before_index", "anchor_index", "after_index",
    "before_head_sha256", "anchor_head_sha256", "after_head_sha256",
    "custody_verified", "graduation_authorized", "witness_signature_hex",
})


class Q7WitnessError(ValueError):
    """Malformed, contradictory, replayed or incorrectly pinned Q7 source."""


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise Q7WitnessError(f"invalid {label} SHA-256")
    return value


def _identity(value: object, label: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise Q7WitnessError(f"invalid {label}")
    return value


def _key(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise Q7WitnessError(f"missing externally pinned {label} Ed25519 key")
    return value


def _index(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value < _MAX_INDEX:
        raise Q7WitnessError(f"invalid {label} index")
    return value


@dataclass(frozen=True)
class PinnedQ7Witness:
    """Expected identities, role keys and CAS heads from an external owner.

    The object itself is caller-owned data, NOT proof that the owner is truly
    independent. Consumers must independently bind its provenance.
    """

    host_id: str
    exam_id: str
    event_id: str
    objective_id: str
    scope_sha256: str
    sealed_plan_sha256: str
    source_sha256: str
    training_inventory_sha256: str
    precommit_signer_key: bytes
    anchor_signer_key: bytes
    training_signer_key: bytes
    witness_signer_key: bytes
    anchor_receipt_sha256: str
    training_start_receipt_sha256: str
    expected_before_index: int
    expected_before_head_sha256: str
    expected_after_head_sha256: str
    consumed_event_sha256s: frozenset[str]

    def __post_init__(self) -> None:
        for name in ("host_id", "exam_id", "event_id", "objective_id"):
            _identity(getattr(self, name), name)
        for name in ("scope_sha256", "sealed_plan_sha256", "source_sha256",
                     "training_inventory_sha256", "anchor_receipt_sha256",
                     "training_start_receipt_sha256", "expected_before_head_sha256",
                     "expected_after_head_sha256"):
            _digest(getattr(self, name), name)
        keys = tuple(_key(getattr(self, name), name) for name in (
            "precommit_signer_key", "anchor_signer_key",
            "training_signer_key", "witness_signer_key"))
        if len(set(keys)) != 4:
            raise Q7WitnessError("four distinct independently pinned role keys required")
        before = _index(self.expected_before_index, "before")
        if before >= _MAX_INDEX - 2:
            raise Q7WitnessError("Q7 witness sequence exhausted")
        if (type(self.consumed_event_sha256s) is not frozenset
                or any(type(x) is not str or _HEX64.fullmatch(x) is None
                       for x in self.consumed_event_sha256s)):
            raise Q7WitnessError("external consumed-event roster required")


class Q7WitnessInspection(NamedTuple):
    """Immutable cryptographic/source classification, never host authority."""

    witness_sha256: str
    event_id: str
    before_index: int
    after_index: int
    after_head_sha256: str
    classification: str = "SIGNED_Q7_CHAIN_SOURCE_ONLY_NO_TEMPORAL_CUSTODY"

    @property
    def temporal_custody_verified(self) -> bool:
        return False

    @property
    def graduation_authorized(self) -> bool:
        return False


def q7_witness_next_head(prior_head_sha256: str, index: int,
                         source_receipt_sha256: str) -> str:
    """Deterministic append-head rule, NOT an actual host append operation."""
    prior = _digest(prior_head_sha256, "prior head")
    receipt = _digest(source_receipt_sha256, "source receipt")
    index = _index(index, "next")
    return sha256(_HEAD_DOMAIN + bytes.fromhex(prior) + index.to_bytes(8, "big")
                  + bytes.fromhex(receipt)).hexdigest()


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise Q7WitnessError("duplicate Q7 witness field")
        value[key] = item
    return value


def _reject_noninteger(_value: str) -> None:
    raise Q7WitnessError("noninteger or nonfinite Q7 witness number")


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def inspect_q7_witnessed_source(
    raw: bytes, *, pins: PinnedQ7Witness,
    precommit: SignedFinalExamPrecommit,
    anchor: SignedQ7SourcePhase,
    training_start: SignedQ7SourcePhase,
) -> Q7WitnessInspection:
    """Check signed public phase/host-head claims against independently supplied pins.

    This does not trust the witness to have a protected clock or store. A
    conforming host implementation must furnish authenticated owner provenance,
    physical pre-training order, an atomic CAS and nonrollback spent register,
    plus post-commit readback before treating Q7 as qualified.
    """
    if type(raw) is not bytes or not 0 < len(raw) <= _MAX_BYTES:
        raise Q7WitnessError("invalid or oversized Q7 witness wire envelope")
    if type(pins) is not PinnedQ7Witness:
        raise Q7WitnessError("external Q7 witness pins required")
    # frozen=True does not stop object.__setattr__; reconstruct a validated
    # local snapshot before evaluating either signature or owner expectation.
    try:
        pins = PinnedQ7Witness(**{
            name: getattr(pins, name) for name in PinnedQ7Witness.__dataclass_fields__})
        if type(precommit) is not SignedFinalExamPrecommit:
            raise Q7WitnessError("signed precommit required")
        if type(anchor) is not SignedQ7SourcePhase or type(training_start) is not SignedQ7SourcePhase:
            raise Q7WitnessError("signed phase receipts required")
        precommit_snapshot = SignedFinalExamPrecommit(
            precommit.public_bytes, precommit.signature_hex, precommit.graduation_authorized)
        anchor_snapshot = SignedQ7SourcePhase(
            anchor.public_bytes, anchor.signature_hex, anchor.graduation_authorized)
        training_snapshot = SignedQ7SourcePhase(
            training_start.public_bytes, training_start.signature_hex,
            training_start.graduation_authorized)
    except AttributeError as exc:
        raise Q7WitnessError("malformed external Q7 input") from exc
    try:
        source = verify_q7_source_phase_order(
            precommit_snapshot,
            pinned_precommit_signer_key=pins.precommit_signer_key,
            expected_exam_id=pins.exam_id,
            expected_event_id=pins.event_id,
            expected_objective_id=pins.objective_id,
            expected_scope_sha256=pins.scope_sha256,
            expected_sealed_plan_sha256=pins.sealed_plan_sha256,
            expected_source_sha256=pins.source_sha256,
            expected_training_inventory_sha256=pins.training_inventory_sha256,
            consumed_event_sha256s=pins.consumed_event_sha256s,
            anchor=anchor_snapshot,
            pinned_anchor_key=pins.anchor_signer_key,
            expected_anchor_receipt_sha256=pins.anchor_receipt_sha256,
            training_start=training_snapshot,
            pinned_training_owner_key=pins.training_signer_key,
            expected_training_start_receipt_sha256=pins.training_start_receipt_sha256,
        )
    except (Q7SourceOrderError, ValueError, TypeError, OverflowError) as exc:
        # The nested precommit decoder currently permits syntactically valid
        # JSON exponents such as 1e309 to reach its canonical serializer as
        # nonfinite floats. Treat that escaped parser ValueError as malformed
        # *untrusted source* at this public boundary, never as a valid phase.
        raise Q7WitnessError("Q7 precommit or source phase failed pinned verification") from exc
    try:
        body = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                          parse_float=_reject_noninteger,
                          parse_constant=_reject_noninteger)
        if (type(body) is not dict or body.keys() != _WIRE_FIELDS
                or _canonical(body) != raw):
            raise Q7WitnessError("noncanonical Q7 witness source shape")
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError) as exc:
        raise Q7WitnessError("malformed or noncanonical Q7 witness source") from exc
    expected = (
        ("schema", SCHEMA), ("host_id", pins.host_id), ("exam_id", pins.exam_id),
        ("event_id", pins.event_id), ("precommit_sha256", source.precommit_sha256),
        ("anchor_receipt_sha256", source.anchor_receipt_sha256),
        ("training_start_receipt_sha256", source.training_start_receipt_sha256),
        ("before_head_sha256", pins.expected_before_head_sha256),
        ("after_head_sha256", pins.expected_after_head_sha256),
    )
    if any(type(body[name]) is not str or body[name] != value for name, value in expected):
        raise Q7WitnessError("Q7 independent witness identity or head pin mismatch")
    before = _index(body["before_index"], "before")
    anchor_index = _index(body["anchor_index"], "anchor")
    after = _index(body["after_index"], "after")
    if (before != pins.expected_before_index or anchor_index != before + 1
            or after != anchor_index + 1):
        raise Q7WitnessError("Q7 source witness indices not consecutive")
    anchor_head = q7_witness_next_head(
        pins.expected_before_head_sha256, anchor_index, source.anchor_receipt_sha256)
    next_head = q7_witness_next_head(
        anchor_head, after, source.training_start_receipt_sha256)
    if (_digest(body["anchor_head_sha256"], "anchor head") != anchor_head
            or next_head != pins.expected_after_head_sha256):
        raise Q7WitnessError("Q7 witness head chain differs from external pin")
    if body["custody_verified"] is not False or body["graduation_authorized"] is not False:
        raise Q7WitnessError("untrusted source must not self-claim host custody or graduation")
    signature = body["witness_signature_hex"]
    if type(signature) is not str or _HEX128.fullmatch(signature) is None:
        raise Q7WitnessError("invalid Q7 witness signature encoding")
    unsigned = {name: value for name, value in body.items()
                if name != "witness_signature_hex"}
    try:
        Ed25519PublicKey.from_public_bytes(pins.witness_signer_key).verify(
            bytes.fromhex(signature), _SIGN_DOMAIN + _canonical(unsigned))
    except Exception as exc:
        raise Q7WitnessError("Q7 witness signature not authenticated") from exc
    return Q7WitnessInspection(
        sha256(raw).hexdigest(), pins.event_id, before, after, next_head)
