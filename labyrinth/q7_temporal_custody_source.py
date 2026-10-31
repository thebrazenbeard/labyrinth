"""Source-only Q7 phase ordering; never a timestamp or graduation authority.

This pure verifier checks independently *pinned* signed byte commitments for
``anchor -> training_start``. It cannot establish when the signers acted,
whether keys were host-custodied, or whether anchors were immutable. Only an
external operator with a protected append-only clock/ledger can establish
those physical facts. No sealed exam questions or answers belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)

from .final_exam_precommit import (
    FinalExamPrecommitError, SignedFinalExamPrecommit,
    verify_pretraining_precommit,
)


SCHEMA = "labyrinth.q7-public-phase-source.v1"
_DOMAIN = b"labyrinth.q7-public-phase-source.v1\0"
_HEX = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_PHASES = ("anchor", "training_start")
_FIELDS = frozenset((
    "schema", "event_id", "precommit_sha256", "phase", "sequence",
    "prior_receipt_sha256", "graduation_authorized",
))
_MAX_BYTES = 1024


class Q7SourceOrderError(ValueError):
    """A proposed public phase sequence cannot be authenticated."""


@dataclass(frozen=True)
class SignedQ7SourcePhase:
    public_bytes: bytes
    signature_hex: str
    graduation_authorized: bool = False


@dataclass(frozen=True)
class Q7SourceOrderSummary:
    event_id: str
    precommit_sha256: str
    anchor_receipt_sha256: str
    training_start_receipt_sha256: str
    temporal_custody_verified: bool = False
    graduation_authorized: bool = False


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise Q7SourceOrderError(f"invalid {name} digest")
    return value


def _canonical(body: dict[str, object]) -> bytes:
    return (json.dumps(body, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise Q7SourceOrderError("duplicate Q7 phase field")
        result[key] = value
    return result


def _reject_noninteger_number(token: str) -> object:
    # The public Q7 schema has only one numeric field (integer sequence).
    # Python's JSON decoder otherwise converts 1e309 to float('inf') and
    # permits NaN/Infinity constants, which break canonical serialization.
    raise Q7SourceOrderError("noninteger or nonfinite Q7 phase number")


def receipt_sha256(receipt: SignedQ7SourcePhase) -> str:
    """Bind the public bytes *and* their signature to an exact source receipt."""
    if (type(receipt) is not SignedQ7SourcePhase
            or receipt.graduation_authorized is not False
            or type(receipt.public_bytes) is not bytes
            or not 0 < len(receipt.public_bytes) <= _MAX_BYTES
            or type(receipt.signature_hex) is not str
            or re.fullmatch(r"[0-9a-f]{128}", receipt.signature_hex) is None):
        raise Q7SourceOrderError("invalid signed Q7 phase")
    return sha256(_DOMAIN + receipt.public_bytes
                  + bytes.fromhex(receipt.signature_hex)).hexdigest()


def create_q7_source_phase(
    *, event_id: str, precommit_sha256: str, phase: str,
    prior_receipt_sha256: str, signing_key: Ed25519PrivateKey,
) -> SignedQ7SourcePhase:
    """Create a source-only phase record. This is **not** an external anchor."""
    if (type(event_id) is not str or _IDENT.fullmatch(event_id) is None
            or type(phase) is not str or phase not in _PHASES
            or not isinstance(signing_key, Ed25519PrivateKey)):
        raise Q7SourceOrderError("invalid Q7 phase signer or identity")
    _digest(precommit_sha256, "precommit")
    _digest(prior_receipt_sha256, "predecessor")
    if phase == "anchor" and prior_receipt_sha256 != "0" * 64:
        raise Q7SourceOrderError("anchor must have zero predecessor")
    if phase == "training_start" and prior_receipt_sha256 == "0" * 64:
        raise Q7SourceOrderError("training start must reference an anchor")
    body = dict(schema=SCHEMA, event_id=event_id,
                precommit_sha256=precommit_sha256, phase=phase,
                sequence=1 if phase == "anchor" else 2,
                prior_receipt_sha256=prior_receipt_sha256,
                graduation_authorized=False)
    public_bytes = _canonical(body)
    if len(public_bytes) > _MAX_BYTES:
        raise Q7SourceOrderError("Q7 phase exceeds size bound")
    return SignedQ7SourcePhase(
        public_bytes=public_bytes,
        signature_hex=signing_key.sign(_DOMAIN + public_bytes).hex(),
    )


def _verified_phase(
    receipt: SignedQ7SourcePhase, public_key: bytes,
) -> dict[str, object]:
    receipt_sha256(receipt)
    if type(public_key) is not bytes or len(public_key) != 32:
        raise Q7SourceOrderError("externally pinned Ed25519 key required")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(receipt.signature_hex), _DOMAIN + receipt.public_bytes)
        body = json.loads(
            receipt.public_bytes, object_pairs_hook=_unique,
            parse_float=_reject_noninteger_number,
            parse_constant=_reject_noninteger_number,
        )
    except (InvalidSignature, ValueError, TypeError, UnicodeError) as exc:
        raise Q7SourceOrderError("Q7 phase signature or encoding invalid") from exc
    if (type(body) is not dict or body.keys() != _FIELDS
            or _canonical(body) != receipt.public_bytes
            or body["schema"] != SCHEMA
            or type(body["event_id"]) is not str
            or _IDENT.fullmatch(body["event_id"]) is None
            or type(body["phase"]) is not str or body["phase"] not in _PHASES
            or type(body["sequence"]) is not int
            or body["graduation_authorized"] is not False):
        raise Q7SourceOrderError("Q7 phase schema invalid")
    _digest(body["precommit_sha256"], "precommit")
    _digest(body["prior_receipt_sha256"], "predecessor")
    return body


def verify_q7_source_phase_order(
    precommit: SignedFinalExamPrecommit, *,
    pinned_precommit_signer_key: bytes, expected_exam_id: str,
    expected_event_id: str, expected_objective_id: str,
    expected_scope_sha256: str, expected_sealed_plan_sha256: str,
    expected_source_sha256: str, expected_training_inventory_sha256: str,
    consumed_event_sha256s: frozenset[str],
    anchor: SignedQ7SourcePhase, pinned_anchor_key: bytes,
    expected_anchor_receipt_sha256: str,
    training_start: SignedQ7SourcePhase, pinned_training_owner_key: bytes,
    expected_training_start_receipt_sha256: str,
) -> Q7SourceOrderSummary:
    """Check cryptographic *source ordering*, never actual elapsed time.

    All expected digests and keys must be owned outside Labyrinth/Minotaur.
    This API does not persist them, confirm independence, or read a host clock.
    """
    try:
        verify_pretraining_precommit(
            precommit, pinned_signer_public_key=pinned_precommit_signer_key,
            expected_exam_id=expected_exam_id, expected_event_id=expected_event_id,
            expected_objective_id=expected_objective_id,
            expected_scope_sha256=expected_scope_sha256,
            expected_sealed_plan_sha256=expected_sealed_plan_sha256,
            expected_source_sha256=expected_source_sha256,
            expected_training_inventory_sha256=expected_training_inventory_sha256,
            consumed_event_sha256s=consumed_event_sha256s,
        )
    except FinalExamPrecommitError as exc:
        raise Q7SourceOrderError("precommit failed independent pins") from exc
    if (type(pinned_anchor_key) is not bytes
            or type(pinned_training_owner_key) is not bytes
            or len(pinned_anchor_key) != 32
            or len(pinned_training_owner_key) != 32
            or len({pinned_anchor_key, pinned_training_owner_key,
                    pinned_precommit_signer_key}) != 3):
        raise Q7SourceOrderError("three distinct externally pinned signers required")
    _digest(expected_anchor_receipt_sha256, "external anchor")
    _digest(expected_training_start_receipt_sha256, "external training start")
    actual_anchor = receipt_sha256(anchor)
    actual_start = receipt_sha256(training_start)
    if (actual_anchor != expected_anchor_receipt_sha256
            or actual_start != expected_training_start_receipt_sha256):
        raise Q7SourceOrderError("Q7 receipt differs from external expected digest")
    a = _verified_phase(anchor, pinned_anchor_key)
    b = _verified_phase(training_start, pinned_training_owner_key)
    precommit_digest = sha256(precommit.public_bytes).hexdigest()
    if (a["event_id"] != expected_event_id
            or b["event_id"] != expected_event_id
            or a["precommit_sha256"] != precommit_digest
            or b["precommit_sha256"] != precommit_digest
            or a["phase"] != "anchor" or a["sequence"] != 1
            or a["prior_receipt_sha256"] != "0" * 64
            or b["phase"] != "training_start" or b["sequence"] != 2
            or b["prior_receipt_sha256"] != actual_anchor):
        raise Q7SourceOrderError("Q7 source phase chain mismatch")
    return Q7SourceOrderSummary(
        expected_event_id, precommit_digest, actual_anchor, actual_start)
