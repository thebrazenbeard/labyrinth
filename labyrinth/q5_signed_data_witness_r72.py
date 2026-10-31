"""Data-only, source-level Ed25519 witness consumer for Q5 evaluation.

This deliberately does not execute a caller-supplied Python verifier. An
external owner must still pin the signer and pre-outcome event *before* any
evaluation, independently hold source cases and the event consumption ledger,
and witness any actual model run. A valid signature here authenticates bytes
under a supplied key, not chronology, custody, non-replay or learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, NamedTuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from labyrinth.five_arm_statistical_screen import SeedOutcome
from labyrinth.q5_evaluation_preregistration_r68 import (
    FrozenEvaluationPrecommit, screen_predeclared_evaluation,
)


SCHEMA = "labyrinth.q5-signed-preoutcome-source-witness.v1"
_DOMAIN = b"labyrinth.q5-signed-preoutcome-source-witness.v1\0"
_FIELDS = frozenset({
    "schema", "event_id", "precommit_sha256", "signer_public_key_sha256",
    "declared_stage",
})
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_MAX_PUBLIC_BYTES = 4096


class Q5SignedWitnessError(ValueError):
    """The signed Q5 source claim fails strict verification."""


@dataclass(frozen=True)
class SignedQ5SourceWitness:
    public_bytes: bytes
    signature_hex: str


class Q5SignedSourceWitnessReceipt(NamedTuple):
    """A source signature check; never an independent host witness receipt."""

    event_id: str
    precommit_sha256: str
    signer_public_key_sha256: str
    public_witness_sha256: str

    @property
    def external_chronology_verified(self) -> bool:
        return False

    @property
    def independent_custody_verified(self) -> bool:
        return False

    @property
    def external_nonreplay_verified(self) -> bool:
        return False

    @property
    def graduation_authorized(self) -> bool:
        return False


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for key, value in pairs:
        if key in body:
            raise Q5SignedWitnessError("duplicate Q5 witness field")
        body[key] = value
    return body


def _reject_nonfinite_json(token: str) -> None:
    # Python's decoder otherwise accepts nonstandard NaN and Infinity tokens;
    # signed JSON must stay within the canonical finite JSON grammar.
    raise Q5SignedWitnessError(f"nonfinite Q5 witness JSON token: {token}")


def _canonical_bytes(body: dict[str, Any]) -> bytes:
    return (json.dumps(body, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _is_digest(value: object) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def verify_q5_signed_source_witness(
    *, witness: SignedQ5SourceWitness,
    pinned_signer_public_key: bytes,
    expected_precommit_sha256: str,
    expected_event_id: str,
    consumed_event_sha256s: frozenset[str],
) -> Q5SignedSourceWitnessReceipt:
    """Check exactly pinned signed bytes using no caller-executable callback.

    `pinned_signer_public_key`, `expected_*`, and `consumed_event_sha256s` are
    caller inputs. This function cannot attest their externally governed origin
    or that the caller has kept a durable one-use register.
    """
    if type(witness) is not SignedQ5SourceWitness:
        raise Q5SignedWitnessError("typed signed Q5 witness required")
    # Snapshot the wrapper's data fields once. Frozen dataclasses can be
    # overwritten by object.__setattr__, so no post-check rereads are allowed.
    public_bytes = witness.public_bytes
    signature_hex = witness.signature_hex
    if (type(public_bytes) is not bytes
            or not 0 < len(public_bytes) <= _MAX_PUBLIC_BYTES
            or type(signature_hex) is not str
            or re.fullmatch(r"[0-9a-f]{128}", signature_hex) is None
            or type(pinned_signer_public_key) is not bytes
            or len(pinned_signer_public_key) != 32
            or not _is_digest(expected_precommit_sha256)
            or type(expected_event_id) is not str
            or _IDENT.fullmatch(expected_event_id) is None
            or type(consumed_event_sha256s) is not frozenset
            or any(not _is_digest(digest) for digest in consumed_event_sha256s)):
        raise Q5SignedWitnessError("Q5 witness source pins are malformed")

    try:
        Ed25519PublicKey.from_public_bytes(pinned_signer_public_key).verify(
            bytes.fromhex(signature_hex), _DOMAIN + public_bytes)
        body = json.loads(public_bytes, object_pairs_hook=_unique_pairs,
                          parse_constant=_reject_nonfinite_json)
    except (InvalidSignature, UnicodeError, ValueError, TypeError) as exc:
        raise Q5SignedWitnessError("Q5 witness signature or canonical encoding invalid") from exc
    if type(body) is not dict:
        raise Q5SignedWitnessError("Q5 witness public source schema changed")
    try:
        # Exponent overflow such as 1e9999 bypasses parse_constant and is
        # decoded as inf; canonical re-encoding must fail with our own public
        # error instead of leaking the JSON serializer's raw ValueError.
        canonical = _canonical_bytes(body)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise Q5SignedWitnessError("Q5 witness canonical encoding invalid") from exc
    if (body.keys() != _FIELDS
            or canonical != public_bytes
            or body["schema"] != SCHEMA
            or body["declared_stage"] != "BEFORE_OUTCOMES"
            or type(body["event_id"]) is not str
            or _IDENT.fullmatch(body["event_id"]) is None
            or not _is_digest(body["precommit_sha256"])
            or not _is_digest(body["signer_public_key_sha256"])):
        raise Q5SignedWitnessError("Q5 witness public source schema changed")
    if (body["event_id"] != expected_event_id
            or body["precommit_sha256"] != expected_precommit_sha256
            or body["signer_public_key_sha256"]
            != sha256(pinned_signer_public_key).hexdigest()):
        raise Q5SignedWitnessError("Q5 witness external identity or digest pin differs")
    spent_digest = sha256((SCHEMA + "\0" + expected_event_id).encode("ascii")).hexdigest()
    if spent_digest in consumed_event_sha256s:
        raise Q5SignedWitnessError("Q5 witness event already consumed by supplied roster")
    return Q5SignedSourceWitnessReceipt(
        expected_event_id, expected_precommit_sha256,
        sha256(pinned_signer_public_key).hexdigest(), sha256(public_bytes).hexdigest())


def screen_q5_with_signed_source_witness(
    *, precommit: FrozenEvaluationPrecommit,
    external_expected_precommit_sha256: str,
    witness: SignedQ5SourceWitness,
    pinned_signer_public_key: bytes,
    expected_event_id: str,
    consumed_event_sha256s: frozenset[str],
    outcomes: tuple[SeedOutcome, ...],
) -> dict[str, object]:
    """Run Q5's source screen after data-only signature verification.

    Unlike the lower-level API, this entry point exposes no caller-provided
    executable verifier. It does not confer external witness custody or any
    authority to train, retain, promote, merge, deploy or graduate.
    """
    source_receipt = verify_q5_signed_source_witness(
        witness=witness, pinned_signer_public_key=pinned_signer_public_key,
        expected_precommit_sha256=external_expected_precommit_sha256,
        expected_event_id=expected_event_id,
        consumed_event_sha256s=consumed_event_sha256s,
    )
    # Only immutable signature-authenticated source bytes are passed into the
    # legacy Q5 source-screen seam. The closure is trusted application code,
    # NOT an arbitrary Python verifier supplied by a caller.
    public_bytes = witness.public_bytes
    if (type(public_bytes) is not bytes
            or sha256(public_bytes).hexdigest() != source_receipt.public_witness_sha256):
        raise Q5SignedWitnessError("Q5 signed witness changed after source verification")

    def _source_checked(data: bytes, expected_digest: str) -> bool:
        return (type(data) is bytes and data == public_bytes
                and expected_digest == source_receipt.precommit_sha256)

    result = screen_predeclared_evaluation(
        precommit=precommit,
        external_expected_precommit_sha256=external_expected_precommit_sha256,
        preoutcome_witness=public_bytes,
        verify_external_preoutcome_witness=_source_checked,
        outcomes=outcomes,
    )
    return {
        **result,
        "signed_witness_source_signature_verified": True,
        "witness_source_record": source_receipt,
    }
