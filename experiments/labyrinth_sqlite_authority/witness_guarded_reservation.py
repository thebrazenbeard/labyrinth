"""Source-only two-principal reservation boundary for Labyrinth/Minotaur.

The SQLite authority is a *local proposal journal*. A separately held,
independently signing monotonic witness is the final one-shot arbiter. A local
receipt is never returned as accepted unless the witness commits its exact
bytes and the witness's pinned Ed25519 signature verifies.

This module cannot establish OS principal separation, monotonic hardware,
immutable witness storage, network authentication or protected key custody.
A witness cloned together with its private signing key can still replay an
event. Consequently Q8 physical qualification remains HOLD/NOT RUN.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Protocol

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from labyrinth.external_monotonic_authority_contract import (
    ExternalAuthorityConflict,
    ExternalAuthorityError,
    ExternalAuthorityHead,
    ExternalAuthorityOutcomeUnknown,
    ExternalAuthorityPort,
    ExternalReservationReceipt,
    ExternalReservationRequest,
    next_head_digest,
)


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_HEX128 = re.compile(r"[0-9a-f]{128}\Z", re.ASCII)
_DOMAIN = b"labyrinth.independent-witness-commit.v1\0"


def _text(value: object, label: str) -> str:
    if type(value) is not str or not _ID.fullmatch(value):
        raise ExternalAuthorityError(f"invalid {label}")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or not _HEX64.fullmatch(value):
        raise ExternalAuthorityError(f"invalid {label}")
    return value


def _key(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise ExternalAuthorityError(f"32-byte {label} public key required")
    return value


def receipt_digest(receipt: ExternalReservationReceipt) -> str:
    """Bind BOTH signed payload and local authority signature, not a label."""
    if type(receipt) is not ExternalReservationReceipt:
        raise ExternalAuthorityError("exact local receipt type required")
    payload = receipt.signed_payload()
    if type(receipt.signature_hex) is not str or not _HEX128.fullmatch(receipt.signature_hex):
        raise ExternalAuthorityError("malformed local receipt signature")
    return sha256(payload + bytes.fromhex(receipt.signature_hex)).hexdigest()


@dataclass(frozen=True)
class WitnessCommit:
    witness_id: str
    provider_id: str
    run_id: str
    event_id: str
    exam_id: str
    operation_nonce: str
    request_sha256: str
    local_receipt_sha256: str
    epoch: int
    fence: int
    commit_index: int
    previous_head_sha256: str
    new_head_sha256: str
    signature_hex: str = ""
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False

    def signed_payload(self) -> bytes:
        for field in ("witness_id", "provider_id", "run_id", "event_id",
                      "exam_id", "operation_nonce"):
            _text(getattr(self, field), field)
        for field in ("request_sha256", "local_receipt_sha256",
                      "previous_head_sha256", "new_head_sha256"):
            _digest(getattr(self, field), field)
        for field in ("epoch", "fence", "commit_index"):
            value = getattr(self, field)
            if type(value) is not int or value < (1 if field != "fence" else 0) or value >= 2**63 - 1:
                raise ExternalAuthorityError(f"invalid {field}")
        if self.graduation_authorized is not False or self.model_execution_authenticated is not False:
            raise ExternalAuthorityError("witness commit cannot authorize graduation/model execution")
        fields = {name: getattr(self, name) for name in self.__dataclass_fields__
                  if name != "signature_hex"}
        return _DOMAIN + (json.dumps(fields, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, ensure_ascii=True) + "\n").encode("ascii")


class IndependentWitness(Protocol):
    """Must be hosted separately, with atomic CAS and nonrollbackable state.

    The witness independently pre-admits exact run/event/exam/nonce/weight
    intent and consumes it once. It signs a WitnessCommit covering the exact
    signed local receipt. A source-only implementation does not prove any of
    these environmental requirements.
    """

    def read_head(self, run_id: str) -> ExternalAuthorityHead: ...

    def commit_receipt_once(self, request: ExternalReservationRequest,
                            receipt: ExternalReservationReceipt) -> WitnessCommit: ...


class WitnessGuardedReservationPort:
    """Never expose a cloned local SQLite reservation as an authorized result.

    The external client reads only the independent witness's head. Local
    reservation happens first to collect its signed proposal, but any error
    AFTER local commit is OUTCOME_UNKNOWN. It cannot be blindly retried.
    """

    def __init__(self, *, local_proposal: ExternalAuthorityPort,
                 witness: IndependentWitness, witness_id: str,
                 provider_id: str, pinned_local_public_key: bytes,
                 pinned_witness_public_key: bytes):
        self._local = local_proposal
        self._witness = witness
        self._witness_id = _text(witness_id, "witness identity")
        self._provider = _text(provider_id, "authority provider")
        self._local_key = _key(pinned_local_public_key, "local authority")
        self._witness_key = _key(pinned_witness_public_key, "external witness")
        if self._local_key == self._witness_key:
            raise ExternalAuthorityError("independent witness signing identity must differ")

    def read_head(self, run_id: str) -> ExternalAuthorityHead:
        _text(run_id, "run")
        head = self._witness.read_head(run_id)
        if type(head) is not ExternalAuthorityHead:
            raise ExternalAuthorityError("untyped independent witness head")
        head.validate()
        if head.provider_id != self._provider or head.run_id != run_id:
            raise ExternalAuthorityConflict("foreign independent witness head")
        return head

    def reserve_once(self, request: ExternalReservationRequest) -> ExternalReservationReceipt:
        if type(request) is not ExternalReservationRequest:
            raise ExternalAuthorityError("exact reservation request required")
        request.validate()
        if request.provider_id != self._provider:
            raise ExternalAuthorityConflict("foreign provider")
        if self.read_head(request.run_id) != request.expected_head():
            raise ExternalAuthorityConflict("stale independent witness head")

        # The SQLite side can commit before any exception or lost ACK. NEVER
        # infer safe retry from an exception crossing this boundary.
        try:
            local = self._local.reserve_once(request)
        except Exception as exc:
            raise ExternalAuthorityOutcomeUnknown(
                "local proposal outcome unknown; independently reconcile; no retry"
            ) from exc
        try:
            if type(local) is not ExternalReservationReceipt:
                raise ExternalAuthorityError("untyped local proposal")
            Ed25519PublicKey.from_public_bytes(self._local_key).verify(
                bytes.fromhex(local.signature_hex), local.signed_payload())
            if (local.provider_id != request.provider_id
                    or local.run_id != request.run_id
                    or local.event_id != request.event_id
                    or local.request_sha256 != request.digest()
                    or local.epoch != request.expected_epoch
                    or local.fence != request.expected_fence
                    or local.commit_index != request.expected_commit_index + 1
                    or local.previous_head_sha256 != request.expected_head_sha256
                    or local.new_head_sha256 != next_head_digest(
                        request, local.reservation_id)):
                raise ExternalAuthorityError("local proposal differs from pinned request")
            local_digest = receipt_digest(local)
            witness_commit = self._witness.commit_receipt_once(request, local)
            if type(witness_commit) is not WitnessCommit:
                raise ExternalAuthorityError("untyped independent witness commitment")
            signed = witness_commit.signed_payload()
            if (type(witness_commit.signature_hex) is not str
                    or not _HEX128.fullmatch(witness_commit.signature_hex)):
                raise ExternalAuthorityError("malformed witness signature")
            Ed25519PublicKey.from_public_bytes(self._witness_key).verify(
                bytes.fromhex(witness_commit.signature_hex), signed)
            if (witness_commit.witness_id != self._witness_id
                    or witness_commit.provider_id != request.provider_id
                    or witness_commit.run_id != request.run_id
                    or witness_commit.event_id != request.event_id
                    or witness_commit.exam_id != request.exam_id
                    or witness_commit.operation_nonce != request.operation_nonce
                    or witness_commit.request_sha256 != request.digest()
                    or witness_commit.local_receipt_sha256 != local_digest
                    or witness_commit.epoch != local.epoch
                    or witness_commit.fence != local.fence
                    or witness_commit.commit_index != local.commit_index
                    or witness_commit.previous_head_sha256 != local.previous_head_sha256
                    or witness_commit.new_head_sha256 != local.new_head_sha256):
                raise ExternalAuthorityError("witness proof does not bind exact local reservation")
            if self.read_head(request.run_id) != local.post_head():
                raise ExternalAuthorityError("witness post-head not committed")
        except Exception as exc:
            # Deliberately fail closed on ALL post-local-commit errors. Even an
            # explicit witness conflict may follow a local commit.
            raise ExternalAuthorityOutcomeUnknown(
                "local reservation may be committed; independently reconcile; no retry"
            ) from exc
        return local
