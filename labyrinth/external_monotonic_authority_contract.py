"""Fail-closed *source contract* for an independently hosted reservation gate.

This module deliberately does not implement a local authority. A qualifying
provider must hold its own private signing key and append-only/monotonic state
outside the learner/trainer's rollback and administrative domain, and perform
the exact candidate/exam/one-shot checks atomically. A caller-provided fake
provider, a signed receipt alone, or this Python client's checks cannot prove
those deployment properties. No graduation decision is made here.

The consumer gets only an externally pinned Ed25519 PUBLIC key. No HMAC or
private key belongs in this process. The optional ``cryptography`` dependency
is required to execute real verification; absent dependencies fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import re
from typing import Protocol


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_HEX128 = re.compile(r"[0-9a-f]{128}\Z", re.ASCII)
_MAX_COUNTER = 2**63 - 1
_REQUEST_DOMAIN = b"labyrinth.external-monotonic-request.v1\0"
_HEAD_DOMAIN = b"labyrinth.external-monotonic-head.v1\0"
_SIGN_DOMAIN = b"labyrinth.external-monotonic-receipt.v1\0"


class ExternalAuthorityError(ValueError):
    """Invalid or unauthenticated external authority assertion."""


class ExternalAuthorityConflict(ExternalAuthorityError):
    """A stale fence/head or already-consumed event must not execute."""


class ExternalAuthorityOutcomeUnknown(ExternalAuthorityError):
    """Provider may have committed; do not blindly retry the operation."""


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or not _ID.fullmatch(value):
        raise ExternalAuthorityError(f"invalid {label}")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or not _HEX64.fullmatch(value):
        raise ExternalAuthorityError(f"invalid {label}")
    return value


def _counter(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value < _MAX_COUNTER:
        raise ExternalAuthorityError(f"invalid {label}")
    return value


def _canonical(value: dict[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


@dataclass(frozen=True)
class ExternalAuthorityHead:
    provider_id: str
    run_id: str
    epoch: int
    fence: int
    commit_index: int
    head_sha256: str

    def validate(self) -> None:
        _identifier(self.provider_id, "authority provider")
        _identifier(self.run_id, "run")
        _counter(self.epoch, "epoch", minimum=1)
        _counter(self.fence, "fence")
        _counter(self.commit_index, "commit index")
        _digest(self.head_sha256, "monotonic head digest")


@dataclass(frozen=True)
class ExternalReservationRequest:
    """Exact host-pinned intent, not a learner-invented graduation request."""

    provider_id: str
    run_id: str
    event_id: str
    exam_id: str
    candidate_artifact_sha256: str
    scope_fingerprint: str
    exam_lineage_sha256: str
    prior_checkpoint_sha256: str
    to_cycle: int
    expected_epoch: int
    expected_fence: int
    expected_commit_index: int
    expected_head_sha256: str
    operation_nonce: str

    def validate(self) -> None:
        for label in ("provider_id", "run_id", "event_id", "exam_id",
                      "operation_nonce"):
            _identifier(getattr(self, label), label)
        for label in ("candidate_artifact_sha256", "scope_fingerprint",
                      "exam_lineage_sha256", "prior_checkpoint_sha256",
                      "expected_head_sha256"):
            _digest(getattr(self, label), label)
        _counter(self.to_cycle, "successor cycle", minimum=1)
        for label in ("expected_fence", "expected_commit_index"):
            _counter(getattr(self, label), label)
        _counter(self.expected_epoch, "expected epoch", minimum=1)

    def payload(self) -> bytes:
        self.validate()
        return _canonical({name: getattr(self, name)
                           for name in self.__dataclass_fields__})

    def digest(self) -> str:
        return sha256(_REQUEST_DOMAIN + self.payload()).hexdigest()

    def expected_head(self) -> ExternalAuthorityHead:
        self.validate()
        return ExternalAuthorityHead(
            self.provider_id, self.run_id, self.expected_epoch,
            self.expected_fence, self.expected_commit_index,
            self.expected_head_sha256)


def next_head_digest(request: ExternalReservationRequest,
                     reservation_id: str) -> str:
    """Public chain commitment; external provider must independently anchor it."""
    _identifier(reservation_id, "reservation")
    return sha256(_HEAD_DOMAIN + request.digest().encode("ascii") + b"\0"
                  + str(request.expected_commit_index + 1).encode("ascii")
                  + b"\0" + reservation_id.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class ExternalReservationReceipt:
    provider_id: str
    run_id: str
    event_id: str
    reservation_id: str
    request_sha256: str
    epoch: int
    fence: int
    commit_index: int
    previous_head_sha256: str
    new_head_sha256: str
    signature_hex: str
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False

    def signed_payload(self) -> bytes:
        for label in ("provider_id", "run_id", "event_id", "reservation_id"):
            _identifier(getattr(self, label), label)
        for label in ("request_sha256", "previous_head_sha256", "new_head_sha256"):
            _digest(getattr(self, label), label)
        _counter(self.epoch, "receipt epoch", minimum=1)
        _counter(self.fence, "receipt fence")
        _counter(self.commit_index, "receipt commit index", minimum=1)
        if (self.graduation_authorized is not False
                or self.model_execution_authenticated is not False):
            raise ExternalAuthorityError("reservation must not claim graduation or model execution")
        return _SIGN_DOMAIN + _canonical({
            name: getattr(self, name) for name in self.__dataclass_fields__
            if name != "signature_hex"
        })

    def post_head(self) -> ExternalAuthorityHead:
        self.signed_payload()
        return ExternalAuthorityHead(
            self.provider_id, self.run_id, self.epoch, self.fence,
            self.commit_index, self.new_head_sha256)


class ExternalAuthorityPort(Protocol):
    """Implement OUTSIDE trainer trust domain with atomic durable consumption.

    Implementations MUST enforce one-shot event+exam+run uniqueness, immutable
    binding of candidate/scope/checkpoint, CAS against epoch/fence/index/head,
    and external monotonicity surviving trainer-side database rollback.
    """

    def read_head(self, run_id: str) -> ExternalAuthorityHead: ...

    def reserve_once(self, request: ExternalReservationRequest) -> ExternalReservationReceipt: ...


def _verify_ed25519(public_key: bytes, payload: bytes, signature: bytes) -> None:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
    except ImportError as exc:
        raise ExternalAuthorityError(
            "external receipt requires cryptography Ed25519 verifier") from exc
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, payload)
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise ExternalAuthorityError("external authority signature invalid") from exc


class ExternalMonotonicReservationClient:
    """One external CAS attempt; a signed receipt is NEVER accepted by itself.

    ``port`` and public key must be provisioned and pinned by the host, never
    selected by the learner. In-process code cannot attest the port's actual
    deployment/rollback domain; physical qualification remains separate.
    """

    def __init__(self, *, port: ExternalAuthorityPort, provider_id: str,
                 pinned_request: ExternalReservationRequest,
                 pinned_ed25519_public_key: bytes):
        self._port = port
        self._provider_id = _identifier(provider_id, "authority provider")
        if type(pinned_request) is not ExternalReservationRequest:
            raise ExternalAuthorityError("host-pinned typed reservation intent required")
        pinned_request.validate()
        if pinned_request.provider_id != self._provider_id:
            raise ExternalAuthorityConflict("host-pinned intent uses foreign provider")
        self._pinned_request = pinned_request
        if type(pinned_ed25519_public_key) is not bytes or len(pinned_ed25519_public_key) != 32:
            raise ExternalAuthorityError("32-byte pinned Ed25519 public key required")
        self._public_key = pinned_ed25519_public_key

    def reserve(self, request: ExternalReservationRequest) -> ExternalReservationReceipt:
        if type(request) is not ExternalReservationRequest:
            raise ExternalAuthorityError("typed reservation request required")
        request.validate()
        if request.provider_id != self._provider_id:
            raise ExternalAuthorityConflict("authority provider not pinned")
        if request != self._pinned_request:
            raise ExternalAuthorityConflict("reservation differs from host-pinned intent")
        try:
            old = self._port.read_head(request.run_id)
        except Exception as exc:
            raise ExternalAuthorityOutcomeUnknown("external authority preflight unavailable") from exc
        if type(old) is not ExternalAuthorityHead:
            raise ExternalAuthorityError("external head must be typed")
        old.validate()
        if old != request.expected_head():
            raise ExternalAuthorityConflict("external monotonic head mismatch; snapshot may be stale")
        # A lost ACK may mean the event was consumed. There is no automatic retry.
        try:
            receipt = self._port.reserve_once(request)
        except ExternalAuthorityConflict:
            raise
        except Exception as exc:
            raise ExternalAuthorityOutcomeUnknown("external reservation outcome unknown") from exc
        # Once reserve_once returns (or its ACK is lost), a durable commit
        # may already exist. An invalid receipt is therefore UNKNOWN, not a
        # safely retryable rejection. Never train based on these bytes.
        try:
            if type(receipt) is not ExternalReservationReceipt:
                raise ExternalAuthorityError("external provider returned untyped receipt")
            payload = receipt.signed_payload()
            if (type(receipt.signature_hex) is not str
                    or not _HEX128.fullmatch(receipt.signature_hex)):
                raise ExternalAuthorityError("malformed external signature")
            if (receipt.provider_id != request.provider_id
                    or receipt.run_id != request.run_id
                    or receipt.event_id != request.event_id
                    or not hmac.compare_digest(receipt.request_sha256, request.digest())
                    or receipt.epoch != request.expected_epoch
                    or receipt.fence != request.expected_fence
                    or receipt.commit_index != request.expected_commit_index + 1
                    or receipt.previous_head_sha256 != request.expected_head_sha256
                    or receipt.new_head_sha256 != next_head_digest(
                        request, receipt.reservation_id)):
                raise ExternalAuthorityConflict("receipt differs from exact reserved request/head")
            _verify_ed25519(self._public_key, payload,
                            bytes.fromhex(receipt.signature_hex))
        except (ExternalAuthorityError, TypeError) as exc:
            raise ExternalAuthorityOutcomeUnknown(
                "unverified post-reservation receipt; reconcile, do not retry") from exc
        try:
            current = self._port.read_head(request.run_id)
        except Exception as exc:
            raise ExternalAuthorityOutcomeUnknown(
                "reservation may have committed; external readback unavailable") from exc
        if type(current) is not ExternalAuthorityHead:
            raise ExternalAuthorityOutcomeUnknown("untyped post-commit head; do not retry")
        try:
            current.validate()
        except ExternalAuthorityError as exc:
            raise ExternalAuthorityOutcomeUnknown(
                "malformed post-commit head; reconcile, do not retry") from exc
        if current != receipt.post_head():
            # An intervening valid writer can advance the head. Conservative
            # fail closed: caller must independently reconcile, not retrain.
            raise ExternalAuthorityOutcomeUnknown(
                "external post-commit head differs; reconcile, do not retry")
        return receipt
