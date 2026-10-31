"""Fail-closed protocol for an *external* monotonic journal authority.

The local MAC-sealed SQLite journal cannot detect replacement with an older
MAC-valid snapshot.  A separately administered compare-and-swap authority must
own the increasing epoch and worker fence.  This module specifies the client
handshake; it does **not** implement or deploy that authority.  In particular,
an in-process fake, a sibling file, and a second database on the same host do
not satisfy the independence requirement.

The authority commits first.  Failure to update the local projection after an
authenticated remote acknowledgment causes HOLD/manual reconciliation; the
client must never retry a consumed operation against its stale local state.
The caller must enforce the current fence *atomically at each artifact sink*;
the read-only worker check below is only a preflight, not a write lease.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import re
from typing import Callable, Protocol


SCHEMA = "labyrinth.external-monotonic-anchor.v1"
_REQUEST_DOMAIN = b"labyrinth.external-monotonic-anchor.request.v1\0"
_RECEIPT_DOMAIN = b"labyrinth.external-monotonic-anchor.receipt.v1\0"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_PHASES = frozenset(("GENESIS", "RUNNING", "COMPLETED"))


class AnchorError(ValueError):
    """A state, signature, transition or fence failed validation."""


class AnchorUnavailable(AnchorError):
    """The independent authority could not be reached; all effects HOLD."""


class AnchorReconciliationRequired(AnchorError):
    """Remote authority may have advanced; local state cannot authorize retry."""


@dataclass(frozen=True)
class AnchorState:
    authority_id: str
    epoch: int
    journal_sha256: str
    run_id: str | None
    event_id: str | None
    fence: int
    checkpoint_sha256: str | None
    phase: str

    def validate(self) -> None:
        if type(self.authority_id) is not str or not _ID.fullmatch(self.authority_id):
            raise AnchorError("invalid independent authority identity")
        if (type(self.epoch) is not int or not 0 <= self.epoch < 2**63 - 1
                or type(self.fence) is not int or not 0 <= self.fence < 2**63 - 1
                or type(self.journal_sha256) is not str
                or not _DIGEST.fullmatch(self.journal_sha256)
                or type(self.phase) is not str
                or self.phase not in _PHASES):
            raise AnchorError("invalid monotonic state")
        if self.phase == "GENESIS":
            if (self.epoch != 0 or self.fence != 0
                    or self.run_id is not None or self.event_id is not None
                    or self.checkpoint_sha256 is not None):
                raise AnchorError("invalid unreserved anchor genesis")
        else:
            if (self.epoch < 1 or type(self.run_id) is not str
                    or not _ID.fullmatch(self.run_id)
                    or type(self.event_id) is not str
                    or not _ID.fullmatch(self.event_id)
                    or type(self.checkpoint_sha256) is not str
                    or not _DIGEST.fullmatch(self.checkpoint_sha256)):
                raise AnchorError("invalid anchored run custody")


@dataclass(frozen=True)
class SignedAnchorOperation:
    operation_id: str
    before: AnchorState
    after: AnchorState
    request_signature_hex: str


@dataclass(frozen=True)
class AuthorityReceipt:
    operation_id: str
    state: AnchorState
    authority_signature_hex: str


class ExternalMonotonicAuthority(Protocol):
    """An independently protected, authenticated, durable CAS service.

    Implementations MUST reject duplicate operation IDs, replays, stale epochs,
    independent run-ID collisions and state rollback.  They must issue an
    authority-authenticated receipt and never acknowledge before durable CAS.
    """

    def read(self, authority_id: str) -> AuthorityReceipt: ...

    def compare_and_swap(self, operation: SignedAnchorOperation) -> AuthorityReceipt: ...


def _key(value: bytes, label: str) -> None:
    if type(value) is not bytes or not 32 <= len(value) <= 4096:
        raise AnchorError(f"invalid {label} HMAC key")


def _distinct_keys(request_key: bytes, receipt_key: bytes) -> None:
    _key(request_key, "request")
    _key(receipt_key, "authority receipt")
    if hmac.compare_digest(request_key, receipt_key):
        raise AnchorError("request and independent authority keys must differ")


def _state_body(state: AnchorState) -> dict[str, object]:
    return {
        "authority_id": state.authority_id, "epoch": state.epoch,
        "journal_sha256": state.journal_sha256,
        "run_id": state.run_id, "event_id": state.event_id,
        "fence": state.fence, "checkpoint_sha256": state.checkpoint_sha256,
        "phase": state.phase,
    }


def _canonical(value: dict[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _operation_bytes(operation_id: str, before: AnchorState,
                     after: AnchorState) -> bytes:
    return _canonical({"schema": SCHEMA, "operation_id": operation_id,
                       "before": _state_body(before), "after": _state_body(after)})


def _receipt_bytes(operation_id: str, state: AnchorState) -> bytes:
    return _canonical({"schema": SCHEMA, "operation_id": operation_id,
                       "state": _state_body(state)})


def validate_transition(before: AnchorState, after: AnchorState) -> None:
    """Bounded reserve, promote and completion; no fence or phase rewind."""
    if type(before) is not AnchorState or type(after) is not AnchorState:
        raise AnchorError("typed anchor states required")
    before.validate()
    after.validate()
    if (after.authority_id != before.authority_id
            or after.epoch != before.epoch + 1
            or after.journal_sha256 == before.journal_sha256):
        raise AnchorError("anchor epoch or journal commitment did not advance")
    if before.phase == "GENESIS":
        if after.phase != "RUNNING" or after.fence != 0:
            raise AnchorError("initial reservation must start at fence zero")
    elif before.phase == "RUNNING":
        if (after.phase not in ("RUNNING", "COMPLETED")
                or after.run_id != before.run_id
                or after.event_id != before.event_id
                or after.fence != before.fence + 1):
            raise AnchorError("run ownership or worker fence changed illegally")
        if (after.phase == "RUNNING"
                and after.checkpoint_sha256 == before.checkpoint_sha256):
            raise AnchorError("checkpoint promotion requires a new checkpoint")
        if (after.phase == "COMPLETED"
                and after.checkpoint_sha256 != before.checkpoint_sha256):
            raise AnchorError("completion cannot silently replace checkpoint")
    else:
        raise AnchorError("completed run cannot be reopened")


def sign_operation(operation_id: str, before: AnchorState, after: AnchorState,
                   *, request_key: bytes, authority_receipt_key: bytes,
                   ) -> SignedAnchorOperation:
    """Host-signed intent; remote authority must still verify independently."""
    _distinct_keys(request_key, authority_receipt_key)
    if type(operation_id) is not str or not _ID.fullmatch(operation_id):
        raise AnchorError("invalid unique operation ID")
    validate_transition(before, after)
    signature = hmac.new(request_key,
                         _REQUEST_DOMAIN + _operation_bytes(operation_id, before, after),
                         sha256).hexdigest()
    return SignedAnchorOperation(operation_id, before, after, signature)


def verify_operation(operation: SignedAnchorOperation, *, request_key: bytes) -> None:
    """Mandatory authority-side request verification before its durable CAS."""
    _key(request_key, "request")
    if type(operation) is not SignedAnchorOperation:
        raise AnchorError("typed anchor operation required")
    validate_transition(operation.before, operation.after)
    if (type(operation.operation_id) is not str
            or not _ID.fullmatch(operation.operation_id)
            or type(operation.request_signature_hex) is not str
            or not _DIGEST.fullmatch(operation.request_signature_hex)):
        raise AnchorError("invalid signed operation identity")
    expected = hmac.new(request_key, _REQUEST_DOMAIN + _operation_bytes(
        operation.operation_id, operation.before, operation.after), sha256).hexdigest()
    if not hmac.compare_digest(expected, operation.request_signature_hex):
        raise AnchorError("anchor operation request signature mismatch")


def verify_receipt(receipt: AuthorityReceipt, *, receipt_key: bytes,
                   authority_id: str, minimum_epoch: int = 0,
                   ) -> AnchorState:
    """Verify externally authenticated readback, not merely client-supplied state."""
    _key(receipt_key, "authority receipt")
    if (type(receipt) is not AuthorityReceipt
            or type(receipt.operation_id) is not str
            or not _ID.fullmatch(receipt.operation_id)
            or type(receipt.authority_signature_hex) is not str
            or not _DIGEST.fullmatch(receipt.authority_signature_hex)
            or type(minimum_epoch) is not int or minimum_epoch < 0):
        raise AnchorError("invalid authority receipt")
    receipt.state.validate()
    if (receipt.state.authority_id != authority_id
            or receipt.state.epoch < minimum_epoch):
        raise AnchorError("stale or mismatched external anchor")
    expected = hmac.new(receipt_key, _RECEIPT_DOMAIN + _receipt_bytes(
        receipt.operation_id, receipt.state), sha256).hexdigest()
    if not hmac.compare_digest(expected, receipt.authority_signature_hex):
        raise AnchorError("external authority receipt authentication failed")
    return receipt.state


def read_anchor(authority: ExternalMonotonicAuthority, *, authority_id: str,
                receipt_key: bytes, minimum_epoch: int = 0) -> AnchorState:
    """Fail closed on partition, stale signed state, or unauthenticated read."""
    try:
        receipt = authority.read(authority_id)
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise AnchorUnavailable("external monotonic authority unreachable") from exc
    return verify_receipt(receipt, receipt_key=receipt_key,
                          authority_id=authority_id, minimum_epoch=minimum_epoch)


def advance_remote_then_project(
    authority: ExternalMonotonicAuthority, operation: SignedAnchorOperation, *,
    receipt_key: bytes,
    local_read: Callable[[], AnchorState],
    local_compare_and_swap: Callable[[AnchorState, AnchorState], bool],
) -> AuthorityReceipt:
    """Remote CAS *precedes* local projection; a torn transition remains spent.

    Local projection must also use an atomic CAS, never overwrite a newer
    value.  A recovery administrator may reconstruct it from an independently
    immutable authority log; this function intentionally cannot roll back.
    """
    if type(operation) is not SignedAnchorOperation:
        raise AnchorError("typed signed request required")
    validate_transition(operation.before, operation.after)
    if local_read() != operation.before:
        raise AnchorReconciliationRequired("local journal differs from planned predecessor")
    observed = read_anchor(authority, authority_id=operation.before.authority_id,
                           receipt_key=receipt_key,
                           minimum_epoch=operation.before.epoch)
    if observed != operation.before:
        raise AnchorReconciliationRequired("authority already advanced or diverged")
    try:
        acknowledged = authority.compare_and_swap(operation)
    except Exception as exc:
        # Even an unexpected provider error may follow a durable remote commit.
        raise AnchorReconciliationRequired(
            "remote commit outcome unknown; do not retry against local state"
        ) from exc
    try:
        accepted = verify_receipt(acknowledged, receipt_key=receipt_key,
                                  authority_id=operation.before.authority_id,
                                  minimum_epoch=operation.after.epoch)
    except AnchorError as exc:
        raise AnchorReconciliationRequired(
            "remote receipt untrusted after potential commit; reconciliation required"
        ) from exc
    if (acknowledged.operation_id != operation.operation_id
            or accepted != operation.after):
        raise AnchorReconciliationRequired("remote acknowledgment differs from planned state")
    try:
        after_remote = read_anchor(authority, authority_id=operation.after.authority_id,
                                   receipt_key=receipt_key,
                                   minimum_epoch=operation.after.epoch)
    except Exception as exc:
        raise AnchorReconciliationRequired(
            "remote readback unavailable after potential commit"
        ) from exc
    if after_remote != operation.after:
        raise AnchorReconciliationRequired("remote state moved before local projection")
    try:
        written = local_compare_and_swap(operation.before, operation.after)
    except Exception as exc:
        raise AnchorReconciliationRequired(
            "remote authority advanced; local commit failed and needs reconciliation"
        ) from exc
    try:
        projected = local_read()
    except Exception as exc:
        raise AnchorReconciliationRequired(
            "remote authority advanced; local readback failed"
        ) from exc
    if written is not True or projected != operation.after:
        raise AnchorReconciliationRequired(
            "remote authority advanced; local projection did not commit exactly"
        )
    try:
        remote_final = read_anchor(authority, authority_id=operation.after.authority_id,
                                   receipt_key=receipt_key,
                                   minimum_epoch=operation.after.epoch)
    except Exception as exc:
        raise AnchorReconciliationRequired(
            "remote authority unavailable after local projection; stop worker"
        ) from exc
    if remote_final != operation.after:
        raise AnchorReconciliationRequired("remote authority changed; local worker must stop")
    return acknowledged


def require_current_worker_fence(
    authority: ExternalMonotonicAuthority, *, authority_id: str,
    receipt_key: bytes, minimum_epoch: int, run_id: str, event_id: str,
    fence: int, checkpoint_sha256: str,
) -> AnchorState:
    """Read-only preflight; artifact sink MUST do an atomic fence check too."""
    state = read_anchor(authority, authority_id=authority_id,
                        receipt_key=receipt_key, minimum_epoch=minimum_epoch)
    if (state.phase != "RUNNING" or state.run_id != run_id
            or state.event_id != event_id or state.fence != fence
            or state.checkpoint_sha256 != checkpoint_sha256):
        raise AnchorError("stale or cross-run worker fence; block output")
    return state
