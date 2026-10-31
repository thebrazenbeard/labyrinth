"""Source-only boundary for *sink-owned*, fence-atomic artifact publication.

This is an interface, NOT a deployed publication service. A worker may write
only to opaque, nonpublic sink staging; the independently administered sink
must own the public destination and perform the epoch/fence/current-digest CAS
in the SAME durable transaction as making the new artifact visible.

Neither a client-side fence read immediately before rename nor a lock inside
the worker process meets this contract. Implementations must verify the
staged bytes against the digest, reject repeated operation IDs and races, and
sign durable receipts with a private key unavailable to worker processes.
The verifier must be operator-pinned and independently administered. In-memory
test doubles and caller-supplied verification functions are source tests,
never evidence of installed authority, protected destinations, or Q8.

Partition/timeout/invalid acknowledgment is UNKNOWN_COMMIT, not authorization
to retry or delete the staged blob. External reconciliation is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Callable, Protocol


SCHEMA = "labyrinth.sink-atomic-artifact-publication.v1"
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class ArtifactPublicationError(ValueError):
    """Malformed intent or untrusted sink acknowledgment."""


class PublicationDenied(ArtifactPublicationError):
    """Sink explicitly rejected a stale/conflicting publication."""


class PublicationOutcomeUnknown(ArtifactPublicationError):
    """Commit might have occurred; stop and reconcile, never retry blindly."""


class SinkConflict(Exception):
    """A conforming sink reports that its *atomic* CAS did not publish."""


@dataclass(frozen=True)
class PublicationIntent:
    sink_id: str
    operation_id: str
    run_id: str
    artifact_id: str
    staged_object_id: str
    epoch: int
    fence: int
    expected_current_sha256: str
    staged_sha256: str

    def validate(self) -> None:
        for name in ("sink_id", "operation_id", "run_id",
                     "artifact_id", "staged_object_id"):
            value = getattr(self, name)
            if type(value) is not str or not _IDENTITY.fullmatch(value):
                raise ArtifactPublicationError(f"invalid {name}")
        if (type(self.epoch) is not int or not 1 <= self.epoch < 2**63 - 1
                or type(self.fence) is not int or not 0 <= self.fence < 2**63 - 1):
            raise ArtifactPublicationError("invalid epoch/fence")
        for name in ("expected_current_sha256", "staged_sha256"):
            value = getattr(self, name)
            if type(value) is not str or not _DIGEST.fullmatch(value):
                raise ArtifactPublicationError(f"invalid {name}")
        if self.staged_sha256 == self.expected_current_sha256:
            raise ArtifactPublicationError("published artifact must advance digest")


@dataclass(frozen=True)
class PublicationReceipt:
    """An independently signed durable acknowledgment from the sink.

    Merely constructing this dataclass does not establish any authenticity;
    the pinned verifier below must reject untrusted bytes.
    """

    intent: PublicationIntent
    transaction_id: str
    current_sha256: str
    durable: bool
    signature_hex: str

    def validate(self) -> None:
        if type(self.intent) is not PublicationIntent:
            raise ArtifactPublicationError("receipt intent type invalid")
        self.intent.validate()
        if (type(self.transaction_id) is not str
                or not _IDENTITY.fullmatch(self.transaction_id)
                or type(self.current_sha256) is not str
                or not _DIGEST.fullmatch(self.current_sha256)
                or self.current_sha256 != self.intent.staged_sha256
                or self.durable is not True
                or type(self.signature_hex) is not str
                or not self.signature_hex):
            raise ArtifactPublicationError("invalid or nondurable receipt")


def canonical_receipt_payload(receipt: PublicationReceipt) -> bytes:
    """Signature input; signature itself is excluded and domain-separated."""
    receipt.validate()
    intent = receipt.intent
    values = {
        "schema": SCHEMA,
        "intent": {name: getattr(intent, name) for name in
                   ("sink_id", "operation_id", "run_id", "artifact_id",
                    "staged_object_id", "epoch", "fence",
                    "expected_current_sha256", "staged_sha256")},
        "transaction_id": receipt.transaction_id,
        "current_sha256": receipt.current_sha256,
        "durable": receipt.durable,
    }
    return (json.dumps(values, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


class IndependentlyOwnedPublicationSink(Protocol):
    """Sink must atomically CAS authority *and* artifact visibility.

    On conflict it raises SinkConflict **without any public artifact effect**.
    On ambiguous transport failure it may have committed, so callers HOLD.
    This interface cannot attest that an implementation meets the contract.
    """

    def commit_if_current(self, intent: PublicationIntent) -> PublicationReceipt: ...


# Production MUST pin a public-key verifier outside the untrusted worker and
# reject any verifier or sink implementation supplied by the model under test.
PinnedReceiptVerifier = Callable[[bytes, str, str], bool]


def publish_with_sink(
    sink: IndependentlyOwnedPublicationSink, intent: PublicationIntent, *,
    expected_sink_id: str,
    verify_receipt_signature: PinnedReceiptVerifier,
) -> PublicationReceipt:
    """Request publication once; never mask unknown commit as rejection.

    A successful return means only that a caller-pinned verifier accepted the
    sink's signed *claim*. Independent operational qualification still requires
    a protected, deployed sink and actual crash/race testing.
    """
    if type(intent) is not PublicationIntent:
        raise ArtifactPublicationError("typed publication intent required")
    intent.validate()
    if (type(expected_sink_id) is not str or not _IDENTITY.fullmatch(expected_sink_id)
            or intent.sink_id != expected_sink_id
            or not callable(verify_receipt_signature)):
        raise ArtifactPublicationError("trusted sink identity/verifier required")
    try:
        receipt = sink.commit_if_current(intent)
    except SinkConflict as exc:
        raise PublicationDenied("stale fence/digest; no publication authorized") from exc
    except Exception as exc:
        raise PublicationOutcomeUnknown(
            "sink commit outcome unknown; reconcile externally before another operation"
        ) from exc
    try:
        if type(receipt) is not PublicationReceipt or receipt.intent != intent:
            raise ArtifactPublicationError("sink returned mismatched receipt")
        receipt.validate()
        accepted = verify_receipt_signature(
            canonical_receipt_payload(receipt), receipt.signature_hex,
            expected_sink_id,
        )
        if accepted is not True:
            raise ArtifactPublicationError("independent sink signature not verified")
    except Exception as exc:
        raise PublicationOutcomeUnknown(
            "commit may have succeeded but acknowledgment is unauthenticated"
        ) from exc
    return receipt
