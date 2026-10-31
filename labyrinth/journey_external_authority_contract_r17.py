"""Versioned SOURCE binding for a native external Journey reservation intent.

This module specifies exactly what ``candidate_artifact_sha256`` means when
the R12 signed Journey uses the separate Ed25519 external-authority protocol:
it is a domain-separated commitment to *all* host, objective, policy, TRAIN,
model-continuity, source and migrated-event-archive identities below. It is
not a bridge to the older HMAC HostLoopReservation signed-byte format.

An external provider must still atomically enforce global event/exam/run
uniqueness, durable monotonicity and the witnessed migration root. No source
check can attest that the migration witness is real, that signing keys are
host-owned, or that a model executed. These helpers never reserve, train,
promote, or authorize graduation. They are a source-only request contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re

from .external_monotonic_authority_contract import (
    ExternalAuthorityError, ExternalAuthorityHead,
    ExternalReservationRequest,
)


SCHEMA = "labyrinth.journey-native-authority-binding.v1"
PURPOSE = "DEVELOPMENT_TRAIN_ONLY"
_DOMAIN = b"labyrinth.journey-native-authority-binding.v1\0"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_SHA = re.compile(r"[a-f0-9]{64}\Z", re.ASCII)
_MAX_COUNTER = 2**63 - 1


class JourneyAuthorityBindingError(ValueError):
    """A host-pinned Journey identity or external request is incompatible."""


def _id(value: object, name: str) -> None:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise JourneyAuthorityBindingError(f"invalid {name}")


def _sha(value: object, name: str) -> None:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise JourneyAuthorityBindingError(f"invalid {name}")


@dataclass(frozen=True)
class JourneyExternalAuthorityBinding:
    """Immutable operator-pinned declarative source material, not a permit.

    The migration root and witness commitments are *declared* inputs. Their
    custody, freshness and relationship to the old consumed-event history
    must be established outside this Python process before a cutover.
    """

    provider_id: str
    host_id: str
    run_id: str
    event_id: str
    exam_id: str
    scope_sha256: str
    objective_contract_sha256: str
    exam_lineage_sha256: str
    journey_sha256: str
    signed_binding_sha256: str
    learner_checkpoint_sha256: str
    prior_training_roster_sha256: str
    next_training_allocation_sha256: str
    host_policy_sha256: str
    source_tree_sha256: str
    fullstate_binding_sha256: str
    consumed_event_ledger_root_sha256: str
    migration_witness_sha256: str
    previous_cycle: int
    next_cycle: int
    purpose: str = PURPOSE
    schema: str = SCHEMA

    def validate(self) -> None:
        if type(self) is not JourneyExternalAuthorityBinding:
            raise JourneyAuthorityBindingError("exact typed Journey binding required")
        for name in ("provider_id", "host_id", "run_id", "event_id", "exam_id"):
            _id(getattr(self, name), name)
        for name in (
            "scope_sha256", "objective_contract_sha256", "exam_lineage_sha256",
            "journey_sha256", "signed_binding_sha256", "learner_checkpoint_sha256",
            "prior_training_roster_sha256", "next_training_allocation_sha256",
            "host_policy_sha256", "source_tree_sha256", "fullstate_binding_sha256",
            "consumed_event_ledger_root_sha256", "migration_witness_sha256",
        ):
            _sha(getattr(self, name), name)
        if (self.schema != SCHEMA or self.purpose != PURPOSE
                or type(self.previous_cycle) is not int
                or not 0 <= self.previous_cycle < _MAX_COUNTER - 1
                or type(self.next_cycle) is not int
                or self.next_cycle != self.previous_cycle + 1):
            raise JourneyAuthorityBindingError("Journey cycle, schema or effect purpose invalid")

    def payload(self) -> bytes:
        """Deterministic exact-field JSON, with an explicit version and purpose."""
        self.validate()
        record = {field: getattr(self, field)
                  for field in self.__dataclass_fields__}
        return (json.dumps(record, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")

    def digest(self) -> str:
        return sha256(_DOMAIN + self.payload()).hexdigest()


def build_journey_external_request(
    binding: JourneyExternalAuthorityBinding,
    expected_head: ExternalAuthorityHead,
    *,
    operation_nonce: str,
) -> ExternalReservationRequest:
    """Construct an exact native Ed25519-provider request; do not send it.

    The candidate-artifact commitment is intentionally the full Journey
    binding digest. Actual provider CAS and signed-receipt readback remain the
    responsibility of ``ExternalMonotonicReservationClient`` plus its host.
    """
    if type(binding) is not JourneyExternalAuthorityBinding:
        raise JourneyAuthorityBindingError("typed Journey binding required")
    binding.validate()
    if type(expected_head) is not ExternalAuthorityHead:
        raise JourneyAuthorityBindingError("typed external monotonic head required")
    try:
        expected_head.validate()
    except ExternalAuthorityError as exc:
        raise JourneyAuthorityBindingError("invalid external monotonic head") from exc
    if (expected_head.provider_id != binding.provider_id
            or expected_head.run_id != binding.run_id):
        raise JourneyAuthorityBindingError("external head and Journey host identities differ")
    _id(operation_nonce, "operation nonce")
    request = ExternalReservationRequest(
        provider_id=binding.provider_id, run_id=binding.run_id,
        event_id=binding.event_id, exam_id=binding.exam_id,
        candidate_artifact_sha256=binding.digest(),
        scope_fingerprint=binding.scope_sha256,
        exam_lineage_sha256=binding.exam_lineage_sha256,
        prior_checkpoint_sha256=binding.learner_checkpoint_sha256,
        to_cycle=binding.next_cycle,
        expected_epoch=expected_head.epoch,
        expected_fence=expected_head.fence,
        expected_commit_index=expected_head.commit_index,
        expected_head_sha256=expected_head.head_sha256,
        operation_nonce=operation_nonce,
    )
    try:
        request.validate()
    except ExternalAuthorityError as exc:
        raise JourneyAuthorityBindingError("derived native authority request invalid") from exc
    return request


def verify_journey_external_request(
    binding: JourneyExternalAuthorityBinding,
    expected_head: ExternalAuthorityHead,
    request: ExternalReservationRequest,
    *,
    operation_nonce: str,
) -> None:
    """Reject *any* request differing from the entire host-pinned binding.

    Success returns no permission object or host receipt; this is a pure
    source contract check prior to independently governed external reservation.
    """
    if type(request) is not ExternalReservationRequest:
        raise JourneyAuthorityBindingError("typed native reservation request required")
    expected = build_journey_external_request(
        binding, expected_head, operation_nonce=operation_nonce)
    try:
        request.validate()
    except ExternalAuthorityError as exc:
        raise JourneyAuthorityBindingError("invalid native reservation request") from exc
    if request != expected:
        raise JourneyAuthorityBindingError("native request differs from full Journey commitment")
