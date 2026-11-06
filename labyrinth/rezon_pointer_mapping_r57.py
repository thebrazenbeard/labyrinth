"""Inert Labyrinth/Rezon *source-observation* compatibility check.

The Rezon source record is supplied by a caller; this module does not parse or
verify Rezon receipts, claim an owner attestation, equate unlike artifact hashes,
or grant any identity, consent, training, merge or deployment authority.

Both expected pin sets MUST be independently acquired through their respective
owners' governed source registries; reconstructing them from offered claims is
not an independent check. This is not a host authorization adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import re
from typing import Any, NamedTuple

from labyrinth.evidence_pointer import (
    ExpectedSourcePinsV1, SourceEvidenceMatchV1,
    verify_source_evidence_pointer,
)


EXPECTED_ARMS = ("correct", "zero_update", "inverted", "zero_lr", "randomized_labels")
TRAINING_PROGRAM = "LABYRINTH_MINILM_FIVE_ARM"
BRIDGE_SEMANTICS = "PROVENANCE_ONLY"
DISPOSITION = "OBSERVED_SOURCE"
QUALIFICATION = "PARTIAL_SOURCE_ALIGNMENT_UNATTESTED"

_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_GIT_FIELDS = ("rezon_source_head", "labyrinth_source_head")
_DIGEST_FIELDS = (
    "receipt_bundle_sha256", "maze_exposure_sha256", "checkpoint_sha256",
    "control_matrix_sha256", "development_event_sha256",
)
_UNRESOLVED_SEMANTIC_PAIRS = (
    ("train_exposure_sha256", "maze_exposure_sha256"),
    ("checkpoint_manifest_sha256", "checkpoint_sha256"),
)


class RezonSourceAlignmentError(ValueError):
    """Incompatible or non-independent Rezon *source* pins; never a host verdict."""


@dataclass(frozen=True, slots=True)
class RezonSourceClaimsV1:
    """Offered source values transcribed from separately preserved Rezon evidence.

    ``receipt_bundle_sha256`` is an externally maintained *opaque bundle
    reference*, NOT a native Rezon ``ExactPins`` field, nor a receipt signature.
    """

    run_id: str
    rezon_source_head: str
    labyrinth_source_head: str
    receipt_bundle_sha256: str
    maze_exposure_sha256: str
    checkpoint_sha256: str
    control_matrix_sha256: str
    development_event_sha256: str
    training_program: str
    bridge_semantics: str
    control_arms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExpectedRezonPinsV1:
    """Owner-governed expected values obtained independently of offered claims.

    This structure alone cannot prove that such an authority was consulted.
    """

    run_id: str
    rezon_source_head: str
    labyrinth_source_head: str
    receipt_bundle_sha256: str
    maze_exposure_sha256: str
    checkpoint_sha256: str
    control_matrix_sha256: str
    development_event_sha256: str
    training_program: str
    bridge_semantics: str
    control_arms: tuple[str, ...]


class RezonSourceAlignmentV1(NamedTuple):
    """Tuple-backed observation; even object.__setattr__ cannot promote it.

    Only source run identifiers are constructor fields. Every status/authority
    property is a fixed constant; no caller-supplied flags can be constructed.
    """

    labyrinth_run_id: str
    rezon_run_id: str

    @property
    def unresolved_semantic_pairs(self) -> tuple[tuple[str, str], ...]:
        return _UNRESOLVED_SEMANTIC_PAIRS

    @property
    def disposition(self) -> str:
        return DISPOSITION

    @property
    def qualification_state(self) -> str:
        return QUALIFICATION

    @property
    def source_receipts_verified(self) -> bool:
        return False

    @property
    def host_execution_verified(self) -> bool:
        return False

    @property
    def checkpoint_verified(self) -> bool:
        return False

    @property
    def model_identity_verified(self) -> bool:
        return False

    @property
    def consent_verified(self) -> bool:
        return False

    @property
    def permissions_verified(self) -> bool:
        return False

    @property
    def cross_model_grants_transferred(self) -> bool:
        return False

    @property
    def can_authorize_training(self) -> bool:
        return False

    @property
    def can_authorize_graduation(self) -> bool:
        return False

    @property
    def can_authorize_merge(self) -> bool:
        return False

    @property
    def can_authorize_deployment(self) -> bool:
        return False

    @property
    def agi_qualification_verified(self) -> bool:
        return False


def _check_rezon_source_pins(value: Any) -> None:
    if type(value) not in (RezonSourceClaimsV1, ExpectedRezonPinsV1):
        raise RezonSourceAlignmentError("exact Rezon source pin type required")
    if type(value.run_id) is not str or _RUN_ID.fullmatch(value.run_id) is None:
        raise RezonSourceAlignmentError("invalid bounded Rezon run identity")
    for name in _GIT_FIELDS:
        item = getattr(value, name)
        if type(item) is not str or _GIT_SHA.fullmatch(item) is None:
            raise RezonSourceAlignmentError(f"invalid Git source field: {name}")
    for name in _DIGEST_FIELDS:
        item = getattr(value, name)
        if type(item) is not str or _DIGEST.fullmatch(item) is None:
            raise RezonSourceAlignmentError(f"invalid SHA-256 source field: {name}")
    if (type(value.training_program) is not str
            or value.training_program != TRAINING_PROGRAM
            or type(value.bridge_semantics) is not str
            or value.bridge_semantics != BRIDGE_SEMANTICS
            or type(value.control_arms) is not tuple
            or value.control_arms != EXPECTED_ARMS):
        raise RezonSourceAlignmentError("wrong five-arm program or provenance-only semantics")


def compare_source_evidence(
    labyrinth_pointer_payload: bytes,
    *,
    expected_labyrinth: ExpectedSourcePinsV1,
    observed_rezon: RezonSourceClaimsV1,
    expected_rezon: ExpectedRezonPinsV1,
) -> RezonSourceAlignmentV1:
    """Compare two source assertions only, leaving divergent semantics unresolved.

    Labyrinth source head, Rezon source head, opaque receipt-bundle reference,
    control matrix and development event are comparable *source* references.
    Rezon's maze exposure and checkpoint hashes describe different object types
    from Labyrinth's training exposure and checkpoint manifest. Neither pair
    receives equality, coverage, physical verification or authorization.
    """
    lab: SourceEvidenceMatchV1 = verify_source_evidence_pointer(
        labyrinth_pointer_payload, expected=expected_labyrinth,
    )
    if type(observed_rezon) is not RezonSourceClaimsV1:
        raise RezonSourceAlignmentError("exact observed Rezon source claims required")
    if type(expected_rezon) is not ExpectedRezonPinsV1:
        raise RezonSourceAlignmentError("independently pinned Rezon expectations required")
    _check_rezon_source_pins(observed_rezon)
    _check_rezon_source_pins(expected_rezon)
    for item in fields(RezonSourceClaimsV1):
        name = item.name
        if getattr(observed_rezon, name) != getattr(expected_rezon, name):
            raise RezonSourceAlignmentError(f"unmatched expected Rezon {name}")

    lab_pointer = lab.pointer
    if lab_pointer.rezon_source_head is None:
        raise RezonSourceAlignmentError("Labyrinth pointer omits Rezon source reference")
    for lab_name, rezon_name in (
        ("source_head", "labyrinth_source_head"),
        ("rezon_source_head", "rezon_source_head"),
        ("rezon_receipt_bundle_sha256", "receipt_bundle_sha256"),
        ("control_matrix_sha256", "control_matrix_sha256"),
        ("development_event_sha256", "development_event_sha256"),
    ):
        if getattr(lab_pointer, lab_name) != getattr(observed_rezon, rezon_name):
            raise RezonSourceAlignmentError(
                f"cross-source reference mismatch: {lab_name} / {rezon_name}")

    return RezonSourceAlignmentV1(
        labyrinth_run_id=lab_pointer.run_id,
        rezon_run_id=observed_rezon.run_id,
    )
