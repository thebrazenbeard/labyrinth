"""Pure source-observation comparison for two separately pinned model adapters.

This is intentionally not a model router, identity verifier, host authorization
service, consent ledger, checkpoint attestation or training/graduation gate.
Expected pins MUST be obtained independently from each owner's governed source
registry; simply copying the offered claims is not an independent check.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
import re
from typing import Any, NamedTuple

from labyrinth.evidence_pointer import ExpectedSourcePinsV1
from labyrinth.rezon_pointer_mapping_r57 import (
    ExpectedRezonPinsV1, RezonSourceClaimsV1, compare_source_evidence,
)


QUALIFICATION = "SOURCE_ONLY_TWO_ADAPTERS_UNATTESTED"
_SCHEMA = "labyrinth.rezon-two-adapter-source-observation.v1"
_ADAPTER_ID = re.compile(r"[a-z][a-z0-9_.:-]{0,63}\Z", re.ASCII)
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_UNRESOLVED_SEMANTIC_PAIRS = (
    ("train_exposure_sha256", "maze_exposure_sha256"),
    ("checkpoint_manifest_sha256", "checkpoint_sha256"),
)


class AdapterSourcePartitionError(ValueError):
    """Invalid or mismatched *source* pins, never a host authority decision."""


@dataclass(frozen=True, slots=True)
class AdapterSourceClaimV1:
    """Offered adapter/model source pins, excluding any security authority."""

    adapter_id: str
    model_source_head: str
    model_artifact_sha256: str
    adapter_contract_sha256: str


@dataclass(frozen=True, slots=True)
class ExpectedAdapterSourceV1:
    """Independently maintained expected source pins for one model adapter."""

    adapter_id: str
    model_source_head: str
    model_artifact_sha256: str
    adapter_contract_sha256: str


class AdapterPairSourceObservationV1(NamedTuple):
    """Untrusted digest of source inputs, with immutable, non-authoritative flags.

    The digest is a comparison fingerprint, not an attested identifier or token.
    Even a caller-constructed or replaced instance cannot carry grant fields.
    """

    source_pair_sha256: str

    @property
    def qualification_state(self) -> str:
        return QUALIFICATION

    @property
    def unresolved_semantic_pairs(self) -> tuple[tuple[str, str], ...]:
        return _UNRESOLVED_SEMANTIC_PAIRS

    @property
    def model_identity_verified(self) -> bool:
        return False

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
    def checkpoint_identity_verified(self) -> bool:
        return False

    @property
    def consent_verified(self) -> bool:
        return False

    @property
    def permissions_verified(self) -> bool:
        return False

    @property
    def principal_verified(self) -> bool:
        return False

    @property
    def cross_model_grants_transferred(self) -> bool:
        return False

    @property
    def cross_model_identity_transferred(self) -> bool:
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


class _AdapterSourceSnapshotV1(NamedTuple):
    """Immutable value copy: source validation and hashing use the SAME pins."""

    adapter_id: str
    model_source_head: str
    model_artifact_sha256: str
    adapter_contract_sha256: str


def _freeze_source_value(value: Any) -> str | None | tuple[str, ...]:
    """Deep copy the complete supported source-value grammar, not references."""
    if type(value) is str or value is None:
        return value
    if type(value) is tuple and all(type(item) is str for item in value):
        return tuple(item for item in value)
    raise AdapterSourcePartitionError("unsupported or mutable source pin type")


def _source_snapshot(value: Any, required_type: type) -> tuple[tuple[str, Any], ...]:
    if type(value) is not required_type:
        raise AdapterSourcePartitionError("exact source claim/expectation type required")
    return tuple((item.name, _freeze_source_value(getattr(value, item.name)))
                 for item in fields(required_type))


def _adapter_snapshot(value: Any, required_type: type) -> _AdapterSourceSnapshotV1:
    captured = dict(_source_snapshot(value, required_type))
    snapshot = _AdapterSourceSnapshotV1(**captured)
    for name, pattern in (
        ("adapter_id", _ADAPTER_ID),
        ("model_source_head", _GIT_SHA),
        ("model_artifact_sha256", _DIGEST),
        ("adapter_contract_sha256", _DIGEST),
    ):
        part = getattr(snapshot, name)
        if type(part) is not str or pattern.fullmatch(part) is None:
            raise AdapterSourcePartitionError(f"invalid adapter source pin: {name}")
    return snapshot


def _require_independent_match(observed: _AdapterSourceSnapshotV1,
                               expected: _AdapterSourceSnapshotV1) -> None:
    for name in _AdapterSourceSnapshotV1._fields:
        if getattr(observed, name) != getattr(expected, name):
            raise AdapterSourcePartitionError(
                f"unmatched expected adapter source pin: {name}")


def compare_two_model_adapter_sources(
    *, labyrinth_pointer_payload: bytes,
    expected_labyrinth: ExpectedSourcePinsV1,
    observed_rezon: RezonSourceClaimsV1,
    expected_rezon: ExpectedRezonPinsV1,
    observed_alpha: AdapterSourceClaimV1,
    expected_alpha: ExpectedAdapterSourceV1,
    observed_beta: AdapterSourceClaimV1,
    expected_beta: ExpectedAdapterSourceV1,
) -> AdapterPairSourceObservationV1:
    """Observe one Labyrinth/Rezon source pair across two distinct model pins.

    All four expectations have separate owner provenance requirements. No
    model invocation, checkpoint decoding, host IO or authority is performed.
    Semantic mismatches in exposure/checkpoint object types remain unresolved.
    """
    # Capture primitive source values ONCE before any comparisons. Frozen
    # dataclasses are not a security boundary: object.__setattr__ and a trace
    # hook can mutate their slots between reads. Every subsequent validation,
    # distinct-model check and digest uses these immutable sealed values.
    sealed_lab_expected = _source_snapshot(expected_labyrinth, ExpectedSourcePinsV1)
    sealed_rezon_observed = _source_snapshot(observed_rezon, RezonSourceClaimsV1)
    sealed_rezon_expected = _source_snapshot(expected_rezon, ExpectedRezonPinsV1)
    alpha = _adapter_snapshot(observed_alpha, AdapterSourceClaimV1)
    alpha_expected = _adapter_snapshot(expected_alpha, ExpectedAdapterSourceV1)
    beta = _adapter_snapshot(observed_beta, AdapterSourceClaimV1)
    beta_expected = _adapter_snapshot(expected_beta, ExpectedAdapterSourceV1)

    alignment = compare_source_evidence(
        labyrinth_pointer_payload,
        expected_labyrinth=expected_labyrinth,
        observed_rezon=observed_rezon,
        expected_rezon=expected_rezon,
    )
    _require_independent_match(alpha, alpha_expected)
    _require_independent_match(beta, beta_expected)
    if alpha.adapter_id == beta.adapter_id:
        raise AdapterSourcePartitionError("two distinct adapter keys required")
    if alpha.model_artifact_sha256 == beta.model_artifact_sha256:
        raise AdapterSourcePartitionError("two distinct model artifact digests required")

    # Full source bytes and Rezon assertions bind this fingerprint to the
    # comparison's inputs; the fingerprint is NEVER a signed identity token.
    material = {
        "schema": _SCHEMA,
        "labyrinth_pointer_payload_sha256": sha256(labyrinth_pointer_payload).hexdigest(),
        "labyrinth_run_id": alignment.labyrinth_run_id,
        "rezon_run_id": alignment.rezon_run_id,
        "rezon_source_claims": dict(sealed_rezon_observed),
        "adapter_alpha": alpha._asdict(),
        "adapter_beta": beta._asdict(),
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    digest = sha256(encoded).hexdigest()

    # Second read is a best-effort fail-closed TOCTOU fence. A mutation after
    # any prior check (including the R57 Labyrinth/Rezon source comparison)
    # invalidates this local observation; no mutable live object contributes
    # bytes to the digest. Hostile code in this same Python process can still
    # monkeypatch classes/functions, so this is NOT authentication.
    for value, required_type, sealed in (
        (expected_labyrinth, ExpectedSourcePinsV1, sealed_lab_expected),
        (observed_rezon, RezonSourceClaimsV1, sealed_rezon_observed),
        (expected_rezon, ExpectedRezonPinsV1, sealed_rezon_expected),
    ):
        if _source_snapshot(value, required_type) != sealed:
            raise AdapterSourcePartitionError("source pins mutated during comparison")
    for value, required_type, sealed in (
        (observed_alpha, AdapterSourceClaimV1, alpha),
        (expected_alpha, ExpectedAdapterSourceV1, alpha_expected),
        (observed_beta, AdapterSourceClaimV1, beta),
        (expected_beta, ExpectedAdapterSourceV1, beta_expected),
    ):
        if _adapter_snapshot(value, required_type) != sealed:
            raise AdapterSourcePartitionError("model-adapter pins mutated during comparison")
    return AdapterPairSourceObservationV1(source_pair_sha256=digest)
