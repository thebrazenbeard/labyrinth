"""Source-only Q6 campaign reservation boundary; never host qualification.

The Q6 toy runner scores public heldouts before it publishes its report. A
caller-supplied previous-exposure set cannot make that operation fresh after a
crash. This module prepares a public, gold-free campaign manifest and gates a
*caller-supplied callback* behind one externally signed, durably read-back
campaign reservation. A missing or ambiguous authority refuses the callback.

There is deliberately no local authority implementation, no private signing
key, no training/evaluation execution, and no claim that individual prompts or
optimizer steps have been reserved. Physical qualification requires an
independently owned, monotonic provider and a pre-reveal per-exposure service;
this Python interface alone cannot establish either. The ordinary unguarded
Q6 toy runner remains explicitly research-only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import hmac
import json
import re
from typing import Any, Callable, Generic, Protocol, TypeVar

from .q6_stratified_toy_study_r93 import (
    SCHEMA as Q6_SOURCE_GENERATOR_SCHEMA, Q6Cohorts,
    construct_q6_cohorts, semantic_case_key, validate_q6_cohorts,
)


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_HEX128 = re.compile(r"[0-9a-f]{128}\Z", re.ASCII)
_MANIFEST_DOMAIN = b"labyrinth.q6-governed-campaign-manifest.v1\0"
_SEMANTIC_DOMAIN = b"labyrinth.q6-arithmetic-semantic-v1\0"
_REQUEST_DOMAIN = b"labyrinth.q6-governed-reservation-request.v1\0"
_HEAD_DOMAIN = b"labyrinth.q6-governed-reservation-head.v1\0"
_RECEIPT_DOMAIN = b"labyrinth.q6-governed-reservation-receipt.v1\0"
_ARMS = ("trained", "zero_update", "inverted_labels")
_EVAL_PHASES = ("initial", "after_a", "after_b")
_SPLITS = (("train_a", "train"), ("train_b", "train"),
           ("heldout", "q6_heldout"), ("transfer", "q6_transfer"))
_MAX_COUNTER = 2**63 - 1
T = TypeVar("T")


class Q6CampaignAuthorityError(ValueError):
    """Malformed, unavailable, or unauthenticated campaign authorization."""


class Q6CampaignOutcomeUnknown(Q6CampaignAuthorityError):
    """Reservation may have committed; no callback and no blind retry."""


def _id(value: object, label: str) -> str:
    if type(value) is not str or not _ID.fullmatch(value):
        raise Q6CampaignAuthorityError(f"invalid {label}")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or not _HEX64.fullmatch(value):
        raise Q6CampaignAuthorityError(f"invalid {label}")
    return value


def _counter(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value < _MAX_COUNTER:
        raise Q6CampaignAuthorityError(f"invalid {label}")
    return value


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


@dataclass(frozen=True)
class Q6PublicCaseCommitment:
    partition: str
    split: str
    case_id: str
    prompt_sha256: str
    arithmetic_semantic_sha256: str


@dataclass(frozen=True)
class Q6PlannedExposure:
    """Declared usage COUNT, not a witnessed individual exposure."""

    partition: str
    case_id: str
    arm: str
    phase: str
    planned_visits: int


@dataclass(frozen=True)
class Q6CampaignManifest:
    schema: str
    source_generator_schema: str
    seed: int
    per_profile: int
    epochs_stage_a: int
    epochs_stage_b: int
    cases: tuple[Q6PublicCaseCommitment, ...]
    planned_exposures: tuple[Q6PlannedExposure, ...]

    def validate(self) -> None:
        if type(self) is not Q6CampaignManifest or self.schema != "labyrinth.q6-governed-campaign.v1":
            raise Q6CampaignAuthorityError("unknown Q6 campaign manifest")
        if self.source_generator_schema != Q6_SOURCE_GENERATOR_SCHEMA:
            raise Q6CampaignAuthorityError("unrecognized pinned Q6 source generator schema")
        if (type(self.seed) is not int or not 0 <= self.seed < 2**31
                or type(self.per_profile) is not int or not 2 <= self.per_profile <= 24):
            raise Q6CampaignAuthorityError("invalid Q6 campaign cohort parameters")
        for label, count in (("stage A epochs", self.epochs_stage_a),
                             ("stage B epochs", self.epochs_stage_b)):
            if type(count) is not int or not 1 <= count <= 1000:
                raise Q6CampaignAuthorityError(f"invalid {label}")
        if (type(self.cases) is not tuple or len(self.cases) != 16 * self.per_profile
                or type(self.planned_exposures) is not tuple):
            raise Q6CampaignAuthorityError("Q6 source manifest incomplete")
        per_partition = {partition: 0 for partition, _ in _SPLITS}
        used_ids: set[str] = set()
        used_prompt_hashes: set[str] = set()
        used_semantics: set[str] = set()
        expected_slots: list[Q6PlannedExposure] = []
        # A self-consistent caller-controlled cohort is NOT evidence of which
        # seed generated it. Reconstruct every ordered full Challenge from the
        # exact pinned generator; changed IDs, gold, metadata, order and seed
        # must fail before a provider or model callback is reached.
        generated = construct_q6_cohorts(seed=self.seed,
                                         per_profile=self.per_profile)
        generated_cases = tuple(
            _public_commitment(challenge, partition, split)
            for partition, split in _SPLITS
            for challenge in getattr(generated, partition)
        )
        if self.cases != generated_cases:
            raise Q6CampaignAuthorityError("Q6 source cohort differs from pinned generator/seed/order")
        for case in self.cases:
            if type(case) is not Q6PublicCaseCommitment:
                raise Q6CampaignAuthorityError("untyped Q6 source case")
            _id(case.case_id, "case ID")
            _digest(case.prompt_sha256, "case prompt")
            _digest(case.arithmetic_semantic_sha256, "case semantic identity")
            if (case.partition not in per_partition
                    or case.split != dict(_SPLITS)[case.partition]
                    or case.case_id in used_ids
                    or case.prompt_sha256 in used_prompt_hashes
                    or case.arithmetic_semantic_sha256 in used_semantics):
                raise Q6CampaignAuthorityError("duplicate, aliased, or invalid Q6 case")
            used_ids.add(case.case_id)
            used_prompt_hashes.add(case.prompt_sha256)
            used_semantics.add(case.arithmetic_semantic_sha256)
            per_partition[case.partition] += 1
            for arm in _ARMS:
                if case.partition in ("train_a", "train_b"):
                    phase = "after_a" if case.partition == "train_a" else "after_b"
                    visits = self.epochs_stage_a if case.partition == "train_a" else self.epochs_stage_b
                    expected_slots.append(Q6PlannedExposure(case.partition, case.case_id,
                                                            arm, phase, visits))
                else:
                    for phase in _EVAL_PHASES:
                        expected_slots.append(Q6PlannedExposure(case.partition, case.case_id,
                                                                arm, phase, 1))
        if (any(value != 4 * self.per_profile for value in per_partition.values())
                or any(type(slot) is not Q6PlannedExposure
                       for slot in self.planned_exposures)
                or tuple(expected_slots) != self.planned_exposures):
            raise Q6CampaignAuthorityError("Q6 exposure plan incomplete or altered")

    def payload(self) -> bytes:
        self.validate()
        # Copy only basic immutable fields, never Python repr or private gold.
        return _canonical({
            "schema": self.schema,
            "source_generator_schema": self.source_generator_schema,
            "seed": self.seed,
            "per_profile": self.per_profile,
            "epochs_stage_a": self.epochs_stage_a,
            "epochs_stage_b": self.epochs_stage_b,
            "cases": [vars(case) for case in self.cases],
            "planned_exposures": [vars(slot) for slot in self.planned_exposures],
        })

    @property
    def sha256(self) -> str:
        return sha256(_MANIFEST_DOMAIN + self.payload()).hexdigest()


def _public_commitment(challenge: Any, partition: str,
                       expected_split: str) -> Q6PublicCaseCommitment:
    semantic = sha256(
        _SEMANTIC_DOMAIN + _canonical(semantic_case_key(challenge))
    ).hexdigest()
    return Q6PublicCaseCommitment(
        partition, expected_split, challenge.id,
        sha256(challenge.prompt.encode("utf-8")).hexdigest(), semantic,
    )


def build_q6_campaign_manifest(
    cohorts: Q6Cohorts, *, seed: int, per_profile: int,
    epochs_stage_a: int, epochs_stage_b: int,
) -> Q6CampaignManifest:
    """Freeze source-admitted cases and all intended arm/phase visit counts.

    No caller historical exposure set is accepted as proof of freshness; the
    externally deployed authority must compare these semantic commitments to
    its complete independent history BEFORE allowing scoring or training.
    """
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise Q6CampaignAuthorityError("invalid study seed")
    if type(per_profile) is not int or not 2 <= per_profile <= 24:
        raise Q6CampaignAuthorityError("invalid profile quota")
    for label, epochs in (("stage A epochs", epochs_stage_a),
                          ("stage B epochs", epochs_stage_b)):
        if type(epochs) is not int or not 1 <= epochs <= 1000:
            raise Q6CampaignAuthorityError(f"invalid {label}")
    validate_q6_cohorts(cohorts, per_profile=per_profile)
    if cohorts != construct_q6_cohorts(seed=seed, per_profile=per_profile):
        raise Q6CampaignAuthorityError(
            "Q6 source cohort differs from pinned generator/seed/order")
    cases: list[Q6PublicCaseCommitment] = []
    slots: list[Q6PlannedExposure] = []
    for partition, expected_split in _SPLITS:
        for challenge in getattr(cohorts, partition):
            cases.append(_public_commitment(challenge, partition, expected_split))
            for arm in _ARMS:
                if partition in ("train_a", "train_b"):
                    epochs = epochs_stage_a if partition == "train_a" else epochs_stage_b
                    phase = "after_a" if partition == "train_a" else "after_b"
                    slots.append(Q6PlannedExposure(partition, challenge.id, arm, phase, epochs))
                else:
                    for phase in _EVAL_PHASES:
                        slots.append(Q6PlannedExposure(partition, challenge.id, arm, phase, 1))
    return Q6CampaignManifest(
        "labyrinth.q6-governed-campaign.v1", Q6_SOURCE_GENERATOR_SCHEMA,
        seed, per_profile,
        epochs_stage_a, epochs_stage_b, tuple(cases), tuple(slots),
    )


@dataclass(frozen=True)
class Q6AuthorityHead:
    provider_id: str
    run_id: str
    epoch: int
    fence: int
    commit_index: int
    head_sha256: str

    def validate(self) -> None:
        _id(self.provider_id, "provider")
        _id(self.run_id, "run")
        _counter(self.epoch, "epoch", minimum=1)
        _counter(self.fence, "fence")
        _counter(self.commit_index, "commit index")
        _digest(self.head_sha256, "monotonic head")


@dataclass(frozen=True)
class Q6CampaignReservationIntent:
    provider_id: str
    run_id: str
    campaign_id: str
    manifest_sha256: str
    candidate_checkpoint_sha256: str
    operation_nonce: str
    expected_epoch: int
    expected_fence: int
    expected_commit_index: int
    expected_head_sha256: str

    def payload(self) -> bytes:
        for field in ("provider_id", "run_id", "campaign_id", "operation_nonce"):
            _id(getattr(self, field), field)
        for field in ("manifest_sha256", "candidate_checkpoint_sha256", "expected_head_sha256"):
            _digest(getattr(self, field), field)
        for field in ("expected_epoch", "expected_fence", "expected_commit_index"):
            _counter(getattr(self, field), field, minimum=1 if field == "expected_epoch" else 0)
        return _canonical(vars(self))

    @property
    def sha256(self) -> str:
        return sha256(_REQUEST_DOMAIN + self.payload()).hexdigest()

    def expected_head(self) -> Q6AuthorityHead:
        self.payload()
        return Q6AuthorityHead(self.provider_id, self.run_id, self.expected_epoch,
                               self.expected_fence, self.expected_commit_index,
                               self.expected_head_sha256)


def next_q6_head_digest(intent: Q6CampaignReservationIntent, reservation_id: str) -> str:
    _id(reservation_id, "reservation")
    return sha256(_HEAD_DOMAIN + intent.sha256.encode("ascii") + b"\0"
                  + str(intent.expected_commit_index + 1).encode("ascii")
                  + b"\0" + reservation_id.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Q6CampaignReservationReceipt:
    provider_id: str
    run_id: str
    campaign_id: str
    reservation_id: str
    request_sha256: str
    manifest_sha256: str
    epoch: int
    fence: int
    commit_index: int
    previous_head_sha256: str
    new_head_sha256: str
    status: str
    signature_hex: str
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False
    individual_exposures_reserved: bool = False

    def signed_payload(self) -> bytes:
        for field in ("provider_id", "run_id", "campaign_id", "reservation_id"):
            _id(getattr(self, field), field)
        for field in ("request_sha256", "manifest_sha256", "previous_head_sha256", "new_head_sha256"):
            _digest(getattr(self, field), field)
        for field in ("epoch", "fence", "commit_index"):
            _counter(getattr(self, field), field, minimum=1 if field in ("epoch", "commit_index") else 0)
        if (self.status != "CAMPAIGN_RESERVED_SOURCE_ONLY"
                or self.graduation_authorized is not False
                or self.model_execution_authenticated is not False
                or self.individual_exposures_reserved is not False):
            raise Q6CampaignAuthorityError("Q6 reservation cannot claim model or individual-exposure custody")
        return _RECEIPT_DOMAIN + _canonical({name: value for name, value in vars(self).items()
                                             if name != "signature_hex"})

    def post_head(self) -> Q6AuthorityHead:
        self.signed_payload()
        return Q6AuthorityHead(self.provider_id, self.run_id, self.epoch,
                               self.fence, self.commit_index, self.new_head_sha256)


class Q6CampaignExternalPort(Protocol):
    """Must be implemented by owner outside learner rollback/admin domain."""

    def read_head(self, run_id: str) -> Q6AuthorityHead: ...
    def reserve_campaign(self, intent: Q6CampaignReservationIntent) -> Q6CampaignReservationReceipt: ...
    def read_reservation(self, campaign_id: str, operation_nonce: str) -> Q6CampaignReservationReceipt: ...


class Q6CampaignReservationClient:
    """Validate an external source assertion; not evidence of host ownership."""

    def __init__(self, *, port: Q6CampaignExternalPort,
                 pinned_intent: Q6CampaignReservationIntent,
                 pinned_manifest: Q6CampaignManifest,
                 pinned_ed25519_public_key: bytes):
        if (type(pinned_intent) is not Q6CampaignReservationIntent
                or type(pinned_manifest) is not Q6CampaignManifest):
            raise Q6CampaignAuthorityError("typed host-pinned Q6 intent and manifest required")
        self._intent_payload = pinned_intent.payload()
        self._manifest_payload = pinned_manifest.payload()
        if pinned_intent.manifest_sha256 != pinned_manifest.sha256:
            raise Q6CampaignAuthorityError("host-pinned manifest commitment mismatch")
        if type(pinned_ed25519_public_key) is not bytes or len(pinned_ed25519_public_key) != 32:
            raise Q6CampaignAuthorityError("pinned Ed25519 public key required")
        self._port = port
        self._public_key = pinned_ed25519_public_key

    def reserve(self, intent: Q6CampaignReservationIntent,
                manifest: Q6CampaignManifest) -> Q6CampaignReservationReceipt:
        if (type(intent) is not Q6CampaignReservationIntent
                or type(manifest) is not Q6CampaignManifest):
            raise Q6CampaignAuthorityError("typed Q6 campaign reservation required")
        # Snapshot before any untrusted port callback. Frozen dataclasses can
        # still be mutated with object.__setattr__ by a hostile caller.
        intent = replace(intent)
        intent_payload = intent.payload()
        manifest_payload = manifest.payload()
        if (not hmac.compare_digest(intent_payload, self._intent_payload)
                or not hmac.compare_digest(manifest_payload, self._manifest_payload)
                or intent.manifest_sha256 != manifest.sha256):
            raise Q6CampaignAuthorityError("reservation differs from host-pinned campaign")
        try:
            pre_head = self._port.read_head(intent.run_id)
        except Exception as exc:
            raise Q6CampaignOutcomeUnknown("external preflight unavailable; no model call") from exc
        if type(pre_head) is not Q6AuthorityHead:
            raise Q6CampaignAuthorityError("untyped external authority head")
        pre_head.validate()
        if pre_head != intent.expected_head():
            raise Q6CampaignAuthorityError("stale or substituted external campaign head")
        try:
            receipt = self._port.reserve_campaign(replace(intent))
        except Exception as exc:
            raise Q6CampaignOutcomeUnknown("external reservation may have committed; reconcile") from exc
        try:
            if type(receipt) is not Q6CampaignReservationReceipt:
                raise Q6CampaignAuthorityError("untyped Q6 reservation receipt")
            receipt = replace(receipt)
            payload = receipt.signed_payload()
            if (receipt.provider_id != intent.provider_id
                    or receipt.run_id != intent.run_id
                    or receipt.campaign_id != intent.campaign_id
                    or receipt.manifest_sha256 != manifest.sha256
                    or receipt.request_sha256 != intent.sha256
                    or receipt.epoch != intent.expected_epoch
                    or receipt.fence != intent.expected_fence
                    or receipt.commit_index != intent.expected_commit_index + 1
                    or receipt.previous_head_sha256 != intent.expected_head_sha256
                    or receipt.new_head_sha256 != next_q6_head_digest(intent, receipt.reservation_id)
                    or type(receipt.signature_hex) is not str
                    or not _HEX128.fullmatch(receipt.signature_hex)):
                raise Q6CampaignAuthorityError("external Q6 receipt differs from exact campaign intent")
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            Ed25519PublicKey.from_public_bytes(self._public_key).verify(
                bytes.fromhex(receipt.signature_hex), payload)
        except Exception as exc:
            raise Q6CampaignOutcomeUnknown("unverified post-reservation receipt; reconcile") from exc
        try:
            stored = self._port.read_reservation(intent.campaign_id, intent.operation_nonce)
            post_head = self._port.read_head(intent.run_id)
        except Exception as exc:
            raise Q6CampaignOutcomeUnknown("external Q6 commit readback unavailable") from exc
        if type(stored) is not Q6CampaignReservationReceipt or type(post_head) is not Q6AuthorityHead:
            raise Q6CampaignOutcomeUnknown("external Q6 readback untyped")
        try:
            stored = replace(stored)
            stored.signed_payload()
            post_head.validate()
        except Q6CampaignAuthorityError as exc:
            raise Q6CampaignOutcomeUnknown("external Q6 readback malformed") from exc
        if stored != receipt or post_head != receipt.post_head():
            raise Q6CampaignOutcomeUnknown("external Q6 readback inconsistent; do not retry")
        if manifest.payload() != manifest_payload:
            raise Q6CampaignOutcomeUnknown("campaign manifest mutated during reservation")
        return receipt


@dataclass(frozen=True)
class Q6SourceExecution(Generic[T]):
    status: str
    callback_invoked: bool
    source_reservation_sha256: str | None = None
    result: T | None = None
    failure: str | None = None


def execute_only_after_campaign_reservation(
    *, client: Q6CampaignReservationClient | None,
    intent: Q6CampaignReservationIntent,
    manifest: Q6CampaignManifest,
    operation: Callable[[], T],
) -> Q6SourceExecution[T]:
    """Never invoke ``operation`` unless a signed external CAS is read back.

    A successful callback is STILL ``SOURCE_ONLY_UNQUALIFIED``: per-case
    pre-reveal and post-score completion CAS are not implemented/wired here.
    """
    if not callable(operation):
        raise TypeError("an explicit model/evaluation callback is required")
    if client is None:
        return Q6SourceExecution("UNQUALIFIED", False, failure="external authority unavailable")
    if type(client) is not Q6CampaignReservationClient:
        return Q6SourceExecution("UNQUALIFIED", False, failure="external authority untrusted")
    try:
        receipt = client.reserve(intent, manifest)
    except Q6CampaignAuthorityError as exc:
        return Q6SourceExecution("UNQUALIFIED", False, failure=str(exc))
    result = operation()
    return Q6SourceExecution(
        "SOURCE_ONLY_UNQUALIFIED", True,
        source_reservation_sha256=sha256(receipt.signed_payload()).hexdigest(),
        result=result,
    )
