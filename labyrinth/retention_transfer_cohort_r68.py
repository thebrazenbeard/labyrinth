"""Source-only fixed retention/structural-transfer cohort manifest (R68 Q6).

The training/evaluation owner must place this manifest digest under genuinely
independent, timestamped custody *before* outcomes are visible. This module
cannot witness that timing, semantic independence, sealed banks, physical
model execution, or host authority. Its only claim is deterministic SOURCE
integrity of an exact roster and declared fixed-size statistical plan.

Each retention BEFORE/AFTER pair is ONE independent semantic-situation unit;
its two prompts never count as two samples. Each distinct structural-transfer
probe is ONE unit. Family and situation identifiers are declarations, not
proof of independent sampling; duplicate identifiers/prompts fail closed.
"""

from __future__ import annotations

from collections import namedtuple
from hashlib import sha256
import hmac
import json
import re
from typing import Any

from labyrinth.retention_qualification import (
    RetentionPair,
    RetentionPolicy,
    _campaign,
    _policy_sha256,
    normalized_prompt_sha256,
    retention_roster_sha256,
)
from labyrinth.retention_sample_planning_r57 import RetentionSamplePlan


SCHEMA = "labyrinth.retention-transfer-fixed-source-cohort.v1"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_TRANSFER_UNITS = 4096
_MAX_MANIFEST_BYTES = 2_097_152


class CohortManifestError(ValueError):
    """Unpinned, selectively sampled, contaminated, or invalid source cohort."""


_TransferFields = namedtuple(
    "_TransferFields",
    ("source_skill_id", "target_skill_id", "case_id", "situation_id",
     "structure_family_id", "prompt_text", "prompt_sha256"),
)


class StructuralTransferUnit(_TransferFields):
    """A source declaration of one novel-target structural-transfer probe."""

    __slots__ = ()


_ReviewFields = namedtuple(
    "_ReviewFields",
    ("manifest_sha256", "retention_roster_sha256", "retention_counts",
     "transfer_counts"),
)


class FixedCohortSourceReview(_ReviewFields):
    """Immutable validation summary. No authority-bearing constructor fields."""

    __slots__ = ()

    @property
    def status(self) -> str:
        return "SOURCE_ONLY_UNATTESTED_FIXED_COHORT"

    @property
    def precommitted_before_outcomes(self) -> bool:
        return False

    @property
    def independent_case_custody_verified(self) -> bool:
        return False

    @property
    def model_execution_authenticated(self) -> bool:
        return False

    @property
    def graduation_authorized(self) -> bool:
        return False


def _identifier(value: Any, name: str) -> None:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise CohortManifestError(f"invalid {name} identifier")


def _digest(value: Any, name: str) -> None:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise CohortManifestError(f"invalid {name} SHA-256")


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _derive_manifest(
    policy: RetentionPolicy,
    retention_roster: tuple[RetentionPair, ...],
    sample_plan: RetentionSamplePlan,
    transfer_targets: tuple[str, ...],
    transfer_target_counts: tuple[int, ...],
    transfer_units: tuple[StructuralTransferUnit, ...],
) -> tuple[bytes, tuple[int, ...], tuple[int, ...], str]:
    """Revalidate ALL underlying declarations at every source boundary."""
    try:
        _campaign(policy, retention_roster)
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise CohortManifestError("invalid or unpinned retention campaign") from exc
    if type(sample_plan) is not RetentionSamplePlan:
        raise CohortManifestError("validated R57 fixed sample plan required")
    try:
        planned_retention_counts = sample_plan.required_units_per_skill
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise CohortManifestError("invalid fixed sample plan") from exc
    if (sample_plan.prior_skill_ids != policy.prior_skill_ids
            or sample_plan.significance_alpha != policy.significance_alpha
            or sample_plan.max_deterioration_rate != policy.max_deterioration_rate
            or sample_plan.minimum_units_per_skill != policy.min_units_per_skill):
        raise CohortManifestError("sample-plan policy does not match retention policy")
    counts = dict.fromkeys(policy.prior_skill_ids, 0)
    for pair in retention_roster:
        counts[pair.skill_id] += 1
    retention_counts = tuple(counts.values())
    # Exact counts prohibit post-hoc over-sampling, under-sampling, optional
    # stopping, and silently repurposing paired forms as extra units.
    if retention_counts != planned_retention_counts:
        raise CohortManifestError("retention count differs from the exact fixed-size plan")

    if (type(transfer_targets) is not tuple or not 1 <= len(transfer_targets) <= 12
            or any(type(target) is not str or _ID_RE.fullmatch(target) is None
                   for target in transfer_targets)
            or len(set(transfer_targets)) != len(transfer_targets)
            or set(transfer_targets).intersection(policy.prior_skill_ids)):
        raise CohortManifestError("one to 12 novel, distinct transfer targets required")
    if (type(transfer_target_counts) is not tuple
            or len(transfer_target_counts) != len(transfer_targets)
            or any(type(n) is not int or not 1 <= n <= 512
                   for n in transfer_target_counts)
            or sum(transfer_target_counts) > _MAX_TRANSFER_UNITS):
        raise CohortManifestError("bounded positive fixed transfer counts required")
    if (type(transfer_units) is not tuple
            or len(transfer_units) != sum(transfer_target_counts)
            or not 1 <= len(transfer_units) <= _MAX_TRANSFER_UNITS
            or any(type(case) is not StructuralTransferUnit for case in transfer_units)):
        raise CohortManifestError("transfer probes differ from fixed sample-size declaration")

    seen_cases = {case_id for pair in retention_roster
                  for case_id in (pair.before_case_id, pair.after_case_id)}
    seen_situations = {pair.situation_id for pair in retention_roster}
    seen_families = set(policy.excluded_training_structure_family_ids)
    seen_families.update(pair.structure_family_id for pair in retention_roster)
    seen_prompts = set(policy.excluded_training_prompt_sha256s)
    for pair in retention_roster:
        seen_prompts.update((pair.before_prompt_sha256, pair.after_prompt_sha256))
    transfer_counts: dict[str, int] = dict.fromkeys(transfer_targets, 0)
    for case in transfer_units:
        for name in ("source_skill_id", "target_skill_id", "case_id",
                     "situation_id", "structure_family_id"):
            _identifier(getattr(case, name), name)
        _digest(case.prompt_sha256, "transfer prompt")
        if case.source_skill_id not in policy.prior_skill_ids:
            raise CohortManifestError("structural-transfer source is not a prior skill")
        if case.target_skill_id not in transfer_counts:
            raise CohortManifestError("undeclared novel structural-transfer target")
        try:
            normalized = normalized_prompt_sha256(case.prompt_text)
        except (ValueError, TypeError, AttributeError) as exc:
            raise CohortManifestError("invalid structural-transfer prompt") from exc
        if normalized != case.prompt_sha256:
            raise CohortManifestError("structural-transfer prompt commitment mismatch")
        if (case.case_id in seen_cases
                or case.situation_id in seen_situations
                or case.structure_family_id in seen_families
                or case.prompt_sha256 in seen_prompts):
            raise CohortManifestError(
                "duplicate or training-contaminated case/situation/family/prompt"
            )
        seen_cases.add(case.case_id)
        seen_situations.add(case.situation_id)
        seen_families.add(case.structure_family_id)
        seen_prompts.add(case.prompt_sha256)
        transfer_counts[case.target_skill_id] += 1
    actual_transfer_counts = tuple(transfer_counts.values())
    if actual_transfer_counts != transfer_target_counts:
        raise CohortManifestError("transfer target counts differ from fixed declaration")

    manifest = {
        "schema": SCHEMA,
        "objective_id": policy.objective_id,
        "retention_policy_sha256": _policy_sha256(policy),
        "retention_roster_sha256": retention_roster_sha256(retention_roster),
        "sample_plan": {
            "prior_skill_ids": list(sample_plan.prior_skill_ids),
            "anticipated_max_deteriorations": list(sample_plan.anticipated_max_deteriorations),
            "minimum_units_per_skill": sample_plan.minimum_units_per_skill,
            "maximum_units_per_skill": sample_plan.maximum_units_per_skill,
            "significance_alpha": sample_plan.significance_alpha,
            "max_deterioration_rate": sample_plan.max_deterioration_rate,
            "exact_retention_counts": list(retention_counts),
        },
        "retention_units": [{
            "skill_id": pair.skill_id,
            "situation_id": pair.situation_id,
            "structure_family_id": pair.structure_family_id,
            "before_case_id": pair.before_case_id,
            "after_case_id": pair.after_case_id,
            "before_prompt_sha256": pair.before_prompt_sha256,
            "after_prompt_sha256": pair.after_prompt_sha256,
        } for pair in retention_roster],
        "transfer_targets": list(transfer_targets),
        "fixed_transfer_counts": list(transfer_target_counts),
        "transfer_units": [{
            key: value for key, value in case._asdict().items()
            if key != "prompt_text"
        } for case in transfer_units],
        "source_only": True,
        "authority_asserted": False,
    }
    payload = _canonical(manifest)
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise CohortManifestError("fixed cohort manifest exceeds bounded source limit")
    # Carry only the digest from the already constructed immutable canonical
    # record. Re-reading the caller's mutable RetentionPair objects after the
    # equality boundary introduces a source TOCTOU inconsistency.
    return (payload, retention_counts, actual_transfer_counts,
            manifest["retention_roster_sha256"])


def build_fixed_cohort_manifest(
    policy: RetentionPolicy,
    retention_roster: tuple[RetentionPair, ...],
    sample_plan: RetentionSamplePlan,
    transfer_targets: tuple[str, ...],
    transfer_target_counts: tuple[int, ...],
    transfer_units: tuple[StructuralTransferUnit, ...],
) -> bytes:
    """Generate immutable canonical bytes to custody-pin BEFORE any outcomes.

    This builder is NOT a trusted timestamp or a precommit acknowledgment.
    An external independent custodian must hold the SHA-256 ahead of outcomes.
    """
    return _derive_manifest(
        policy, retention_roster, sample_plan, transfer_targets,
        transfer_target_counts, transfer_units,
    )[0]


def verify_fixed_cohort_manifest(
    expected_manifest_sha256: str,
    manifest_bytes: bytes,
    policy: RetentionPolicy,
    retention_roster: tuple[RetentionPair, ...],
    sample_plan: RetentionSamplePlan,
    transfer_targets: tuple[str, ...],
    transfer_target_counts: tuple[int, ...],
    transfer_units: tuple[StructuralTransferUnit, ...],
) -> FixedCohortSourceReview:
    """Match exact canonical source bytes to a separately supplied prior pin.

    Caller provenance/time are not authenticated: supplying a freshly computed
    digest from the same process does not establish genuine preregistration.
    """
    _digest(expected_manifest_sha256, "independently held expected manifest")
    if (type(manifest_bytes) is not bytes
            or not 1 <= len(manifest_bytes) <= _MAX_MANIFEST_BYTES):
        raise CohortManifestError("bounded exact manifest bytes required")
    rebuilt, retention_counts, transfer_counts, validated_roster_digest = _derive_manifest(
        policy, retention_roster, sample_plan, transfer_targets,
        transfer_target_counts, transfer_units,
    )
    actual_digest = sha256(manifest_bytes).hexdigest()
    if (not hmac.compare_digest(expected_manifest_sha256, actual_digest)
            or not hmac.compare_digest(rebuilt, manifest_bytes)):
        raise CohortManifestError("manifest differs from the externally expected fixed cohort")
    return FixedCohortSourceReview(
        manifest_sha256=actual_digest,
        retention_roster_sha256=validated_roster_digest,
        retention_counts=retention_counts,
        transfer_counts=transfer_counts,
    )


def verify_fixed_cohort_outcome_selection(
    expected_manifest_sha256: str,
    manifest_bytes: bytes,
    policy: RetentionPolicy,
    retention_roster: tuple[RetentionPair, ...],
    sample_plan: RetentionSamplePlan,
    transfer_targets: tuple[str, ...],
    transfer_target_counts: tuple[int, ...],
    transfer_units: tuple[StructuralTransferUnit, ...],
    *,
    retention_situation_ids: tuple[str, ...],
    transfer_case_ids: tuple[str, ...],
) -> FixedCohortSourceReview:
    """Reject outcome sets with dropped, duplicated, substituted, or late IDs.

    It verifies *identities and coverage*, never prediction correctness,
    inference runs, blinded allocation, semantic independence, or custody.
    """
    review = verify_fixed_cohort_manifest(
        expected_manifest_sha256, manifest_bytes, policy,
        retention_roster, sample_plan, transfer_targets,
        transfer_target_counts, transfer_units,
    )
    # This canonical source record was byte-for-byte verified just above.
    # Caller-owned RetentionPair dataclasses remain mutable through
    # object.__setattr__; never construct the expected outcome identities by
    # rereading them after the manifest equality boundary.
    pinned_units = json.loads(manifest_bytes)
    for supplied, expected, name in (
        (retention_situation_ids,
         {unit["situation_id"] for unit in pinned_units["retention_units"]}, "retention"),
        (transfer_case_ids,
         {unit["case_id"] for unit in pinned_units["transfer_units"]}, "structural transfer"),
    ):
        if (type(supplied) is not tuple
                or len(supplied) != len(expected)
                or any(type(item) is not str for item in supplied)
                or len(set(supplied)) != len(supplied)
                or set(supplied) != expected):
            raise CohortManifestError(
                f"{name} outcome selection does not cover exact predeclared identities"
            )
    return review
