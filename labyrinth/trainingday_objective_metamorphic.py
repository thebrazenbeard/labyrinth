"""Public formal-contract metamorphic checks; never a Minotaur graduation exam."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


def _id(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _sha(value: object) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


@dataclass(frozen=True)
class FormalSkillCriterion:
    skill_id: str
    metric_id: str
    min_cases: int
    min_correct: int


@dataclass(frozen=True)
class FormalObjectiveContract:
    objective_id: str
    objective_kind: str
    skills: tuple[FormalSkillCriterion, ...]
    excluded_skill_ids: tuple[str, ...]
    evaluator_sha256: str
    conditions_sha256: str
    public_context_sha256: str
    presentation_label: str = ""


@dataclass(frozen=True)
class PublicDevelopmentObservation:
    case_id: str
    skill_id: str
    correct: bool
    format_valid: bool
    split: str = "public_development"


@dataclass(frozen=True)
class MetamorphicVerdict:
    status: str
    reasons: tuple[str, ...]
    formal_fingerprint: str | None
    original_public_mastery: bool | None
    variant_public_mastery: bool | None
    graduation_authorized: bool = False
    promotion_allowed: bool = False


def _reject(status: str, reason: str, fingerprint: str | None = None) -> MetamorphicVerdict:
    return MetamorphicVerdict(status, (reason,), fingerprint, None, None)


def _formal_fingerprint(contract: FormalObjectiveContract) -> str | None:
    if (type(contract) is not FormalObjectiveContract
            or not _id(contract.objective_id)
            or type(contract.objective_kind) is not str
            or contract.objective_kind not in ("skill_mastery", "training_effect")
            or type(contract.skills) is not tuple
            or not 1 <= len(contract.skills) <= 32
            or type(contract.excluded_skill_ids) is not tuple
            or len(contract.excluded_skill_ids) > 32
            or not all(_sha(x) for x in (
                contract.evaluator_sha256,
                contract.conditions_sha256,
                contract.public_context_sha256,
            ))
            or type(contract.presentation_label) is not str
            or len(contract.presentation_label) > 300):
        return None
    criteria: list[dict[str, object]] = []
    skill_ids: set[str] = set()
    for criterion in contract.skills:
        if (type(criterion) is not FormalSkillCriterion
                or not _id(criterion.skill_id) or not _id(criterion.metric_id)
                or criterion.skill_id in skill_ids
                or type(criterion.min_cases) is not int
                or not 1 <= criterion.min_cases <= 500
                or type(criterion.min_correct) is not int
                or not 1 <= criterion.min_correct <= criterion.min_cases):
            return None
        skill_ids.add(criterion.skill_id)
        criteria.append({
            "skill_id": criterion.skill_id,
            "metric_id": criterion.metric_id,
            "min_cases": criterion.min_cases,
            "min_correct": criterion.min_correct,
        })
    excluded = contract.excluded_skill_ids
    if (any(not _id(x) for x in excluded)
            or len(excluded) != len(set(excluded))
            or skill_ids.intersection(excluded)):
        return None
    # Display wording is non-normative. The identity, skill boundaries, exact
    # numeric rubric, evaluator and context hashes are normative.
    canonical = {
        "objective_id": contract.objective_id,
        "objective_kind": contract.objective_kind,
        "skills": sorted(criteria, key=lambda item: item["skill_id"]),
        "excluded_skill_ids": sorted(excluded),
        "evaluator_sha256": contract.evaluator_sha256,
        "conditions_sha256": contract.conditions_sha256,
        "public_context_sha256": contract.public_context_sha256,
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(b"labyrinth.public-formal-metamorphic.v1\0" + raw).hexdigest()


def _validate_public_cases(
    cases: tuple[PublicDevelopmentObservation, ...],
    contract: FormalObjectiveContract,
) -> tuple[str, dict[str, PublicDevelopmentObservation] | None]:
    if type(cases) is not tuple or not 1 <= len(cases) <= 4096:
        return "MALFORMED_PUBLIC_OBSERVATIONS", None
    declared = {skill.skill_id for skill in contract.skills}
    by_id: dict[str, PublicDevelopmentObservation] = {}
    for case in cases:
        if (type(case) is not PublicDevelopmentObservation
                or not _id(case.case_id) or not _id(case.skill_id)
                or case.skill_id not in declared
                or type(case.correct) is not bool
                or type(case.format_valid) is not bool
                or type(case.split) is not str):
            return "MALFORMED_PUBLIC_OBSERVATIONS", None
        if case.split != "public_development":
            return "UNSUPPORTED_OBSERVATION_SPLIT", None
        if case.case_id in by_id:
            return "MALFORMED_PUBLIC_OBSERVATIONS", None
        by_id[case.case_id] = case
    return "OK", by_id


def _public_practice_mastery(
    contract: FormalObjectiveContract,
    cases: dict[str, PublicDevelopmentObservation],
) -> bool:
    for skill in contract.skills:
        matching = [case for case in cases.values() if case.skill_id == skill.skill_id]
        correct = sum(case.correct and case.format_valid for case in matching)
        if len(matching) < skill.min_cases or correct < skill.min_correct:
            return False
    return True


def compare_public_objective_metamorphs(
    frozen: FormalObjectiveContract,
    variant: FormalObjectiveContract,
    original_cases: tuple[PublicDevelopmentObservation, ...],
    variant_cases: tuple[PublicDevelopmentObservation, ...],
) -> MetamorphicVerdict:
    """Compare frozen public *formal* invariants; never infer prose semantics.

    Only permutations of exact formal skill/case identifiers and nonnormative
    presentation wording count as equivalent. Changing a skill, threshold,
    evaluator, assessment conditions or excluded scope requires an entirely
    new objective/examination protocol. The public practice score is NEVER a
    Minotaur exam, neural-improvement receipt or promotion authorization.
    """
    original_fp = _formal_fingerprint(frozen)
    variant_fp = _formal_fingerprint(variant)
    if original_fp is None or variant_fp is None:
        return _reject("MALFORMED_FORMAL_CONTRACT", "malformed formal objective")
    if original_fp != variant_fp:
        return _reject("SCOPE_OR_RUBRIC_CHANGED", "normative objective changed", original_fp)
    original_status, original = _validate_public_cases(original_cases, frozen)
    if original is None:
        return _reject(original_status, "original public observations rejected", original_fp)
    variant_status, observed = _validate_public_cases(variant_cases, variant)
    if observed is None:
        return _reject(variant_status, "variant public observations rejected", original_fp)
    if original != observed:
        return _reject("PUBLIC_CASE_MUTATION", "nonidentical public case evidence", original_fp)
    initial_mastery = _public_practice_mastery(frozen, original)
    variant_mastery = _public_practice_mastery(variant, observed)
    if initial_mastery != variant_mastery:
        # Defense in depth against future differing grading implementations.
        return _reject("PUBLIC_GRADE_INCONSISTENCY", "public grades differ", original_fp)
    return MetamorphicVerdict("PUBLIC_FORMAL_INVARIANT_ONLY",
                              ("public development practice only; no independent exam",),
                              original_fp, initial_mastery, variant_mastery)
