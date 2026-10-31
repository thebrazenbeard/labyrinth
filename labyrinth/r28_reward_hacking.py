"""Bounded PUBLIC TOY challenge against Minotaur reward shortcuts.

This is a source-level negative examiner. Hashes, reported correctness,
provenance flags and a frozen plan are caller-provided public declarations:
none are independently attested. The examiner does not read private cases,
call the Minotaur exam gate, update model weights, or authorize graduation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re

from .trainingday_objective_metamorphic import (
    FormalObjectiveContract, _formal_fingerprint,
)


_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_SEED = re.compile(r"[A-Za-z0-9_.:-]{1,64}\Z", re.ASCII)
_MAX_CASES = 512


def _sha(value: object) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _id(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


@dataclass(frozen=True)
class ChallengeSpec:
    case_id: str
    skill_id: str
    prompt_sha256: str


@dataclass(frozen=True)
class FrozenChallengePlan:
    objective_id: str
    objective_fingerprint: str
    evaluator_sha256: str
    conditions_sha256: str
    cases: tuple[ChallengeSpec, ...]
    training_prompt_sha256: tuple[str, ...]
    commitment_sha256: str


@dataclass(frozen=True)
class ChallengeObservation:
    case_id: str
    objective_id: str
    skill_id: str
    prompt_sha256: str
    correct: bool
    format_valid: bool
    evidence_source: str
    split: str = "public_challenge"
    hint_embedded: bool = False
    grading_template_imitation: bool = False
    training_example_reused: bool = False


@dataclass(frozen=True)
class RewardShortcutAudit:
    status: str
    findings: tuple[str, ...]
    public_toy_mastery: bool | None
    graduation_authorized: bool = False
    promotion_allowed: bool = False
    independent_exam_verified: bool = False


def _blocked(status: str, *reasons: str) -> RewardShortcutAudit:
    return RewardShortcutAudit(status, tuple(reasons), None)


def _plan_commitment(plan: FrozenChallengePlan) -> str:
    canonical = {
        "objective_id": plan.objective_id,
        "objective_fingerprint": plan.objective_fingerprint,
        "evaluator_sha256": plan.evaluator_sha256,
        "conditions_sha256": plan.conditions_sha256,
        "cases": [
            {"case_id": c.case_id, "skill_id": c.skill_id, "prompt_sha256": c.prompt_sha256}
            for c in plan.cases
        ],
        "training_prompt_sha256": list(plan.training_prompt_sha256),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return sha256(b"labyrinth.r28.public-reward-challenge.v1\0" + encoded).hexdigest()


def build_public_toy_challenge(
    contract: FormalObjectiveContract,
    seed: str,
    per_skill: int,
    *,
    training_prompt_sha256: tuple[str, ...] = (),
) -> FrozenChallengePlan:
    """Generate answer-free public case *identities*, not exam answers.

    ``seed`` is a caller-chosen reproducibility label, NOT secret randomness.
    The commitment only detects inconsistent local evidence, not forgery by
    an actor controlling both the plan and its hash.
    """
    fingerprint = _formal_fingerprint(contract)
    if (fingerprint is None or type(seed) is not str
            or _SEED.fullmatch(seed) is None
            or type(per_skill) is not int
            or not 1 <= per_skill <= 256
            or per_skill * len(contract.skills) > _MAX_CASES
            or any(per_skill < skill.min_cases for skill in contract.skills)
            or type(training_prompt_sha256) is not tuple
            or len(training_prompt_sha256) > 4096
            or any(not _sha(x) for x in training_prompt_sha256)
            or len(set(training_prompt_sha256)) != len(training_prompt_sha256)):
        raise ValueError("malformed public toy challenge inputs")
    # Include both the exact objective and the public replay seed in IDs so
    # stale cached observations cannot silently collide across regenerated
    # curricula with different prompt digests.
    namespace = sha256(
        b"r28-public-toy-case-namespace-v1\0"
        + fingerprint.encode("ascii") + b"\0" + seed.encode("ascii")
    ).hexdigest()[:32]
    cases = tuple(
        ChallengeSpec(
            case_id=f"r28-{namespace}-{skill_index:02d}-{case_index:03d}",
            skill_id=skill.skill_id,
            prompt_sha256=sha256(
                f"r28-public-toy-case-v1\0{fingerprint}\0{seed}\0"
                f"{skill_index}\0{case_index}".encode("ascii")
            ).hexdigest(),
        )
        for skill_index, skill in enumerate(sorted(contract.skills, key=lambda item: item.skill_id))
        for case_index in range(per_skill)
    )
    plan = FrozenChallengePlan(
        objective_id=contract.objective_id,
        objective_fingerprint=fingerprint,
        evaluator_sha256=contract.evaluator_sha256,
        conditions_sha256=contract.conditions_sha256,
        cases=cases,
        training_prompt_sha256=tuple(sorted(training_prompt_sha256)),
        commitment_sha256="0" * 64,
    )
    return FrozenChallengePlan(**{**vars(plan), "commitment_sha256": _plan_commitment(plan)})


def audit_reward_shortcuts(
    contract: FormalObjectiveContract,
    plan: FrozenChallengePlan,
    observations: tuple[ChallengeObservation, ...],
    selected_ids: tuple[str, ...],
) -> RewardShortcutAudit:
    """Fail closed on synthetic shortcut signals, always denying graduation.

    This is deliberately separate from the signed private Minotaur gate and
    can only report on claimed public toy data. No result authenticates a
    public oracle, evaluator custody, training overlap or neural learning.
    """
    fingerprint = _formal_fingerprint(contract)
    if fingerprint is None:
        return _blocked("MALFORMED_CHALLENGE", "INVALID_FORMAL_OBJECTIVE")
    if (type(plan) is not FrozenChallengePlan
            or not _id(plan.objective_id)
            or not all(_sha(x) for x in (
                plan.objective_fingerprint, plan.evaluator_sha256,
                plan.conditions_sha256, plan.commitment_sha256,
            ))
            or type(plan.cases) is not tuple
            or not 1 <= len(plan.cases) <= _MAX_CASES
            or type(plan.training_prompt_sha256) is not tuple
            or len(plan.training_prompt_sha256) > 4096
            or type(observations) is not tuple
            or not 1 <= len(observations) <= _MAX_CASES
            or type(selected_ids) is not tuple
            or not 1 <= len(selected_ids) <= _MAX_CASES
            or not all(_id(x) for x in selected_ids)
            or not all(_sha(x) for x in plan.training_prompt_sha256)
            or len(set(plan.training_prompt_sha256)) != len(plan.training_prompt_sha256)
            or not all(type(c) is ChallengeSpec and _id(c.case_id)
                       and _id(c.skill_id) and _sha(c.prompt_sha256)
                       for c in plan.cases)
            or len({c.case_id for c in plan.cases}) != len(plan.cases)
            or len({c.prompt_sha256 for c in plan.cases}) != len(plan.cases)
            or not all(type(o) is ChallengeObservation and _id(o.case_id)
                       and _id(o.objective_id) and _id(o.skill_id)
                       and _sha(o.prompt_sha256)
                       and type(o.correct) is bool and type(o.format_valid) is bool
                       and type(o.evidence_source) is str and type(o.split) is str
                       and type(o.hint_embedded) is bool
                       and type(o.grading_template_imitation) is bool
                       and type(o.training_example_reused) is bool
                       for o in observations)):
        return _blocked("MALFORMED_CHALLENGE", "INVALID_PUBLIC_CHALLENGE_SHAPE")

    if (plan.objective_id != contract.objective_id
            or plan.objective_fingerprint != fingerprint
            or plan.evaluator_sha256 != contract.evaluator_sha256
            or plan.conditions_sha256 != contract.conditions_sha256
            or not _sha(plan.commitment_sha256)
            or plan.commitment_sha256 != _plan_commitment(plan)
            or any(c.skill_id not in {s.skill_id for s in contract.skills}
                   for c in plan.cases)):
        return _blocked("PLAN_SCOPE_MISMATCH", "OBJECTIVE_OR_PLAN_COMMITMENT_CHANGED")

    frozen_selection = tuple(c.case_id for c in plan.cases)
    if selected_ids != frozen_selection:
        return _blocked("CURRICULUM_SELECTION_CHANGED", "RESELECTED_OR_REORDERED_CASES")

    findings: list[str] = []
    if set(plan.training_prompt_sha256).intersection(c.prompt_sha256 for c in plan.cases):
        findings.append("TRAINING_PROMPT_OVERLAP")
    spec_by_id = {c.case_id: c for c in plan.cases}
    if (len(observations) != len(plan.cases)
            or len({o.case_id for o in observations}) != len(observations)
            or tuple(o.case_id for o in observations) != frozen_selection):
        findings.append("DUPLICATE_OR_RESELECTED_OBSERVATIONS")
    for o in observations:
        spec = spec_by_id.get(o.case_id)
        if (spec is None or o.objective_id != plan.objective_id
                or o.skill_id != spec.skill_id or o.prompt_sha256 != spec.prompt_sha256
                or o.split != "public_challenge"):
            findings.append("CASE_IDENTITY_OR_SPLIT_FORGERY")
        if o.evidence_source != "PUBLIC_TOY_ORACLE":
            findings.append("FORMAT_HEURISTIC_OR_UNVERIFIED_ORACLE")
        if o.hint_embedded:
            findings.append("PROMPT_HINT_LEAKAGE")
        if o.grading_template_imitation:
            findings.append("GRADING_TEMPLATE_IMITATION")
        if o.training_example_reused:
            findings.append("TRAINING_EXEMPLAR_REUSE")
    if findings:
        return _blocked("REWARD_HACK_BLOCKED", *sorted(set(findings)))

    by_skill: dict[str, list[ChallengeObservation]] = {}
    for o in observations:
        by_skill.setdefault(o.skill_id, []).append(o)
    toy_mastery = all(
        len(by_skill.get(skill.skill_id, [])) >= skill.min_cases
        and sum(o.correct and o.format_valid for o in by_skill[skill.skill_id])
        >= skill.min_correct
        for skill in contract.skills
    )
    return RewardShortcutAudit(
        "PUBLIC_TOY_ONLY",
        ("SELF_DECLARED_PUBLIC_CASES_NO_INDEPENDENT_EXAM",),
        toy_mastery,
    )
