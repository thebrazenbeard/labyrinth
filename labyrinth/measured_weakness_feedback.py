"""Authenticate scored *development* failures before steering a new Maze.

This optional source companion reuses the separately keyed signed exam lineage
contract.  The keys and precommitments MUST be independently held by an
evaluator/runtime; this Python code cannot prove that isolation.  Training
exposure rosters must also come from a host-owned append-only source.  This
module does not install a runtime hook, execute a model, grade private final
examinations, or authorize graduation.

In particular, a model/operator cannot replace measured weakness with a
free-form hint and call it an exam result.  An independently controlled
deployment still needs a mandatory invocation, replay protection, and fenced
checkpoint publication before it can rely on this source-level bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import math
import re
from typing import Any

from .blind_evaluation_custody import verify_assessment
from .hf_development_transition import verify_development_transition
from .regenerative_maze import MazePlan, generate_maze_plan
from .signed_exam_lineage import SignedExamLineageBinding, verify_exam_lineage_binding
from .training_scope import TrainingScopeSpec


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ROSTER_DOMAIN = b"labyrinth.training-exposure-roster.v1\0"
_MAX_EXPOSURES = 65536
_VERIFICATION_FIELDS = frozenset({
    "scope", "context", "transition", "scored_receipt_bytes",
    "scored_signature_hex", "candidate_lineage_bytes",
    "candidate_lineage_signature_hex", "candidate_run_id",
    "candidate_artifact_sha256", "expected_evaluator_roster_sha256",
    "expected_development_roster_sha256", "failure_threshold_percent",
    "binding_signer_id", "expected_nonce_sha256",
    "independent_evaluator_key", "independent_lineage_key",
    "independent_transition_key", "independent_binding_key",
})


@dataclass(frozen=True)
class TrainingExposureRef:
    """TRAIN-only case identity; no prompt, labels, or evaluation answers."""

    case_id: str
    prompt_sha256: str


def training_roster_commitment(exposures: tuple[TrainingExposureRef, ...]) -> str:
    """Domain-separated canonical commitment, externally pinned *before* exam."""
    if (type(exposures) is not tuple or not 1 <= len(exposures) <= _MAX_EXPOSURES
            or any(type(item) is not TrainingExposureRef for item in exposures)):
        raise ValueError("bounded TRAIN exposure roster required")
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    for case in exposures:
        if (type(case.case_id) is not str or not _ID.fullmatch(case.case_id)
                or type(case.prompt_sha256) is not str
                or not _SHA256.fullmatch(case.prompt_sha256)):
            raise ValueError("invalid TRAIN exposure identity")
        if case.case_id in seen_ids or case.prompt_sha256 in seen_prompts:
            raise ValueError("duplicate TRAIN exposure identity or prompt")
        seen_ids.add(case.case_id)
        seen_prompts.add(case.prompt_sha256)
    canonical = json.dumps(
        sorted((case.case_id, case.prompt_sha256) for case in exposures),
        ensure_ascii=True, allow_nan=False, separators=(",", ":"),
    ).encode("utf-8")
    return sha256(_ROSTER_DOMAIN + canonical).hexdigest()


def _wilson_95(correct: int, total: int) -> tuple[float, float]:
    """Bounded descriptive interval, not proof of independent sampling."""
    z = 1.959963984540054
    observed = correct / total
    scale = 1 + z * z / total
    centre = (observed + z * z / (2 * total)) / scale
    radius = z * math.sqrt(observed * (1 - observed) / total
                           + z * z / (4 * total * total)) / scale
    return max(0.0, centre - radius), min(1.0, centre + radius)


@dataclass(frozen=True)
class MeasuredRemediation:
    """Development-only successor descriptor; never a qualification receipt."""

    maze: MazePlan
    weak_skills: tuple[str, ...]
    uncertain_skills: tuple[str, ...]
    coverage_by_skill: tuple[tuple[str, int, int], ...]
    exam_id: str
    event_id: str
    from_cycle: int
    to_cycle: int
    roster_sha256: str
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False
    evaluator_host_independence_verified: bool = False


def derive_measured_remediation(
    *, binding: SignedExamLineageBinding,
    training_exposures: tuple[TrainingExposureRef, ...],
    expected_training_roster_sha256: str,
    **verification_inputs: Any,
) -> MeasuredRemediation:
    """Use the *signed* per-skill score counts to select a regenerated Maze.

    All four keys and precommitments are required through the exact options of
    ``verify_exam_lineage_binding``; no operator weakness override is allowed.
    External callers must independently pin the TRAIN roster and secrets,
    durably consume the signed event, and keep all exam data out of training.
    """
    if set(verification_inputs) != _VERIFICATION_FIELDS:
        raise ValueError("exact signed development lineage inputs required; no overrides")
    if (type(expected_training_roster_sha256) is not str
            or not _SHA256.fullmatch(expected_training_roster_sha256)):
        raise ValueError("host-pinned TRAIN roster digest required")
    actual_training_sha256 = training_roster_commitment(training_exposures)
    if not hmac.compare_digest(actual_training_sha256, expected_training_roster_sha256):
        raise ValueError("TRAIN exposures differ from expected host-pinned roster")

    body = verify_exam_lineage_binding(binding, **verification_inputs)
    scope = verification_inputs["scope"]
    context = verification_inputs["context"]
    transition = verification_inputs["transition"]
    if type(scope) is not TrainingScopeSpec:
        raise ValueError("declaration identity required")
    scored = verify_assessment(
        verification_inputs["scored_receipt_bytes"],
        verification_inputs["scored_signature_hex"],
        evaluator_key=verification_inputs["independent_evaluator_key"],
        run_id=verification_inputs["candidate_run_id"],
        checkpoint_sha256=verification_inputs["candidate_artifact_sha256"],
    )
    verified = verify_development_transition(
        scope, context, transition.receipt_bytes, transition.signature_hex,
        independent_development_key=verification_inputs["independent_transition_key"],
    )
    if body["scored_receipt_sha256"] != sha256(
            verification_inputs["scored_receipt_bytes"]).hexdigest():
        raise ValueError("scored receipt identity mismatch")

    public = transition.development_public_roster
    exam_ids = {case.case_id for case in public}
    exam_prompts = {case.prompt_sha256 for case in public}
    if (len(exam_ids) != len(public) or len(exam_prompts) != len(public)
            or any(case.case_id in exam_ids or case.prompt_sha256 in exam_prompts
                   for case in training_exposures)):
        raise ValueError("TRAIN exposure and development exam partitions overlap")
    groups = {(group["skill"], group["split"]): group for group in scored["groups"]}
    expected_groups = {(skill, split) for skill in scope.skills
                       for split in ("heldout", "transfer")}
    if set(groups) != expected_groups:
        raise ValueError("incomplete or out-of-scope signed development split coverage")
    roster_counts = {skill: sum(case.skill == skill for case in public)
                     for skill in scope.skills}
    if (sum(group["total"] for group in scored["groups"]) != len(public)
            or any(sum(groups[(skill, split)]["total"]
                       for split in ("heldout", "transfer")) != roster_counts[skill]
                   for skill in scope.skills)):
        raise ValueError("scored aggregate totals do not match signed public exam roster")

    threshold = verification_inputs["failure_threshold_percent"] / 100
    weak = tuple(skill for skill in scope.skills if any(
        groups[(skill, split)]["correct"] / groups[(skill, split)]["total"] < threshold
        for split in ("heldout", "transfer")
    ))
    if not weak or weak != verified.weak_skills:
        raise ValueError("signed development weaknesses disagree with scored measurements")
    uncertain_skills = []
    for skill in scope.skills:
        for split in ("heldout", "transfer"):
            group = groups[(skill, split)]
            lower, upper = _wilson_95(group["correct"], group["total"])
            if lower <= threshold <= upper:
                uncertain_skills.append(skill)
                break
    counts = tuple((skill, groups[(skill, "heldout")]["total"],
                    groups[(skill, "transfer")]["total"]) for skill in scope.skills)
    maze = generate_maze_plan(scope, context.cycle_index, weak_skills=weak)
    if (maze.scope_fingerprint != body["scope_fingerprint"]
            or maze.cycle_index != body["to_cycle"]
            or body["from_cycle"] + 1 != body["to_cycle"]):
        raise ValueError("regenerated Maze does not match signed cycle")
    return MeasuredRemediation(
        maze=maze, weak_skills=weak, uncertain_skills=tuple(uncertain_skills),
        coverage_by_skill=counts, exam_id=verified.exam_id,
        event_id=verified.event_id, from_cycle=body["from_cycle"],
        to_cycle=body["to_cycle"], roster_sha256=actual_training_sha256,
    )
