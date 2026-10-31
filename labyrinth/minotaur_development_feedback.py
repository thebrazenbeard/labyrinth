"""Development-only Minotaur diagnostics from a signed, scored exam lineage.

The training taskmaster cannot certify its own observations. This module
consumes the existing four-key signed development failure + independently
pinned TRAIN roster gate, then projects per-skill errors and a *TRAIN-only*
tactic recommendation. No prompt, gold answer, or final-exam material is
returned. The host's possession of keys and actual model inference remain
unattested by this Python source; no graduate/merge authority is created.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .blind_evaluation_custody import verify_assessment
from .measured_weakness_feedback import (
    MeasuredRemediation, TrainingExposureRef, derive_measured_remediation,
)
from .minotaur_model import TACTICS
from .signed_exam_lineage import SignedExamLineageBinding


@dataclass(frozen=True)
class SkillDevelopmentObservation:
    """Exact signed aggregate counts; no case prompts, answers, or labels."""

    skill: str
    heldout_correct: int
    heldout_total: int
    heldout_errors: int
    transfer_correct: int
    transfer_total: int
    transfer_errors: int
    uncertain: bool
    weak: bool
    suggested_train_tactic: str | None


@dataclass(frozen=True)
class MinotaurDevelopmentFeedback:
    """An internal development recommendation, never an exit credential."""

    exam_id: str
    event_id: str
    from_cycle: int
    to_cycle: int
    # This is the signed adapter *file-byte* digest, not Journey's LoRA
    # parameter-value checkpoint digest. Keep the types distinct at handoff.
    candidate_artifact_sha256: str
    observations: tuple[SkillDevelopmentObservation, ...]
    remediation: MeasuredRemediation
    source_signature_verified: bool = False
    evaluator_host_independence_verified: bool = False
    matched_control_benefit_verified: bool = False
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False


def _tactic(heldout_correct: int, heldout_total: int,
            transfer_correct: int, transfer_total: int) -> str:
    """Bounded diagnostic suggestion, not a trained taskmaster policy."""
    # Compare rates by cross multiplication, not a rounded float or a
    # pooled average that could conceal a badly failing transfer split.
    heldout_vs_transfer = heldout_correct * transfer_total - (
        transfer_correct * heldout_total)
    if heldout_vs_transfer > 0:
        return "transfer"  # transfer is weaker
    if heldout_vs_transfer < 0:
        return "practice"  # heldout is weaker
    return "contrast"  # both splits equally challenging


def derive_minotaur_development_feedback(
    *,
    binding: SignedExamLineageBinding,
    training_exposures: tuple[TrainingExposureRef, ...],
    expected_training_roster_sha256: str,
    **verification_inputs: Any,
) -> MinotaurDevelopmentFeedback:
    """Project actual scored development failures into internal diagnostics.

    All signing inputs and independent roster precommit are required by
    derive_measured_remediation; unknown options (including an operator-
    declared weak skill or proposed PASS) fail closed. This wrapper does not
    trust a standalone, unsigned MinotaurExamReceipt.
    """
    remediation = derive_measured_remediation(
        binding=binding,
        training_exposures=training_exposures,
        expected_training_roster_sha256=expected_training_roster_sha256,
        **verification_inputs,
    )
    receipt = verify_assessment(
        verification_inputs["scored_receipt_bytes"],
        verification_inputs["scored_signature_hex"],
        evaluator_key=verification_inputs["independent_evaluator_key"],
        run_id=verification_inputs["candidate_run_id"],
        checkpoint_sha256=verification_inputs["candidate_artifact_sha256"],
    )
    scope = verification_inputs["scope"]
    groups = {
        (row["skill"], row["split"]): row
        for row in receipt["groups"]
    }
    observations = []
    for skill in scope.skills:
        heldout = groups[(skill, "heldout")]
        transfer = groups[(skill, "transfer")]
        weak = skill in remediation.weak_skills
        tactic = _tactic(
            heldout["correct"], heldout["total"],
            transfer["correct"], transfer["total"],
        ) if weak else None
        if tactic is not None and tactic not in TACTICS:
            raise ValueError("unapproved Minotaur training tactic")
        observations.append(SkillDevelopmentObservation(
            skill=skill,
            heldout_correct=heldout["correct"],
            heldout_total=heldout["total"],
            heldout_errors=heldout["total"] - heldout["correct"],
            transfer_correct=transfer["correct"],
            transfer_total=transfer["total"],
            transfer_errors=transfer["total"] - transfer["correct"],
            uncertain=skill in remediation.uncertain_skills,
            weak=weak,
            suggested_train_tactic=tactic,
        ))
    return MinotaurDevelopmentFeedback(
        exam_id=remediation.exam_id,
        event_id=remediation.event_id,
        from_cycle=remediation.from_cycle,
        to_cycle=remediation.to_cycle,
        candidate_artifact_sha256=verification_inputs["candidate_artifact_sha256"],
        observations=tuple(observations),
        remediation=remediation,
        source_signature_verified=True,
    )
