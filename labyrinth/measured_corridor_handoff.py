"""Development-only state handoff from *signed measured* failure to new Maze.

The existing measured-feedback verifier authenticates the development score
and plans a successor maze. This companion binds that plan to the Journey's
actual waiting learner checkpoint, training exposures, and sealed exam
history. It never creates a model update or authorizes private graduation.

The Journey learner checkpoint here identifies the *LoRA parameter values*,
as required by the existing HF Journey successor contract. The separately
signed candidate artifact digest identifies the adapter *file bytes*, and
the two digests must not be equated.

An independent host must still provision signing keys, consume event IDs,
measure the LoRA/file-byte correspondence, protect checkpoints, and make
this entry point mandatory. A valid signature passed by an in-process caller
is source verification, not hosted authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .journey import (
    AWAITING_EXAM, TRAINING, JourneyAttempt, JourneyState, _validate_state,
)
from .measured_weakness_feedback import (
    MeasuredRemediation, TrainingExposureRef, derive_measured_remediation,
)
from .hf_resume_bridge import ResumeContext
from .signed_exam_lineage import SignedExamLineageBinding


@dataclass(frozen=True)
class MeasuredCorridorHandoff:
    """One bounded development-FAIL transition, never a graduation receipt."""

    journey: JourneyState
    remediation: MeasuredRemediation
    graduation_authorized: bool = False
    independent_evaluator_host_verified: bool = False
    checkpoint_publication_fenced: bool = False


def advance_from_measured_development_failure(
    state: JourneyState,
    *,
    binding: SignedExamLineageBinding,
    training_exposures: tuple[TrainingExposureRef, ...],
    expected_training_roster_sha256: str,
    **verification_inputs: Any,
) -> MeasuredCorridorHandoff:
    """Keep learner state when an independently signed *development* exam fails.

    Uses the exact four-key verification inputs of derive_measured_remediation.
    The next maze's weakness selection is derived from signed score counts.
    No operator-supplied weak-skill override is accepted.

    Exam case/prompt digests are sealed in the resulting JourneyState; no
    grader answers or raw exam prompt text enter a new training room.
    """
    _validate_state(state)
    if state.phase != AWAITING_EXAM:
        raise ValueError("measured remediation requires a completed TRAIN cycle awaiting exam")
    if not isinstance(binding, SignedExamLineageBinding):
        raise ValueError("signed measured development exam binding required")
    scope = verification_inputs.get("scope")
    context = verification_inputs.get("context")
    if (scope != state.scope or type(context) is not ResumeContext
            or context.scope_fingerprint != state.scope.fingerprint
            or context.cycle_index != state.cycle_index + 1
            or state.checkpoint_sha256 != context.predecessor_lora_value_sha256):
        raise ValueError("signed LoRA learner checkpoint differs from current journey")
    # _validate_state checks attempt count and indices, not every predecessor
    # edge. Require the exact canonical chain before making a new attempt.
    if state.attempts:
        if (state.attempts[0].checkpoint_before_sha256
                != state.baseline_checkpoint_sha256
                or any(a.checkpoint_after_sha256 != b.checkpoint_before_sha256
                       for a, b in zip(state.attempts, state.attempts[1:]))):
            raise ValueError("historical learner checkpoint chain is discontinuous")
        expected_before = state.attempts[-1].checkpoint_after_sha256
    else:
        expected_before = state.baseline_checkpoint_sha256
    if (state.pending_checkpoint_before_sha256 != expected_before
            or state.pending_checkpoint_before_sha256 == state.checkpoint_sha256):
        raise ValueError("pending predecessor does not match authoritative learner lineage")
    if (type(training_exposures) is not tuple or not training_exposures
            or any(type(case) is not TrainingExposureRef for case in training_exposures)):
        raise ValueError("TRAIN roster must contain typed exposure identities")
    if (tuple(case.case_id for case in training_exposures)
            != state.pending_training_case_ids
            or tuple(case.prompt_sha256 for case in training_exposures)
            != state.pending_training_prompt_sha256):
        raise ValueError("host-pinned TRAIN roster differs from current journey completion")

    remediation = derive_measured_remediation(
        binding=binding, training_exposures=training_exposures,
        expected_training_roster_sha256=expected_training_roster_sha256,
        **verification_inputs,
    )
    if (remediation.from_cycle != state.cycle_index
            or remediation.to_cycle != state.cycle_index + 1
            or remediation.maze.scope_fingerprint != state.scope.fingerprint
            or remediation.maze.objective_id != state.scope.objective_id):
        raise ValueError("signed successor Maze does not follow current journey")
    if remediation.exam_id in state.sealed_exam_ids:
        raise ValueError("development examination identity replayed")

    transition = verification_inputs["transition"]
    exam_ids = tuple(case.case_id for case in transition.development_public_roster)
    exam_prompts = tuple(case.prompt_sha256 for case in transition.development_public_roster)
    if (len(set(exam_ids)) != len(exam_ids)
            or len(set(exam_prompts)) != len(exam_prompts)
            or set(exam_ids) & (set(state.used_training_case_ids)
                                | set(state.sealed_exam_case_ids))
            or set(exam_prompts) & (set(state.used_training_prompt_sha256)
                                    | set(state.sealed_exam_prompt_sha256))):
        raise ValueError("development exam overlaps or replays a sealed journey partition")

    attempt = JourneyAttempt(
        cycle_index=state.cycle_index, exam_id=remediation.exam_id,
        checkpoint_before_sha256=state.pending_checkpoint_before_sha256,
        checkpoint_after_sha256=state.checkpoint_sha256,
        training_case_ids=state.pending_training_case_ids,
        training_prompt_sha256=state.pending_training_prompt_sha256,
        exam_case_ids=exam_ids, exam_prompt_sha256=exam_prompts,
        exam_status="FAIL", weak_skills=remediation.weak_skills,
    )
    successor = replace(
        state, phase=TRAINING, maze=remediation.maze,
        attempts=state.attempts + (attempt,),
        sealed_exam_ids=state.sealed_exam_ids + (remediation.exam_id,),
        exam_event_statuses=state.exam_event_statuses + ("FAIL",),
        sealed_exam_case_ids=state.sealed_exam_case_ids + exam_ids,
        sealed_exam_prompt_sha256=state.sealed_exam_prompt_sha256 + exam_prompts,
        pending_training_case_ids=(), pending_training_prompt_sha256=(),
        pending_checkpoint_before_sha256="", inconclusive_exam_count=0,
    )
    _validate_state(successor)
    if (successor.checkpoint_sha256 != state.checkpoint_sha256
            or successor.baseline_checkpoint_sha256 != state.baseline_checkpoint_sha256):
        raise ValueError("learner checkpoint was reset during Maze regeneration")
    return MeasuredCorridorHandoff(journey=successor, remediation=remediation)
