"""Opt-in development exam intake for a regenerated adaptive Journey.

This adapter closes one source-level trust-label gap: caller-controlled
comparison methods must not make private-final or train evidence appear to be
the held-out/transfer development examination that steers regeneration.

It is not an authenticated evaluator, host authority, graduation gate, or a
replacement for the separately signed development lineage. Do not import
private prompts, labels, or answer keys into a training corridor.
"""

from __future__ import annotations

from dataclasses import dataclass

from .adaptive_corridors import AdaptiveWalkState, begin_adaptive_walk_from_journey
from .journey import (
    TRAINING, JourneyAttempt, JourneyState, _validate_state as validate_journey,
    face_minotaur,
)
from .minotaur_exam import ExamCaseResult, MinotaurExamReceipt
from .regenerative_maze import MazePlan, MazeRoom
from .training_scope import TrainingScopeSpec


@dataclass(frozen=True)
class ExamIntakeTransition:
    """Internal development transition; deliberately grants no effects."""

    journey: JourneyState
    adaptive_walk: AdaptiveWalkState | None
    graduation_authorized: bool = False
    evaluator_host_independence_verified: bool = False
    neural_execution_verified: bool = False


def _validate_exact_state_claims(state: JourneyState) -> None:
    """Reject caller-supplied comparison/hash behavior at the exam boundary.

    The legacy Journey validator uses membership, equality, and set overlap
    over state history. ``str`` subclasses can falsify these predicates even
    after a frozen dataclass has been constructed. This exact-type boundary
    covers all state-history and pending-evidence tuples before any evaluation
    or regeneration decision is made; it does not authenticate that history.
    """
    if type(state.phase) is not str:
        raise ValueError("exact journey phase string required")
    if type(state.scope) is not TrainingScopeSpec or type(state.maze) is not MazePlan:
        raise ValueError("exact journey scope and maze records required")
    if type(state.maze.rooms) is not tuple:
        raise ValueError("exact maze rooms tuple required")
    for room in state.maze.rooms:
        if type(room) is not MazeRoom or type(room.purpose) is not str:
            raise ValueError("exact maze room and purpose required")
    for name in (
        "used_training_case_ids", "used_training_prompt_sha256",
        "sealed_exam_ids", "exam_event_statuses", "sealed_exam_case_ids",
        "sealed_exam_prompt_sha256", "pending_training_case_ids",
        "pending_training_prompt_sha256",
    ):
        values = getattr(state, name)
        if type(values) is not tuple or any(type(value) is not str for value in values):
            raise ValueError(f"exact {name} primitive string tuple required")
    if type(state.pending_checkpoint_before_sha256) is not str:
        raise ValueError("exact pending checkpoint string required")
    if type(state.attempts) is not tuple:
        raise ValueError("exact attempt history tuple required")
    for attempt in state.attempts:
        if type(attempt) is not JourneyAttempt or type(attempt.cycle_index) is not int:
            raise ValueError("exact journey attempt required")
        for name in (
            "exam_id", "checkpoint_before_sha256", "checkpoint_after_sha256",
            "exam_status",
        ):
            if type(getattr(attempt, name)) is not str:
                raise ValueError(f"exact attempt {name} string required")
        for name in (
            "training_case_ids", "training_prompt_sha256", "exam_case_ids",
            "exam_prompt_sha256", "weak_skills",
        ):
            values = getattr(attempt, name)
            if type(values) is not tuple or any(type(value) is not str for value in values):
                raise ValueError(f"exact attempt {name} string tuple required")


def _primitive_snapshot(receipt: MinotaurExamReceipt) -> MinotaurExamReceipt:
    """Copy exactly typed primitive claims before invoking legacy evaluators.

    Existing frozen dataclasses can still be changed by ``object.__setattr__``.
    Revalidation alone is insufficient if an object supplies hostile equality
    methods. This guard admits only built-in immutable field types and passes
    newly constructed records into the established Journey state machine.
    """
    if type(receipt) is not MinotaurExamReceipt:
        raise ValueError("exact MinotaurExamReceipt required")
    if type(receipt.exam_kind) is not str or receipt.exam_kind != "development":
        raise ValueError("exact development exam_kind required")
    for name in ("exam_id", "objective_id", "checkpoint_sha256",
                 "baseline_sha256", "evaluator_id"):
        if type(getattr(receipt, name)) is not str:
            raise ValueError(f"exact {name} string required")
    if type(receipt.cycle_index) is not int or type(receipt.cases) is not tuple:
        raise ValueError("exact exam cycle and case tuple required")

    copied = []
    for case in receipt.cases:
        if type(case) is not ExamCaseResult:
            raise ValueError("exact development case required")
        if type(case.split) is not str or case.split not in ("heldout", "transfer"):
            raise ValueError("exact development split required")
        for name in ("case_id", "skill", "prompt_sha256"):
            if type(getattr(case, name)) is not str:
                raise ValueError(f"exact {name} string required")
        if type(case.score) not in (int, float) or type(case.baseline_score) not in (int, float):
            raise ValueError("exact numeric development scores required")
        copied.append(ExamCaseResult(
            case_id=case.case_id,
            skill=case.skill,
            split=case.split,
            score=case.score,
            baseline_score=case.baseline_score,
            prompt_sha256=case.prompt_sha256,
        ))
    return MinotaurExamReceipt(
        exam_id=receipt.exam_id,
        objective_id=receipt.objective_id,
        cycle_index=receipt.cycle_index,
        checkpoint_sha256=receipt.checkpoint_sha256,
        baseline_sha256=receipt.baseline_sha256,
        evaluator_id=receipt.evaluator_id,
        cases=tuple(copied),
        exam_kind="development",
    )


def admit_development_exam(
    state: JourneyState,
    receipt: MinotaurExamReceipt,
) -> ExamIntakeTransition:
    """Take a development-only exam to Journey's existing adaptive boundary.

    On a failed development exam, only case identities and prompt commitments
    pass into the regenerated train-only corridor. The existing Journey
    machine decides whether remediation, another exam, or exhaustion follows.
    A provisional development PASS does not create a training handoff or a
    graduation authorization.
    """
    if type(state) is not JourneyState:
        raise ValueError("exact JourneyState required")
    _validate_exact_state_claims(state)
    validate_journey(state)
    snapshot = _primitive_snapshot(receipt)
    next_journey = face_minotaur(state, snapshot)
    adaptive_walk = (
        begin_adaptive_walk_from_journey(next_journey)
        if next_journey.phase == TRAINING else None
    )
    return ExamIntakeTransition(next_journey, adaptive_walk)
