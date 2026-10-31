"""Regenerative Labyrinth journey state machine, with no hidden neural effects.

The learner enters with a frozen training scope and checkpoint identity. Each
maze requires explicit *training-only* room visits and a changed checkpoint
receipt before an internal Minotaur exam may be considered. A failed exam
rerolls the maze but never erases the learner's checkpoint. Exam cases are
sealed against subsequent training. This orchestration is not independent
model execution, evaluator attestation, or formal graduation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Sequence

from .training_scope import TrainingScopeSpec
from .regenerative_maze import MazePlan, MazeRoom, generate_maze_plan
from .minotaur_exam import ExamCaseResult, MinotaurExamReceipt, evaluate_minotaur


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
TRAINING = "TRAINING"
AWAITING_EXAM = "AWAITING_EXAM"
DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION = "development_passed_pending_private_qualification"
EXHAUSTED = "EXHAUSTED"
MAX_INCONCLUSIVE_EXAMS_PER_CYCLE = 2


def _digest(value: str, label: str) -> None:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 identity")


@dataclass(frozen=True)
class TrainingCompletionReceipt:
    """Host-observed training metadata; does not authenticate neural execution.

    Each (room_id, case_id, prompt_sha256) triple binds a completed room to a
    concrete train case and exact prompt commitment. A distinct before/after
    digest is required, but hashes alone do not
    establish learning. Controls with intentionally zero updates are separate.
    """

    objective_id: str
    cycle_index: int
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str
    room_case_ids: tuple[tuple[str, str, str], ...]
    split: str = "train"

    def __post_init__(self) -> None:
        if type(self.objective_id) is not str or not _ID.fullmatch(self.objective_id):
            raise ValueError("invalid objective_id")
        if type(self.cycle_index) is not int or self.cycle_index < 0:
            raise ValueError("invalid training cycle index")
        _digest(self.checkpoint_before_sha256, "checkpoint_before_sha256")
        _digest(self.checkpoint_after_sha256, "checkpoint_after_sha256")
        if type(self.split) is not str or self.split != "train":
            raise ValueError("only training evidence may finish a maze")
        if type(self.room_case_ids) is not tuple or not self.room_case_ids:
            raise ValueError("room_case_ids must be a nonempty immutable tuple")
        for pair in self.room_case_ids:
            if (type(pair) is not tuple or len(pair) != 3
                    or any(type(value) is not str or not _ID.fullmatch(value) for value in pair[:2])):
                raise ValueError("invalid training room/case identity")
            _digest(pair[2], "training prompt")
        if len({pair[0] for pair in self.room_case_ids}) != len(self.room_case_ids):
            raise ValueError("duplicate training room")
        if len({pair[1] for pair in self.room_case_ids}) != len(self.room_case_ids):
            raise ValueError("duplicate training case")
        if len({pair[2] for pair in self.room_case_ids}) != len(self.room_case_ids):
            raise ValueError("duplicate training prompt commitment")


@dataclass(frozen=True)
class JourneyAttempt:
    cycle_index: int
    exam_id: str
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str
    training_case_ids: tuple[str, ...]
    training_prompt_sha256: tuple[str, ...]
    exam_case_ids: tuple[str, ...]
    exam_prompt_sha256: tuple[str, ...]
    exam_status: str
    weak_skills: tuple[str, ...]


@dataclass(frozen=True)
class JourneyState:
    scope: TrainingScopeSpec
    baseline_checkpoint_sha256: str
    checkpoint_sha256: str
    maze: MazePlan
    phase: str = TRAINING
    attempts: tuple[JourneyAttempt, ...] = ()
    used_training_case_ids: tuple[str, ...] = ()
    used_training_prompt_sha256: tuple[str, ...] = ()
    sealed_exam_ids: tuple[str, ...] = ()
    exam_event_statuses: tuple[str, ...] = ()
    sealed_exam_case_ids: tuple[str, ...] = ()
    sealed_exam_prompt_sha256: tuple[str, ...] = ()
    pending_training_case_ids: tuple[str, ...] = ()
    pending_training_prompt_sha256: tuple[str, ...] = ()
    pending_checkpoint_before_sha256: str = ""
    inconclusive_exam_count: int = 0

    @property
    def cycle_index(self) -> int:
        return self.maze.cycle_index


def _validate_state(state: JourneyState) -> None:
    if not isinstance(state, JourneyState):
        raise ValueError("invalid journey state")
    if not isinstance(state.scope, TrainingScopeSpec):
        raise ValueError("invalid training scope")
    TrainingScopeSpec.__post_init__(state.scope)
    _digest(state.baseline_checkpoint_sha256, "baseline_checkpoint_sha256")
    _digest(state.checkpoint_sha256, "checkpoint_sha256")
    if not isinstance(state.maze, MazePlan):
        raise ValueError("invalid maze")
    MazePlan.__post_init__(state.maze)
    if (not state.maze.rooms or any(room.skill not in state.scope.skills for room in state.maze.rooms)
            or not set(state.scope.skills).issubset({room.skill for room in state.maze.rooms})):
        raise ValueError("maze does not cover declared scope")
    if (not isinstance(state.maze, MazePlan)
            or state.maze.scope_fingerprint != state.scope.fingerprint
            or state.maze.objective_id != state.scope.objective_id
            or not 0 <= state.cycle_index < state.scope.max_cycles
            or state.phase not in (TRAINING, AWAITING_EXAM,
                                    DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION, EXHAUSTED)):
        raise ValueError("journey state has invalid scope, maze or phase")
    if len(set(state.used_training_case_ids)) != len(state.used_training_case_ids):
        raise ValueError("duplicate training case in journey history")
    if len(set(state.used_training_prompt_sha256)) != len(state.used_training_prompt_sha256):
        raise ValueError("duplicate training prompt in journey history")
    if len(set(state.sealed_exam_case_ids)) != len(state.sealed_exam_case_ids):
        raise ValueError("duplicate exam case in journey history")
    if len(set(state.sealed_exam_ids)) != len(state.sealed_exam_ids):
        raise ValueError("duplicate exam event ID in journey history")
    if (len(state.sealed_exam_ids) != len(state.exam_event_statuses)
            or any(status not in ("PASS", "FAIL", "INCONCLUSIVE")
                   for status in state.exam_event_statuses)):
        raise ValueError("invalid exam event status history")
    if len(set(state.sealed_exam_prompt_sha256)) != len(state.sealed_exam_prompt_sha256):
        raise ValueError("duplicate exam prompt in journey history")
    if set(state.used_training_case_ids) & set(state.sealed_exam_case_ids):
        raise ValueError("exam cases cannot become training cases")
    if set(state.used_training_prompt_sha256) & set(state.sealed_exam_prompt_sha256):
        raise ValueError("exam prompts cannot become training prompts")
    if len(state.used_training_case_ids) != len(state.used_training_prompt_sha256):
        raise ValueError("training case and prompt counts differ")
    if len(state.sealed_exam_case_ids) != len(state.sealed_exam_prompt_sha256):
        raise ValueError("exam case and prompt counts differ")
    if state.phase == AWAITING_EXAM:
        _digest(state.pending_checkpoint_before_sha256, "pending_checkpoint_before_sha256")
        if not state.pending_training_case_ids:
            raise ValueError("missing completed training cases for exam")
        if len(state.pending_training_case_ids) != len(state.pending_training_prompt_sha256):
            raise ValueError("missing training prompt evidence")
    elif (state.pending_training_case_ids or state.pending_training_prompt_sha256
          or state.pending_checkpoint_before_sha256):
        raise ValueError("pending training evidence outside exam phase")
    if len(state.attempts) > state.scope.max_cycles or any(
        attempt.cycle_index != index for index, attempt in enumerate(state.attempts)
    ):
        raise ValueError("invalid attempt history")
    if (type(state.inconclusive_exam_count) is not int
            or not 0 <= state.inconclusive_exam_count <= MAX_INCONCLUSIVE_EXAMS_PER_CYCLE
            or (state.phase == TRAINING and state.inconclusive_exam_count != 0)
            or (state.phase == AWAITING_EXAM
                and state.inconclusive_exam_count == MAX_INCONCLUSIVE_EXAMS_PER_CYCLE)):
        raise ValueError("invalid inconclusive exam count")
    if state.phase == TRAINING and len(state.attempts) != state.cycle_index:
        raise ValueError("active maze must follow the previous completed attempts")
    if state.phase == AWAITING_EXAM and len(state.attempts) != state.cycle_index:
        raise ValueError("exam must follow the previous completed attempts")
    if (state.phase in (DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION, EXHAUSTED)
            and len(state.attempts) != state.cycle_index + 1):
        raise ValueError("finished journey requires terminal attempt")


def begin_journey(scope: TrainingScopeSpec, *, checkpoint_sha256: str) -> JourneyState:
    """Freeze an objective and enter the first deterministic training maze."""
    if not isinstance(scope, TrainingScopeSpec):
        raise ValueError("expected TrainingScopeSpec")
    TrainingScopeSpec.__post_init__(scope)
    _digest(checkpoint_sha256, "checkpoint_sha256")
    state = JourneyState(
        scope=scope,
        baseline_checkpoint_sha256=checkpoint_sha256,
        checkpoint_sha256=checkpoint_sha256,
        maze=generate_maze_plan(scope, 0),
    )
    _validate_state(state)
    return state


def complete_maze(state: JourneyState, receipt: TrainingCompletionReceipt) -> JourneyState:
    """Accept one complete train-only maze traversal and a changed model hash."""
    # Preserve TRAIN/evaluation isolation even when a caller mutates a frozen
    # state or receipt after construction. The snapshot is in-process routing
    # integrity, never proof that physical neural weights were updated.
    state = _snapshot_journey_claims(state)
    if type(receipt) is not TrainingCompletionReceipt:
        raise ValueError("exact TrainingCompletionReceipt required")
    receipt = replace(receipt)
    if state.phase != TRAINING:
        raise ValueError("journey is not accepting training evidence")
    if not isinstance(receipt, TrainingCompletionReceipt):
        raise ValueError("expected TrainingCompletionReceipt")
    TrainingCompletionReceipt.__post_init__(receipt)
    if (receipt.objective_id != state.scope.objective_id
            or receipt.cycle_index != state.cycle_index
            or receipt.checkpoint_before_sha256 != state.checkpoint_sha256):
        raise ValueError("training receipt does not match current objective, cycle and learner")
    if receipt.checkpoint_after_sha256 == state.checkpoint_sha256:
        raise ValueError("no changed checkpoint: cannot assert neural training progress")
    room_ids = tuple(room.room_id for room in state.maze.rooms)
    received_ids = tuple(pair[0] for pair in receipt.room_case_ids)
    if len(received_ids) != len(room_ids) or set(received_ids) != set(room_ids):
        raise ValueError("every maze room must be completed exactly once")
    case_ids = tuple(pair[1] for pair in receipt.room_case_ids)
    prompts = tuple(pair[2] for pair in receipt.room_case_ids)
    if set(case_ids) & (set(state.used_training_case_ids) | set(state.sealed_exam_case_ids)):
        raise ValueError("training case was reused or came from an exam")
    if set(prompts) & (set(state.used_training_prompt_sha256) | set(state.sealed_exam_prompt_sha256)):
        raise ValueError("training prompt was reused or came from an exam")
    next_state = replace(
        state,
        checkpoint_sha256=receipt.checkpoint_after_sha256,
        phase=AWAITING_EXAM,
        used_training_case_ids=state.used_training_case_ids + case_ids,
        used_training_prompt_sha256=state.used_training_prompt_sha256 + prompts,
        pending_training_case_ids=case_ids,
        pending_training_prompt_sha256=prompts,
        pending_checkpoint_before_sha256=receipt.checkpoint_before_sha256,
    )
    _validate_state(next_state)
    return next_state


def _snapshot_journey_claims(state: JourneyState) -> JourneyState:
    """Copy primitive journey claims before TRAIN or exam set comparisons.

    Frozen in-process records can be mutated with ``object.__setattr__``.
    A string subclass can impersonate AWAITING_EXAM, or compare equal while
    hashing differently to hide an already-used private/train case. These
    checks protect routing only; they cannot authenticate a real evaluator,
    prior history, model checkpoint, or examiner custody.
    """
    if type(state) is not JourneyState:
        raise ValueError("exact JourneyState required")
    if type(state.phase) is not str or type(state.pending_checkpoint_before_sha256) is not str:
        raise ValueError("exact journey phase and pending checkpoint required")
    if type(state.scope) is not TrainingScopeSpec or type(state.maze) is not MazePlan:
        raise ValueError("exact journey scope and MazePlan required")
    if type(state.maze.rooms) is not tuple or any(
        type(room) is not MazeRoom for room in state.maze.rooms
    ):
        raise ValueError("exact MazeRoom tuple required")
    for name in (
        "used_training_case_ids", "used_training_prompt_sha256", "sealed_exam_ids",
        "exam_event_statuses", "sealed_exam_case_ids", "sealed_exam_prompt_sha256",
        "pending_training_case_ids", "pending_training_prompt_sha256",
    ):
        values = getattr(state, name)
        if type(values) is not tuple or any(type(value) is not str for value in values):
            raise ValueError(f"exact {name} primitive string tuple required")
    if type(state.attempts) is not tuple:
        raise ValueError("exact journey attempt tuple required")
    for attempt in state.attempts:
        if type(attempt) is not JourneyAttempt or type(attempt.cycle_index) is not int:
            raise ValueError("exact JourneyAttempt required")
        for name in (
            "exam_id", "checkpoint_before_sha256", "checkpoint_after_sha256", "exam_status",
        ):
            if type(getattr(attempt, name)) is not str:
                raise ValueError(f"exact JourneyAttempt {name} string required")
        for name in (
            "training_case_ids", "training_prompt_sha256", "exam_case_ids",
            "exam_prompt_sha256", "weak_skills",
        ):
            values = getattr(attempt, name)
            if type(values) is not tuple or any(type(value) is not str for value in values):
                raise ValueError(f"exact JourneyAttempt {name} string tuple required")
    # New records revalidate the nested scope, Maze, room and attempt values.
    snap_state = replace(
        state,
        scope=replace(state.scope),
        maze=replace(state.maze, rooms=tuple(replace(room) for room in state.maze.rooms)),
        attempts=tuple(replace(attempt) for attempt in state.attempts),
    )
    _validate_state(snap_state)
    return snap_state


def _snapshot_development_exam_inputs(
    state: JourneyState, receipt: MinotaurExamReceipt,
) -> tuple[JourneyState, MinotaurExamReceipt]:
    """Copy exactly typed development evidence for the internal Minotaur."""
    snap_state = _snapshot_journey_claims(state)
    if type(receipt) is not MinotaurExamReceipt:
        raise ValueError("exact MinotaurExamReceipt required")
    if type(receipt.exam_kind) is not str or receipt.exam_kind != "development":
        raise ValueError("only development exam kind with exact primitive string allowed")
    if type(receipt.cases) is not tuple or any(
        type(case) is not ExamCaseResult or type(case.split) is not str
        for case in receipt.cases
    ):
        raise ValueError("exact development exam cases and splits required")
    snap_receipt = replace(
        receipt, cases=tuple(replace(case) for case in receipt.cases),
    )
    return snap_state, snap_receipt


def face_minotaur(state: JourneyState, receipt: MinotaurExamReceipt) -> JourneyState:
    """Apply an independent *internal* mastery gate, never formal graduation.

    `evaluate_minotaur` checks objective coverage, scores and controls but
    cannot authenticate an evaluator or qualify neural execution. A separate
    trusted graduation gate is still required for external claims.
    """
    state, receipt = _snapshot_development_exam_inputs(state, receipt)
    if state.phase != AWAITING_EXAM:
        raise ValueError("Minotaur exam requires completed maze training")
    if not isinstance(receipt, MinotaurExamReceipt):
        raise ValueError("expected MinotaurExamReceipt")
    if receipt.exam_kind != "development":
        # Final/private qualification material must never produce targeted
        # remediation hints that can subsequently steer training.
        raise ValueError("only development exams may drive maze regeneration")
    if (receipt.objective_id != state.scope.objective_id
            or receipt.cycle_index != state.cycle_index
            or receipt.checkpoint_sha256 != state.checkpoint_sha256
            or receipt.baseline_sha256 != state.baseline_checkpoint_sha256):
        raise ValueError("exam receipt does not match current learning attempt")
    if receipt.exam_id in state.sealed_exam_ids:
        raise ValueError("reused Minotaur exam identity")
    exam_ids = tuple(item.case_id for item in receipt.cases)
    exam_prompts = tuple(item.prompt_sha256 for item in receipt.cases)
    if len(set(exam_ids)) != len(exam_ids):
        raise ValueError("duplicate exam cases")
    if set(exam_ids) & (set(state.used_training_case_ids) | set(state.sealed_exam_case_ids)):
        raise ValueError("exam case overlaps training or an earlier exam")
    if set(exam_prompts) & (set(state.used_training_prompt_sha256) | set(state.sealed_exam_prompt_sha256)):
        raise ValueError("exam prompt overlaps training or an earlier exam")
    decision = evaluate_minotaur(state.scope, receipt)
    if decision.status not in ("PASS", "FAIL", "INCONCLUSIVE"):
        raise ValueError("unknown internal exam decision")
    sealed = state.sealed_exam_case_ids + exam_ids
    sealed_prompts = state.sealed_exam_prompt_sha256 + exam_prompts
    sealed_events = state.sealed_exam_ids + (receipt.exam_id,)
    if decision.status == "INCONCLUSIVE":
        # Allow one new development exam if evidence is incomplete; a second
        # inconclusive attempt exhausts this journey instead of allowing an
        # unbounded stream of fresh exposed cases or an accidental pass.
        count = state.inconclusive_exam_count + 1
        if count < MAX_INCONCLUSIVE_EXAMS_PER_CYCLE:
            next_state = replace(state, sealed_exam_case_ids=sealed,
                                 sealed_exam_prompt_sha256=sealed_prompts,
                                 sealed_exam_ids=sealed_events,
                                 exam_event_statuses=state.exam_event_statuses + (decision.status,),
                                 inconclusive_exam_count=count)
        else:
            summary = JourneyAttempt(
                cycle_index=state.cycle_index,
                exam_id=receipt.exam_id,
                checkpoint_before_sha256=state.pending_checkpoint_before_sha256,
                checkpoint_after_sha256=state.checkpoint_sha256,
                training_case_ids=state.pending_training_case_ids,
                training_prompt_sha256=state.pending_training_prompt_sha256,
                exam_case_ids=exam_ids,
                exam_prompt_sha256=exam_prompts,
                exam_status="INCONCLUSIVE",
                weak_skills=(),
            )
            next_state = replace(
                state, phase=EXHAUSTED, attempts=state.attempts + (summary,),
                sealed_exam_case_ids=sealed, sealed_exam_prompt_sha256=sealed_prompts,
                sealed_exam_ids=sealed_events,
                exam_event_statuses=state.exam_event_statuses + (decision.status,),
                pending_training_case_ids=(), pending_training_prompt_sha256=(),
                pending_checkpoint_before_sha256="", inconclusive_exam_count=count,
            )
        _validate_state(next_state)
        return next_state
    summary = JourneyAttempt(
        cycle_index=state.cycle_index,
        exam_id=receipt.exam_id,
        checkpoint_before_sha256=state.pending_checkpoint_before_sha256,
        checkpoint_after_sha256=state.checkpoint_sha256,
        training_case_ids=state.pending_training_case_ids,
        training_prompt_sha256=state.pending_training_prompt_sha256,
        exam_case_ids=exam_ids,
        exam_prompt_sha256=exam_prompts,
        exam_status=decision.status,
        weak_skills=tuple(decision.weak_skills),
    )
    attempts = state.attempts + (summary,)
    base = replace(state, attempts=attempts, sealed_exam_case_ids=sealed,
                   sealed_exam_prompt_sha256=sealed_prompts,
                   sealed_exam_ids=sealed_events,
                   exam_event_statuses=state.exam_event_statuses + (decision.status,),
                   pending_training_case_ids=(), pending_training_prompt_sha256=(),
                   pending_checkpoint_before_sha256="", inconclusive_exam_count=0)
    if decision.status == "PASS":
        next_state = replace(base, phase=DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION)
    elif state.cycle_index + 1 == state.scope.max_cycles:
        next_state = replace(base, phase=EXHAUSTED)
    else:
        rerolled = generate_maze_plan(state.scope, state.cycle_index + 1,
                                     weak_skills=decision.weak_skills)
        next_state = replace(base, phase=TRAINING, maze=rerolled)
    _validate_state(next_state)
    return next_state
