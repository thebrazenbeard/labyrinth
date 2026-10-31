"""Train-only chamber-level adaptive routing inside one Labyrinth maze.

Unlike end-of-cycle Minotaur regeneration, the next *training* chamber may
change immediately after a scored training experience. Feedback is a local
trainer receipt, not authenticated model improvement; no examination results,
holdout prompts, gold labels, or private-final weakness hints enter this API.

An assignment is a new train-only room descriptor with a distinct case seed
and a bounded suggested difficulty. Approved providers must materialize its
challenge, keep answer keys trainer-side and report the exact training prompt
commitment. The suggested difficulty is not evidence that every provider
honors that target. This opt-in mechanism does not mutate JourneyState.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import re

from .regenerative_maze import MazePlan, MazeRoom
from .training_scope import TrainingScopeSpec


_HEX32 = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_CASE_ID = re.compile(r"train-[A-Za-z0-9_.:-]{1,121}\Z", re.ASCII)
_ANY_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_MAX_NONCES = 16
_MAX_SEALED_COMMITMENTS = 4096


def _digest(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(piece) for piece in parts).encode("utf-8")).hexdigest()


def _sha(value: str, name: str) -> None:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class PendingChamber:
    """Trainer-owned routing decision; contains no challenge answer key."""

    origin_room_id: str
    room_id: str
    skill: str
    case_seed: int
    difficulty: int
    visit_index: int
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        for name in ("origin_room_id", "room_id"):
            value = getattr(self, name)
            if type(value) is not str or not _HEX32.fullmatch(value):
                raise ValueError(f"invalid {name}")
        if type(self.skill) is not str or not self.skill:
            raise ValueError("chamber skill must be nonempty")
        if type(self.case_seed) is not int or not 0 <= self.case_seed < 2**31:
            raise ValueError("invalid case seed")
        if type(self.difficulty) is not int or not 1 <= self.difficulty <= 5:
            raise ValueError("difficulty must be in [1,5]")
        if type(self.visit_index) is not int or not 0 <= self.visit_index < 128:
            raise ValueError("invalid visit index")
        _sha(self.checkpoint_sha256, "assigned checkpoint")

    @property
    def room(self) -> MazeRoom:
        """Pass this descriptor to an operator-approved train-only provider."""
        self.__post_init__()
        return MazeRoom(self.room_id, self.skill, self.case_seed)


@dataclass(frozen=True)
class TrainingRoomFeedback:
    """Self-reported *train* feedback; no private exam evidence allowed."""

    room_id: str
    case_id: str
    prompt_sha256: str
    score: int
    split: str
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str

    def __post_init__(self) -> None:
        if type(self.room_id) is not str or _HEX32.fullmatch(self.room_id) is None:
            raise ValueError("invalid room identity")
        if type(self.case_id) is not str or _CASE_ID.fullmatch(self.case_id) is None:
            raise ValueError("only approved train-prefixed case identities accepted")
        _sha(self.prompt_sha256, "training prompt")
        if type(self.score) is not int or self.score not in (0, 1):
            raise ValueError("train score must be exactly 0 or 1")
        if type(self.split) is not str or self.split != "train":
            raise ValueError("only train feedback can steer a corridor")
        _sha(self.checkpoint_before_sha256, "checkpoint before room")
        _sha(self.checkpoint_after_sha256, "checkpoint after room")


@dataclass(frozen=True)
class ChamberEvent:
    assignment: PendingChamber
    feedback: TrainingRoomFeedback


@dataclass(frozen=True)
class AdaptiveWalkState:
    scope: TrainingScopeSpec
    plan: MazePlan
    initial_checkpoint_sha256: str
    checkpoint_sha256: str
    history: tuple[ChamberEvent, ...] = ()
    pending: PendingChamber | None = None
    forbidden_case_ids: tuple[str, ...] = ()
    forbidden_prompt_sha256: tuple[str, ...] = ()

    @property
    def remaining_count(self) -> int:
        _validate_state(self)
        return len(self.plan.rooms) - len(self.history)

    @property
    def trail_sha256(self) -> str:
        """Reproducible visit-order commitment, **not** a trusted attestation."""
        _validate_state(self)
        digest = _digest("adaptive-walk-v1", self.scope.fingerprint,
                         self.plan.cycle_index, self.initial_checkpoint_sha256)
        for event in self.history:
            a, f = event.assignment, event.feedback
            digest = _digest(digest, a.origin_room_id, a.room_id, a.skill,
                             a.case_seed, a.difficulty, f.case_id,
                             f.prompt_sha256, f.score, f.checkpoint_after_sha256)
        return digest


def _validate_state(state: AdaptiveWalkState) -> None:
    if type(state) is not AdaptiveWalkState:
        raise ValueError("invalid corridor state")
    if type(state.scope) is not TrainingScopeSpec or type(state.plan) is not MazePlan:
        raise ValueError("invalid scope or maze plan")
    TrainingScopeSpec.__post_init__(state.scope)
    MazePlan.__post_init__(state.plan)
    if (state.plan.objective_id != state.scope.objective_id
            or state.plan.scope_fingerprint != state.scope.fingerprint
            or not 0 <= state.plan.cycle_index < state.scope.max_cycles
            or len(state.plan.rooms) != state.scope.total_chambers
            or set(room.skill for room in state.plan.rooms) != set(state.scope.skills)):
        raise ValueError("maze does not match declared training scope")
    _sha(state.initial_checkpoint_sha256, "initial checkpoint")
    _sha(state.checkpoint_sha256, "retained checkpoint")
    if type(state.forbidden_case_ids) is not tuple or type(state.forbidden_prompt_sha256) is not tuple:
        raise ValueError("forbidden evidence commitments must be immutable tuples")
    if (len(state.forbidden_case_ids) > _MAX_SEALED_COMMITMENTS
            or len(state.forbidden_prompt_sha256) > _MAX_SEALED_COMMITMENTS):
        raise ValueError("sealed evidence commitment budget exceeded")
    if len(state.forbidden_case_ids) != len(state.forbidden_prompt_sha256):
        raise ValueError("sealed case and prompt commitments must be paired")
    if (state.plan.cycle_index > 0
            and (not state.forbidden_case_ids or not state.forbidden_prompt_sha256)):
        raise ValueError("regenerated corridor missing prior-cycle exam custody")
    if any(type(case) is not str or _ANY_CASE_ID.fullmatch(case) is None
           for case in state.forbidden_case_ids):
        raise ValueError("invalid sealed case identity")
    for prompt in state.forbidden_prompt_sha256:
        _sha(prompt, "sealed prompt")
    if (len(set(state.forbidden_case_ids)) != len(state.forbidden_case_ids)
            or len(set(state.forbidden_prompt_sha256)) != len(state.forbidden_prompt_sha256)):
        raise ValueError("duplicate sealed evidence commitment")
    if type(state.history) is not tuple or len(state.history) > len(state.plan.rooms):
        raise ValueError("invalid chamber history")
    originals = {room.room_id: room.skill for room in state.plan.rooms}
    origins: set[str] = set()
    routed_ids: set[str] = set()
    case_seeds: set[int] = set()
    case_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    checkpoint = state.initial_checkpoint_sha256
    for index, event in enumerate(state.history):
        if type(event) is not ChamberEvent:
            raise ValueError("invalid chamber event")
        assignment, receipt = event.assignment, event.feedback
        if type(assignment) is not PendingChamber or type(receipt) is not TrainingRoomFeedback:
            raise ValueError("invalid chamber event payload")
        PendingChamber.__post_init__(assignment)
        TrainingRoomFeedback.__post_init__(receipt)
        if assignment != _expected_assignment(state.scope, state.plan,
                                              state.history[:index], checkpoint):
            raise ValueError("recorded chamber deviates from deterministic train-only route")
        if (assignment.origin_room_id not in originals
                or assignment.skill != originals[assignment.origin_room_id]
                or assignment.origin_room_id in origins
                or assignment.room_id in routed_ids
                or assignment.case_seed in case_seeds
                or assignment.visit_index != index
                or assignment.checkpoint_sha256 != checkpoint
                or receipt.room_id != assignment.room_id
                or receipt.checkpoint_before_sha256 != checkpoint
                or receipt.case_id in case_ids or receipt.prompt_sha256 in prompt_hashes
                or receipt.case_id in state.forbidden_case_ids
                or receipt.prompt_sha256 in state.forbidden_prompt_sha256):
            raise ValueError("inconsistent or repeated training evidence")
        origins.add(assignment.origin_room_id)
        routed_ids.add(assignment.room_id)
        case_seeds.add(assignment.case_seed)
        case_ids.add(receipt.case_id)
        prompt_hashes.add(receipt.prompt_sha256)
        checkpoint = receipt.checkpoint_after_sha256
    if state.checkpoint_sha256 != checkpoint:
        raise ValueError("checkpoint continuity broken across visited chambers")
    if state.pending is not None:
        if type(state.pending) is not PendingChamber:
            raise ValueError("invalid pending chamber")
        PendingChamber.__post_init__(state.pending)
        if state.pending != _expected_assignment(
            state.scope, state.plan, state.history, state.checkpoint_sha256
        ):
            raise ValueError("pending chamber deviates from deterministic train-only route")
        if (state.pending.origin_room_id not in originals
                or state.pending.skill != originals[state.pending.origin_room_id]
                or state.pending.origin_room_id in origins
                or state.pending.room_id in routed_ids
                or state.pending.case_seed in case_seeds
                or state.pending.visit_index != len(state.history)
                or state.pending.checkpoint_sha256 != state.checkpoint_sha256):
            raise ValueError("pending chamber does not belong to active corridor")


def begin_adaptive_walk(
    scope: TrainingScopeSpec,
    plan: MazePlan,
    *,
    checkpoint_sha256: str,
    forbidden_case_ids: tuple[str, ...] | None = None,
    forbidden_prompt_sha256: tuple[str, ...] | None = None,
) -> AdaptiveWalkState:
    """Enter cycle zero only; regenerated corridors require Journey custody.

    Caller-supplied forbidden commitments cannot establish a complete prior
    history.  Cycle >0 must use :func:`begin_adaptive_walk_from_journey`.
    """
    if type(plan) is not MazePlan:
        raise ValueError("expected a validated MazePlan")
    MazePlan.__post_init__(plan)
    if ((forbidden_case_ids is None) != (forbidden_prompt_sha256 is None)):
        raise ValueError("sealed case IDs and prompt hashes must be supplied together")
    if plan.cycle_index > 0:
        raise ValueError("regenerated corridors require a validated JourneyState")
    case_ids = () if forbidden_case_ids is None else forbidden_case_ids
    prompts = () if forbidden_prompt_sha256 is None else forbidden_prompt_sha256
    if type(case_ids) is not tuple or type(prompts) is not tuple or len(case_ids) != len(prompts):
        raise ValueError("historical evidence commitments must be paired immutable tuples")
    state = AdaptiveWalkState(
        scope=scope, plan=plan,
        initial_checkpoint_sha256=checkpoint_sha256,
        checkpoint_sha256=checkpoint_sha256,
        forbidden_case_ids=case_ids,
        forbidden_prompt_sha256=prompts,
    )
    _validate_state(state)
    return state


def begin_adaptive_walk_from_journey(journey: object) -> AdaptiveWalkState:
    """Bind prior train/dev exam commitments directly to a valid JourneyState.

    This imports no exam content: only identifiers and SHA-256 commitments
    cross into the training corridor. Neither the signed private graduation
    evaluator nor its held-out gold belongs here. The source-level Journey
    receipt is self-reported; this binding does not independently authenticate
    a neural training effect.
    """
    from .journey import JourneyState, TRAINING, _validate_state as validate_journey

    if type(journey) is not JourneyState:
        raise ValueError("expected a JourneyState")
    validate_journey(journey)
    if journey.phase != TRAINING:
        raise ValueError("adaptive training can only begin in the TRAINING phase")
    state = AdaptiveWalkState(
        scope=journey.scope, plan=journey.maze,
        initial_checkpoint_sha256=journey.checkpoint_sha256,
        checkpoint_sha256=journey.checkpoint_sha256,
        forbidden_case_ids=journey.used_training_case_ids + journey.sealed_exam_case_ids,
        forbidden_prompt_sha256=(journey.used_training_prompt_sha256
                                 + journey.sealed_exam_prompt_sha256),
    )
    _validate_state(state)
    return state


def _expected_assignment(
    scope: TrainingScopeSpec,
    plan: MazePlan,
    history: tuple[ChamberEvent, ...],
    checkpoint_sha256: str,
) -> PendingChamber:
    """Derive the one authorized route without relying on a stored assignment.

    Also used when validating restored history, so this helper deliberately
    performs no recursive state validation.
    """
    if len(history) == len(plan.rooms):
        raise ValueError("authorized corridor budget exhausted")
    visited = {event.assignment.origin_room_id for event in history}
    remaining = [room for room in plan.rooms if room.room_id not in visited]
    last = history[-1] if history else None
    preferred = None
    if last is not None:
        previous_skill = last.assignment.skill
        if last.feedback.score == 0:
            preferred = next((room for room in remaining if room.skill == previous_skill), None)
        else:
            preferred = next((room for room in remaining if room.skill != previous_skill), None)
    room = preferred if preferred is not None else remaining[0]
    # Difficulty follows evidence for *this* skill only: success on arithmetic
    # cannot justify advancing programming difficulty without a programming
    # observation. Missing skill evidence starts at the declared specificity.
    difficulty = scope.specificity
    last_same_skill = next(
        (event for event in reversed(history) if event.assignment.skill == room.skill), None
    )
    if last_same_skill is not None:
        difficulty = max(1, min(5, last_same_skill.assignment.difficulty
                                 + (1 if last_same_skill.feedback.score else -1)))
    feedback_commitment = _digest(
        "adaptive-journey-history", scope.fingerprint, plan.cycle_index,
        *(_digest(event.assignment.room_id, event.feedback.case_id,
                  event.feedback.prompt_sha256, event.feedback.score) for event in history),
    )
    used_room_ids = {event.assignment.room_id for event in history}
    used_seeds = {event.assignment.case_seed for event in history}
    initial_room_ids = {entry.room_id for entry in plan.rooms}
    initial_seeds = {entry.case_seed for entry in plan.rooms}
    assignment = None
    for nonce in range(_MAX_NONCES):
        tag = _digest("adaptive-chamber", scope.fingerprint, plan.cycle_index,
                      room.room_id, len(history), feedback_commitment, nonce)
        new_id = tag[:32]
        new_seed = int(_digest("adaptive-seed", tag)[:8], 16) % (2**31)
        if (new_id in used_room_ids or new_id in initial_room_ids
                or new_seed in used_seeds or new_seed in initial_seeds):
            continue
        return PendingChamber(
            origin_room_id=room.room_id,
            room_id=new_id,
            skill=room.skill,
            case_seed=new_seed,
            difficulty=difficulty,
            visit_index=len(history),
            checkpoint_sha256=checkpoint_sha256,
        )
    raise ValueError("unable to assign unique train-only chamber within bounded attempts")


def offer_next_chamber(state: AdaptiveWalkState) -> AdaptiveWalkState:
    """Select the next train-only chamber from *observed train* feedback.

    Failure drills the same skill when any such room remains. Success seeks a
    different skill to maintain breadth. A source-controlled fallback picks
    the first remaining room. The suggested difficulty falls after a failure
    and rises after success. Every branch changes the next room's seed.
    """
    _validate_state(state)
    if state.pending is not None:
        raise ValueError("must complete pending chamber before requesting another")
    assignment = _expected_assignment(
        state.scope, state.plan, state.history, state.checkpoint_sha256
    )
    result = replace(state, pending=assignment)
    _validate_state(result)
    return result


def record_training_feedback(
    state: AdaptiveWalkState, receipt: TrainingRoomFeedback
) -> AdaptiveWalkState:
    """Seal a completed train chamber and retain its learner checkpoint.

    No independent runtime attestation is asserted: an operator must verify
    actual model update/reload before using these receipts as evidence.
    """
    _validate_state(state)
    if state.pending is None:
        raise ValueError("no pending train chamber")
    if type(receipt) is not TrainingRoomFeedback:
        raise ValueError("expected TrainingRoomFeedback")
    TrainingRoomFeedback.__post_init__(receipt)
    if (receipt.room_id != state.pending.room_id
            or receipt.checkpoint_before_sha256 != state.checkpoint_sha256
            or receipt.case_id in state.forbidden_case_ids
            or receipt.prompt_sha256 in state.forbidden_prompt_sha256):
        raise ValueError("training feedback does not match authorized chamber")
    result = replace(
        state,
        checkpoint_sha256=receipt.checkpoint_after_sha256,
        history=state.history + (ChamberEvent(state.pending, receipt),),
        pending=None,
    )
    _validate_state(result)
    return result
