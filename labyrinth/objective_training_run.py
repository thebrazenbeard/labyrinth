"""Execute declared objectives through an allowlisted, TRAIN-only provider.

An ObjectiveContract is a statement of intent; this runner gives the existing
arithmetic and program-reasoning contracts an actual finite sequence of public
prompts, local scores, and a JourneyState-compatible TRAIN completion receipt.
It deliberately leaves training supervision with the trusted operator.

This is source-side practice, not a neural optimizer, independent examiner,
durable exposure authority, signed checkpoint provenance, or graduation gate.
The in-process exposure ledger must not be mistaken for Q6 host custody.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Callable

from .approved_train_provider_r90 import (
    LocalTrainScore, RoutedObjectiveTrainRoom,
    _MAX_OBJECTIVE_REDRAW_ATTEMPTS,
    rate_objective_train_response, route_objective_train_room,
)
from .journey import (
    JourneyState, TRAINING, TrainingCompletionReceipt,
    _validate_state as validate_journey,
)
from .journey_challenges import TraineeRoomView
from .objective_contract import ObjectiveContract


_MAX_REDRAWS = _MAX_OBJECTIVE_REDRAW_ATTEMPTS
_PRACTICE = re.compile(
    r"Calculate ([1-9][0-9]*) ([+-]) ([1-9][0-9]*)\.\n"
    r"Respond with exactly one line: ANSWER: <integer>\.\Z",
    re.ASCII,
)


def _source_train_semantic_key(routed: RoutedObjectiveTrainRoom) -> tuple[str, ...]:
    """Conservative arithmetic aliases for *local* TRAIN selection only.

    This is not a Q6 cross-host exposure ledger. The approved static provider
    emits one exact grammar, allowing commutative addition to be identified
    across different case IDs and operand order before a new case is opened.
    """
    view = routed.trainee_view
    if routed.provider_id != "arithmetic_static_train_v1":
        return ("public-prompt", view.prompt_sha256)
    parsed = _PRACTICE.fullmatch(view.prompt)
    if parsed is None:
        raise ValueError("source arithmetic TRAIN prompt has unsupported grammar")
    a, operation, b = int(parsed[1]), parsed[2], int(parsed[3])
    if operation == "+":
        a, b = sorted((a, b))
    return ("arithmetic-practice", operation, str(a), str(b))


@dataclass(frozen=True)
class ObjectiveTrainObservation:
    """A local score; no learner output, answer, or private exam contents."""

    provider_id: str
    room_id: str
    case_id: str
    prompt_sha256: str
    correct: bool
    points: int
    split: str = "train"

    @property
    def independent_exam_evidence(self) -> bool:
        return False

    @property
    def graduation_authorized(self) -> bool:
        return False


class ObjectiveTrainingRun:
    """Single in-process TRAIN maze traversal with explicit provider binding.

    Only :meth:`next_training_prompt` and the callback to
    :meth:`run_training_chamber` are learner-facing. The runner itself contains
    private TRAIN supervision and belongs exclusively to the trusted operator.
    A failed learner callback keeps its prompt pending and exposed; there is no
    replacement or retry on the next room to conceal an exposure.
    """

    def __init__(self, contract: ObjectiveContract, journey: JourneyState,
                 *, provider_id: str | None = None) -> None:
        if type(contract) is not ObjectiveContract or type(journey) is not JourneyState:
            raise ValueError("exact ObjectiveContract and JourneyState required")
        ObjectiveContract.__post_init__(contract)
        validate_journey(journey)
        if journey.phase != TRAINING or contract.scope != journey.scope:
            raise ValueError("objective run requires its current TRAINING journey")
        # Make an exact field snapshot against later caller-side frozen-record
        # mutation. Neither copying nor frozen dataclasses provide host trust.
        self._contract = deepcopy(contract)
        self._journey = deepcopy(journey)
        self._provider_id = provider_id
        # Fail unsupported/forged contracts before accepting learner exposure.
        try:
            route_objective_train_room(
                self._contract, self._journey.maze, self._journey.maze.rooms[0],
                provider_id=self._provider_id, journey=self._journey,
            )
        except ValueError as exc:
            if (self._journey.maze.rooms[0].skill != "arithmetic" or str(exc) !=
                    "approved challenge overlaps prior training or development evidence"):
                raise
            # This is a valid approved provider but its first deterministic
            # case is already sealed. No bytes were opened: next_training_prompt
            # will search alternate *source-generated* candidates before
            # exposing anything, or fail closed if none exists.
        self._pending: RoutedObjectiveTrainRoom | None = None
        # Internal ledger is primitive, immutable tuples. A frozen dataclass
        # returned to the caller is NOT immutable against object.__setattr__.
        # Never retain a caller-visible observation in the source ledger.
        # Seventh private scalar records the exact source redraw attempt.
        # It never enters the learner view or claims evaluation authority.
        self._observations: tuple[tuple[str, str, str, str, bool, int, int], ...] = ()
        self._opened_case_ids: set[str] = set()
        self._opened_prompt_sha256: set[str] = set()
        self._opened_semantics: set[tuple[str, ...]] = set()
        # Do not retain/return the same mutable-via-object.__setattr__ receipt.
        # (objective, cycle, before, after, room/case/prompt triples)
        self._completion_snapshot: tuple[
            str, int, str, str, tuple[tuple[str, str, str], ...]
        ] | None = None

    @property
    def observations(self) -> tuple[ObjectiveTrainObservation, ...]:
        """Only sanitized, self-reported, local practice observations."""
        return tuple(ObjectiveTrainObservation(*fields[:6]) for fields in self._observations)

    @property
    def is_complete(self) -> bool:
        return (self._pending is None
                and len(self._observations) == len(self._journey.maze.rooms))

    @property
    def opened_count(self) -> int:
        """Includes pending exposure, even if a learner callback failed."""
        return len(self._opened_case_ids)

    def next_training_prompt(self) -> TraineeRoomView:
        """Open the next TRAIN room; admit case and prompt before returning it."""
        if self._pending is not None:
            raise ValueError("complete the already exposed TRAIN prompt first")
        if self._completion_snapshot is not None:
            raise ValueError("completed TRAIN run is sealed")
        position = len(self._observations)
        if position >= len(self._journey.maze.rooms):
            raise ValueError("objective TRAIN maze is exhausted")
        room = self._journey.maze.rooms[position]
        # Keep every candidate private until *all* local evidence/semantic
        # checks pass. A duplicate does not consume a slot or trigger a model
        # call; source-controlled, bounded retries regenerate novel TRAIN.
        # Existing sealed prior-case checks are re-applied at each attempt.
        redraw_budget = (_MAX_REDRAWS if room.skill == "arithmetic" else 0)
        for attempt in range(redraw_budget + 1):
            try:
                routed = route_objective_train_room(
                    self._contract, self._journey.maze, room,
                    provider_id=self._provider_id, journey=self._journey,
                    redraw_attempt=attempt,
                )
            except ValueError as exc:
                if (room.skill == "arithmetic" and str(exc) ==
                        "approved challenge overlaps prior training or development evidence"):
                    continue
                raise
            view = routed.trainee_view
            semantic = _source_train_semantic_key(routed)
            if (view.challenge_id not in self._opened_case_ids
                    and view.prompt_sha256 not in self._opened_prompt_sha256
                    and semantic not in self._opened_semantics):
                break
        else:
            raise ValueError("source TRAIN redraw budget exhausted without novel evidence")
        self._opened_case_ids.add(view.challenge_id)
        self._opened_prompt_sha256.add(view.prompt_sha256)
        self._opened_semantics.add(semantic)
        self._pending = routed
        return view

    def submit_training_response(self, response: str) -> ObjectiveTrainObservation:
        """Score trainer-private TRAIN gold with a fixed source-owned rater."""
        pending = self._pending
        if pending is None:
            raise ValueError("no exposed TRAIN prompt to score")
        if type(response) is not str or len(response) > 8192:
            raise ValueError("learner response must be bounded ordinary text")
        room = self._journey.maze.rooms[len(self._observations)]
        score: LocalTrainScore = rate_objective_train_response(
            self._contract, self._journey.maze, room, response,
            provider_id=self._provider_id, journey=self._journey,
            redraw_attempt=pending.redraw_attempt,
        )
        view = pending.trainee_view
        if (score.provider_id != pending.provider_id
                or score.room_id != view.room_id
                or score.challenge_id != view.challenge_id
                or score.prompt_sha256 != view.prompt_sha256):
            raise ValueError("scored TRAIN room differs from its opened prompt")
        observed = ObjectiveTrainObservation(
            provider_id=score.provider_id, room_id=score.room_id,
            case_id=score.challenge_id, prompt_sha256=score.prompt_sha256,
            correct=score.correct, points=score.points,
        )
        self._observations += ((
            str(observed.provider_id), str(observed.room_id),
            str(observed.case_id), str(observed.prompt_sha256),
            bool(observed.correct), int(observed.points), int(pending.redraw_attempt),
        ),)
        self._pending = None
        return observed

    def run_training_chamber(
        self, learner: Callable[[TraineeRoomView], str],
    ) -> ObjectiveTrainObservation:
        """Invoke operator-approved inference with a public view, never gold."""
        if not callable(learner):
            raise ValueError("learner must be an operator-approved callable")
        view = self.next_training_prompt()
        response = learner(view)
        return self.submit_training_response(response)

    def self_reported_completion_receipt(
        self, *, checkpoint_after_sha256: str,
    ) -> TrainingCompletionReceipt:
        """Create a source-level handoff; do not assert a verified weight change."""
        if not self.is_complete:
            raise ValueError("cannot complete an unfinished TRAIN maze")
        if self._completion_snapshot is not None:
            if checkpoint_after_sha256 != self._completion_snapshot[3]:
                raise ValueError("completion checkpoint already pinned in this run")
            return TrainingCompletionReceipt(*self._completion_snapshot)
        # Regenerate canonical TRAIN identities from the source-owned provider,
        # not from externally held Observation instances or callback output.
        for index, room in enumerate(self._journey.maze.rooms):
            expected = route_objective_train_room(
                self._contract, self._journey.maze, room,
                provider_id=self._provider_id, journey=self._journey,
                redraw_attempt=self._observations[index][6],
            )
            view = expected.trainee_view
            entry = self._observations[index]
            if entry[:4] != (expected.provider_id, room.room_id,
                             view.challenge_id, view.prompt_sha256):
                raise ValueError("TRAIN observation ledger differs from source-generated case")
        receipt = TrainingCompletionReceipt(
            objective_id=self._journey.scope.objective_id,
            cycle_index=self._journey.cycle_index,
            checkpoint_before_sha256=self._journey.checkpoint_sha256,
            checkpoint_after_sha256=checkpoint_after_sha256,
            room_case_ids=tuple(
                (room_id, case_id, prompt_sha256)
                for _provider_id, room_id, case_id, prompt_sha256, _correct, _points, _attempt
                in self._observations
            ),
        )
        if receipt.checkpoint_before_sha256 == receipt.checkpoint_after_sha256:
            raise ValueError("changed checkpoint digest must be externally reported")
        self._completion_snapshot = (
            receipt.objective_id, receipt.cycle_index,
            receipt.checkpoint_before_sha256, receipt.checkpoint_after_sha256,
            receipt.room_case_ids,
        )
        return TrainingCompletionReceipt(*self._completion_snapshot)
