"""Execute an approved cognitive-perspective objective as source-only TRAIN.

The learner sees only prompt-only ``TraineeRoomView`` records and returns an
A/B/C choice. The trainer privately scores each choice against the fixed,
synthetic provider. None of the scores, source commitments, self-reported
receipts, or prior-exposure lists authenticate real neural weight updates,
private examination, host custody, or graduation authority. A same-process
Python learner callback can inspect trainer frames/closures or mutate private
objects: the prompt-only interface is API hygiene, not an isolation boundary.
Only an independently isolated process or host can handle an untrusted learner.

Do not use the arithmetic/program runner's ``ANSWER: <integer>`` scorer for
this lane. Cognitive scenarios have a distinct bounded 100/0 choice rubric.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from .approved_cognitive_empathy_train_r90 import (
    CognitiveTrainingRoom, PROVIDER_ID, _forbidden_hashes,
    prepare_cognitive_empathy_train_rooms,
)
from .journey import (
    JourneyState, TRAINING, TrainingCompletionReceipt, _snapshot_journey_claims,
)
from .journey_challenges import TraineeRoomView
from .objective_contract import ObjectiveContract
from .objective_training_run import ObjectiveTrainObservation, ObjectiveTrainingRun
from .regenerative_maze import generate_maze_plan


@dataclass(frozen=True)
class CognitiveTrainFeedbackState:
    """Publicly reportable local practice counts, never a mastery verdict."""

    cycle_index: int
    completed_rooms: int
    total_rooms: int
    correct_rooms: int
    incorrect_rooms: int

    @property
    def independent_exam_evidence(self) -> bool:
        return False

    @property
    def graduation_authorized(self) -> bool:
        return False


class CognitiveObjectiveTrainingRun:
    """A bounded, two-cycle, explicitly approved synthetic cognitive TRAIN run.

    An operator must provide externally owned prior semantic exclusions where
    applicable. In-process prior JourneyState case/prompt commitments also
    have to match regenerated source identities; mismatches fail closed.
    Neither record is host-verified, signed, or evaluator-attested.
    """

    def __init__(
        self, contract: ObjectiveContract, journey: JourneyState, *,
        provider_id: str = PROVIDER_ID,
        forbidden_semantic_sha256: tuple[str, ...] | frozenset[str] = (),
    ) -> None:
        if type(contract) is not ObjectiveContract or type(journey) is not JourneyState:
            raise ValueError("exact ObjectiveContract and JourneyState required")
        ObjectiveContract.__post_init__(contract)
        snap = _snapshot_journey_claims(journey)
        if snap.phase != TRAINING or contract.scope != snap.scope:
            raise ValueError("cognitive objective requires its current TRAINING journey")
        if type(provider_id) is not str or provider_id != PROVIDER_ID:
            raise ValueError("unapproved cognitive TRAIN provider")

        self._contract = deepcopy(contract)
        self._journey = snap
        self._provider_id = PROVIDER_ID
        # Prior cycle attempts must preserve their exact source case/prompt
        # commitments before we accept newly exposed TRAIN cases. This is
        # deterministic source consistency, not cross-process replay authority.
        prior_semantics: set[str] = set()
        for attempt in self._journey.attempts:
            prior_plan = generate_maze_plan(
                self._journey.scope, attempt.cycle_index,
                weak_skills=(self._journey.attempts[attempt.cycle_index - 1].weak_skills
                             if attempt.cycle_index else ()),
            )
            prior_rooms = prepare_cognitive_empathy_train_rooms(
                self._contract, prior_plan, provider_id=self._provider_id,
            )
            if (tuple(room.trainee_view.challenge_id for room in prior_rooms)
                    != attempt.training_case_ids
                    or tuple(room.trainee_view.prompt_sha256 for room in prior_rooms)
                    != attempt.training_prompt_sha256):
                raise ValueError("prior cognitive TRAIN exposure does not match approved source")
            prior_semantics.update(room.semantic_sha256 for room in prior_rooms)

        # Validate exact operator input *before* coalescing with source-derived
        # prior-cycle semantics: duplicates/invalid hashes must never vanish
        # behind a set conversion that would accept a forged alias list.
        operator_semantics = _forbidden_hashes(
            forbidden_semantic_sha256, "forbidden_semantic_sha256",
        )
        self._forbidden_semantics = tuple(sorted(prior_semantics | operator_semantics))
        forbidden_case_ids = (
            self._journey.used_training_case_ids + self._journey.sealed_exam_case_ids
        )
        forbidden_prompts = (
            self._journey.used_training_prompt_sha256
            + self._journey.sealed_exam_prompt_sha256
        )
        self._forbidden_case_ids = forbidden_case_ids
        self._forbidden_prompts = forbidden_prompts
        self._rooms: tuple[CognitiveTrainingRoom, ...] = prepare_cognitive_empathy_train_rooms(
            self._contract, self._journey.maze, provider_id=self._provider_id,
            forbidden_semantic_sha256=self._forbidden_semantics,
            forbidden_prompt_sha256=forbidden_prompts,
            forbidden_case_ids=forbidden_case_ids,
        )
        self._pending_index: int | None = None
        # Primitive-only private ledger: (room, case, prompt, semantic, correct, points).
        # Never retain returned frozen records; object.__setattr__ can mutate them.
        self._observations: tuple[tuple[str, str, str, str, bool, int], ...] = ()
        self._opened_case_ids: set[str] = set()
        self._opened_prompt_sha256: set[str] = set()
        self._opened_semantics: set[str] = set()
        self._completion_snapshot: tuple[
            str, int, str, str, tuple[tuple[str, str, str], ...]
        ] | None = None

    @property
    def observations(self) -> tuple[ObjectiveTrainObservation, ...]:
        return tuple(ObjectiveTrainObservation(
            self._provider_id, room_id, case_id, prompt_sha, correct, points,
        ) for room_id, case_id, prompt_sha, _semantic, correct, points
                     in self._observations)

    @property
    def feedback_state(self) -> CognitiveTrainFeedbackState:
        good = sum(1 for entry in self._observations if entry[4])
        done = len(self._observations)
        return CognitiveTrainFeedbackState(
            cycle_index=self._journey.cycle_index, completed_rooms=done,
            total_rooms=len(self._rooms), correct_rooms=good, incorrect_rooms=done-good,
        )

    @property
    def is_complete(self) -> bool:
        return self._pending_index is None and len(self._observations) == len(self._rooms)

    @property
    def opened_count(self) -> int:
        return len(self._opened_case_ids)

    @property
    def authorizes_host_execution(self) -> bool:
        return False

    @property
    def authorizes_training(self) -> bool:
        return False

    @property
    def authorizes_exam(self) -> bool:
        return False

    @property
    def authorizes_graduation(self) -> bool:
        return False

    def _regenerated_roster(self) -> tuple[CognitiveTrainingRoom, ...]:
        """Refuse silent source replacement before scoring or completion."""
        regenerated = prepare_cognitive_empathy_train_rooms(
            self._contract, self._journey.maze, provider_id=self._provider_id,
            forbidden_semantic_sha256=self._forbidden_semantics,
            forbidden_prompt_sha256=self._forbidden_prompts,
            forbidden_case_ids=self._forbidden_case_ids,
        )
        if len(regenerated) != len(self._rooms):
            raise ValueError("cognitive TRAIN source roster changed")
        for source, expected in zip(regenerated, self._rooms):
            if (source.trainee_view != expected.trainee_view
                    or source.semantic_sha256 != expected.semantic_sha256
                    or source.prepared.challenge.answer
                    != expected.prepared.challenge.answer):
                raise ValueError("cognitive TRAIN source room changed")
        return regenerated

    def next_training_prompt(self) -> TraineeRoomView:
        if self._pending_index is not None:
            raise ValueError("complete the already exposed TRAIN prompt first")
        if self._completion_snapshot is not None:
            raise ValueError("completed cognitive TRAIN run is sealed")
        index = len(self._observations)
        if index >= len(self._rooms):
            raise ValueError("cognitive TRAIN scenario bank exhausted for this run")
        source = self._regenerated_roster()[index]
        view = source.trainee_view
        if (view.challenge_id in self._opened_case_ids
                or view.prompt_sha256 in self._opened_prompt_sha256
                or source.semantic_sha256 in self._opened_semantics):
            raise ValueError("cognitive TRAIN source exposure repeated")
        self._opened_case_ids.add(view.challenge_id)
        self._opened_prompt_sha256.add(view.prompt_sha256)
        self._opened_semantics.add(source.semantic_sha256)
        self._pending_index = index
        return view

    def submit_training_response(self, response: str) -> ObjectiveTrainObservation:
        index = self._pending_index
        if index is None:
            raise ValueError("no exposed cognitive TRAIN prompt to score")
        source = self._regenerated_roster()[index]
        view = source.trainee_view
        # The fixed binary choice rubric accepts only <=32 ordinary text
        # characters, and never uses arithmetic/program response parsing.
        points = source.score_train_choice(response)
        self._observations += ((
            view.room_id, view.challenge_id, view.prompt_sha256,
            source.semantic_sha256, points == 100, points,
        ),)
        self._pending_index = None
        return ObjectiveTrainObservation(
            self._provider_id, view.room_id, view.challenge_id,
            view.prompt_sha256, points == 100, points,
        )

    def run_training_chamber(
        self, learner: Callable[[TraineeRoomView], str],
    ) -> ObjectiveTrainObservation:
        if not callable(learner):
            raise ValueError("learner must be an operator-approved callable")
        public_view = self.next_training_prompt()
        return self.submit_training_response(learner(public_view))

    def self_reported_completion_receipt(
        self, *, checkpoint_after_sha256: str,
    ) -> TrainingCompletionReceipt:
        """Only a source-level handoff, not an observed checkpoint update."""
        if not self.is_complete:
            raise ValueError("cannot complete an unfinished cognitive TRAIN maze")
        if self._completion_snapshot is not None:
            if checkpoint_after_sha256 != self._completion_snapshot[3]:
                raise ValueError("completion checkpoint already pinned in this run")
            return TrainingCompletionReceipt(*self._completion_snapshot)
        expected = self._regenerated_roster()
        for entry, source in zip(self._observations, expected):
            view = source.trainee_view
            if entry[:4] != (
                view.room_id, view.challenge_id, view.prompt_sha256,
                source.semantic_sha256,
            ):
                raise ValueError("cognitive TRAIN observation differs from canonical source")
        receipt = TrainingCompletionReceipt(
            objective_id=self._journey.scope.objective_id,
            cycle_index=self._journey.cycle_index,
            checkpoint_before_sha256=self._journey.checkpoint_sha256,
            checkpoint_after_sha256=checkpoint_after_sha256,
            room_case_ids=tuple(
                (room_id, case_id, prompt_sha)
                for room_id, case_id, prompt_sha, _semantic, _correct, _points
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


def open_approved_objective_training_run(
    contract: ObjectiveContract, journey: JourneyState, *,
    provider_id: str | None = None,
    forbidden_semantic_sha256: tuple[str, ...] | frozenset[str] = (),
) -> ObjectiveTrainingRun | CognitiveObjectiveTrainingRun:
    """Explicitly route only an allowlisted objective to its own TRAIN rubric.

    No arbitrary provider, tool import, learner string, or generic goal parser
    can create a new execution lane. Host-owned admission remains separate.
    """
    if type(contract) is not ObjectiveContract or type(journey) is not JourneyState:
        raise ValueError("exact objective contract and journey required")
    ObjectiveContract.__post_init__(contract)
    if contract.scope.skills == ("cognitive_empathy",):
        return CognitiveObjectiveTrainingRun(
            contract, journey, provider_id=(PROVIDER_ID if provider_id is None
                                            else provider_id),
            forbidden_semantic_sha256=forbidden_semantic_sha256,
        )
    if forbidden_semantic_sha256:
        raise ValueError("semantic cognitive exclusions only apply to cognitive TRAIN")
    return ObjectiveTrainingRun(contract, journey, provider_id=provider_id)
