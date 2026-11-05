"""TRAIN-only live corridor execution with a persistent, bounded Minotaur taskmasker.

This narrow arithmetic integration closes the gap between *suggesting* a tactic
and actually presenting it to a learner. Each subsequent chamber is generated
after feedback from the preceding chamber, so the policy and the adaptive
corridor can alter the next prompt. A trusted in-process arithmetic generator
owns training gold; the learner callback receives a prompt-only view.

This is NOT the examiner, a neural Minotaur model, a neural learner trainer,
independent evaluation, or graduation. Scores and model-checkpoint digests are
operator-side training receipts, not attestations of parameter improvement.
The caller must additionally seal *all* previously exposed examination case
and prompt commitments before regenerating a maze. Python callbacks are
operator-trusted code, not sandboxed model-generated instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

from .adaptive_corridors import (
    AdaptiveWalkState, TrainingRoomFeedback, begin_adaptive_walk,
    begin_adaptive_walk_from_journey, offer_next_chamber, record_training_feedback,
)
from .journey_challenges import TraineeRoomView
from .journey import JourneyState, TrainingCompletionReceipt
from .minotaur_model import RoomContext, RoomPlan, plan_training_room
from .minotaur_policy_learning import (
    MinotaurTrainingPolicy, TrainCaseApproval, TrainRoomOutcome,
)
from .regenerative_maze import MazePlan
from .taskmaster_routing import RoutedTrainingRoom, _arithmetic_challenge
from .training_scope import TrainingScopeSpec


_ANSWER = re.compile(r"ANSWER: ([+-]?\d+)\Z", re.ASCII)


@dataclass(frozen=True)
class ObservedTrainingChamber:
    """Audit receipt with no answer key, response content, or examiner data."""

    room_id: str
    case_id: str
    skill: str
    tactic: str
    difficulty: int
    prompt_sha256: str
    correct: bool
    policy_before_sha256: str
    policy_after_sha256: str
    checkpoint_after_sha256: str


class TrainingCorridorRunner:
    """A finite, operator-run training corridor; never an autonomous examiner.

    To avoid policy steering an answer key, the taskmaster receives only public
    RoomContext metadata. The generator creates the challenge, then the trusted
    runner admits its exact TRAIN identity and prompt hash *before* collecting
    the learner response. There is no channel for exam labels into the policy.
    """

    def __init__(
        self, state: AdaptiveWalkState, *,
        policy: MinotaurTrainingPolicy | None = None,
        admissions: tuple[TrainCaseApproval, ...] = (),
        journey_provenance: JourneyState | None = None,
    ) -> None:
        # The supplied state is revalidated by these public interfaces, not
        # trusted solely because it is a frozen dataclass.
        _ = state.remaining_count
        if state.plan.cycle_index > 0:
            if (type(journey_provenance) is not JourneyState
                    or begin_adaptive_walk_from_journey(journey_provenance) != state):
                raise ValueError("regenerated runner requires an exact validated JourneyState binding")
        elif journey_provenance is not None:
            raise ValueError("initial corridor does not accept exam provenance")
        if any(skill != "arithmetic" for skill in state.scope.skills):
            raise ValueError("live taskmaster corridor currently supports arithmetic only")
        if policy is not None:
            if type(policy) is not MinotaurTrainingPolicy:
                raise ValueError("expected exact trainer-owned taskmasker policy")
            if (policy.scope != state.scope
                    or type(admissions) is not tuple
                    or tuple(policy._approved_cases) != admissions):
                raise ValueError("taskmaster and external TRAIN admission ledger disagree")
            _ = policy.model_hash
        elif admissions:
            raise ValueError("TRAIN approvals without an existing policy")
        self.state = state
        self._policy = policy
        self._admissions = admissions
        self._in_flight: RoutedTrainingRoom | None = None
        self._observations: list[ObservedTrainingChamber] = []

    @classmethod
    def begin(
        cls, scope: TrainingScopeSpec, plan: MazePlan, *,
        checkpoint_sha256: str,
    ) -> "TrainingCorridorRunner":
        """Start TRAIN cycle zero; later cycles require bound JourneyState."""
        if type(plan) is not MazePlan or plan.cycle_index != 0:
            raise ValueError("regenerated cycles require validated JourneyState custody")
        state = begin_adaptive_walk(
            scope, plan, checkpoint_sha256=checkpoint_sha256,
        )
        return cls(state)

    @property
    def policy(self) -> MinotaurTrainingPolicy | None:
        """Trainer-owned mutable policy for external write-once save_policy()."""
        return self._policy

    @property
    def training_registry(self) -> tuple[TrainCaseApproval, ...]:
        """Externally retain this manifest for load_policy() after restart."""
        return self._admissions

    @property
    def observations(self) -> tuple[ObservedTrainingChamber, ...]:
        return tuple(self._observations)

    @property
    def is_complete(self) -> bool:
        return self.state.pending is None and self.state.remaining_count == 0

    def next_training_prompt(self) -> TraineeRoomView:
        """Materialize exactly one next chamber using *current* policy state.

        The returned view is the sole learner-facing object; never return a
        Challenge, RoutedTrainingRoom, policy registry or trainer gold.
        """
        if self._in_flight is not None or self.state.pending is not None:
            raise ValueError("complete the pending TRAIN chamber first")
        assigned_state = offer_next_chamber(self.state)
        assigned = assigned_state.pending
        if assigned is None:  # Defensive; offer_next_chamber always creates it.
            raise ValueError("no eligible training chamber")
        context = RoomContext(
            objective_id=self.state.scope.objective_id,
            domain=self.state.scope.domain,
            skill=assigned.skill,
            room_id=assigned.room_id,
            cycle_index=self.state.plan.cycle_index,
            case_seed=assigned.case_seed,
            specificity=self.state.scope.specificity,
            scale=self.state.scope.scale,
        )
        if self._policy is None:
            # A policy needs at least one source-generated, preapproved TRAIN
            # case to exist. The first chamber bootstraps ordinary practice.
            selection = RoomPlan(
                assigned.room_id, assigned.difficulty, "practice",
                "initial_train_bootstrap", None,
            )
        else:
            proposed = plan_training_room(context, taskmaster=self._policy)
            if proposed.source != "untrusted_model_suggestion":
                raise ValueError("persistent taskmaster returned an invalid suggestion")
            # The per-skill corridor measures difficulty against specificity;
            # the taskmaster adjusts on its own histogram. Honor *both* with
            # a bounded additive modifier, then actually generate the prompt.
            resolved = max(1, min(5, proposed.difficulty
                                   + assigned.difficulty - self.state.scope.specificity))
            selection = RoomPlan(
                assigned.room_id, resolved, proposed.tactic,
                "policy_plus_adaptive_train", None,
            )
        challenge = _arithmetic_challenge(assigned.room, self.state.scope, selection)
        routed = RoutedTrainingRoom(assigned.room_id, assigned.skill, selection, challenge)
        view = routed.trainee_view
        admission = TrainCaseApproval(
            room_id=assigned.room_id,
            case_id=challenge.id,
            skill=assigned.skill,
            tactic=selection.tactic,
            difficulty=selection.difficulty,
            prompt_sha256=view.prompt_sha256,
        )
        if (admission.case_id in self.state.forbidden_case_ids
                or admission.prompt_sha256 in self.state.forbidden_prompt_sha256
                or any(admission.case_id == previous.case_id
                       or admission.prompt_sha256 == previous.prompt_sha256
                       for previous in self._admissions)):
            raise ValueError("TRAIN provider generated reused/sealed evidence")

        if self._policy is None:
            policy = MinotaurTrainingPolicy(
                scope=self.state.scope,
                approved_training_cases=(admission,),
            )
        else:
            policy = self._policy
            # Registry admission itself changes model_hash, but is **not** a
            # learned observation; only submit_training_response may append
            # an ObservedTrainingChamber after scoring approved TRAIN data.
            policy.extend_training_registry((admission,))
        self._policy = policy
        self._admissions += (admission,)
        self._in_flight = routed
        self.state = assigned_state
        return view

    def submit_training_response(
        self, response: str, *, checkpoint_after_sha256: str | None = None,
    ) -> ObservedTrainingChamber:
        """Score TRAIN gold locally, then update corridor and routing policy.

        A checkpoint digest is **caller-reported**, not an attestation of any
        neural update. Omitting it retains the previous digest. No graduation
        or exam status is asserted from this observation.
        """
        routed = self._in_flight
        if routed is None or self.state.pending is None or self._policy is None:
            raise ValueError("no pending TRAIN prompt")
        if type(response) is not str or len(response) > 4096:
            raise ValueError("learner response must be bounded text")
        match = _ANSWER.fullmatch(response.strip())
        correct = match is not None and str(int(match.group(1))) == routed.challenge.answer
        view = routed.trainee_view
        checkpoint = (self.state.checkpoint_sha256 if checkpoint_after_sha256 is None
                      else checkpoint_after_sha256)
        receipt = TrainingRoomFeedback(
            room_id=routed.room_id,
            case_id=routed.challenge.id,
            prompt_sha256=view.prompt_sha256,
            score=int(correct), split="train",
            checkpoint_before_sha256=self.state.checkpoint_sha256,
            checkpoint_after_sha256=checkpoint,
        )
        next_state = record_training_feedback(self.state, receipt)
        outcome = TrainRoomOutcome(
            objective_id=self.state.scope.objective_id,
            room_id=routed.room_id, case_id=routed.challenge.id,
            skill=routed.skill, tactic=routed.suggestion.tactic,
            difficulty=routed.suggestion.difficulty,
            prompt_sha256=view.prompt_sha256,
            was_correct=correct,
        )
        # The policy itself checks exact TRAIN registry admission. Failure
        # cannot advance the corridor because state is assigned afterward.
        before = self._policy.model_hash
        update = self._policy.observe(outcome)
        observed = ObservedTrainingChamber(
            room_id=routed.room_id, case_id=routed.challenge.id,
            skill=routed.skill, tactic=routed.suggestion.tactic,
            difficulty=routed.suggestion.difficulty,
            prompt_sha256=view.prompt_sha256,
            correct=correct,
            policy_before_sha256=before,
            policy_after_sha256=update["after_hash"],
            checkpoint_after_sha256=next_state.checkpoint_sha256,
        )
        self.state = next_state
        self._in_flight = None
        self._observations.append(observed)
        return observed

    def run_training_chamber(
        self, learner: Callable[[TraineeRoomView], str], *,
        checkpoint_after_sha256: str | None = None,
    ) -> ObservedTrainingChamber:
        """Pass only a prompt-only view to operator-approved learner code."""
        if not callable(learner):
            raise ValueError("learner must be an operator-approved callable")
        view = self.next_training_prompt()
        response = learner(view)
        return self.submit_training_response(
            response, checkpoint_after_sha256=checkpoint_after_sha256,
        )

    def self_reported_completion_receipt(self) -> TrainingCompletionReceipt:
        """Map adaptive TRAIN visits to original maze rooms for JourneyState.

        This self-reported receipt alone cannot establish changed neural
        weights. The host must independently inspect the actual adapter's
        checkpoint before using it to infer parameter adaptation.
        """
        if not self.is_complete:
            raise ValueError("cannot finish a maze with unvisited TRAIN chambers")
        if self.state.checkpoint_sha256 == self.state.initial_checkpoint_sha256:
            raise ValueError("no changed checkpoint digest was reported")
        return TrainingCompletionReceipt(
            objective_id=self.state.scope.objective_id,
            cycle_index=self.state.plan.cycle_index,
            checkpoint_before_sha256=self.state.initial_checkpoint_sha256,
            checkpoint_after_sha256=self.state.checkpoint_sha256,
            room_case_ids=tuple(
                (event.assignment.origin_room_id, event.feedback.case_id,
                 event.feedback.prompt_sha256)
                for event in self.state.history
            ),
        )

    def regenerate(
        self, journey: JourneyState,
    ) -> "TrainingCorridorRunner":
        """Start next TRAIN maze while keeping learned taskmaster statistics.

        The validated prior JourneyState (from an operator-controlled
        development exam) must already seal prior TRAIN and exam identities
        and prompt hashes. Its receipts remain source-level, not independently
        authenticated. This code does not conduct an exam, import its content,
        grant graduation, or allow arbitrary caller-supplied seal tuples.
        """
        if not self.is_complete or self._policy is None:
            raise ValueError("finish all authorized TRAIN chambers before regeneration")
        if type(journey) is not JourneyState:
            raise ValueError("regeneration requires exact validated JourneyState")
        next_state = begin_adaptive_walk_from_journey(journey)
        if next_state.plan.cycle_index != self.state.plan.cycle_index + 1:
            raise ValueError("regeneration must advance exactly one maze cycle")
        if (next_state.scope != self.state.scope
                or next_state.checkpoint_sha256 != self.state.checkpoint_sha256):
            raise ValueError("exam handoff changed declared scope or retained checkpoint")
        retained_ids = set(next_state.forbidden_case_ids)
        retained_prompts = set(next_state.forbidden_prompt_sha256)
        if any(event.feedback.case_id not in retained_ids
               or event.feedback.prompt_sha256 not in retained_prompts
               for event in self.state.history):
            raise ValueError("journey handoff omitted prior training exposure")
        if (not set(self.state.forbidden_case_ids).issubset(retained_ids)
                or not set(self.state.forbidden_prompt_sha256).issubset(retained_prompts)):
            raise ValueError("journey handoff lost sealed historical evidence")
        return TrainingCorridorRunner(
            next_state, policy=self._policy, admissions=self._admissions,
            journey_provenance=journey,
        )
