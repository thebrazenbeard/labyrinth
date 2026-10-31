"""Run an adaptive Minotaur corridor with *measured* toy parameter updates.

The TRAIN generator retains the gold; only prompt-only views reach inference.
After all approved practice chambers have been scored against the unchanged
pre-training checkpoint, the trusted operator performs a real four-parameter
optimizer pass and hashes the learner directly before/after. Unscored registry
admissions are recorded separately from scored policy updates.
There is no pretrained/HF/LoRA update or independent examiner in this path.
"""

from __future__ import annotations

import hashlib
import math

from .journey_policy_runner import TrainingCorridorRunner
from .regenerative_maze import MazePlan, generate_maze_plan
from .toy_backend import ToyLinearAdapter
from .training_scope import TrainingScopeSpec


SCHEMA = "labyrinth.minotaur-measured-toy-corridor.v1"


def run_measured_toy_corridor(
    scope: TrainingScopeSpec, plan: MazePlan, learner: ToyLinearAdapter, *,
    epochs: int = 8, learning_rate: float = 0.5,
) -> dict:
    """Train a real toy learner against dynamically admitted TRAIN cases.

    This intentionally narrower path cannot adapt LoRA or claim learning
    improvement: only a *changed toy parameter hash* and optimizer report
    are established. Python implementation and trainer are operator-trusted.
    """
    if type(scope) is not TrainingScopeSpec or type(plan) is not MazePlan:
        raise ValueError("exact source scope and maze plan required")
    TrainingScopeSpec.__post_init__(scope)
    MazePlan.__post_init__(plan)
    if (scope.skills != ("arithmetic",) or scope.total_chambers < 2
            or plan != generate_maze_plan(scope, 0)):
        raise ValueError("toy learner needs an exact initial two-or-more-room arithmetic maze")
    if type(learner) is not ToyLinearAdapter:
        raise ValueError("expected exact operator-owned ToyLinearAdapter")
    if type(epochs) is not int or not 1 <= epochs <= 1000:
        raise ValueError("epochs must be in [1,1000]")
    if (type(learning_rate) not in (int, float)
            or not math.isfinite(learning_rate) or not 0 < learning_rate <= 1):
        raise ValueError("learning_rate must be finite and in (0,1]")

    baseline = learner.model_hash
    runner = TrainingCorridorRunner.begin(scope, plan, checkpoint_sha256=baseline)
    approved = []
    for _ in range(scope.total_chambers):
        trainee_view = runner.next_training_prompt()
        routed = runner._in_flight  # Operator-side private TRAIN gold; never export it.
        if (routed is None or routed.challenge.split != "train"
                or routed.challenge.kind != "arithmetic"
                or routed.suggestion.tactic != "practice"
                or routed.challenge.tier != 1
                or routed.challenge.id != trainee_view.challenge_id
                or hashlib.sha256(routed.challenge.prompt.encode("utf-8")).hexdigest()
                != trainee_view.prompt_sha256):
            # Fail before touching toy parameters if dynamically chosen tactics
            # are outside this backend's deliberately narrow grammar.
            raise ValueError("toy backend cannot train the selected Minotaur tactic")
        approved.append(routed.challenge)
        response = learner.generate(trainee_view.prompt)
        runner.submit_training_response(response)

    observations = runner.observations
    transitions = runner.registry_transitions
    if (not runner.is_complete or runner.state.checkpoint_sha256 != baseline
            or len(runner.observations) != scope.total_chambers):
        raise RuntimeError("toy corridor did not finish against its pre-training checkpoint")
    if len(transitions) != len(observations) - 1:
        raise RuntimeError("missing non-learning registry transitions")
    for index, transition in enumerate(transitions):
        if (transition.before_policy_sha256 != observations[index].policy_after_sha256
                or transition.after_policy_sha256 != observations[index + 1].policy_before_sha256
                or transition.added_case_ids != (observations[index + 1].case_id,)
                or transition.train_observation is not False
                or transition.authorizes_model_update is not False):
            raise RuntimeError("Minotaur policy lineage has an unaccounted transition")
    if learner.model_hash != baseline:
        raise RuntimeError("toy learner was changed outside admitted TRAIN sequence")
    train_result = learner.train(approved, epochs=epochs,
                                 learning_rate=float(learning_rate))
    after_train = learner.model_hash
    if (train_result.get("before_model_hash") != baseline
            or train_result.get("after_model_hash") != after_train
            or type(train_result.get("parameter_updates")) is not int
            or train_result["parameter_updates"] <= 0
            or after_train == baseline):
        raise RuntimeError("no consistent measured parameter update")
    return {
        "schema": SCHEMA,
        "objective_id": scope.objective_id,
        "scope_fingerprint": scope.fingerprint,
        "checkpoint_before_sha256": baseline,
        "checkpoint_after_sha256": after_train,
        "parameter_updates": train_result["parameter_updates"],
        "optimizer_event": {
            "phase": "after_train_corridor",
            "checkpoint_before_sha256": baseline,
            "checkpoint_after_sha256": after_train,
            "parameter_updates": train_result["parameter_updates"],
        },
        "training_case_ids": [case.id for case in approved],
        "training_prompt_sha256": [hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()
                                  for case in approved],
        "registry_transitions": [{
            "transition_kind": transition.transition_kind,
            "train_observation": transition.train_observation,
            "authorizes_model_update": transition.authorizes_model_update,
            "before_policy_sha256": transition.before_policy_sha256,
            "after_policy_sha256": transition.after_policy_sha256,
            "before_registry_sha256": transition.before_registry_sha256,
            "after_registry_sha256": transition.after_registry_sha256,
            "added_case_ids": list(transition.added_case_ids),
        } for transition in transitions],
        "observations": [{
            "room_id": observation.room_id,
            "case_id": observation.case_id,
            "prompt_sha256": observation.prompt_sha256,
            "tactic": observation.tactic,
            "difficulty": observation.difficulty,
            "policy_before_sha256": observation.policy_before_sha256,
            "policy_after_sha256": observation.policy_after_sha256,
            "checkpoint_after_sha256": observation.checkpoint_after_sha256,
            "correct": observation.correct,
        } for observation in observations],
        "source_status": "MEASURED_TOY_PARAMETER_UPDATE_ONLY",
        "pretrained_improvement_established": False,
        "graduation_authorized": False,
        "limitations": [
            "Toy four-parameter arithmetic update, not pretrained/HF/LoRA adaptation.",
            "All TRAIN responses are scored internally; no independent evaluation.",
            "Optimizer was run after the bounded corridor, not after every chamber.",
            "Every TRAIN score was generated before optimizer updates using the initial checkpoint.",
            "Registry admissions are unscored policy transitions, not parameter learning.",
            "Parameter hash readback proves local value change, not generalization or custody.",
        ],
    }
