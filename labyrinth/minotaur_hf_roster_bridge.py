"""Two-stage source-only bridge from adaptive Minotaur TRAIN to HF/LoRA SGD.

The room selection is feedback-conditioned: a complete HF minibatch schedule
cannot be committed until this source-controlled TRAIN corridor has generated
its cases. This bridge **first** collects TRAIN-only cases while a pre-existing
learner responds to prompts, **then** pins an ordered roster for HF `_train_steps`.
Consequently it is not online neural-driven curriculum adaptation, an actual
HF `run_pilot` entrypoint, a host precommit, a physical optimizer witness, or
evidence of transfer/graduation. Only operator-trusted code receives gold.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable

from .hf_pilot import PilotConfig, _training_visit_exposures
from .journey_challenges import TraineeRoomView
from .journey_policy_runner import TrainingCorridorRunner
from .maze import Challenge


@dataclass(frozen=True)
class MinotaurHfTrainRoster:
    """Operator-private catalog usable by `_train_steps` after full staging."""

    scope_sha256: str
    cycle_index: int
    train_cases: tuple[Challenge, ...]
    ordered_optimizer_exposures: tuple[tuple[str, str], ...]

    @property
    def physically_authenticated(self) -> bool:
        """Source-only roster can never claim independent physical custody."""
        return False

    @property
    def graduation_authorized(self) -> bool:
        """Materializing TRAIN is not an authorization to graduate."""
        return False


def collect_minotaur_hf_roster(
    runner: TrainingCorridorRunner, config: PilotConfig,
    learner: Callable[[TraineeRoomView], str],
) -> MinotaurHfTrainRoster:
    """Stage exact approved TRAIN material and pin HF optimizer exposure order.

    The caller must own a fresh, single-worker runner and a trusted learner
    callback. This function advances the runner, so an exception after chamber
    creation aborts the stage and the caller must not reuse its partial roster.
    The returned private cases may be fed to `_train_steps` with
    `expected_visit_exposures=ordered_optimizer_exposures`. `run_pilot` does not
    yet accept this catalog and no external host controls are created here.
    """
    if type(runner) is not TrainingCorridorRunner or type(config) is not PilotConfig:
        raise ValueError("exact runner and HF PilotConfig required")
    if not callable(learner):
        raise ValueError("trusted operator learner callback required")
    if (runner.observations or runner.state.pending is not None
            or runner._in_flight is not None or runner.is_complete):
        raise ValueError("fresh TRAIN corridor required for HF roster staging")
    scope = runner.state.scope
    if (scope.skills != ("arithmetic",) or config.tiers != (1,)
            or type(config.seed) is not int or not 0 <= config.seed < 2**31
            or type(config.steps) is not int or not 1 <= config.steps <= 128
            or type(config.train_count) is not int
            or config.train_count != runner.state.remaining_count
            or not 2 <= config.train_count <= 64):
        raise ValueError("HF TRAIN roster requires matching bounded arithmetic scope and budget")
    cases: list[Challenge] = []
    for _ in range(config.train_count):
        view = runner.next_training_prompt()
        case = runner.operator_pending_train_case(view)
        # Only this prompt-only view goes to the learner, never the Challenge.
        response = learner(view)
        outcome = runner.submit_training_response(response)
        digest = hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()
        if (type(case) is not Challenge or case.split != "train"
                or outcome.case_id != case.id or outcome.prompt_sha256 != digest
                or view.challenge_id != case.id or view.prompt_sha256 != digest):
            raise RuntimeError("adaptive TRAIN case differs from scored admission")
        cases.append(case)
    if (not runner.is_complete or len(runner.observations) != len(cases)
            or len({case.id for case in cases}) != len(cases)
            or len({case.prompt for case in cases}) != len(cases)):
        raise ValueError("staged TRAIN catalog incomplete or duplicated")
    registered = {approval.case_id: approval for approval in runner.training_registry}
    for case in cases:
        admission = registered.get(case.id)
        if (admission is None or admission.split != "train"
                or admission.prompt_sha256
                != hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()):
            raise ValueError("TRAIN catalog is not approved by the current registry")
    return MinotaurHfTrainRoster(
        scope_sha256=scope.fingerprint,
        cycle_index=runner.state.plan.cycle_index,
        train_cases=tuple(cases),
        ordered_optimizer_exposures=_training_visit_exposures(
            cases, seed=config.seed, steps=config.steps,
        ),
    )
