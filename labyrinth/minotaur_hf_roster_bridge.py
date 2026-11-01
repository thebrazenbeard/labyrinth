"""Two-stage source-only bridge from adaptive Minotaur TRAIN to HF SGD.

Feedback from a prompt-only learner affects the next TRAIN chamber during
staging. Only after all TRAIN cases are collected is their optimizer visit
order pinned for an HF run. No neural weights are updated by staging, and the
roster is neither independent exposure custody nor graduation authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
from typing import Callable

from .hf_pilot import PilotConfig, _training_visit_exposures
from .journey_challenges import TraineeRoomView
from .journey_policy_runner import TrainingCorridorRunner, _ANSWER
from .maze import Challenge


@dataclass(frozen=True)
class MinotaurHfTrainRoster:
    """Trainer-private staged TRAIN catalog and exact optimizer visit roster."""

    scope_sha256: str
    cycle_index: int
    train_cases: tuple[Challenge, ...] = field(repr=False)
    ordered_optimizer_exposures: tuple[tuple[str, str], ...]

    @property
    def physically_authenticated(self) -> bool:
        return False

    @property
    def graduation_authorized(self) -> bool:
        return False


def _primitive_train_snapshot(case: Challenge) -> Challenge:
    """Detach the complete trusted case as immutable primitive supervision.

    A frozen dataclass alone is not custody: same-process code can rewrite its
    attributes with ``object.__setattr__``. In particular, copying only the
    case ID and public prompt hash would miss changed historical gold.
    """
    if (type(case) is not Challenge
            or any(type(getattr(case, name)) is not str
                   for name in ("id", "kind", "split", "prompt", "answer"))
            or type(case.tier) is not int
            or type(case.metadata) is not tuple
            or any(type(pair) is not tuple or len(pair) != 2
                   or any(type(part) is not str for part in pair)
                   for pair in case.metadata)):
        raise RuntimeError("mutated TRAIN material has nonprimitive fields")
    return Challenge(
        id=case.id, kind=case.kind, tier=case.tier, split=case.split,
        prompt=case.prompt, answer=case.answer,
        metadata=tuple((key, value) for key, value in case.metadata),
    )


def _assert_historical_cases_unchanged(
    live_cases: list[Challenge], snapshots: list[Challenge],
) -> None:
    for live, snapshot in zip(live_cases, snapshots, strict=True):
        try:
            unchanged = _primitive_train_snapshot(live) == snapshot
        except RuntimeError as exc:
            raise RuntimeError("historical TRAIN material changed after staging") from exc
        if not unchanged:
            raise RuntimeError("historical TRAIN material changed after staging")


def _bounded_int(value: object, minimum: int, maximum: int) -> bool:
    return type(value) is int and minimum <= value <= maximum


def collect_minotaur_hf_roster(
    runner: TrainingCorridorRunner,
    config: PilotConfig,
    learner: Callable[[TraineeRoomView], str],
) -> MinotaurHfTrainRoster:
    """Stage exact approved TRAIN cases, then pin their HF visit order.

    The caller owns a fresh single-worker runner and a trusted callback. If a
    callback or admission fails after materializing a chamber, discard the
    partial stage; never replay or treat its incomplete roster as training.
    The private cases can feed ``_train_steps`` with the returned ordered
    visits, but ordinary ``run_pilot`` does not yet consume this catalog.
    """
    if type(runner) is not TrainingCorridorRunner or type(config) is not PilotConfig:
        raise ValueError("exact runner and HF PilotConfig required")
    if not callable(learner):
        raise ValueError("trusted operator learner callback required")
    if (runner.observations or runner.state.pending is not None
            or runner._in_flight is not None or runner.is_complete):
        raise ValueError("fresh TRAIN corridor required for HF roster staging")
    scope = runner.state.scope
    # Match HF's scalar resource contract without inspecting model/output
    # paths or importing a model; staging is a source-only operation.
    if (scope.skills != ("arithmetic",) or type(config.tiers) is not tuple
            or config.tiers != (1,)
            or not _bounded_int(config.seed, 0, 2**31 - 1)
            or not _bounded_int(config.steps, 1, 128)
            or not _bounded_int(config.train_count, 2, 32)
            or config.train_count != runner.state.remaining_count
            or not _bounded_int(config.eval_count, 2, 32)
            or not _bounded_int(config.max_tokens, 32, 1024)
            or not _bounded_int(config.max_new_tokens, 1, 32)
            or not _bounded_int(config.lora_rank, 1, 16)
            or not _bounded_int(config.lora_alpha, 1, 64)
            or type(config.learning_rate) is not float
            or not math.isfinite(config.learning_rate)
            or not 1e-7 <= config.learning_rate <= 0.001
            or type(config.device) is not str
            or config.device not in ("cpu", "cuda")
            or type(config.answer_only) is not bool):
        raise ValueError("HF TRAIN roster requires matching bounded arithmetic scope and budget")

    cases: list[Challenge] = []
    snapshots: list[Challenge] = []
    for _ in range(config.train_count):
        view = runner.next_training_prompt()
        case = runner.operator_pending_train_case(view)
        # Python frozen dataclasses can still be rewritten using
        # object.__setattr__. Keep detached pre-callback values and refuse a
        # mutated public view or trainer-private case before scoring feedback.
        original_view = replace(view)
        original_case = _primitive_train_snapshot(case)
        routed_before = runner._in_flight
        pending_before = runner.state.pending
        if routed_before is None or pending_before is None:
            raise RuntimeError("pending TRAIN scoring material is missing")
        routed_snapshot = replace(
            routed_before, suggestion=replace(routed_before.suggestion),
            challenge=original_case,
        )
        pending_snapshot = replace(pending_before)
        state_before = runner.state
        response = learner(view)  # The learner sees no Challenge or answer.
        if view != original_view or _primitive_train_snapshot(case) != original_case:
            raise RuntimeError("learner callback mutated TRAIN material")
        _assert_historical_cases_unchanged(cases, snapshots)
        if (runner._in_flight is not routed_before
                or routed_before.challenge is not case
                or routed_before != routed_snapshot
                or runner.state is not state_before
                or runner.state.pending is not pending_before
                or pending_before != pending_snapshot):
            raise RuntimeError("pending TRAIN scoring material changed during callback")
        outcome = runner.submit_training_response(response)
        match = _ANSWER.fullmatch(response.strip())
        expected_correct = (match is not None
                            and str(int(match.group(1))) == original_case.answer)
        prompt_sha256 = hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()
        if (type(case) is not Challenge or case.split != "train"
                or outcome.case_id != case.id
                or outcome.prompt_sha256 != prompt_sha256
                or view.challenge_id != case.id
                or view.prompt_sha256 != prompt_sha256):
            raise RuntimeError("adaptive TRAIN case differs from scored admission")
        if outcome.correct != expected_correct:
            raise RuntimeError("scored TRAIN feedback differs from pinned supervision")
        cases.append(case)
        snapshots.append(original_case)

    _assert_historical_cases_unchanged(cases, snapshots)
    if (not runner.is_complete or len(runner.observations) != len(cases)
            or len({case.id for case in cases}) != len(cases)
            or len({case.prompt for case in cases}) != len(cases)):
        raise ValueError("staged TRAIN catalog incomplete or duplicated")
    registered = {entry.case_id: entry for entry in runner.training_registry}
    for case in snapshots:
        approval = registered.get(case.id)
        if (approval is None or approval.split != "train"
                or approval.prompt_sha256
                != hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()):
            raise ValueError("TRAIN catalog is not approved by the current registry")

    return MinotaurHfTrainRoster(
        scope_sha256=scope.fingerprint,
        cycle_index=runner.state.plan.cycle_index,
        train_cases=tuple(snapshots),
        ordered_optimizer_exposures=_training_visit_exposures(
            snapshots, seed=config.seed, steps=config.steps,
        ),
    )
