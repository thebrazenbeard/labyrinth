"""Bounded model-chosen actions inside the developmental Labyrinth.

The policy sees public story observations and legal actions only. It may walk
and elect TRAIN/Minotaur confrontations; it cannot write supervision, examiner
gold, verdicts or checkpoint identities. This is an interface for a real model
action policy, not evidence that such a policy was run or that neural learning
occurred.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Protocol

from .model_saga_runtime import SagaSession, StepResult


_ACTION_LINE = re.compile(r"ACTION: (look|left|right|forward|back|train|face)\Z", re.ASCII)
_PUBLIC_HISTORY_CAP = 8


@dataclass(frozen=True)
class TheseusDecisionContext:
    """Only public, bounded details; never TRAIN answer or exam material."""

    objective_id: str
    maze_cycle_index: int
    decision_index: int
    observation: str
    available_actions: tuple[str, ...]
    recent_events: tuple[str, ...]


class TheseusActionPolicy(Protocol):
    def choose_action(self, context: TheseusDecisionContext) -> str: ...


@dataclass(frozen=True)
class TheseusActionReceipt:
    decision_index: int
    action: str
    observation_sha256: str
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str
    effect_event: str
    maze_cycle_before: int
    maze_cycle_after: int


@dataclass(frozen=True)
class TheseusEpisodeReceipt:
    """One local, self-attested run; not a private qualification receipt."""

    status: str
    development_status: str
    decisions: tuple[TheseusActionReceipt, ...]
    final_checkpoint_sha256: str
    maze_cycle_index: int
    qualification: str = "NOT_ESTABLISHED"


def drive_theseus(
    session: SagaSession, policy: TheseusActionPolicy, *, max_decisions: int = 128,
) -> TheseusEpisodeReceipt:
    """Ask an untrusted model policy to drive actual source-guarded steps.

    Invalid policy output is a non-effectful stop. If a step/trainer raises,
    let the exception propagate: a partial external TRAIN write may have
    occurred, so automatic retry or a clean success receipt would be unsafe.
    No implicit fallback chooses the route for the model.
    """
    if not isinstance(session, SagaSession):
        raise ValueError("requires a real SagaSession, not an arbitrary grade stub")
    if not callable(getattr(policy, "choose_action", None)):
        raise ValueError("a model action policy is required")
    if type(max_decisions) is not int or not 1 <= max_decisions <= 512:
        raise ValueError("bounded model action budget must be in [1, 512]")

    receipts: list[TheseusActionReceipt] = []
    recent: list[str] = []
    status = "HALTED_DECISION_BUDGET"
    for index in range(max_decisions):
        observation: StepResult = session.observe()
        if observation.terminated:
            status = "DEVELOPMENT_EPISODE_ENDED"
            break
        if observation.truncated:
            status = "HALTED_SOURCE_ACTION_BUDGET"
            break
        if not observation.available_actions:
            status = "HALTED_INDETERMINATE"
            break
        context = TheseusDecisionContext(
            objective_id=session.scope.objective_id,
            maze_cycle_index=session.journey.cycle_index,
            decision_index=index,
            observation=observation.observation,
            available_actions=observation.available_actions,
            recent_events=tuple(recent[-_PUBLIC_HISTORY_CAP:]),
        )
        response = policy.choose_action(context)
        if type(response) is not str or len(response) > 64:
            status = "HALTED_INVALID_POLICY_OUTPUT"
            break
        match = _ACTION_LINE.fullmatch(response)
        if match is None or match.group(1) not in observation.available_actions:
            status = "HALTED_INVALID_POLICY_OUTPUT"
            break
        action = match.group(1)
        cycle_before = session.journey.cycle_index
        before_sha = observation.checkpoint_sha256
        result = session.step(action)
        receipts.append(TheseusActionReceipt(
            decision_index=index,
            action=action,
            observation_sha256=hashlib.sha256(observation.observation.encode("utf-8")).hexdigest(),
            checkpoint_before_sha256=before_sha,
            checkpoint_after_sha256=result.checkpoint_sha256,
            effect_event=result.event,
            maze_cycle_before=cycle_before,
            maze_cycle_after=session.journey.cycle_index,
        ))
        recent.append(result.event)
        if result.terminated:
            status = "DEVELOPMENT_EPISODE_ENDED"
            break
        if result.truncated:
            status = "HALTED_SOURCE_ACTION_BUDGET"
            break
    return TheseusEpisodeReceipt(
        status=status,
        development_status=session.development_status,
        decisions=tuple(receipts),
        final_checkpoint_sha256=session.checkpoint.weights_sha256,
        maze_cycle_index=session.journey.cycle_index,
    )


__all__ = [
    "TheseusDecisionContext", "TheseusActionPolicy", "TheseusActionReceipt",
    "TheseusEpisodeReceipt", "drive_theseus",
]
