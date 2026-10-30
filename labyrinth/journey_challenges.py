"""Trusted trainer-side materialization of regenerative training rooms.

Room descriptions have no answer key. This module is the boundary where an
approved skill provider generates TRAIN cases and retains private supervision.
Only the prompt-only view is suitable for a trainee inference adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Mapping

from .maze import Challenge, generate_maze
from .regenerative_maze import MazePlan


_BUILTIN_TIERS = {
    "arithmetic": 1,
    "operation_chain": 2,
    "source_priority": 3,
    "program_reasoning": 4,
}


@dataclass(frozen=True)
class TraineeRoomView:
    room_id: str
    skill: str
    challenge_id: str
    prompt: str
    prompt_sha256: str


@dataclass(frozen=True)
class PreparedTrainingRoom:
    """Holds evaluator-only gold; never pass this object to the trainee."""

    room_id: str
    skill: str
    challenge: Challenge

    @property
    def trainee_view(self) -> TraineeRoomView:
        return TraineeRoomView(
            room_id=self.room_id,
            skill=self.skill,
            challenge_id=self.challenge.id,
            prompt=self.challenge.prompt,
            prompt_sha256=hashlib.sha256(self.challenge.prompt.encode("utf-8")).hexdigest(),
        )


def prepare_training_rooms(
    plan: MazePlan,
    *,
    providers: Mapping[str, Callable[[int], Challenge]] | None = None,
) -> tuple[PreparedTrainingRoom, ...]:
    """Materialize only registered skills; unsupported goals fail closed.

    External skill providers are injected by a trusted trainer, never loaded
    from text in a learner-controlled scope declaration.
    """
    if not isinstance(plan, MazePlan):
        raise ValueError("expected validated MazePlan")
    # Even frozen dataclasses may have been mutated through object.__setattr__.
    plan.__post_init__()
    if providers is not None and not isinstance(providers, Mapping):
        raise ValueError("providers must be an explicit skill mapping")
    supplied = {} if providers is None else dict(providers)
    if any(not isinstance(key, str) or not key or not callable(value) for key, value in supplied.items()):
        raise ValueError("invalid skill provider")
    materialized: list[PreparedTrainingRoom] = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    for room in plan.rooms:
        if room.purpose != "train":
            raise ValueError("exam/holdout room cannot be materialized for training")
        if room.skill in supplied:
            challenge = supplied[room.skill](room.case_seed)
        elif room.skill in _BUILTIN_TIERS:
            challenge = generate_maze(
                seed=room.case_seed, tiers=(_BUILTIN_TIERS[room.skill],),
                count=1, split="train",
            )[0]
        else:
            raise ValueError(f"no approved training provider for skill {room.skill!r}")
        if (
            not isinstance(challenge, Challenge)
            or challenge.split != "train"
            or not isinstance(challenge.id, str)
            or not challenge.id
            or not isinstance(challenge.prompt, str)
            or not challenge.prompt
            or not isinstance(challenge.answer, str)
            or not challenge.answer
        ):
            raise ValueError("provider returned an invalid/non-training challenge")
        prompt_hash = hashlib.sha256(challenge.prompt.encode("utf-8")).hexdigest()
        if challenge.id in seen_ids or prompt_hash in seen_prompts:
            raise ValueError("duplicate challenge id or prompt in generated maze")
        seen_ids.add(challenge.id)
        seen_prompts.add(prompt_hash)
        materialized.append(PreparedTrainingRoom(room.room_id, room.skill, challenge))
    return tuple(materialized)
