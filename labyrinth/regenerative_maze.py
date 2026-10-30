"""Reproducible, scoped *training-room descriptors* for a regenerative maze.

Rooms are dispatch instructions to a separate skill-specific challenge provider,
not generated questions, answers, training effects, or graduation evidence.
This boundary prevents a maze planner from claiming to teach any domain merely
by assigning its name to a room.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Sequence

from .training_scope import TrainingScopeSpec


MAX_ROOMS = 128
_SEED_MODULUS = 2**31
_HEX32 = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


def _digest(*pieces: object) -> str:
    """Portable domain-separated digest for deterministic room identities."""
    return hashlib.sha256("\x1f".join(str(piece) for piece in pieces).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MazeRoom:
    """A unique, train-only location in one particular Labyrinth cycle."""

    room_id: str
    skill: str
    case_seed: int
    purpose: str = "train"

    def __post_init__(self) -> None:
        if type(self.room_id) is not str or _HEX32.fullmatch(self.room_id) is None:
            raise ValueError("room_id must be a 128-bit hexadecimal identity")
        if type(self.skill) is not str or not self.skill:
            raise ValueError("room skill must be nonempty")
        if type(self.case_seed) is not int or not 0 <= self.case_seed < _SEED_MODULUS:
            raise ValueError("room case_seed must be a signed-31-bit-safe integer")
        if self.purpose != "train":
            raise ValueError("maze room purpose must be train, never evaluation")


@dataclass(frozen=True)
class MazePlan:
    """A bounded immutable maze; no answer keys or exam prompts are held here."""

    objective_id: str
    scope_fingerprint: str
    cycle_index: int
    rooms: tuple[MazeRoom, ...]

    def __post_init__(self) -> None:
        if type(self.objective_id) is not str or not self.objective_id:
            raise ValueError("maze requires a declared objective")
        if type(self.scope_fingerprint) is not str or _HEX64.fullmatch(self.scope_fingerprint) is None:
            raise ValueError("maze requires a scope fingerprint")
        if type(self.cycle_index) is not int or not 0 <= self.cycle_index <= 7:
            raise ValueError("invalid maze cycle_index")
        if type(self.rooms) is not tuple or not 1 <= len(self.rooms) <= MAX_ROOMS:
            raise ValueError("maze room count out of bounds")
        for room in self.rooms:
            if not isinstance(room, MazeRoom):
                raise ValueError("maze rooms must be MazeRoom records")
            MazeRoom.__post_init__(room)
        if (len({room.room_id for room in self.rooms}) != len(self.rooms)
                or len({room.case_seed for room in self.rooms}) != len(self.rooms)):
            raise ValueError("maze contains duplicate room identities or seeds")

    @property
    def counts_by_skill(self) -> dict[str, int]:
        return {skill: sum(room.skill == skill for room in self.rooms)
                for skill in sorted({room.skill for room in self.rooms})}


def _reweighted_counts(skills: tuple[str, ...], base: int, weak: set[str]) -> dict[str, int]:
    counts = {skill: base for skill in skills}
    if not weak or len(weak) == len(skills) or base == 1:
        return counts

    # Shift bounded training capacity from better-performing skills to weaker
    # ones, while preserving every declared skill's representation. The budget
    # is unchanged; failure cannot silently create more training work.
    donors = sorted(set(skills) - weak)
    recipients = sorted(weak)
    max_shift = max(1, base // 2) * len(recipients)
    for turn in range(max_shift):
        recipient = recipients[turn % len(recipients)]
        available = [skill for skill in donors if counts[skill] > 1]
        if not available:
            break
        donor = sorted(available, key=lambda skill: (-counts[skill], skill))[0]
        counts[donor] -= 1
        counts[recipient] += 1
    return counts


def generate_maze_plan(
    scope: TrainingScopeSpec,
    cycle_index: int,
    *,
    weak_skills: Sequence[str] = (),
) -> MazePlan:
    """Build one new training maze from a predeclared scope and prior failures.

    A cycle's content seeds and IDs are never reused in another cycle of the
    same scope. The specification's seed is for reproducibility, **not** for
    cryptographic unpredictability. Exam questions must come from an isolated
    trusted evaluator and must never be installed as room content.
    """
    if not isinstance(scope, TrainingScopeSpec):
        raise ValueError("scope must be TrainingScopeSpec")
    # Frozen dataclasses can be mutated with object.__setattr__; revalidate
    # restored objects to prevent unbounded room counts or malformed IDs.
    TrainingScopeSpec.__post_init__(scope)
    if type(cycle_index) is not int or not 0 <= cycle_index < scope.max_cycles:
        raise ValueError("cycle_index must be within the scope's authorized cycles")
    if (
        not isinstance(weak_skills, Sequence)
        or isinstance(weak_skills, (str, bytes))
        or any(type(skill) is not str or not skill for skill in weak_skills)
        or len(set(weak_skills)) != len(weak_skills)
    ):
        raise ValueError("weak_skills must be a sequence of distinct nonempty skill IDs")
    unknown = set(weak_skills) - set(scope.skills)
    if unknown:
        raise ValueError("failure feedback references a skill outside the declared scope")

    base = scope.chamber_count_per_skill
    counts = _reweighted_counts(scope.skills, base, set(weak_skills))
    if sum(counts.values()) > MAX_ROOMS:
        raise ValueError("maze exceeds the bounded training-room budget")

    descriptors = [
        (skill, position)
        for skill in scope.skills
        for position in range(counts[skill])
    ]
    # Seeded order is stable across platforms and independent of global RNG.
    descriptors.sort(key=lambda pair: (_digest("ordering", scope.fingerprint, cycle_index, *pair), pair))
    seed_key = int(_digest("case-seed-key", scope.fingerprint)[:8], 16) % _SEED_MODULUS
    rooms = tuple(
        MazeRoom(
            room_id=_digest("maze-room", scope.fingerprint, cycle_index, skill, position)[:32],
            skill=skill,
            # Affine permutation modulo 2^31 (odd multiplier). Global
            # ordinal < 8*128, so no case_seed repeats across cycles.
            case_seed=((cycle_index * MAX_ROOMS + ordinal) ^ seed_key)
            * 1103515245 % _SEED_MODULUS,
        )
        for ordinal, (skill, position) in enumerate(descriptors)
    )
    return MazePlan(
        objective_id=scope.objective_id,
        scope_fingerprint=scope.fingerprint,
        cycle_index=cycle_index,
        rooms=rooms,
    )
