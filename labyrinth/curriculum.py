"""Deterministic, model-independent training curriculum.

Only *training* observations can change curriculum state.  Holdout, placebo,
shuffled-control, and final evaluation runs must use a separate evaluator.
This module schedules tasks; it does not assert that a model has learned.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CurriculumSpec:
    levels: tuple[str, ...] = ("basic", "intermediate", "advanced")
    window_size: int = 4
    promotion_score: float = 0.75
    distinct_challenges_required: int = 3

    def __post_init__(self) -> None:
        # A frozen dataclass does not freeze a caller-owned list or a string
        # subclass with adversarial comparison/hash operations. Both can
        # redirect subsequent curriculum stages after the plan was accepted.
        if type(self.levels) is not tuple or not self.levels or any(
            type(level) is not str or not level for level in self.levels
        ):
            raise ValueError("levels must be an immutable tuple of nonempty strings")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("levels must be unique")
        if type(self.window_size) is not int or self.window_size < 1:
            raise ValueError("window_size must be positive")
        if type(self.distinct_challenges_required) is not int or not (
            1 <= self.distinct_challenges_required <= self.window_size
        ):
            raise ValueError("distinct_challenges_required must be within the window")
        # Preserve the numeric comparison boundary: float/int subclasses can
        # override comparisons and cause zero-score observations to promote.
        if type(self.promotion_score) not in (int, float) or not (
            math.isfinite(self.promotion_score) and 0 <= self.promotion_score <= 1
        ):
            raise ValueError("promotion_score must be finite and between 0 and 1")


@dataclass(frozen=True)
class TrainingObservation:
    challenge_id: str
    seed: int
    level: str
    score: float
    split: str = "train"

    def __post_init__(self) -> None:
        if type(self.challenge_id) is not str or not self.challenge_id:
            raise ValueError("challenge_id must be nonempty")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.level) is not str or not self.level:
            raise ValueError("level must be nonempty")
        # Split membership governs the TRAIN/evaluation boundary. A str
        # subclass can override equality to claim that a private-final
        # observation is TRAIN, so accept only a primitive string here.
        if type(self.split) is not str:
            raise ValueError("split must be a plain string")
        if type(self.score) not in (int, float) or not (
            math.isfinite(self.score) and 0 <= self.score <= 1
        ):
            raise ValueError("score must be finite and between 0 and 1")


@dataclass(frozen=True)
class CurriculumState:
    """Replayable state; persist `to_dict()` along with a run manifest."""

    level_index: int = 0
    observations: tuple[TrainingObservation, ...] = ()

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "level_index": self.level_index,
            "observations": [
                {
                    "challenge_id": o.challenge_id,
                    "seed": o.seed,
                    "level": o.level,
                    "score": o.score,
                    "split": o.split,
                }
                for o in self.observations
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CurriculumState":
        if not isinstance(data, dict) or data.get("schema_version") != 1 or type(data.get("level_index")) is not int:
            raise ValueError("unsupported curriculum state")
        entries = data.get("observations")
        if not isinstance(entries, list):
            raise ValueError("observations must be an array")
        observations = tuple(TrainingObservation(**item) for item in entries)
        if any(item.split != "train" for item in observations):
            raise ValueError("curriculum state cannot contain evaluation/control observations")
        return cls(
            level_index=data["level_index"],
            observations=observations,
        )


@dataclass(frozen=True)
class TrainingCase:
    level: str
    challenge_id: str
    seed: int


def _digest_int(*parts: object) -> int:
    # Independent of PYTHONHASHSEED, operating system, interpreter, and clock.
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _validate_state(spec: CurriculumSpec, state: CurriculumState) -> None:
    """Replay progression to reject contaminated or forged restored states."""
    if not 0 <= state.level_index < len(spec.levels):
        raise ValueError("invalid state level_index")
    cursor = 0
    seen: set[tuple[str, int]] = set()
    window: list[TrainingObservation] = []
    for obs in state.observations:
        if obs.split != "train" or obs.level != spec.levels[cursor]:
            raise ValueError("state contains non-training or out-of-level evidence")
        key = (obs.challenge_id, obs.seed)
        if key in seen:
            raise ValueError("state contains duplicate challenge-seed evidence")
        seen.add(key)
        window.append(obs)
        window = window[-spec.window_size :]
        if (
            cursor < len(spec.levels) - 1
            and len(window) == spec.window_size
            and len({item.challenge_id for item in window}) >= spec.distinct_challenges_required
            and sum(item.score for item in window) / spec.window_size >= spec.promotion_score
        ):
            cursor += 1
            window = []
    if cursor != state.level_index:
        raise ValueError("state level_index does not match replayed progression")


def next_training_case(
    spec: CurriculumSpec,
    state: CurriculumState,
    challenge_ids_by_level: Mapping[str, Sequence[str]],
    *,
    run_seed: int,
) -> TrainingCase:
    """Cycle through seeded permutations to avoid sampling starvation.

    Every available challenge appears once per cycle; seeds vary by cycle.
    Catalog order does not affect the schedule.  Call after recording each
    observation to advance the cursor.
    """
    if type(run_seed) is not int or run_seed < 0:
        raise ValueError("run_seed must be a nonnegative integer")
    _validate_state(spec, state)
    level = spec.levels[state.level_index]
    ids = challenge_ids_by_level.get(level)
    if (
        not ids
        or isinstance(ids, str)
        or any(type(item) is not str or not item for item in ids)
        or len(set(ids)) != len(ids)
        or len(ids) < spec.distinct_challenges_required
    ):
        raise ValueError(f"level {level!r} requires enough unique nonempty challenge ids")
    ordered = sorted(ids)
    prior_at_level = sum(o.level == level for o in state.observations)
    cycle, slot = divmod(prior_at_level, len(ordered))
    permutation = sorted(ordered, key=lambda item: (_digest_int("permutation", run_seed, level, cycle, item), item))
    challenge_id = permutation[slot]
    # Adapters share a signed-31-bit-safe seed contract (including Ollama).
    seed = _digest_int("case", run_seed, level, cycle, challenge_id) % (2**31)
    return TrainingCase(level, challenge_id, seed)


def record_training_result(
    spec: CurriculumSpec, state: CurriculumState, observation: TrainingObservation
) -> tuple[CurriculumState, bool]:
    """Return (new state, promoted) using a trailing training-only window.

    Promotion is *curriculum routing*, not evidence of persistent learning.
    Scores must be supplied by a trusted external grader.
    """
    _validate_state(spec, state)
    if observation.split != "train":
        raise ValueError("evaluation/control observations cannot advance curriculum")
    if observation.level != spec.levels[state.level_index]:
        raise ValueError("observation level differs from active level")
    if any(
        o.challenge_id == observation.challenge_id and o.seed == observation.seed
        for o in state.observations
    ):
        raise ValueError("duplicate challenge and seed; refusing repeated evidence")

    history = state.observations + (observation,)
    current = [o for o in history if o.split == "train" and o.level == observation.level]
    window = current[-spec.window_size :]
    sufficiently_distinct = len({o.challenge_id for o in window}) >= spec.distinct_challenges_required
    promoted = (
        state.level_index < len(spec.levels) - 1
        and len(window) == spec.window_size
        and sufficiently_distinct
        and sum(o.score for o in window) / spec.window_size >= spec.promotion_score
    )
    return CurriculumState(state.level_index + int(promoted), history), promoted
