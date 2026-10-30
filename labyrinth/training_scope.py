"""Frozen, model-independent declaration of what a Labyrinth journey teaches.

A scope controls training challenge selection and episode size. It does not
authorize access to evaluation labels or confer Minotaur graduation. A goal
label is metadata, never executable code or a prompt trusted as policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any


SCHEMA = "labyrinth.training-scope.v1"
MAX_SCOPE_JSON_BYTES = 16_384
MAX_ROOMS = 128
_OBJECTIVE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_LABEL = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z", re.ASCII)
_RESERVED_SPLITS = frozenset((
    "eval", "evaluation", "control", "holdout", "heldout", "final",
    "placebo", "shuffled", "test", "reserved",
))


def _rooms_per_skill(scale: int, specificity: int) -> int:
    # These are training chambers, not private examination cases. The scaling
    # is deliberately small, deterministic, and independent of machine RAM.
    return scale * (1 + specificity // 2)


@dataclass(frozen=True)
class TrainingScopeSpec:
    """Immutable training intention, bounded before generating any chambers.

    `specificity` is how narrowly to exercise each declared skill (1..5),
    while `scale` controls episode length (1..5). Narrowness must be expressed
    through new variations, not repetitions of memorized test fixtures.
    """

    objective_id: str
    domain: str
    skills: tuple[str, ...]
    specificity: int
    scale: int
    max_cycles: int
    seed: int

    def __post_init__(self) -> None:
        if type(self.objective_id) is not str or not _OBJECTIVE.fullmatch(self.objective_id):
            raise ValueError("invalid objective_id")
        if type(self.domain) is not str or not _LABEL.fullmatch(self.domain):
            raise ValueError("domain must be a lowercase identifier")
        if type(self.skills) is not tuple or not 1 <= len(self.skills) <= 12:
            raise ValueError("skills must be a nonempty tuple of at most 12 entries")
        if any(type(skill) is not str or not _LABEL.fullmatch(skill) for skill in self.skills):
            raise ValueError("skills must contain lowercase identifiers")
        if len(set(self.skills)) != len(self.skills):
            raise ValueError("duplicate skills are forbidden")
        if any(
            skill in _RESERVED_SPLITS
            or skill.startswith(("eval_", "control_", "holdout_", "heldout_", "final_"))
            for skill in self.skills
        ):
            raise ValueError("split and control names cannot be training skills")
        if type(self.specificity) is not int or not 1 <= self.specificity <= 5:
            raise ValueError("specificity must be an integer in [1, 5]")
        if type(self.scale) is not int or not 1 <= self.scale <= 5:
            raise ValueError("scale must be an integer in [1, 5]")
        if type(self.max_cycles) is not int or not 1 <= self.max_cycles <= 8:
            raise ValueError("max_cycles must be an integer in [1, 8]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be a nonnegative signed-31-bit integer")
        if len(self.skills) * _rooms_per_skill(self.scale, self.specificity) > MAX_ROOMS:
            raise ValueError("declared journey exceeds 128 training chambers per cycle")

    @property
    def chamber_count_per_skill(self) -> int:
        self.__post_init__()  # Also reject corrupted frozen objects before use.
        return _rooms_per_skill(self.scale, self.specificity)

    @property
    def total_chambers(self) -> int:
        return len(self.skills) * self.chamber_count_per_skill

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "schema": SCHEMA,
            "objective_id": self.objective_id,
            "domain": self.domain,
            "skills": list(self.skills),
            "specificity": self.specificity,
            "scale": self.scale,
            "max_cycles": self.max_cycles,
            "seed": self.seed,
        }

    @property
    def fingerprint(self) -> str:
        """SHA-256 of canonical bytes; identifies content, not author/trust."""
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingScopeSpec":
        expected = {
            "schema", "objective_id", "domain", "skills", "specificity",
            "scale", "max_cycles", "seed",
        }
        if type(data) is not dict or set(data) != expected or data["schema"] != SCHEMA:
            raise ValueError("unsupported or ambiguous training scope schema")
        if type(data["skills"]) is not list:
            raise ValueError("skills must be a JSON array")
        return cls(
            objective_id=data["objective_id"],
            domain=data["domain"],
            skills=tuple(data["skills"]),
            specificity=data["specificity"],
            scale=data["scale"],
            max_cycles=data["max_cycles"],
            seed=data["seed"],
        )


def parse_scope_json(raw: bytes) -> TrainingScopeSpec:
    """Bounded strict JSON import for predeclared scope manifests."""
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_SCOPE_JSON_BYTES:
        raise ValueError("scope JSON must be nonempty bytes <= 16 KiB")

    def unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key in training scope JSON")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise ValueError(f"nonfinite JSON number: {value}")

    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid UTF-8 JSON training scope") from exc
    return TrainingScopeSpec.from_dict(data)
