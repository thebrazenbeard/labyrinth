"""Predeclared observable learning intent, separate from execution and grading.

An ObjectiveContract elaborates a legacy TrainingScopeSpec with human-written
focus and behavioral targets. It is **data**, not an executable prompt, rubric
implementation, dataset, training authorization, or Minotaur graduation grant.
Callers must separately bind operator-approved task generators and evaluators.
The focus-selection helper informs a future curriculum provider; it does not
itself alter challenge content or certify that a provider honored the target.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .training_scope import TrainingScopeSpec


SCHEMA = "labyrinth.objective-contract.v1"
MAX_OBJECTIVE_JSON_BYTES = 16_384
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z", re.ASCII)
_BANNED_NAMES = frozenset((
    "eval", "evaluation", "test", "holdout", "heldout", "control",
    "final", "private_final", "shuffled", "placebo", "reserved",
))


def _identifier(raw: str, label: str) -> None:
    if type(raw) is not str or not _IDENTIFIER.fullmatch(raw) or raw in _BANNED_NAMES:
        raise ValueError(f"{label} must be a nonreserved lowercase identifier")


def _text(raw: str, label: str, *, max_chars: int = 384) -> None:
    if (type(raw) is not str or not 3 <= len(raw) <= max_chars
            or raw != raw.strip() or any(ord(char) < 32 or ord(char) == 127 for char in raw)
            or len(raw.encode("utf-8")) > max_chars * 4):
        raise ValueError(f"{label} must be trimmed, single-line, bounded text")


def _text_tuple(values: tuple[str, ...], label: str, *, min_size: int, max_size: int) -> None:
    if type(values) is not tuple or not min_size <= len(values) <= max_size:
        raise ValueError(f"{label} must be an immutable bounded tuple")
    for value in values:
        _text(value, label)
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicates")


@dataclass(frozen=True)
class SkillTarget:
    """One declared skill with concrete observable, separately graded targets.

    Numeric thresholds describe the operator's intended target only. Actual
    evaluator policy still requires independent qualification and custody.
    """

    skill: str
    focus_tags: tuple[str, ...]
    observable_behaviors: tuple[str, ...]
    rubric_id: str
    minimum_score_percent: int
    minimum_gain_percent: int

    def __post_init__(self) -> None:
        _identifier(self.skill, "skill")
        _identifier(self.rubric_id, "rubric_id")
        if type(self.focus_tags) is not tuple or not 1 <= len(self.focus_tags) <= 5:
            raise ValueError("focus_tags must contain 1..5 declared focus identifiers")
        for tag in self.focus_tags:
            _identifier(tag, "focus tag")
        if len(set(self.focus_tags)) != len(self.focus_tags):
            raise ValueError("duplicate focus tags")
        _text_tuple(self.observable_behaviors, "observable_behaviors", min_size=1, max_size=8)
        if (type(self.minimum_score_percent) is not int
                or not 0 <= self.minimum_score_percent <= 100
                or type(self.minimum_gain_percent) is not int
                or not 0 <= self.minimum_gain_percent <= 100):
            raise ValueError("target percentages must be integers in [0,100]")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "skill": self.skill,
            "focus_tags": list(self.focus_tags),
            "observable_behaviors": list(self.observable_behaviors),
            "rubric_id": self.rubric_id,
            "minimum_score_percent": self.minimum_score_percent,
            "minimum_gain_percent": self.minimum_gain_percent,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SkillTarget":
        keys = {
            "skill", "focus_tags", "observable_behaviors", "rubric_id",
            "minimum_score_percent", "minimum_gain_percent",
        }
        if type(raw) is not dict or set(raw) != keys:
            raise ValueError("unknown or missing skill target fields")
        if type(raw["focus_tags"]) is not list or type(raw["observable_behaviors"]) is not list:
            raise ValueError("focus_tags and observable_behaviors must be JSON arrays")
        return cls(
            skill=raw["skill"],
            focus_tags=tuple(raw["focus_tags"]),
            observable_behaviors=tuple(raw["observable_behaviors"]),
            rubric_id=raw["rubric_id"],
            minimum_score_percent=raw["minimum_score_percent"],
            minimum_gain_percent=raw["minimum_gain_percent"],
        )


@dataclass(frozen=True)
class ObjectiveContract:
    """Immutable semantic target attached to an unchanged training scope.

    More specificity selects fewer, more concentrated focus tags. This only
    communicates intent to a trusted provider; no generator is automatically
    synthesized from natural language and no external grade is authorized.
    """

    scope: TrainingScopeSpec
    intention: str
    targets: tuple[SkillTarget, ...]
    excluded_scopes: tuple[str, ...]
    forbidden_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.scope) is not TrainingScopeSpec:
            raise ValueError("scope must be a valid TrainingScopeSpec")
        TrainingScopeSpec.__post_init__(self.scope)
        _text(self.intention, "intention", max_chars=600)
        if type(self.targets) is not tuple or len(self.targets) != len(self.scope.skills):
            raise ValueError("targets must cover the declared skill set exactly")
        for target in self.targets:
            if type(target) is not SkillTarget:
                raise ValueError("each target must be a SkillTarget")
            SkillTarget.__post_init__(target)
        if tuple(target.skill for target in self.targets) != self.scope.skills:
            raise ValueError("targets must match scope skills in declared order, without duplicates")
        _text_tuple(self.excluded_scopes, "excluded_scopes", min_size=0, max_size=16)
        _text_tuple(self.forbidden_claims, "forbidden_claims", min_size=1, max_size=16)

    @property
    def training_scope(self) -> TrainingScopeSpec:
        """Exact unchanged legacy scope for existing planner/JourneyState APIs."""
        self.__post_init__()
        return self.scope

    def selected_focus(self, skill: str) -> tuple[str, ...]:
        """Predeclared narrowness policy: 1=broadest, 5=first declared focus.

        Intent selection is metadata. A caller must verify any actual generated
        challenge reflects it; returning tags does not prove conditioning.
        """
        self.__post_init__()
        for target in self.targets:
            if target.skill == skill:
                return target.focus_tags[:max(1, 6 - self.scope.specificity)]
        raise ValueError("skill outside declared objective contract")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        result = {
            "schema": SCHEMA,
            "scope": self.scope.to_dict(),
            "intention": self.intention,
            "targets": [target.to_dict() for target in self.targets],
            "excluded_scopes": list(self.excluded_scopes),
            "forbidden_claims": list(self.forbidden_claims),
        }
        if len(_canonical_bytes(result)) > MAX_OBJECTIVE_JSON_BYTES:
            raise ValueError("objective contract exceeds JSON size limit")
        return result

    @property
    def fingerprint(self) -> str:
        """Canonical content hash, not authorship or rubric-evaluator attestation."""
        return hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ObjectiveContract":
        keys = {"schema", "scope", "intention", "targets", "excluded_scopes", "forbidden_claims"}
        if type(raw) is not dict or set(raw) != keys or raw.get("schema") != SCHEMA:
            raise ValueError("unsupported or ambiguous objective contract schema")
        if (type(raw["targets"]) is not list or type(raw["excluded_scopes"]) is not list
                or type(raw["forbidden_claims"]) is not list):
            raise ValueError("targets, excluded_scopes and forbidden_claims must be JSON arrays")
        return cls(
            scope=TrainingScopeSpec.from_dict(raw["scope"]),
            intention=raw["intention"],
            targets=tuple(SkillTarget.from_dict(target) for target in raw["targets"]),
            excluded_scopes=tuple(raw["excluded_scopes"]),
            forbidden_claims=tuple(raw["forbidden_claims"]),
        )


def _canonical_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def parse_objective_json(raw: bytes) -> ObjectiveContract:
    """Reject oversized, ambiguous, nonfinite, or non-UTF8 declarations."""
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_OBJECTIVE_JSON_BYTES:
        raise ValueError("objective JSON must be nonempty bytes <=16 KiB")

    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key, value in items:
            if key in data:
                raise ValueError("duplicate key in objective JSON")
            data[key] = value
        return data

    def invalid_constant(value: str) -> Any:
        raise ValueError(f"nonfinite JSON number: {value}")

    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                          parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid UTF-8 objective contract") from exc
    return ObjectiveContract.from_dict(data)
