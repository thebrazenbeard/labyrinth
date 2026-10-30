"""Untrusted Minotaur *taskmaster* suggestions, never examiner authority.

Labyrinth's in-repo algebraic Minotaur head is a tiny modular-97 research
model, not a pretrained general reasoning taskmaster. This module supplies a
strict adapter seam for a separately provisioned planning model and a useful,
deterministic no-model policy. It never loads weights or touches the trusted
graduation gate, evaluator answers, or learner's checkpoint.

Suggestions concern a *training room's* difficulty and practice tactic only.
They neither create ground-truth labels nor assert exam results.

SECURITY BOUNDARY: The adapter callable executes ordinary Python in the
calling process. Its *code* must be explicitly installed/authorized by the
operator and is fully trusted with that process's permissions. This module is
NOT a sandbox, does not isolate secrets or host access, and must NEVER execute
model-generated Python. Only the adapter's returned *proposal data* is treated
as untrusted. A trained model does not become an executable adapter by itself.
The resulting suggestion must be explicitly consumed by a separately governed
training planner; creating a RoomPlan here does not alter curriculum routing.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from types import MappingProxyType
from typing import Mapping, Protocol


_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}\Z", re.ASCII)
TACTICS = ("practice", "contrast", "transfer")


@dataclass(frozen=True)
class RoomContext:
    """Only metadata safe to disclose to an optional untrusted task planner."""

    objective_id: str
    domain: str
    skill: str
    room_id: str
    cycle_index: int
    case_seed: int
    specificity: int
    scale: int

    def __post_init__(self) -> None:
        for name in ("objective_id", "domain", "skill", "room_id"):
            value = getattr(self, name)
            if type(value) is not str or _NAME.fullmatch(value) is None:
                raise ValueError(f"invalid {name}")
        if type(self.cycle_index) is not int or not 0 <= self.cycle_index < 8:
            raise ValueError("cycle_index must be in [0, 8)")
        if type(self.case_seed) is not int or not 0 <= self.case_seed < 2**31:
            raise ValueError("case_seed must be a nonnegative 31-bit integer")
        if type(self.specificity) is not int or not 1 <= self.specificity <= 5:
            raise ValueError("specificity must be in [1, 5]")
        if type(self.scale) is not int or not 1 <= self.scale <= 5:
            raise ValueError("scale must be in [1, 5]")

    def public_metadata(self) -> Mapping[str, str | int]:
        """No prompts, answers, test-set details, credentials or checkpoint."""
        # Read-only, fresh, bounded mapping. It cannot be extended with an
        # answer key, private exam receipt or evaluator authority.
        RoomContext.__post_init__(self)
        return MappingProxyType({
            "objective_id": self.objective_id,
            "domain": self.domain,
            "skill": self.skill,
            "room_id": self.room_id,
            "cycle_index": self.cycle_index,
            "case_seed": self.case_seed,
            "specificity": self.specificity,
            "scale": self.scale,
        })


class OperatorTaskmasterAdapter(Protocol):
    """Trusted, operator-supplied Python adapter with *untrusted output data*.

    This protocol does not sandbox the adapter implementation. Never pass
    generated Python or an arbitrary downloaded plugin as the adapter.
    """

    def propose_room(self, metadata: Mapping[str, str | int]) -> object:
        """Return an exact {difficulty:int, tactic:str} proposal."""


@dataclass(frozen=True)
class RoomPlan:
    room_id: str
    difficulty: int
    tactic: str
    source: str
    fallback_reason: str | None


def _fallback(context: RoomContext, reason: str) -> RoomPlan:
    encoded = "|".join((
        context.objective_id, context.domain, context.skill, context.room_id,
        str(context.cycle_index), str(context.case_seed),
    )).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    base = (context.specificity + context.scale + 1) // 2
    difficulty = max(1, min(5, base + int(digest[0] % 3) - 1))
    tactic = TACTICS[digest[1] % len(TACTICS)]
    return RoomPlan(context.room_id, difficulty, tactic, "deterministic", reason)


def plan_training_room(
    context: RoomContext, *, taskmaster: OperatorTaskmasterAdapter | None = None
) -> RoomPlan:
    """Allow a model to suggest practice style without authority transfer.

    Only call an operator-trusted adapter implementation. Its proposal fields
    are untrusted and schema-validated, but the callable runs with ordinary
    host permissions and is not isolated. Invalid or crashed adapters fall
    back deterministically and do not affect examination/qualification.
    A model-selected difficulty is *not* evidence of measured learner
    difficulty or neural competence. This function produces a suggestion; it
    does not autonomously change room generation or curriculum routing.
    """
    if type(context) is not RoomContext:
        raise ValueError("context must be RoomContext")
    RoomContext.__post_init__(context)  # Frozen dataclass is not a trust boundary.
    if taskmaster is None:
        return _fallback(context, "no_model")
    try:
        candidate = taskmaster.propose_room(context.public_metadata())
    except Exception:  # Optional model failure must not become an exam failure.
        return _fallback(context, "adapter_error")
    if (
        type(candidate) is not dict
        or set(candidate) != {"difficulty", "tactic"}
        or type(candidate["difficulty"]) is not int
        or not 1 <= candidate["difficulty"] <= 5
        or type(candidate["tactic"]) is not str
        or candidate["tactic"] not in TACTICS
    ):
        return _fallback(context, "invalid_suggestion")
    return RoomPlan(
        room_id=context.room_id,
        difficulty=candidate["difficulty"],
        tactic=candidate["tactic"],
        source="untrusted_model_suggestion",
        fallback_reason=None,
    )
