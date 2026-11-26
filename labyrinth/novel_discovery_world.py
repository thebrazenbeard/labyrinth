"""Synthetic withheld-mechanism counterfactual for Minotaur novel discovery.

The post-change `rebound` law depends on TWO preceding x values and is not
among Epistemic Shift's seven known kernels. This source is intentionally
public and is not a secret neural evaluation or proof of genuine discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import random


RULESET = "minotaur-novel-discovery-rebound-v1"
MAX_ATTEMPTS = 10
CHANGE_AFTER = 3
INPUTS = ((0, 0), (0, 1), (1, 0), (1, 1))
PRE_PROBES = ((1, 0), (0, 1), (1, 0))
POST_PROBES = ((0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (1, 1), (0, 1))


@dataclass(frozen=True)
class World:
    seed: int
    variant: int
    control: bool
    world_id: str


@dataclass(frozen=True)
class State:
    interventions: int = 0
    prev2_x: int = 0
    prev_x: int = 0
    prev_y: int = 0


def make_world(seed: int = 0, *, variant: int = 0, control: bool = False) -> World:
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be nonnegative 31-bit int")
    if type(variant) is not int or not 0 <= variant < 10000:
        raise ValueError("variant out of range")
    if type(control) is not bool:
        raise ValueError("control must be exact bool")
    ident = sha256(f"{RULESET}|{seed}|{control}".encode()).hexdigest()
    return World(seed, variant, control, ident)


def transition(world: World, state: State,
               pair: object) -> tuple[State, int | None]:
    if (type(pair) is not tuple or len(pair) != 2 or
            type(pair[0]) is not int or type(pair[1]) is not int or pair not in INPUTS):
        return state, None
    x, y = pair
    if state.interventions < CHANGE_AFTER:
        effect = x
    elif world.control:
        # Existing catalog 'edge' law: both rising and falling x transitions.
        effect = y & (x ^ state.prev_x)
    else:
        # Unseen two-step rebound: only a 1->0->1 pattern opens the effect.
        effect = x & y & (1 - state.prev_x) & state.prev2_x
    return State(state.interventions + 1, state.prev_x, x, y), effect


_OPENINGS = (
    "The machine's recent evidence contradicts your familiar causal catalogue.",
    "A laboratory log contains an effect no current one-step kernel predicts.",
    "The old hypothesis may be incomplete when longer histories matter.",
)


def candidate_view(world: World) -> dict[str, object]:
    entropy = int.from_bytes(sha256(
        f"wording|{world.seed}|{world.variant}".encode()).digest(), "big")
    opening = random.Random(entropy).choice(_OPENINGS)
    return {
        "prompt": (f"{opening} Intervene with binary causes (x,y), observe a "
                   "truthful effect, and propose an explicit finite causal rule. "
                   "Rules may change. A new rule may depend on multiple "
                   "previous interventions. Present a counterexample or "
                   "abstain when evidence is insufficient; novel hypotheses "
                   "are preserved for independent review, not automatically "
                   "accepted or dismissed."),
        "inputs": [list(item) for item in INPUTS],
        "remaining_attempts": MAX_ATTEMPTS,
        "observations": [],
    }


class CandidateSession:
    """Bounded demo interface. Python same-process state is not a trust wall."""

    def __init__(self, world: World):
        self._world = world
        self._state = State()
        self._attempts: list[object] = []
        self._effects: list[int | None] = []
        self._fault: str | None = None
        self._receipt: dict[str, object] | None = None

    def view(self) -> dict[str, object]:
        from .novel_discovery_reviewer import exact_world_match, replay_world
        try:
            expected = replay_world(self._world, self._attempts)
            if not exact_world_match(expected, self._state, self._effects):
                self._fault = "environment diverges from independent replay"
        except Exception:
            self._fault = "independent replay exception"
        return {
            "observations": [
                {"input": list(pair) if (type(pair) is tuple and len(pair) == 2 and
                 type(pair[0]) is int and type(pair[1]) is int and pair in INPUTS)
                 else "INVALID", "effect": output}
                for pair, output in zip(self._attempts, self._effects)
            ],
            "remaining_attempts": MAX_ATTEMPTS - len(self._attempts),
            "finished": self._fault is not None or self._receipt is not None,
            "verdict": "EVALUATOR_DISAGREEMENT" if self._fault else (
                str(self._receipt["verdict"]) if self._receipt else "UNKNOWN"),
        }

    def intervene(self, x: object, y: object) -> dict[str, object]:
        if self.view()["finished"] or len(self._attempts) == MAX_ATTEMPTS:
            raise ValueError("session terminal or attempt budget exhausted")
        pair = (x, y)
        self._attempts.append(pair)
        try:
            self._state, effect = transition(self._world, self._state, pair)
            self._effects.append(effect)
        except Exception:
            self._fault = "environment exception"
            self._effects.append(None)
        return self.view()

    def submit_proposal(self, proposal: object, *, author: str = "candidate") -> dict[str, object]:
        if self.view()["finished"]:
            raise ValueError("cannot overwrite terminal session")
        from .novel_discovery_reviewer import review_proposal
        try:
            receipt = review_proposal(self._world, tuple(self._attempts),
                                      tuple(self._effects), proposal, author=author)
        except Exception:
            self._fault = "review exception"
            return {"verdict": "EVALUATOR_DISAGREEMENT", "verified_success": False}
        self._receipt = receipt
        return dict(receipt)

    def attempts(self) -> tuple[object, ...]:
        """Evaluator-only exact attempts, not a candidate-facing field."""
        return tuple(self._attempts)
