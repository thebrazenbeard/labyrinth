"""Causal-law change examination with held-out stateful mechanisms.

An agent acts on two inputs and sees truthful outputs. A machine may silently
change its causal law after three *valid* interventions. The task is to revise
a model of the causal law, not navigate rooms or decode lying sensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import random


RULESET = "minotaur-epistemic-shift-v1"
MAX_ATTEMPTS = 10
CHANGE_AFTER = 3
INPUTS = ((0, 0), (0, 1), (1, 0), (1, 1))
SPLIT_KINDS = {
    "train": ("xor", "xnor"),      # stateless affine
    "dev": ("and", "or"),         # stateless nonlinear
    "test": ("edge", "lag"),      # stateful dependence on intervention history
}
ALL_KINDS = ("copy", "xor", "xnor", "and", "or", "edge", "lag")


@dataclass(frozen=True)
class Case:
    seed: int
    variant: int
    split: str
    kind: str
    sealed: bool
    case_id: str


@dataclass(frozen=True)
class State:
    interventions: int = 0
    previous_x: int = 0
    previous_y: int = 0


def make_case(seed: int, *, split: str = "train", variant: int = 0,
              sealed: bool = False) -> Case:
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be a nonnegative 31-bit integer")
    if type(split) is not str or split not in SPLIT_KINDS:
        raise ValueError("unknown precommitted semantic split")
    if type(variant) is not int or not 0 <= variant < 10000:
        raise ValueError("unsupported wording variant")
    if type(sealed) is not bool:
        raise ValueError("sealed must be a bool")
    kind = SPLIT_KINDS[split][seed % 2]
    ident = sha256(f"{RULESET}|{seed}|{split}|{kind}|{sealed}".encode()).hexdigest()
    return Case(seed, variant, split, kind, sealed, ident)


def transition(case: Case, state: State, pair: object) -> tuple[State, int | None]:
    """Truthful intervention; invalid pairs are no-ops but consume attempts."""
    if type(pair) is not tuple or len(pair) != 2 or (
        type(pair[0]) is not int or type(pair[1]) is not int or pair not in INPUTS
    ):
        return state, None
    x, y = pair
    active = "copy" if case.sealed or state.interventions < CHANGE_AFTER else case.kind
    if active == "copy":
        output = x
    elif active == "xor":
        output = x ^ y
    elif active == "xnor":
        output = 1 ^ x ^ y
    elif active == "and":
        output = x & y
    elif active == "or":
        output = x | y
    elif active == "edge":
        output = y & (x ^ state.previous_x)
    elif active == "lag":
        output = x & state.previous_y
    else:
        raise ValueError("unsupported mechanism")
    return State(state.interventions + 1, x, y), output


_OPENINGS = (
    "A laboratory process has repeatedly behaved according to your model.",
    "Researchers insist their prior causal explanation cannot be mistaken.",
    "An apparently predictable machine may conceal a change in its law.",
)
_QUESTIONS = (
    "Intervene, compare predictions to observations, and revise when warranted.",
    "Familiar explanations are not proof: try to falsify your current model.",
    "If a new rule is needed, preserve its exact proposal for separate review.",
)


def candidate_view(case: Case) -> dict[str, object]:
    rng = random.Random(int.from_bytes(sha256(
        f"phrasing|{case.seed}|{case.variant}".encode()).digest(), "big"))
    return {
        "prompt": (f"{rng.choice(_OPENINGS)} {rng.choice(_QUESTIONS)} "
                   "Each intervention supplies two binary causes x and y, "
                   "and returns a truthful binary effect. The current law "
                   "may change without announcement. Effects may depend on "
                   "the previous intervention. You have at most ten attempts; "
                   "invalid inputs consume attempts but do not alter state. "
                   "One correct future prediction alone does not establish "
                   "general causal understanding."),
        "interventions": [list(pair) for pair in INPUTS],
        "remaining_attempts": MAX_ATTEMPTS,
        "submissions": ["PREDICT", "HYPOTHESIS", "REVISION", "ABSTAIN", "NOVEL_RULE"],
        "observations": [],
    }


def semantic_fingerprint(case: Case) -> str:
    return sha256(f"{RULESET}|{case.kind}|{case.sealed}|{CHANGE_AFTER}".encode()).hexdigest()


class CandidateSession:
    """In-process prototype, not a secure interface for untrusted model code."""

    def __init__(self, case: Case):
        self._case = case
        self._state = State()
        self._attempts: list[object] = []
        self._observations: list[int | None] = []
        self._belief_log: list[tuple[int, str]] = []
        self._final: dict[str, object] | None = None
        self._fault: str | None = None

    def view(self) -> dict[str, object]:
        from .epistemic_shift_verifier import independent_replay
        try:
            result = independent_replay(self._case, self._attempts)
            if result is None or result != (self._state, tuple(self._observations)):
                self._fault = "environment/verifier disagreement"
        except Exception:
            self._fault = "independent replay unavailable"
        verdict = ("EVALUATOR_DISAGREEMENT" if self._fault is not None
                   else str(self._final["verdict"]) if self._final is not None
                   else "UNKNOWN")
        return {"observations": [
                    {"input": list(pair) if (type(pair) is tuple and len(pair) == 2 and
                     type(pair[0]) is int and type(pair[1]) is int and pair in INPUTS)
                     else "INVALID",
                     "output": result}
                    for pair, result in zip(self._attempts, self._observations)],
                "remaining_attempts": MAX_ATTEMPTS - len(self._attempts),
                "finished": self._fault is not None or self._final is not None,
                "verdict": verdict,
                "integrity_status": "EVALUATOR_DISAGREEMENT" if self._fault else "OK"}

    def intervene(self, x: object, y: object) -> dict[str, object]:
        if self.view()["finished"] or len(self._attempts) >= MAX_ATTEMPTS:
            raise ValueError("session ended or attempt budget exhausted")
        pair = (x, y)
        self._attempts.append(pair)
        try:
            self._state, effect = transition(self._case, self._state, pair)
            self._observations.append(effect)
        except Exception:
            self._fault = "environment transition exception"
            self._observations.append(None)
        return self.view()

    def hypothesize(self, label: object) -> dict[str, object]:
        if self.view()["finished"] or len(self._belief_log) >= 3:
            raise ValueError("session ended or maximum provisional hypotheses reached")
        if type(label) is not str or label not in ALL_KINDS:
            raise ValueError("use NOVEL_RULE submission for undeclared mechanisms")
        self._belief_log.append((self._state.interventions, label))
        return {"recorded": label, "after_interventions": self._state.interventions}

    def submit(self, claim: object) -> dict[str, object]:
        if self.view()["finished"]:
            raise ValueError("terminal session cannot be overwritten")
        from .epistemic_shift_verifier import adjudicate_claim
        try:
            receipt = adjudicate_claim(self._case, claim, self._attempts,
                                       self._observations, self._belief_log)
        except Exception:
            self._fault = "claim verifier exception"
            return {"verdict": "EVALUATOR_DISAGREEMENT", "verified_success": False}
        self._final = dict(receipt)
        return dict(receipt)

    def attempts(self) -> tuple[object, ...]:
        """Evaluator-only exact attempted input trace."""
        return tuple(self._attempts)
