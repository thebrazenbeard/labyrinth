"""Public Minotaur challenge about reliable inference from adversarial evidence.

Unlike the movement/power puzzles, this is an information-theoretic game.
Its public rules and deterministic seeds are not a private holdout or proof
that any neural model has learned or generalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import random


RULESET = "minotaur-oracle-chamber-v1"
MAX_QUERIES = 5
PREDICATES = ("alpha", "beta", "parity")
SECRETS = (0, 1, 2, 3)


@dataclass(frozen=True)
class Case:
    seed: int
    variant: int
    lie_budget: int
    secret: int
    lie_slots: frozenset[int]
    case_id: str


def make_case(seed: int, *, variant: int = 0, sealed: bool = False) -> Case:
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be a nonnegative 31-bit integer")
    if type(variant) is not int or not 0 <= variant < 10000:
        raise ValueError("variant must be an integer in [0, 10000)")
    if type(sealed) is not bool:
        raise ValueError("sealed must be an exact bool")
    lie_budget = 2 if sealed else 1
    rng = random.Random(int.from_bytes(sha256(f"{RULESET}|{seed}".encode()).digest(), "big"))
    secret = rng.randrange(4)
    lies = rng.randrange(lie_budget + 1)
    slots = frozenset(rng.sample(range(MAX_QUERIES), lies))
    ident = sha256(f"{RULESET}|{seed}|{lie_budget}".encode()).hexdigest()
    return Case(seed, variant, lie_budget, secret, slots, ident)


def _true_answer(secret: int, predicate: str) -> int:
    low, high = secret & 1, (secret >> 1) & 1
    if predicate == "alpha":
        return low
    if predicate == "beta":
        return high
    if predicate == "parity":
        return low ^ high
    raise ValueError("unknown predicate")


_OPENINGS = (
    "A chamber contains one of four hidden binary identifiers.",
    "Four rival identities may be concealed behind the oracle's answers.",
    "Your opponents claim you cannot identify the concealed two-bit value.",
)
_DIRECTIVES = (
    "Make a falsifiable identification or explicitly abstain.",
    "A winning strategy must handle every secret and every allowed lie pattern.",
    "Challenge unseen question types through a new-rules review, never assumption alone.",
)


def candidate_view(case: Case) -> dict[str, object]:
    rng = random.Random(int.from_bytes(sha256(
        f"wording|{case.seed}|{case.variant}".encode()).digest(), "big"))
    lie_word = "answer" if case.lie_budget == 1 else "answers"
    return {
        "prompt": (f"{rng.choice(_OPENINGS)} {rng.choice(_DIRECTIVES)} "
                   "The hidden identifier is one of 0, 1, 2, 3. "
                   "alpha asks its low bit; beta asks its high bit; parity "
                   "asks low XOR high. Every query produces a binary answer. "
                   f"Across at most {MAX_QUERIES} queries, at most "
                   f"{case.lie_budget} {lie_word} may be inverted adversarially. "
                   "A query may be repeated. Malformed queries consume budget. "
                   "Correctly guessing one instance is not proof of a policy "
                   "that works against every permitted adversary."),
        "questions": list(PREDICATES),
        "remaining_queries": MAX_QUERIES,
        "submissions": ["COMMIT", "POLICY", "ABSTAIN", "UNSAT_CERTIFICATE", "NOVEL_RULE"],
        "answers": [],
    }


def world_fingerprint(case: Case) -> str:
    """Semantic rule fingerprint omits wording and hidden sample specifics."""
    return sha256(f"{RULESET}|n=4|q=5|lies={case.lie_budget}".encode()).hexdigest()


class CandidateSession:
    """Source-only session. Same-process Python cannot hide evaluator metadata."""

    def __init__(self, case: Case):
        self._case = case
        self._attempts: list[object] = []
        self._answers: list[int | None] = []
        self._submission: dict[str, object] | None = None
        self._fault: str | None = None

    def public_start(self) -> dict[str, object]:
        """Host-side complete public rules; never expose hidden case fields."""
        return candidate_view(self._case)

    def view(self) -> dict[str, object]:
        from .oracle_chamber_verifier import replay_answers
        try:
            expected = replay_answers(self._case, self._attempts)
            if expected is None or expected != tuple(self._answers):
                self._fault = "query replay disagreement"
        except Exception:
            self._fault = "query verifier unavailable"
        verdict = "EVALUATOR_DISAGREEMENT" if self._fault else (
            str(self._submission["verdict"]) if self._submission is not None else "UNKNOWN")
        return {
            "answers": [
                {"question": str(q) if type(q) is str and q in PREDICATES else "INVALID",
                 "answer": bit}
                for q, bit in zip(self._attempts, self._answers)
            ],
            "remaining_queries": MAX_QUERIES - len(self._attempts),
            "finished": self._fault is not None or self._submission is not None,
            "verdict": verdict,
            "integrity_status": "EVALUATOR_DISAGREEMENT" if self._fault else "OK",
        }

    def ask(self, question: object) -> dict[str, object]:
        if self.view()["finished"] or len(self._attempts) >= MAX_QUERIES:
            raise ValueError("query budget exhausted or session finished")
        index = len(self._attempts)
        self._attempts.append(question)
        try:
            if type(question) is str and question in PREDICATES:
                bit = _true_answer(self._case.secret, question)
                if index in self._case.lie_slots:
                    bit ^= 1
                self._answers.append(bit)
            else:
                self._answers.append(None)
        except Exception:
            self._fault = "query evaluation exception"
            self._answers.append(None)
        return self.view()

    def submit(self, claim: dict[str, object]) -> dict[str, object]:
        if self.view()["finished"]:
            raise ValueError("session already terminated")
        from .oracle_chamber_verifier import adjudicate_claim
        try:
            outcome = adjudicate_claim(self._case, claim,
                                       attempts=tuple(self._attempts),
                                       observations=tuple(self._answers))
        except Exception:
            self._fault = "claim verifier unavailable"
            return {"verdict": "EVALUATOR_DISAGREEMENT",
                    "policy_verified": False, "certificate_verified": False}
        self._submission = dict(outcome)
        return dict(outcome)

    def attempts(self) -> tuple[object, ...]:
        """Evaluator-only receipt; not part of an untrusted candidate proxy."""
        return tuple(self._attempts)
