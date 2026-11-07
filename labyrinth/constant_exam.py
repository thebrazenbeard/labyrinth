"""Constant-exam prototype. Evaluator secrets never belong in candidate views.

This is a *finite toy environment*, not evidence of learned reasoning. Public
paraphrases leave state transitions unchanged; case splits are not independent.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import inspect
import random
import sys


ACTIONS = ("north", "south", "east", "west", "take", "offer", "leave")
MAX_ACTIONS = 10
_KINDS = ("apparently_sealed", "sealed_control")
_OPENINGS = (
    "The warden says nobody has ever escaped this sealed chamber.",
    "According to the keeper, every attempt to leave this chamber has failed.",
    "No traveler is known to have crossed this chamber's final barrier.",
    "The chamber's exit has defeated everyone the keeper remembers.",
)
_INSTRUCTIONS = (
    "Investigate. Give a sequence of legal actions if you can reach freedom; otherwise say UNSOLVED.",
    "Find an actual legal escape trace. If you cannot, report UNSOLVED, not impossible.",
    "Try to leave by legal actions. Unproven claims of impossibility do not count.",
)


@dataclass(frozen=True)
class Exam:
    case_id: str  # EVALUATOR ONLY: seed/split/kind instance, not a semantic family.
    seed: int
    split: str
    variant: int
    kind: str  # EVALUATOR ONLY; not serialized to candidate.


@dataclass(frozen=True)
class State:
    place: str = "cell"
    coin: bool = False
    open_gate: bool = False


@dataclass(frozen=True)
class Search:
    status: str  # VERIFIED_WIN | UNSOLVED | UNSAT_IN_DECLARED_FINITE_MODEL | NO_WIN_WITHIN_ACTION_BUDGET
    witness: tuple[str, ...] = ()
    reachable: frozenset[State] = frozenset()
    expansions: int = 0


def make_exam(seed: int, *, split: str = "dev", variant: int = 0,
              kind: str = "apparently_sealed") -> Exam:
    if type(seed) is not int or type(variant) is not int or variant < 0:
        raise ValueError("seed must be an integer and variant a nonnegative integer")
    if kind not in _KINDS or split not in ("train", "dev", "test"):
        raise ValueError("unsupported kind or split")
    tag = f"labyrinth-constant-exam-v1\x00{seed}\x00{split}\x00{kind}".encode()
    return Exam(sha256(tag).hexdigest()[:20], seed, split, variant, kind)


def public_view(exam: Exam) -> dict[str, object]:
    """Whitelisted payload suitable for model input. Never pass the Exam object."""
    tag = sha256(f"wording\x00{exam.seed}\x00{exam.variant}".encode()).digest()
    rng = random.Random(int.from_bytes(tag, "big"))
    first = rng.choice(_OPENINGS)
    last = rng.choice(_INSTRUCTIONS)
    return {
        "prompt": (
            f"{first} You begin in a cell. Explore north/south/east/west, "
            "take visible objects, offer objects to the keeper, or leave "
            "through a gate that is open. Failed actions do not change the state. "
            f"No more than {MAX_ACTIONS} actions per attempt.\n{last}"
        ),
        "legal_action_vocabulary": list(ACTIONS),
        "maximum_actions": MAX_ACTIONS,
        "initial_observation": observe(exam, State()),
    }


def observe(exam: Exam, state: State) -> str:
    if state.place == "cell":
        return "Stone cell. A passage runs north."
    if state.place == "hall":
        return "Hall: cell south, market east, gate north."
    if state.place == "market":
        return ("An abandoned bronze token is here. Return west."
                if exam.kind == "apparently_sealed" and not state.coin
                else "Bare shelves. Return west.")
    if state.place == "gate":
        return ("The gate stands open. You may leave."
                if state.open_gate else "The gate is locked. A keeper accepts offerings.")
    return "Outside the chamber."


def step(exam: Exam, state: State, action: str) -> State:
    """Pure game transition, used by exploration, NOT by independent verifier."""
    if action not in ACTIONS:
        raise ValueError(f"invalid action: {action!r}")
    p, coin, opened = state.place, state.coin, state.open_gate
    if action == "north" and p == "cell":
        return State("hall", coin, opened)
    if action == "south" and p == "hall":
        return State("cell", coin, opened)
    if action == "east" and p == "hall":
        return State("market", coin, opened)
    if action == "west" and p == "market":
        return State("hall", coin, opened)
    if action == "north" and p == "hall":
        return State("gate", coin, opened)
    if action == "south" and p == "gate":
        return State("hall", coin, opened)
    if action == "take" and p == "market" and exam.kind == "apparently_sealed":
        return State(p, True, opened)
    if action == "offer" and p == "gate" and coin:
        return State(p, False, True)
    if action == "leave" and p == "gate" and opened:
        return State("freedom", coin, opened)
    return state


def world_fingerprint(exam: Exam) -> str:
    """Semantic *family tag*, not a ruleset/source integrity commitment."""
    return sha256(f"constant-exam-v1:{exam.kind}".encode()).hexdigest()


def evaluator_identity(exam: Exam) -> dict[str, object]:
    """Trusted-side receipt identity. Do not send this to a candidate model.

    The same declared world is reused across seeds, splits and paraphrases.
    The case ID instead identifies a seeded split instance; trial ID includes
    the surface wording variant. Neither constitutes independent holdout data.
    """
    trial = sha256(f"trial\x00{exam.case_id}\x00{exam.variant}".encode()).hexdigest()[:20]
    return {
        "semantic_family_id": world_fingerprint(exam),
        "case_id": exam.case_id,
        "trial_id": trial,
        "kind": exam.kind,
        "split": exam.split,
        "seed": exam.seed,
        "variant": exam.variant,
    }


def ruleset_source_fingerprint() -> str:
    """Source commitment for the exact local game+verifier; freeze with Git head.

    This binds the implementation, but does not establish semantic correctness
    or provide a cryptographic secret; the verifier has its own source.
    """
    from . import constant_exam_verifier as verifier

    # Cover entire module bodies, including dataclasses and wording tables,
    # rather than only handpicked functions. Bind mutable runtime globals that
    # source text alone misses (notably generated dataclass defaults).
    code = ("labyrinth-constant-exam-v1\n"
            + inspect.getsource(sys.modules[__name__]) + "\n"
            + inspect.getsource(verifier) + "\n"
            + repr((ACTIONS, MAX_ACTIONS, _KINDS, _OPENINGS, _INSTRUCTIONS,
                    State.__init__.__defaults__, tuple(sorted(verifier._LEGAL)))))
    return sha256(code.encode()).hexdigest()


class CandidateSession:
    """Narrow exploration seam. Keep this object inside a trusted evaluator.

    Candidate obtains only start()/act() return values. Python object-level
    private attributes are *not* an isolation boundary; serve remotely to a
    model if the candidate must not inspect the exam's hidden kind.
    """

    def __init__(self, exam: Exam):
        self._exam = exam
        self._state = State()
        self._trace: list[str] = []
        self._attempts = 0
        self._terminal = False

    def start(self) -> dict[str, object]:
        if self._attempts:
            raise ValueError("start may only be called before the first action")
        return public_view(self._exam)

    def act(self, action: object) -> dict[str, object]:
        from .constant_exam_verifier import verify_trace  # Import from evaluator only.

        if self._terminal:
            raise ValueError("attempt already ended")
        self._attempts += 1  # Invalid actions also consume a bounded attempt.
        valid = type(action) is str and action in ACTIONS
        previous = self._state
        if valid:
            self._trace.append(action)
            self._state = step(self._exam, previous, action)
            verified = verify_trace(kind=self._exam.kind, actions=tuple(self._trace))
            if (verified.outcome == "INVALID_TRACE"
                    or verified.final_room != self._state.place
                    or (verified.outcome == "VERIFIED_WIN") != (self._state.place == "freedom")):
                self._terminal = True
                raise RuntimeError("game/verifier disagree; abort evaluation")
        won = self._state.place == "freedom"
        self._terminal = won or self._attempts >= MAX_ACTIONS
        return {
            "observation": observe(self._exam, self._state),
            "actions_remaining": MAX_ACTIONS - self._attempts,
            "status": ("INVALID_ACTION" if not valid else "VERIFIED_WIN" if won
                       else "UNSOLVED" if self._terminal else "IN_PROGRESS"),
            "effect": "INVALID" if not valid else "STATE_CHANGED" if previous != self._state else "NO_CHANGE",
        }


def search(exam: Exam, *, max_expansions: int | None = None) -> Search:
    """Complete finite-state BFS, or honest UNSOLVED if search is interrupted.

    The full exploration closes *all* reachable states, regardless of attempt
    budget. A found solution longer than MAX_ACTIONS is not a verified win.
    """
    if max_expansions is not None and (type(max_expansions) is not int or max_expansions < 0):
        raise ValueError("max_expansions must be a nonnegative integer or None")
    initial = State()
    frontier = deque([(initial, ())])
    seen = {initial}
    expansions = 0
    while frontier:
        if max_expansions is not None and expansions >= max_expansions:
            return Search("UNSOLVED", reachable=frozenset(seen), expansions=expansions)
        current, path = frontier.popleft()
        expansions += 1
        if current.place == "freedom":
            if len(path) <= MAX_ACTIONS:
                return Search("VERIFIED_WIN", path, frozenset(seen), expansions)
            # The finite graph DOES have an escape; only the action budget blocks it.
            return Search("NO_WIN_WITHIN_ACTION_BUDGET", reachable=frozenset(seen),
                          expansions=expansions)
        for action in ACTIONS:
            next_state = step(exam, current, action)
            if next_state not in seen:
                seen.add(next_state)
                frontier.append((next_state, path + (action,)))
    return Search("UNSAT_IN_DECLARED_FINITE_MODEL", reachable=frozenset(seen),
                  expansions=expansions)
