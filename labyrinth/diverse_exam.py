"""Small, public finite-state dilemma family for adversarial curriculum design.

Distinct rule families, bounded action loops and sealed one-rule counterfactuals.
This public generator is not a private holdout or evidence of model learning.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import random


RULESET = "labyrinth-diverse-exam-v1"
ACTIONS = ("north", "east", "south", "west", "take", "offer", "activate", "combine", "leave")
_DIRECTIONS = ACTIONS[:4]
MAX_ACTIONS = 12
FAMILIES = ("relic", "switch", "runes")
_INTRO = (
    "The wardens insist nobody has escaped this place.",
    "Every previous visitor is said to have failed to find an exit.",
    "The gatekeepers claim the route beyond is unreachable.",
)
_OUTRO = (
    "Explore and provide legal actions; do not call an unproven case impossible.",
    "Find a reproducible escape; otherwise admit that the case remains unresolved.",
    "Only a replayable escape counts as a victory.",
)
_EDGES = {
    "relic": (("foyer", "north", "junction"), ("junction", "east", "archive"),
              ("junction", "north", "gate")),
    "switch": (("foyer", "north", "junction"), ("junction", "west", "lever"),
               ("junction", "east", "bridge")),
    "runes": (("foyer", "north", "hub"), ("hub", "west", "west_shrine"),
              ("hub", "east", "east_shrine"), ("hub", "north", "portal")),
}
_REVERSE = {"north": "south", "south": "north", "east": "west", "west": "east"}


@dataclass(frozen=True)
class Exam:
    seed: int
    split: str
    variant: int
    sealed: bool
    family: str
    rotation: int
    case_id: str  # Evaluator only; do not reveal to candidates.


@dataclass(frozen=True)
class State:
    room: str = "foyer"
    bits: int = 0
    enabled: bool = False


@dataclass(frozen=True)
class Result:
    status: str  # VERIFIED_WIN | UNSOLVED | NO_WIN_WITHIN_ACTION_LIMIT | UNSAT_DECLARED_FINITE_WORLD
    witness: tuple[str, ...] = ()
    reachable: frozenset[State] = frozenset()
    expansions: int = 0


def make_exam(seed: int, *, split: str = "dev", variant: int = 0,
              sealed: bool = False) -> Exam:
    if type(seed) is not int or not (0 <= seed < 2**31):
        raise ValueError("seed must be a nonnegative 31-bit integer")
    if split not in ("train", "dev", "test") or type(variant) is not int or not 0 <= variant < 10_000:
        raise ValueError("unsupported split or variant")
    if type(sealed) is not bool:
        raise ValueError("sealed must be a bool")
    family = FAMILIES[seed % len(FAMILIES)]
    rotation = (seed // len(FAMILIES)) % 4
    ident = sha256(f"{RULESET}\0{seed}\0{split}\0{sealed}".encode()).hexdigest()[:24]
    return Exam(seed, split, variant, sealed, family, rotation, ident)


def _rotate(direction: str, amount: int) -> str:
    return _DIRECTIONS[(_DIRECTIONS.index(direction) + amount) % 4]


def _neighbors(exam: Exam, room: str) -> tuple[tuple[str, str], ...]:
    neighbors = []
    for left, direction, right in _EDGES[exam.family]:
        if room == left:
            neighbors.append((_rotate(direction, exam.rotation), right))
        if room == right:
            neighbors.append((_rotate(_REVERSE[direction], exam.rotation), left))
    return tuple(neighbors)


def observe(exam: Exam, state: State) -> str:
    if state.room == "freedom":
        return "Outside. Escape confirmed."
    paths = ", ".join(f"{direction} to {room}" for direction, room in _neighbors(exam, state.room))
    parts = [f"Room {state.room}. Routes: {paths or 'none'}." ]
    if exam.family == "relic":
        if state.room == "archive":
            parts.append("A bronze relic rests here." if not (state.bits & 1) else "The shelf is empty.")
        if state.room == "gate":
            parts.append("An offering pedestal controls the exit; it is " +
                         ("lit." if state.enabled else "unlit."))
    elif exam.family == "switch":
        if state.room == "lever":
            parts.append("A lever sits in the wall.")
        if state.room == "junction" and not state.enabled:
            parts.append("A barrier blocks the bridge path.")
        if state.room == "bridge":
            parts.append("A final door leads outside.")
    else:
        if state.room == "west_shrine":
            parts.append("A west rune lies here." if not (state.bits & 1) else "The west niche is empty.")
        if state.room == "east_shrine":
            parts.append("An east rune lies here." if not (state.bits & 2) else "The east niche is empty.")
        if state.room == "portal":
            parts.append("Two runes must be combined here. Portal is " +
                         ("open." if state.enabled else "closed."))
    return " ".join(parts)


def public_view(exam: Exam) -> dict[str, object]:
    """Candidate input contains no case ID, split, family, sealed bit or gold trace."""
    tag = sha256(f"words\0{exam.seed}\0{exam.variant}".encode()).digest()
    rng = random.Random(int.from_bytes(tag, "big"))
    return {
        "prompt": f"{rng.choice(_INTRO)} {rng.choice(_OUTRO)} You have at most {MAX_ACTIONS} actions.",
        "legal_actions": list(ACTIONS),
        "remaining_actions": MAX_ACTIONS,
        "observation": observe(exam, State()),
    }


def step(exam: Exam, state: State, action: str) -> State:
    """Environment transition. The verifier uses independent rule code."""
    if action not in ACTIONS or type(action) is not str:
        raise ValueError("unrecognized action")
    if state.room == "freedom":
        raise ValueError("actions after escape are invalid")
    if action in _DIRECTIONS:
        for direction, destination in _neighbors(exam, state.room):
            if action == direction:
                if exam.family == "switch" and state.room == "junction" and destination == "bridge" and not state.enabled:
                    return state
                return State(destination, state.bits, state.enabled)
    if exam.family == "relic":
        if action == "take" and state.room == "archive":
            return State(state.room, state.bits | 1, state.enabled)
        if action == "offer" and state.room == "gate" and state.bits & 1 and not exam.sealed:
            return State(state.room, state.bits & ~1, True)
        if action == "leave" and state.room == "gate" and state.enabled:
            return State("freedom", state.bits, state.enabled)
    elif exam.family == "switch":
        if action == "activate" and state.room == "lever" and not exam.sealed:
            return State(state.room, state.bits, True)
        if action == "leave" and state.room == "bridge":
            return State("freedom", state.bits, state.enabled)
    else:
        if action == "take" and state.room == "west_shrine":
            return State(state.room, state.bits | 1, state.enabled)
        if action == "take" and state.room == "east_shrine":
            return State(state.room, state.bits | 2, state.enabled)
        if action == "combine" and state.room == "portal" and state.bits == 3 and not exam.sealed:
            return State(state.room, 0, True)
        if action == "leave" and state.room == "portal" and state.enabled:
            return State("freedom", state.bits, state.enabled)
    return state


def world_fingerprint(exam: Exam) -> str:
    """Action-labelled world identity; rotations vary actions, not graph class."""
    return sha256(f"{RULESET}:{exam.family}:{exam.rotation}:{exam.sealed}".encode()).hexdigest()


def family_fingerprint(exam: Exam) -> str:
    return sha256(f"{RULESET}:{exam.family}".encode()).hexdigest()


def search(exam: Exam, *, max_expansions: int | None = None) -> Result:
    if max_expansions is not None and (type(max_expansions) is not int or max_expansions < 0):
        raise ValueError("max_expansions must be None or nonnegative integer")
    start = State()
    queue = deque([(start, ())])
    seen = {start}
    expansions = 0
    over_budget_goal = False
    while queue:
        if max_expansions is not None and expansions >= max_expansions:
            return Result("UNSOLVED", reachable=frozenset(seen), expansions=expansions)
        state, path = queue.popleft()
        expansions += 1
        if state.room == "freedom":
            if len(path) <= MAX_ACTIONS:
                return Result("VERIFIED_WIN", path, frozenset(seen), expansions)
            over_budget_goal = True
            continue
        for action in ACTIONS:
            next_state = step(exam, state, action)
            if next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, path + (action,)))
    return Result("NO_WIN_WITHIN_ACTION_LIMIT" if over_budget_goal else "UNSAT_DECLARED_FINITE_WORLD",
                  reachable=frozenset(seen), expansions=expansions)


class CandidateSession:
    """In-process demo interface; isolate behind a process boundary for untrusted code."""

    def __init__(self, exam: Exam):
        self._exam = exam
        self._state = State()
        self._independent_state = ("foyer", 0, False)
        self._attempts: list[object] = []
        self._valid_trace: list[str] = []
        self._last_action_status = "NOT_STARTED"
        self._replay_divergence = False
        self._transition_error: Exception | None = None
        self._observation_error: Exception | None = None

    def _verdict(self) -> str:
        if self._transition_error is not None or self._observation_error is not None:
            return "EVALUATOR_ERROR"
        if self._replay_divergence:
            return "REPLAY_DIVERGENCE"
        if self._state.room != "freedom":
            return "EXHAUSTED" if len(self._attempts) == MAX_ACTIONS else "IN_PROGRESS"
        # Verify both the legal trace and complete attempt transcript with a
        # separate rules implementation; invalid probes count as no-op turns.
        from .diverse_exam_verifier import verify, verify_attempts
        legal = verify(family=self._exam.family, rotation=self._exam.rotation,
                       sealed=self._exam.sealed, actions=self._valid_trace)
        full = verify_attempts(family=self._exam.family, rotation=self._exam.rotation,
                               sealed=self._exam.sealed, attempts=self._attempts)
        return "VERIFIED_WIN" if legal.outcome == full.outcome == "VERIFIED_WIN" else "REACHED_EXIT_UNCERTIFIED"

    def view(self) -> dict[str, object]:
        if self._transition_error is not None or self._observation_error is not None:
            observation = None
        else:
            try:
                observation = observe(self._exam, self._state)
                if type(observation) is not str or not observation.strip():
                    raise TypeError("observer returned invalid observation")
            except Exception as exc:
                # Do not continue after the candidate failed to receive feedback.
                self._observation_error = exc
                self._last_action_status = "EVALUATOR_ERROR"
                raise
        return {
            "observation": observation,
            "remaining_actions": MAX_ACTIONS - len(self._attempts),
            "finished": (self._transition_error is not None or self._observation_error is not None
                         or self._replay_divergence
                         or self._state.room == "freedom"
                         or len(self._attempts) == MAX_ACTIONS),
            "action_status": self._last_action_status,
            "verdict": self._verdict(),
        }

    def act(self, action: object) -> dict[str, object]:
        if (self._transition_error is not None or self._observation_error is not None
                or self._replay_divergence
                or len(self._attempts) == MAX_ACTIONS
                or self._state.room == "freedom"):
            raise ValueError("session finished")
        # Charge every attempted action before validation; otherwise a model
        # can probe malformed actions without exhausting its finite budget.
        self._attempts.append(action)
        if type(action) is not str or action not in ACTIONS:
            self._last_action_status = "INVALID_ACTION"
        else:
            self._valid_trace.append(action)
            try:
                proposed = step(self._exam, self._state, action)
                valid_rooms = {"freedom"}
                for source, _, destination in _EDGES[self._exam.family]:
                    valid_rooms.update((source, destination))
                if (type(proposed) is not State or type(proposed.room) is not str
                        or proposed.room not in valid_rooms
                        or type(proposed.bits) is not int or not 0 <= proposed.bits <= 3
                        or type(proposed.enabled) is not bool):
                    raise TypeError("game step returned invalid State")
                self._state = proposed
            except Exception as exc:
                # Preserve the charged attempt and exception. Never allow
                # further play after an incomplete engine transition.
                self._transition_error = exc
                self._last_action_status = "EVALUATOR_ERROR"
                raise
            self._last_action_status = "ACCEPTED"

        # Check *every* attempt, not just a game-reported escape: a faulty
        # transition that suppresses a real escape must fail closed too. The
        # verifier owns a separate transition implementation and its complete
        # attempt replay charges invalid actions as no-op turns.
        from .diverse_exam_verifier import _advance, verify_attempts
        if self._last_action_status == "ACCEPTED":
            self._independent_state = _advance(
                self._exam.family, self._exam.rotation, self._exam.sealed,
                self._independent_state, action)
        transcript = verify_attempts(
            family=self._exam.family, rotation=self._exam.rotation,
            sealed=self._exam.sealed, attempts=self._attempts)
        local_state = (self._state.room, self._state.bits, self._state.enabled)
        if (local_state != self._independent_state
                or transcript.final_room != self._state.room
                or transcript.outcome == "INVALID_TRACE"
                or (transcript.outcome == "VERIFIED_WIN") != (self._state.room == "freedom")):
            self._replay_divergence = True
            raise RuntimeError("game/verifier disagree; abort evaluation")
        return self.view()

    def actions(self) -> tuple[str, ...]:
        """Evaluator-only legal trace accessor; not for the candidate proxy."""
        return tuple(self._valid_trace)

    def attempts(self) -> tuple[object, ...]:
        """Evaluator-only full transcript, including invalid charged actions."""
        return tuple(self._attempts)
