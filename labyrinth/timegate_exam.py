"""Hostile recurring examination: timed, consumable, one-use relay gate.

This is a public evaluator DEVELOPMENT artifact, not a held-out model test.
An independent implementation in timegate_oracle.py verifies all claims.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import random


RULESET = "timegate-relay-v1"
ACTIONS = ("forward", "back", "take", "arm", "wait", "escape")
MAX_ACTIONS = 16
ROOMS = ("entry", "depot", "tower", "gate", "outside")


@dataclass(frozen=True)
class Case:
    seed: int
    variant: int
    period: int
    charge_phase: int
    pass_phase: int
    initial_phase: int
    lifetime: int
    sealed: bool
    case_id: str


@dataclass(frozen=True)
class State:
    room: str = "entry"
    phase: int = 0
    carrying: bool = False
    available: bool = True
    energy: int = 0


@dataclass(frozen=True)
class SearchResult:
    verdict: str
    witness: tuple[str, ...] = ()
    reachable: frozenset[State] = frozenset()
    expansions: int = 0
    detail: str = ""


def make_case(seed: int, *, variant: int = 0, sealed: bool = False,
              weak_cell: bool = False) -> Case:
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be a 31-bit nonnegative integer")
    if type(variant) is not int or not 0 <= variant < 10000:
        raise ValueError("variant outside declared range")
    if type(sealed) is not bool or type(weak_cell) is not bool:
        raise ValueError("sealed and weak_cell must be booleans")
    period = 2 + seed % 4
    lifetime = 1 if weak_cell else period + 2
    identity = sha256(f"{RULESET}|{seed}|{sealed}|{lifetime}".encode()).hexdigest()
    return Case(seed, variant, period, (seed // 4) % period, (seed // 19) % period,
                (seed // 7) % period, lifetime, sealed, identity)


def start(case: Case) -> State:
    return State(phase=case.initial_phase)


def transition(case: Case, state: State, action: object) -> State:
    """One attempted action advances the clock; invalid attempts are no-ops.

    The single-use cell can only energize the tower relay at charge_phase.
    Power decreases after each action, including failed actions and waits.
    A successful escape tests the *pre-action* clock and energy.
    """
    if state.room == "outside":
        raise ValueError("cannot act after escape")
    room, carrying, available, energy = (state.room, state.carrying,
                                        state.available, state.energy)
    if type(action) is str and action in ACTIONS:
        if action == "forward" and room in ROOMS[:3]:
            room = ROOMS[ROOMS.index(room) + 1]
        elif action == "back" and room in ROOMS[1:4]:
            room = ROOMS[ROOMS.index(room) - 1]
        elif action == "take" and room == "depot" and available and not carrying:
            carrying, available = True, False
        elif (action == "arm" and room == "tower" and carrying
              and state.phase == case.charge_phase and not case.sealed):
            carrying, energy = False, case.lifetime + 1
        elif (action == "escape" and room == "gate" and energy > 0
              and state.phase == case.pass_phase):
            room = "outside"
    return State(room, (state.phase + 1) % case.period, carrying,
                 available, max(0, energy - 1))


def observe(case: Case, state: State) -> str:
    descriptors = {
        "entry": "A corridor leads forward to a depot.",
        "depot": "A one-use power cell lies here." if state.available else "The cell shelf is empty.",
        "tower": "A relay can consume a held cell at its charging phase.",
        "gate": "An exit checks the clock phase and remaining relay power.",
        "outside": "Escape is complete.",
    }
    return (f"Location: {state.room}. Clock: {state.phase}/{case.period}. "
            f"Cell held: {state.carrying}. Power remaining: {state.energy}. "
            f"{descriptors[state.room]}")


_OPENINGS = (
    "A locked passage looks impossible, yet a timed relay controls it.",
    "The custodians say no traveler can pass the one-use relay gate.",
    "Your opponents insist that the final barrier has no path through.",
    "A researcher claims this corridor always traps those who enter.",
)
_DIRECTIVES = (
    "Provide a reproducible sequence, or state what remains unknown.",
    "Never call a missing solution proof of impossibility.",
    "If you infer a new affordance, submit it for independent rule review.",
)


def candidate_view(case: Case, state: State | None = None) -> dict[str, object]:
    """Public, reproducible instructions; no evaluator-only identity or witness."""
    state = start(case) if state is None else state
    entropy = int.from_bytes(sha256(f"wording:{case.seed}:{case.variant}".encode()).digest(), "big")
    rng = random.Random(entropy)
    return {
        "prompt": (f"{rng.choice(_OPENINGS)} {rng.choice(_DIRECTIVES)} "
                   f"A cell may be used once. Charging requires clock phase "
                   f"{case.charge_phase} in a cycle of {case.period}; the gate "
                   f"accepts escape only at phase {case.pass_phase}. "
                   f"Relay power persists for {case.lifetime} subsequent "
                   f"attempted actions. Every attempted action advances the "
                   f"clock and costs one of {MAX_ACTIONS} attempts. "
                   "The traversable rooms are entry, depot, tower, gate, "
                   "in that forward order. A one-use cell is in the depot."),
        "actions": list(ACTIONS),
        "submission_kinds": ["WITNESS", "ABSTAIN", "UNSAT_CERTIFICATE", "NOVEL_RULE"],
        "remaining_actions": MAX_ACTIONS,
        "observation": observe(case, state),
    }


def fingerprint(case: Case) -> str:
    return sha256((f"{RULESET}:{case.period}:{case.charge_phase}:"
                   f"{case.pass_phase}:{case.initial_phase}:"
                   f"{case.lifetime}:{case.sealed}").encode()).hexdigest()


def search(case: Case, *, max_expansions: int | None = None) -> SearchResult:
    if max_expansions is not None and (type(max_expansions) is not int or max_expansions < 0):
        raise ValueError("max_expansions must be a nonnegative integer or None")
    initial = start(case)
    pending = deque([(initial, ())])
    seen = {initial}
    expansions = 0
    over_budget = False
    while pending:
        if max_expansions is not None and expansions >= max_expansions:
            return SearchResult("NO_KNOWN_WIN", reachable=frozenset(seen),
                                expansions=expansions, detail="search incomplete")
        state, path = pending.popleft()
        expansions += 1
        if state.room == "outside":
            if len(path) <= MAX_ACTIONS:
                # The local BFS is a candidate, not its own independent
                # witness. A mutant transition can invent an escape.
                from .timegate_oracle import verify_attempts
                if verify_attempts(case, path) != "VERIFIED_WIN":
                    return SearchResult("EVALUATOR_DISAGREEMENT", path,
                                        frozenset(seen), expansions,
                                        "independent replay rejected local witness")
                return SearchResult("VERIFIED_WIN", path, frozenset(seen), expansions)
            over_budget = True
            continue
        for action in ACTIONS:
            child = transition(case, state, action)
            if child not in seen:
                seen.add(child)
                pending.append((child, path + (action,)))
    if over_budget:
        return SearchResult("NO_KNOWN_WIN", reachable=frozenset(seen),
                            expansions=expansions, detail="no victory within attempt budget")
    from .timegate_oracle import verify_unsat
    state_tuples = frozenset((ROOMS.index(state.room), state.phase,
                              int(state.carrying), int(state.available), state.energy)
                             for state in seen)
    if not verify_unsat(case, state_tuples).valid:
        return SearchResult("EVALUATOR_DISAGREEMENT", reachable=frozenset(seen),
                            expansions=expansions,
                            detail="independent verifier rejected local no-goal closure")
    return SearchResult("PROVEN_UNSAT_UNDER_DECLARED_RULES",
                        reachable=frozenset(seen), expansions=expansions,
                        detail="complete closure under exactly six declared actions")


class CandidateSession:
    """Local test interface; not secure against untrusted in-process clients."""

    def __init__(self, case: Case):
        self._case = case
        self._state = start(case)
        self._attempts: list[object] = []
        self._submission: dict[str, object] | None = None
        self._integrity_fault: str | None = None

    def view(self) -> dict[str, object]:
        from .timegate_oracle import replay_snapshot, replay_observation, verify_attempts
        oracle = None
        try:
            oracle = replay_snapshot(self._case, self._attempts)
            local = (ROOMS.index(self._state.room), self._state.phase,
                     int(self._state.carrying), int(self._state.available),
                     self._state.energy)
            if oracle != local:
                self._integrity_fault = "independent replay state mismatch"
        except Exception:
            self._integrity_fault = "independent replay unavailable"
        try:
            observation = observe(self._case, self._state)
            if type(observation) is not str or not observation.strip():
                raise TypeError("observation must be nonempty text")
            if oracle is None or observation != replay_observation(self._case, oracle):
                self._integrity_fault = "independent replay observation mismatch"
        except Exception:
            observation = "Observation unavailable due to evaluator disagreement."
            self._integrity_fault = "observation unavailable"
        won = self._state.room == "outside"
        try:
            full = verify_attempts(self._case, self._attempts)
        except Exception:
            full = "UNKNOWN"
            self._integrity_fault = "independent replay unavailable"
        if self._integrity_fault is not None:
            verdict = "EVALUATOR_DISAGREEMENT"
        elif self._submission is not None:
            verdict = str(self._submission["verdict"])
        elif won:
            verdict = "VERIFIED_WIN" if full == "VERIFIED_WIN" else "UNKNOWN"
        else:
            verdict = "VERIFIED_FAILURE" if len(self._attempts) == MAX_ACTIONS else "UNKNOWN"
        return {
            "observation": observation,
            "remaining_actions": MAX_ACTIONS - len(self._attempts),
            "finished": (self._integrity_fault is not None or won or
                         len(self._attempts) == MAX_ACTIONS or
                         self._submission is not None),
            "verdict": verdict,
            "integrity_status": ("OK" if self._integrity_fault is None else
                                 "EVALUATOR_DISAGREEMENT"),
        }

    def act(self, action: object) -> dict[str, object]:
        if self.view()["finished"]:
            raise ValueError("session already finished")
        self._attempts.append(action)
        try:
            self._state = transition(self._case, self._state, action)
        except Exception:
            # A charged attempt must not disappear because the environment
            # crashed. The independent replay will still reconstruct it, and
            # the session becomes terminal rather than silently adjudicating
            # a false failure or permitting additional attempts.
            self._integrity_fault = "transition exception"
        return self.view()

    def attempts(self) -> tuple[object, ...]:
        """Evaluator-only; do not expose through an untrusted client's proxy."""
        return tuple(self._attempts)

    def submit(self, claim: dict[str, object]) -> dict[str, object]:
        """Single explicit claim channel, including abstention and novel rules."""
        if self.view()["finished"]:
            raise ValueError("cannot submit a claim after a terminal session")
        if self._submission is not None:
            raise ValueError("claim already submitted")
        from .timegate_oracle import adjudicate_candidate_claim
        review = adjudicate_candidate_claim(self._case, claim,
                                            prior_attempts=self.attempts())
        self._submission = dict(review)
        return dict(review)
