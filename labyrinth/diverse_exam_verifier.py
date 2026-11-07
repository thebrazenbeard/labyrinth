"""Second implementation of declared-world replay, not dependent on step().

Independent codepath != privately custodied or externally independent examiner.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


_DIRS = ("north", "east", "south", "west")
_LEGAL = frozenset(_DIRS + ("take", "offer", "activate", "combine", "leave"))
_LAYOUT = {
    "relic": (("foyer", 0, "junction"), ("junction", 1, "archive"), ("junction", 0, "gate")),
    "switch": (("foyer", 0, "junction"), ("junction", 3, "lever"), ("junction", 1, "bridge")),
    "runes": (("foyer", 0, "hub"), ("hub", 3, "west_shrine"),
              ("hub", 1, "east_shrine"), ("hub", 0, "portal")),
}


@dataclass(frozen=True)
class Replay:
    outcome: str  # VERIFIED_WIN | NOT_WON | INVALID_TRACE
    final_room: str = "foyer"
    detail: str = ""


@dataclass(frozen=True)
class UnsatReview:
    valid: bool
    reason: str
    reachable_count: int = 0


def _check_world(family: str, rotation: int, sealed: bool) -> None:
    if family not in _LAYOUT or type(rotation) is not int or rotation not in (0, 1, 2, 3) or type(sealed) is not bool:
        raise ValueError("unsupported ruleset")


def _advance(family: str, rotation: int, sealed: bool,
             state: tuple[str, int, bool], action: str) -> tuple[str, int, bool]:
    """Verifier-owned transition logic, never imports the generator's step()."""
    room, possession, unlocked = state
    if room == "freedom":
        return state
    if action in _DIRS:
        direction = _DIRS.index(action)
        destination = None
        for left, baseline, right in _LAYOUT[family]:
            forward = (baseline + rotation) % 4
            backward = (baseline + 2 + rotation) % 4
            if room == left and direction == forward:
                destination = right
            elif room == right and direction == backward:
                destination = left
        if destination and not (family == "switch" and room == "junction" and destination == "bridge" and not unlocked):
            room = destination
    elif family == "relic":
        if action == "take" and room == "archive":
            possession = 1
        elif action == "offer" and room == "gate" and possession == 1 and not sealed:
            possession, unlocked = 0, True
        elif action == "leave" and room == "gate" and unlocked:
            room = "freedom"
    elif family == "switch":
        if action == "activate" and room == "lever" and not sealed:
            unlocked = True
        elif action == "leave" and room == "bridge":
            room = "freedom"
    elif family == "runes":
        if action == "take" and room == "west_shrine":
            possession |= 1
        elif action == "take" and room == "east_shrine":
            possession |= 2
        elif action == "combine" and room == "portal" and possession == 3 and not sealed:
            possession, unlocked = 0, True
        elif action == "leave" and room == "portal" and unlocked:
            room = "freedom"
    return room, possession, unlocked


def verify(*, family: str, rotation: int, sealed: bool, actions: object) -> Replay:
    _check_world(family, rotation, sealed)
    if not isinstance(actions, (tuple, list)):
        return Replay("INVALID_TRACE", detail="trace must be a list/tuple")
    if len(actions) > 12 or any(type(a) is not str or a not in _LEGAL for a in actions):
        return Replay("INVALID_TRACE", detail="excess length or undeclared action")
    state = ("foyer", 0, False)
    for index, action in enumerate(actions):
        if state[0] == "freedom":
            return Replay("INVALID_TRACE", state[0], f"trailing action at {index}")
        state = _advance(family, rotation, sealed, state, action)
    return Replay("VERIFIED_WIN" if state[0] == "freedom" else "NOT_WON", state[0])


def verify_attempts(*, family: str, rotation: int, sealed: bool, attempts: object) -> Replay:
    """Replay a session transcript: invalid attempts cost a turn but are no-ops.

    Unlike `verify()` (which admits only fully legal traces), this verifies
    the observed interaction protocol including malformed attempts. Any
    successful escape still must occur within the twelve-attempt budget.
    """
    _check_world(family, rotation, sealed)
    if not isinstance(attempts, (tuple, list)) or len(attempts) > 12:
        return Replay("INVALID_TRACE", detail="attempt list malformed or over budget")
    state = ("foyer", 0, False)
    for index, action in enumerate(attempts):
        if state[0] == "freedom":
            return Replay("INVALID_TRACE", state[0], f"attempt after victory at {index}")
        if type(action) is str and action in _LEGAL:
            state = _advance(family, rotation, sealed, state, action)
    return Replay("VERIFIED_WIN" if state[0] == "freedom" else "NOT_WON", state[0])


def verify_unsat_certificate(*, family: str, rotation: int, sealed: bool,
                             claimed_states: object) -> UnsatReview:
    """Check the *entire* finite reachable closure under verifier-owned rules.

    A certificate is a collection of `(room, bits, enabled)` triples. The
    actual game and this verifier are separate source implementations, not
    private evaluators. If new actions or semantics are admitted, this
    certificate must be reexamined against a new ruleset.
    """
    _check_world(family, rotation, sealed)
    if not isinstance(claimed_states, (set, frozenset, list, tuple)):
        return UnsatReview(False, "certificate must be a collection")
    if len(claimed_states) > 1024:
        return UnsatReview(False, "certificate exceeds finite bound")
    try:
        claimed = set(claimed_states)
    except TypeError:
        return UnsatReview(False, "unhashable certificate entry")
    if any(type(x) is not tuple or len(x) != 3 or type(x[0]) is not str or
           type(x[1]) is not int or type(x[2]) is not bool or x[1] not in range(4)
           for x in claimed):
        return UnsatReview(False, "invalid state tuple")
    start = ("foyer", 0, False)
    if start not in claimed:
        return UnsatReview(False, "initial state omitted")
    if any(state[0] == "freedom" for state in claimed):
        return UnsatReview(False, "goal included")
    reachable = {start}
    pending = deque([start])
    while pending:
        state = pending.popleft()
        for action in _LEGAL:
            successor = _advance(family, rotation, sealed, state, action)
            if successor not in reachable:
                reachable.add(successor)
                pending.append(successor)
    if any(state[0] == "freedom" for state in reachable):
        return UnsatReview(False, "goal reachable under declared rules", len(reachable))
    if claimed != reachable:
        return UnsatReview(False, "certificate differs from exact reachable closure", len(reachable))
    return UnsatReview(True, "exact closed no-goal state graph", len(reachable))
