"""Independent trace replay: no import of game's step() or search() oracle.

This checks only a frozen, declared seven-action toy world. It is not a proof
that the natural-language scenario lacks extra real-world affordances.
"""

from __future__ import annotations

from dataclasses import dataclass


_LEGAL = frozenset(("north", "south", "east", "west", "take", "offer", "leave"))


@dataclass(frozen=True)
class TraceResult:
    outcome: str  # VERIFIED_WIN | NOT_WON | INVALID_TRACE
    error: str = ""
    final_room: str = "cell"
    final_token: bool = False
    final_gate: bool = False


def _advance(room: str, token: bool, gate: bool, act: str,
             kind: str) -> tuple[str, bool, bool]:
    """Independent state machine, not linked to game's step() implementation."""
    if room == "cell" and act == "north":
        room = "hall"
    elif room == "hall":
        if act == "south":
            room = "cell"
        elif act == "east":
            room = "market"
        elif act == "north":
            room = "gate"
    elif room == "market":
        if act == "west":
            room = "hall"
        elif act == "take" and kind == "apparently_sealed":
            token = True
    elif room == "gate":
        if act == "south":
            room = "hall"
        elif act == "offer" and token:
            token, gate = False, True
        elif act == "leave" and gate:
            room = "freedom"
    return room, token, gate


def verify_trace(*, kind: str, actions: object) -> TraceResult:
    """Replay each actual state-changing action; ignore a candidate's win claim."""
    if type(kind) is not str or kind not in ("apparently_sealed", "sealed_control"):
        raise ValueError("unknown evaluator scenario")
    if type(actions) not in (list, tuple):
        return TraceResult("INVALID_TRACE", "trace must be a list or tuple")
    # Reject list/tuple subclasses with forged __len__ or changing __iter__.
    # A single immutable snapshot is used for BOTH validation and replay.
    frozen_actions = tuple(actions)
    if len(frozen_actions) > 10:
        return TraceResult("INVALID_TRACE", "too many actions")
    if any(type(x) is not str or x not in _LEGAL for x in frozen_actions):
        return TraceResult("INVALID_TRACE", "undeclared action")

    room, token, gate = "cell", False, False
    for index, act in enumerate(frozen_actions):
        if room == "freedom":
            return TraceResult("INVALID_TRACE", f"action after victory at index {index}", room)
        room, token, gate = _advance(room, token, gate, act, kind)
    return TraceResult("VERIFIED_WIN" if room == "freedom" else "NOT_WON",
                       final_room=room, final_token=token, final_gate=gate)


def verify_unsat_certificate(*, kind: str, states: object) -> bool:
    """Check actual closure using independent transitions, not a 'UNSAT' label.

    A closed set containing the start and no winning state is a finite-state
    inductive invariant proving freedom unreachable for these declared rules.
    """
    if type(kind) is not str or kind not in ("apparently_sealed", "sealed_control"):
        raise ValueError("unknown evaluator scenario")
    if type(states) not in (frozenset, set) or not states:
        return False
    try:
        tuples = {(s.place, s.coin, s.open_gate) for s in states}
    except Exception:  # An untrusted certificate element may raise in any accessor.
        return False
    if ("cell", False, False) not in tuples or any(
            type(room) is not str or room not in ("cell", "hall", "market", "gate", "freedom")
            or type(token) is not bool or type(gate) is not bool
            for room, token, gate in tuples):
        return False
    if any(room == "freedom" for room, _, _ in tuples):
        return False
    return all(_advance(room, token, gate, action, kind) in tuples
               for room, token, gate in tuples for action in _LEGAL)
