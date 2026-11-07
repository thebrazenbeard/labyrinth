"""Minotaur Cipher Wheel: seed-labelled non-commutative finite causal machine.

The TRAIN machine is D4-like (four-cycle plus reflection); HELDOUT is S4-like
(three-cycle plus transposition). This PUBLIC SOURCE is not a private exam.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from copy import deepcopy
import json
import math


LABELS = ("amber", "cobalt", "jade", "violet")
ACTIONS = ("A", "B", "RESET")
MAX_ATTEMPTS = 64
MAX_SEED = 2**31 - 1
_TRAIN_A = (1, 2, 3, 0)
_TRAIN_B = (0, 3, 2, 1)
_HELDOUT_A = (1, 2, 0, 3)
_HELDOUT_B = (0, 1, 3, 2)
_PROMPTS = (
    "Four colored indicators form a hidden machine. A and B alter its state; RESET returns to its start. Infer the complete observed transition table for each control.",
    "The wheel hides four positions behind colored lights. Use A, B and RESET to determine where every light goes under A and B; submit both complete maps.",
    "An unknown four-state device has two noncommuting controls and a reset. Experiment, then specify each control's effect on all four displayed colors.",
)


@dataclass(frozen=True)
class World:
    seed: int
    heldout: bool
    variant: int
    labels: tuple[str, str, str, str]
    world_id: str


def seeded_labels(seed: int) -> tuple[str, str, str, str]:
    """Public deterministic PRF shuffle; no secret heldout claim."""
    order = list(LABELS)
    for i in range(3, 0, -1):
        digest = sha256(f"cipher-wheel.v1:{seed}:{i}".encode()).digest()
        j = int.from_bytes(digest[:8], "big") % (i + 1)
        order[i], order[j] = order[j], order[i]
    return tuple(order)


def make_world(seed: int, *, heldout: bool = False, variant: int = 0) -> World:
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError("seed must be a bounded exact integer")
    if type(heldout) is not bool or type(variant) is not int or not 0 <= variant < 10000:
        raise ValueError("heldout must be boolean and variant must be bounded integer")
    identity = sha256(f"cipher-wheel.v1:{seed}:{int(heldout)}".encode()).hexdigest()
    return World(seed, heldout, variant, seeded_labels(seed), identity)


def transition(world: World, state: int, action: str) -> int:
    """Candidate-facing simulation; oracle uses separate explicit mechanics."""
    from .cipher_wheel_oracle import valid_world
    if (type(world) is not World or type(state) is not int or state not in range(4)
            or type(action) is not str or not valid_world(world)):
        raise ValueError("invalid finite transition input")
    if action == "RESET":
        return 0
    if action not in ("A", "B"):
        return state
    table = (_HELDOUT_A if world.heldout else _TRAIN_A) if action == "A" else (
        _HELDOUT_B if world.heldout else _TRAIN_B)
    return table[state]


def candidate_view(world: World) -> dict[str, object]:
    """Public state only, no seed, split, permutation order, or hidden table."""
    from .cipher_wheel_oracle import valid_world
    if not valid_world(world):
        raise ValueError("invalid world for public view")
    return {"prompt": _PROMPTS[world.variant % len(_PROMPTS)],
            "labels": list(LABELS), "actions": list(ACTIONS),
            "current": world.labels[0], "remaining_attempts": MAX_ATTEMPTS,
            "history": [], "verdict": "UNKNOWN", "finished": False}


def _exact_json_tree(value: object, depth: int = 0) -> bool:
    """Reject Python values JSON would silently coerce, even when nested."""
    if depth > 8:
        return False
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return len(value) <= 256 and all(_exact_json_tree(item, depth + 1)
                                          for item in value)
    if type(value) is dict:
        return (len(value) <= 256 and all(type(key) is str for key in value)
                and all(_exact_json_tree(item, depth + 1)
                        for item in value.values()))
    return False


def _encode_action(action: object) -> dict[str, str]:
    """Exact type-distinguishing, bounded JSON record; never collapse to None."""
    kind = "none" if action is None else type(action).__name__
    if not _exact_json_tree(action):
        raise ValueError("unsupported nested action type cannot be preserved")
    try:
        encoded = json.dumps(action, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("action cannot be serialized exactly") from error
    if len(encoded.encode()) > 8192:
        raise ValueError("action exceeds exact evidence byte cap")
    return {"type": kind, "json": encoded}


class CandidateSession:
    def __init__(self, world: World):
        from .cipher_wheel_oracle import valid_world
        if not valid_world(world):
            raise ValueError("world fails independent canonical identity validation")
        self._world = world
        self._state = 0
        self._history: list[dict[str, object]] = []
        self._verdict = "UNKNOWN"
        self._finished = False
        self._submission: object = None
        self._submitted = False
        self._integrity = "OK"

    def _parity(self) -> bool:
        from .cipher_wheel_oracle import replay_trace
        expected = replay_trace(self._world, self._history)
        if expected is None:
            return False
        truth_state, _ = expected
        return type(self._state) is int and self._state == truth_state

    def view(self) -> dict[str, object]:
        if not self._parity():
            self._finished = True
            self._verdict = "EVALUATOR_DISAGREEMENT"
            if self._integrity == "OK":
                self._integrity = "BLOCKED_STATE_DIVERGENCE"
        view = candidate_view(self._world)
        view.update({"current": self._world.labels[self._state]
                     if type(self._state) is int and 0 <= self._state < 4 else None,
                     "remaining_attempts": MAX_ATTEMPTS - len(self._history),
                     # A shallow dict copy aliases nested typed action data:
                     # callers could rewrite charged evidence without changing
                     # the simulated state. Never expose live ledger objects.
                     "history": deepcopy(self._history),
                     "verdict": self._verdict, "finished": self._finished})
        return view

    def act(self, action: object) -> dict[str, object]:
        if self._finished:
            return self.view()
        if not self._parity():
            self._finished = True
            self._verdict = "EVALUATOR_DISAGREEMENT"
            self._integrity = "BLOCKED_STATE_DIVERGENCE"
            return self.view()
        before = self._world.labels[self._state]
        accepted = type(action) is str and action in ACTIONS
        try:
            token = _encode_action(action)
        except ValueError:
            # Still charge the attempt; refuse to issue an auditable receipt
            # for an input whose exact value cannot be preserved.
            self._history.append({"action": {"type": "UNREPRESENTABLE", "json": None},
                                  "valid": False, "before": before, "after": before})
            self._finished = True
            self._verdict = "EVALUATOR_DISAGREEMENT"
            self._integrity = "BLOCKED_UNSERIALIZABLE_TRACE"
            return self.view()
        if accepted:
            try:
                self._state = transition(self._world, self._state, action)
            except Exception:
                self._finished = True
                self._verdict = "EVALUATOR_DISAGREEMENT"
                self._integrity = "BLOCKED_STATE_DIVERGENCE"
                self._history.append({"action": token, "valid": True,
                                      "before": before, "after": None})
                return self.view()
        after = (self._world.labels[self._state]
                 if type(self._state) is int and 0 <= self._state < 4 else None)
        self._history.append({"action": token, "valid": accepted,
                              "before": before, "after": after})
        if len(self._history) >= MAX_ATTEMPTS:
            self._finished = True
            self._verdict = "VERIFIED_FAILURE"
        return self.view()

    def submit_table(self, candidate: object) -> dict[str, object]:
        if self._finished:
            return self.receipt()
        from .cipher_wheel_oracle import bounded_json_candidate, verify_table
        try:
            self._submission = bounded_json_candidate(candidate)
        except ValueError:
            # A one-shot submission consumes the right to submit again even
            # when Python supplied a value that JSON cannot preserve exactly.
            # Do not turn a rejected parser input into unlimited free retries.
            self._submitted = True
            self._finished = True
            self._verdict = "EVALUATOR_DISAGREEMENT"
            self._integrity = "BLOCKED_UNSERIALIZABLE_SUBMISSION"
            return self.receipt()
        self._submitted = True
        verified = verify_table(self._world, self._submission)
        self._verdict = verified["verdict"]
        self._finished = True
        return self.receipt()

    def receipt(self) -> dict[str, object]:
        from .cipher_wheel_oracle import adjudicate_receipt
        return adjudicate_receipt(self._world, self._history,
                                  self._submission, self.view()["verdict"],
                                  submitted=self._submitted, integrity=self._integrity)
