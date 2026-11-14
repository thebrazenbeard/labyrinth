"""Generated finite causal register with genuine mid-episode rule revision.

Five distinct arithmetic/Boolean mechanism generators, not seed relabelings.
This is PUBLIC SOURCE for hostile evaluations, not a private learning exam.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256


FAMILIES = ("feedback_shift", "rotate_add", "threshold_gate", "carry_fold",
            "majority_mask")
CONTROLS = ("0", "1", "WAIT", "COMMIT")
MAX_MOVES = 9
MAX_PROBES = 16
MAX_SEED = 2**31 - 1


@dataclass(frozen=True)
class Rule:
    family: int
    mask: int
    bias: int
    threshold: int


@dataclass(frozen=True)
class Case:
    seed: int
    before: Rule
    after: Rule
    initial: int
    target: int
    switch_at: int
    max_pulses: int
    mechanic_signature: str
    case_id: str


@dataclass(frozen=True)
class State:
    register: int
    ticks: int = 0
    pulses: int = 0
    injected_after: bool = False
    committed: bool = False
    won: bool = False


def _digest(seed: int, tag: str) -> bytes:
    return sha256(f"switchback-circuit.v1:{seed}:{tag}".encode()).digest()


def _rule(seed: int, tag: str) -> Rule:
    data = _digest(seed, tag)
    return Rule(data[0] % len(FAMILIES), 1 + data[1] % 7,
                data[2] % 8, 1 + data[3] % 3)


def kernel(register: int, bit: int, rule: Rule) -> int:
    """DUT path; independent oracle implements each mechanism separately."""
    if (type(register) is not int or register not in range(8)
            or type(bit) is not int or bit not in (0, 1)
            or type(rule) is not Rule):
        raise ValueError("invalid finite circuit input")
    family, mask, bias, threshold = (rule.family, rule.mask,
                                     rule.bias, rule.threshold)
    if family == 0:
        return (((register << 1) | bit) ^
                (((register & mask).bit_count() & 1) << 2) ^ bias) & 7
    if family == 1:
        return ((((register << 1) | (register >> 2)) & 7) + bit + bias) & 7
    if family == 2:
        return (((register << 1) | bit) ^
                (mask if register.bit_count() >= threshold else bias)) & 7
    if family == 3:
        return (register + bit + (register & mask).bit_count() + bias) & 7
    if family == 4:
        return (register ^ (mask if (register & mask).bit_count() >= threshold
                            else 0) ^ bit ^ bias) & 7
    raise ValueError("unsupported rule family")


def _signature(before: Rule, after: Rule, switch_at: int, max_pulses: int) -> str:
    truth = bytes(kernel(r, b, rule) for rule in (before, after)
                  for r in range(8) for b in (0, 1))
    return sha256(b"switchback.mechanic.v1\0" + bytes([switch_at, max_pulses])
                  + truth).hexdigest()


def make_case(seed: int) -> Case:
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError("seed must be a bounded exact int")
    first = _rule(seed, "pre")
    second = _rule(seed, "post")
    # Deterministic rejection of identical truth tables. A rule revision must
    # change actual observable mechanics, not merely source labels.
    for _ in range(8):
        if tuple(kernel(r, b, first) for r in range(8) for b in (0, 1)) != tuple(
                kernel(r, b, second) for r in range(8) for b in (0, 1)):
            break
        second = Rule(second.family, second.mask, (second.bias + 1) % 8,
                      second.threshold)
    else:
        raise ValueError("unable to generate distinct source kernels")
    data = _digest(seed, "case")
    initial, target = data[0] % 8, data[1] % 8
    switch_at, max_pulses = 2 + data[2] % 3, 1 + data[3] % 4
    mechanic = _signature(first, second, switch_at, max_pulses)
    identity = sha256(f"switchback.case.v1:{seed}:{mechanic}:{initial}:{target}".encode()).hexdigest()
    return Case(seed, first, second, initial, target, switch_at,
                max_pulses, mechanic, identity)


def public_view(case: Case) -> dict[str, object]:
    return {"prompt": "A three-bit causal register changes its update law partway through the episode. Inputs 0 and 1 update it; WAIT advances time without input. COMMIT succeeds only on the target value after the change, with a post-change input. Submit any legal winning plan or a bounded uncertainty claim.",
            "initial_register": format(case.initial, "03b"),
            "target_register": format(case.target, "03b"),
            "plan_actions": list(CONTROLS),
            "probe_actions": ["0", "1", "WAIT", "RESET"],
            "maximum_moves": MAX_MOVES,
            "switch_at": case.switch_at, "maximum_pulses": case.max_pulses}


def step(case: Case, state: State, action: str) -> State:
    if type(case) is not Case or type(state) is not State or type(action) is not str:
        raise ValueError("invalid step input")
    if state.committed or state.ticks >= MAX_MOVES or action not in CONTROLS:
        return state
    if action == "COMMIT":
        return State(state.register, state.ticks, state.pulses, state.injected_after,
                     True, state.ticks >= case.switch_at and state.injected_after
                     and state.register == case.target)
    if action == "WAIT":
        return State(state.register, state.ticks + 1, state.pulses,
                     state.injected_after)
    if state.pulses >= case.max_pulses:
        return state
    revised = state.ticks >= case.switch_at
    register = kernel(state.register, int(action), case.after if revised else case.before)
    return State(register, state.ticks + 1, state.pulses + 1,
                 state.injected_after or revised)


class ProbeSession:
    """A bounded public JSON observation channel, not a secure host sandbox."""

    def __init__(self, case: Case):
        from .switchback_oracle import _valid_case
        if not _valid_case(case):
            raise ValueError("invalid source case")
        self._case = case
        self._state = State(case.initial)
        self._history: list[dict[str, object]] = []
        self._finished = False
        self._result: dict[str, object] | None = None

    def view(self) -> dict[str, object]:
        public = public_view(self._case)
        public.update({"current_register": format(self._state.register, "03b"),
                       "ticks": self._state.ticks, "pulses_used": self._state.pulses,
                       "remaining_probes": MAX_PROBES - len(self._history),
                       "observations": deepcopy(self._history),
                       "finished": self._finished})
        return public

    def probe(self, action: object) -> dict[str, object]:
        if self._finished:
            return self.view()
        if len(self._history) >= MAX_PROBES:
            # Sixteenth probe is legal and a final answer remains available;
            # a seventeenth probe is a terminal budget violation.
            self._finished = True
            self._result = {"verdict": "VERIFIED_FAILURE",
                            "reason": "PROBE_BUDGET_EXCEEDED",
                            "verified_success": False}
            return self.view()
        if type(action) is not str or len(action) > 32:
            self._history.append({"action": "UNREPRESENTABLE", "valid": False,
                                  "applied": False,
                                  "before": format(self._state.register, "03b"),
                                  "after": None})
            self._finished = True
            self._result = {"verdict": "EVALUATOR_DISAGREEMENT",
                            "reason": "BLOCKED_UNSERIALIZABLE_ACTION",
                            "verified_success": False}
            return self.view()
        before = format(self._state.register, "03b")
        valid = action in ("0", "1", "WAIT", "RESET")
        applied = (action == "RESET" or
                   (valid and self._state.ticks < MAX_MOVES and
                    (action == "WAIT" or self._state.pulses < self._case.max_pulses)))
        if valid:
            try:
                from .switchback_oracle import State as OracleState, oracle_step
                if action == "RESET":
                    advanced = State(self._case.initial)
                    expected = OracleState(self._case.initial)
                else:
                    advanced = step(self._case, self._state, action)
                    expected = oracle_step(self._case, self._state, action)
                if (type(advanced) is not type(expected)
                        or type(advanced.register) is not int
                        or not 0 <= advanced.register < 8
                        or type(advanced.ticks) is not int
                        or not 0 <= advanced.ticks <= MAX_MOVES
                        or type(advanced.pulses) is not int
                        or not 0 <= advanced.pulses <= self._case.max_pulses
                        or any(type(flag) is not bool for flag in (
                            advanced.injected_after, advanced.committed, advanced.won))):
                    raise ValueError("malformed DUT probe transition")
                # Independent per-transition parity, rather than only replay
                # after a final plan. A faulty DUT observation fails closed.
                if advanced != expected:
                    raise ValueError("DUT probe transition disagrees with oracle")
                self._state = advanced
            except Exception:
                # A crashing/mutated evaluator must never grant free retries
                # or permit a later valid plan to wash out the failed probe.
                self._history.append({"action": action, "valid": True,
                                      "applied": False, "before": before, "after": None})
                self._finished = True
                self._result = {"verdict": "EVALUATOR_DISAGREEMENT",
                                "reason": "BLOCKED_PROBE_EXECUTION",
                                "verified_success": False}
                return self.view()
        self._history.append({"action": action, "valid": valid, "applied": applied,
                              "before": before, "after": format(self._state.register, "03b")})
        return self.view()

    def submit(self, proposal: object) -> dict[str, object]:
        from .switchback_oracle import (adjudicate_session, verify_probe_trace,
                                        verify_receipt_replay)
        if self._finished:
            return deepcopy(self._result) if self._result is not None else {
                "verdict": "UNKNOWN", "verified_success": False}
        self._finished = True
        try:
            if not verify_probe_trace(self._case, self._history):
                self._result = {"verdict": "EVALUATOR_DISAGREEMENT",
                                "reason": "INDEPENDENT_PROBE_REPLAY_DIVERGED",
                                "verified_success": False}
            else:
                candidate = adjudicate_session(self._case, self._history, proposal)
                if (type(candidate) is not dict or candidate.get("schema") !=
                        "labyrinth.switchback-session-receipt.v1" or
                        candidate.get("case_id") != self._case.case_id or
                        not verify_receipt_replay(candidate)):
                    raise ValueError("submission receipt failed independent replay")
                self._result = candidate
        except Exception:
            self._result = {"verdict": "EVALUATOR_DISAGREEMENT",
                            "reason": "BLOCKED_SUBMISSION_EVALUATOR",
                            "verified_success": False}
        return deepcopy(self._result)
