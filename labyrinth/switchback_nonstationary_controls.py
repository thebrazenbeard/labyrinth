"""Bounded source-public controls for a changing Switchback causal rule.

Three deterministic ZERO-UPDATE Python baselines compare: (a) a source-aware
open-loop plan with no observations, (b) feedback with an explicit rule-revision
reset, and (c) feedback that incorrectly retains the prechange hypothesis.
Candidate code receives only public `view/probe/submit` and cannot access
Case, rules, seed, hidden state, or a privileged solver.

The source law is public: knowing its five generator templates is not a
private, blind, independently custodied neural generalization experiment.
Crucially, the public prompt explicitly declares `switch_at`; this policy
does not discover a surprise change point from experience. It merely obeys
the disclosed one and adapts its postchange source hypothesis using feedback.
Every result denies model training, graduation and independent family claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Protocol

from .switchback_circuit import Rule, kernel


class PublicSwitchbackPort(Protocol):
    def view(self) -> dict[str, object]: ...
    def probe(self, action: str) -> dict[str, object]: ...
    def submit(self, proposal: dict[str, object]) -> dict[str, object]: ...


@dataclass(frozen=True)
class NonstationaryResult:
    policy: str
    verdict: str
    proposal: str
    probes: tuple[str, ...]
    plan: tuple[str, ...] | None
    remaining_rule_functions: int
    contradicted_stationary_assumption: bool
    outcome: dict[str, object]
    model_checkpoint: str = "NONE"
    optimizer_updates: int = 0
    neural_learning_established: bool = False
    graduation_authority: str = "none"
    independent_mechanic_families_verified: bool = False


# Deduplicate parameterizations that happen to produce identical truth tables.
# These are source-public programs, not source-hidden target rule instances.
_TRANSITION_TABLES = tuple(dict.fromkeys(
    tuple(tuple(kernel(register, bit, Rule(family, mask, bias, threshold))
                for bit in (0, 1)) for register in range(8))
    for family, mask, bias, threshold in product(range(5), range(1, 8),
                                                range(8), range(1, 4))
))
SOURCE_RULE_FUNCTIONS = len(_TRANSITION_TABLES)
_ALL = tuple(range(SOURCE_RULE_FUNCTIONS))


def _bits(value: object) -> int:
    if (type(value) is not str or len(value) != 3
            or any(char not in "01" for char in value)):
        raise ValueError("public register must contain exactly three bits")
    return int(value, 2)


def _parameters(port: PublicSwitchbackPort) -> tuple[int, int, int, int, int]:
    view = port.view()
    start, target = _bits(view["initial_register"]), _bits(view["target_register"])
    switch_at, max_pulses, max_moves = (view["switch_at"],
                                       view["maximum_pulses"], view["maximum_moves"])
    if (any(type(number) is not int for number in
            (switch_at, max_pulses, max_moves))
            or not 2 <= switch_at <= 4 or not 1 <= max_pulses <= 4
            or max_moves != 9):
        raise ValueError("unsupported public Switchback rule/budget")
    return start, target, switch_at, max_pulses, max_moves


def _finish(port: PublicSwitchbackPort, policy: str, probes: list[str],
            plan: list[str] | None, remaining: int, contradiction: bool
            ) -> NonstationaryResult:
    proposal = {"kind": "PLAN", "actions": list(plan)} if plan is not None else {
        "kind": "ABSTAIN"}
    response = port.submit(proposal)
    if type(response) is not dict or type(response.get("verdict")) is not str:
        raise ValueError("public examiner failed to provide a typed verdict")
    return NonstationaryResult(policy, response["verdict"], proposal["kind"],
                               tuple(probes), tuple(plan) if plan else None,
                               remaining, contradiction, response)


def open_loop_source_plan(port: PublicSwitchbackPort) -> NonstationaryResult:
    """Best fixed post-switch action sequence in source-public rule ensemble.

    No public probe is made. The plan may be WRONG; that failure is preserved
    in the evaluator response. No target source seed, rule object or oracle.
    """
    initial, target, switch_at, max_pulses, max_moves = _parameters(port)
    max_bits = min(max_pulses, max_moves - switch_at - 1)
    best_score = -1
    best_seq: tuple[int, ...] = ()
    for length in range(1, max_bits + 1):
        for bits in product((0, 1), repeat=length):
            hits = 0
            for table in _TRANSITION_TABLES:
                current = initial
                for bit in bits:
                    current = table[current][bit]
                hits += int(current == target)
            if hits > best_score:
                best_score, best_seq = hits, bits
    proposed = ["WAIT"] * switch_at + [str(bit) for bit in best_seq] + ["COMMIT"]
    return _finish(port, "open_loop_source_plan", [], proposed,
                   SOURCE_RULE_FUNCTIONS, False)


def revision_aware_feedback(port: PublicSwitchbackPort) -> NonstationaryResult:
    """Observe one prechange step, RESET, then learn post-change afresh."""
    return _feedback_policy(port, allow_revision=True)


def falsely_stationary_feedback(port: PublicSwitchbackPort) -> NonstationaryResult:
    """Hostile ablation: wrongly keep prechange rule posterior after switch."""
    return _feedback_policy(port, allow_revision=False)


def _groups(candidates: tuple[int, ...], state: int, bit: int
            ) -> dict[int, tuple[int, ...]]:
    buckets: dict[int, list[int]] = {}
    for index in candidates:
        output = _TRANSITION_TABLES[index][state][bit]
        buckets.setdefault(output, []).append(index)
    return {output: tuple(indices) for output, indices in buckets.items()}


def _best_bit(candidates: tuple[int, ...], state: int, target: int,
              remaining: int) -> int:
    @lru_cache(maxsize=10000)
    def optimal_hits(posterior: tuple[int, ...], register: int, horizon: int) -> int:
        if horizon <= 0 or not posterior:
            return 0
        totals = []
        for bit in (0, 1):
            score = sum(len(group) if output == target else
                        optimal_hits(group, output, horizon - 1)
                        for output, group in _groups(posterior, register, bit).items())
            totals.append(score)
        return max(totals)

    return max((0, 1), key=lambda bit: (
        sum(len(group) if output == target else
            optimal_hits(group, output, remaining - 1)
            for output, group in _groups(candidates, state, bit).items()),
        -bit))


def _feedback_policy(port: PublicSwitchbackPort, *, allow_revision: bool
                     ) -> NonstationaryResult:
    initial, target, switch_at, max_pulses, max_moves = _parameters(port)
    policy = ("revision_aware_feedback" if allow_revision
              else "falsely_stationary_feedback")
    probes: list[str] = []

    # Same prechange observation costs for both policies. The experiment
    # differs only in whether its posterior is reset at the declared switch.
    pre = port.probe("1")
    probes.append("1")
    pre_output = _bits(pre["current_register"])
    pre_posterior = _groups(_ALL, initial, 1).get(pre_output, ())
    port.probe("RESET")
    probes.append("RESET")
    plan: list[str] = []
    for _ in range(switch_at):
        port.probe("WAIT")
        probes.append("WAIT")
        plan.append("WAIT")

    candidates = _ALL if allow_revision else pre_posterior
    contradiction = False
    pulse_limit = min(max_pulses, max_moves - switch_at - 1)
    for step in range(pulse_limit):
        if not candidates:
            contradiction = True
            break
        visible = port.view()
        if visible.get("finished") or visible.get("remaining_probes", 0) <= 0:
            break
        before = _bits(visible["current_register"])
        bit = _best_bit(candidates, before, target, pulse_limit - step)
        view = port.probe(str(bit))
        probes.append(str(bit))
        plan.append(str(bit))
        after = _bits(view["current_register"])
        candidates = _groups(candidates, before, bit).get(after, ())
        if not candidates:
            contradiction = True
        # A directly observed valid path is decisive even when the false
        # stationary hypothesis class is contradicted by this same evidence.
        if after == target:
            return _finish(port, policy, probes, plan + ["COMMIT"],
                           len(candidates), contradiction)
    return _finish(port, policy, probes, None, len(candidates), contradiction)
