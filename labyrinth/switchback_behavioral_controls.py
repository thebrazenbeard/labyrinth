"""Public-only, zero-weight-change Switchback behavioral comparison controls.

These are Python algorithms, NOT trained or untrained neural model executions.
They receive a limited observation/response interface, never a Case, seed,
Rule, case_id or mechanism signature. The actual Python process is not an
independently secured evaluator; callers must preserve separate custody.

The policies can witness some plans but cannot prove bounded UNSAT. Do not
promote their wins, source-aware priors or seeded variation to model learning,
structural-family independence, or graduation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol


class PublicProbePort(Protocol):
    """Candidate-visible methods only. This type is not a security wall."""

    def view(self) -> dict[str, object]: ...
    def probe(self, action: str) -> dict[str, object]: ...
    def submit(self, proposal: dict[str, object]) -> dict[str, object]: ...


@dataclass(frozen=True)
class BehavioralResult:
    policy: str
    policy_seed: int | None
    proposal: str
    verdict: str
    probe_actions: tuple[str, ...]
    submitted_plan: tuple[str, ...] | None
    outcome: dict[str, object]
    optimizer_updates: int = 0
    model_checkpoint: str = "NONE"
    evidence_class: str = "SYNTHETIC_SOURCE_TEST"
    neural_learning_established: bool = False
    graduation_authority: str = "none"


_MEMORIZED_PLAN = ("0", "0", "WAIT", "0", "COMMIT")


def _finish(port: PublicProbePort, *, policy: str, policy_seed: int | None,
            actions: list[str], plan: tuple[str, ...] | None) -> BehavioralResult:
    proposal: dict[str, object] = (
        {"kind": "PLAN", "actions": list(plan)} if plan is not None
        else {"kind": "ABSTAIN"}
    )
    outcome = port.submit(proposal)
    if type(outcome) is not dict or type(outcome.get("verdict")) is not str:
        raise ValueError("public examiner did not provide a verdict")
    return BehavioralResult(policy, policy_seed, proposal["kind"],
                            outcome["verdict"], tuple(actions), plan, outcome)


def blind_memorized_plan(port: PublicProbePort) -> BehavioralResult:
    """A source-public five-action script. Expected false wins and failures."""
    return _finish(port, policy="blind_memorized_plan", policy_seed=None,
                   actions=[], plan=_MEMORIZED_PLAN)


def deterministic_zero_witness(port: PublicProbePort) -> BehavioralResult:
    """WAIT to switch, repeatedly pulse 0, submit only if target is observed."""
    return _witness_policy(port, policy="deterministic_zero_witness",
                           policy_seed=None)


def seeded_feedback_witness(port: PublicProbePort, *, policy_seed: int = 17
                            ) -> BehavioralResult:
    """Seeded state-feedback heuristic with no learned weights or rule access.

    The seed is a publicly declared strategy RNG seed, NOT a target Case seed.
    It is precommitted and should not be optimized against held-out labels.
    """
    if type(policy_seed) is not int or not 0 <= policy_seed < 2**31:
        raise ValueError("policy seed must be a bounded exact int")
    return _witness_policy(port, policy="seeded_feedback_witness",
                           policy_seed=policy_seed)


def _visible_three_bit(value: object) -> int:
    if (type(value) is not str or len(value) != 3
            or any(bit not in "01" for bit in value)):
        raise ValueError("expected exactly three public register bits")
    return int(value, 2)


def _witness_policy(port: PublicProbePort, *, policy: str,
                    policy_seed: int | None) -> BehavioralResult:
    public = port.view()
    initial = _visible_three_bit(public["initial_register"])
    target = _visible_three_bit(public["target_register"])
    switch = public["switch_at"]
    max_pulses = public["maximum_pulses"]
    max_moves = public["maximum_moves"]
    if (any(type(number) is not int for number in (switch, max_pulses, max_moves))
            or switch < 0 or max_pulses < 1 or max_moves < 2
            or type(public.get("remaining_probes")) is not int):
        raise ValueError("invalid declared public time/probe budgets")
    plan: list[str] = []
    attempts: list[str] = []
    # A future COMMIT consumes one permitted plan action; no unknown-rule
    # UNSAT claim and no pulse overrun are permitted.
    max_post_pulses = min(max_pulses, max_moves - switch - 1)
    if max_post_pulses < 1 or switch > public["remaining_probes"]:
        return _finish(port, policy=policy, policy_seed=policy_seed,
                       actions=attempts, plan=None)
    for _ in range(switch):
        view = port.probe("WAIT")
        plan.append("WAIT")
        attempts.append("WAIT")
        if view.get("finished"):
            return _finish(port, policy=policy, policy_seed=policy_seed,
                           actions=attempts, plan=None)

    for index in range(max_post_pulses):
        view = port.view()
        if view.get("finished") or view["remaining_probes"] <= 0:
            break
        current = _visible_three_bit(view["current_register"])
        if policy_seed is None:
            bit = "0"
        else:
            payload = (f"switchback.behavior.v1|{policy_seed}|{initial}|"
                       f"{target}|{switch}|{current}|{index}")
            bit = str(sha256(payload.encode()).digest()[0] & 1)
        observed = port.probe(bit)
        attempts.append(bit)
        plan.append(bit)
        record = observed.get("observations", [])
        if (observed.get("finished") or not record
                or record[-1].get("action") != bit
                or record[-1].get("valid") is not True
                or record[-1].get("applied") is not True):
            return _finish(port, policy=policy, policy_seed=policy_seed,
                           actions=attempts, plan=None)
        if _visible_three_bit(observed["current_register"]) == target:
            return _finish(port, policy=policy, policy_seed=policy_seed,
                           actions=attempts, plan=tuple(plan + ["COMMIT"]))
    return _finish(port, policy=policy, policy_seed=policy_seed,
                   actions=attempts, plan=None)
