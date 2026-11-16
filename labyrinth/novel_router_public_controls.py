"""Zero-update, source-public behavioral controls for delayed Novel Router.

These policies run against candidate-visible view/probe/submit only. They are
not neural models and never receive target Case/seed/rules or call make_case,
step, search, adjudicate, or the privileged source verifier. Python in-process
method access is not secure evaluator custody. Public source is NOT heldout.

All success is a witnessed plan or directly scored blind guess; unsupported
UNSAT claims are forbidden. No neural training, checkpoint or graduation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol


class PublicRouterPort(Protocol):
    def view(self) -> dict[str, object]: ...
    def probe(self, action: str) -> dict[str, object]: ...
    def submit(self, proposal: dict[str, object]) -> dict[str, object]: ...


@dataclass(frozen=True)
class RouterControlResult:
    policy: str
    verdict: str
    proposal: str
    probes: tuple[str, ...]
    plan: tuple[str, ...] | None
    outcome: dict[str, object]
    optimizer_updates: int = 0
    model_checkpoint: str = "NONE"
    neural_learning_established: bool = False
    graduation_authority: str = "none"
    independent_custody: bool = False


def _public_int(value: object, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("invalid typed public router parameter")
    return value


def _params(port: PublicRouterPort) -> tuple[int, int, int, tuple[int, int]]:
    view = port.view()
    switch = _public_int(view["switch_at"], 2, 4)
    loads = _public_int(view["maximum_loads"], 1, 3)
    routes = _public_int(view["maximum_routes"], 1, 2)
    target = view["target_levels"]
    if (type(target) is not list or len(target) != 2 or
            any(type(level) is not int or not 0 <= level <= 3 for level in target)
            or view.get("maximum_moves") != 11 or
            view.get("maximum_probes") != 20):
        raise ValueError("unexpected public router target or budget")
    return switch, loads, routes, (target[0], target[1])


def _finish(port: PublicRouterPort, policy: str, probes: list[str],
            plan: list[str] | None) -> RouterControlResult:
    proposal = ({"kind": "PLAN", "actions": list(plan)} if plan is not None
                else {"kind": "ABSTAIN"})
    result = port.submit(proposal)
    if type(result) is not dict or type(result.get("verdict")) is not str:
        raise ValueError("examiner did not return typed verdict")
    return RouterControlResult(policy, result["verdict"], proposal["kind"],
                               tuple(probes), tuple(plan) if plan else None,
                               result)


def blind_delayed_delivery(port: PublicRouterPort) -> RouterControlResult:
    """Submit a fixed legal after-switch delivery without any observations."""
    switch, _, _, _ = _params(port)
    plan = ["TICK"] * switch + ["LOAD", "TICK", "COMMIT"]
    return _finish(port, "blind_delayed_delivery", [], plan)


def _witness(view: dict[str, object], target: tuple[int, int]) -> bool:
    levels = view.get("current_levels")
    return (type(levels) is list and levels == list(target) and
            view.get("pending_packet") == 0 and
            view.get("post_delivery_observed") is True and
            type(view.get("ticks")) is int and view["ticks"] >= view["switch_at"])


def _probe_path(port: PublicRouterPort, actions: list[str], probes: list[str],
                target: tuple[int, int]) -> list[str] | None:
    """Replay a public path; return winning PREFIX only after seeing proof."""
    accepted: list[str] = []
    for action in actions:
        remaining = port.view().get("remaining_probes")
        if type(remaining) is not int or remaining <= 0:
            return None
        result = port.probe(action)
        probes.append(action)
        trace = result.get("observations")
        if (result.get("finished") or type(trace) is not list or not trace or
                trace[-1].get("action") != action or
                trace[-1].get("valid") is not True or
                trace[-1].get("applied") is not True):
            return None
        accepted.append(action)
        if _witness(result, target):
            return accepted + ["COMMIT"]
    return None


def deterministic_trial_witness(port: PublicRouterPort) -> RouterControlResult:
    """Up to three predeclared legal trials; no hidden-rule inference."""
    switch, loads, routes, target = _params(port)
    recipes = [
        ["TICK"] * switch + ["LOAD", "TICK"],
        ["TICK"] * switch + ["ROUTE", "LOAD", "TICK"],
    ]
    if loads >= 2:
        recipes.append(["TICK"] * switch + ["LOAD", "TICK", "LOAD", "TICK"])
    probes: list[str] = []
    for trial, recipe in enumerate(recipes):
        overhead = 1 if trial else 0
        if len(probes) + overhead + len(recipe) > port.view()["remaining_probes"] + len(probes):
            break
        if trial:
            reset = port.probe("RESET")
            probes.append("RESET")
            if reset.get("finished"):
                break
        witness = _probe_path(port, recipe, probes, target)
        if witness is not None:
            return _finish(port, "deterministic_trial_witness", probes, witness)
    return _finish(port, "deterministic_trial_witness", probes, None)


def seeded_feedback_witness(port: PublicRouterPort, *, policy_seed: int = 7
                            ) -> RouterControlResult:
    """A seeded observation-dependent route choice with no neural updates."""
    _public_int(policy_seed, 0, 2**31 - 1)
    switch, max_loads, max_routes, target = _params(port)
    probes: list[str] = []
    path: list[str] = []
    waiting = ["TICK"] * switch
    witness = _probe_path(port, waiting, probes, target)
    path.extend(waiting)
    if witness:
        return _finish(port, "seeded_feedback_witness", probes, witness)
    for load in range(max_loads):
        view = port.view()
        if view.get("finished") or view.get("ticks", 11) >= 9:
            break
        levels = view["current_levels"]
        material = (f"router.feedback.v1|{policy_seed}|{load}|"
                    f"{levels[0]},{levels[1]}|{target[0]},{target[1]}|"
                    f"{view['orientation']}")
        toggle = sha256(material.encode()).digest()[0] & 1
        if (toggle and view["routes_used"] < max_routes and
                len(path) + 3 < 11):
            witness = _probe_path(port, ["ROUTE"], probes, target)
            path.append("ROUTE")
            if witness:
                return _finish(port, "seeded_feedback_witness", probes,
                               path + ["COMMIT"])
        if len(path) + 3 > 11:
            break
        segment = ["LOAD", "TICK"]
        witness = _probe_path(port, segment, probes, target)
        if witness is not None:
            return _finish(port, "seeded_feedback_witness", probes,
                           path + witness)
        path.extend(segment)
    return _finish(port, "seeded_feedback_witness", probes, None)
