"""Source-public *novel mechanism* challenge: delayed two-lane routing.

Independent of the five published three-bit Switchback kernel families.
This is a finite synthetic source challenge, NOT a private heldout exam,
graduation evidence, or a neural-training result.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json


MAX_MOVES = 11
MAX_PROBES = 20
MAX_SEED = 2**31 - 1
PLAN_ACTIONS = ("LOAD", "ROUTE", "TICK", "PURGE", "COMMIT")
PROBE_ACTIONS = ("LOAD", "ROUTE", "TICK", "PURGE", "RESET")


@dataclass(frozen=True)
class RoutingRule:
    lane_bias: int
    coupling: int
    payload_shift: int


@dataclass(frozen=True)
class Case:
    seed: int
    before: RoutingRule
    after: RoutingRule
    initial_left: int
    initial_right: int
    initial_orientation: int
    target_left: int
    target_right: int
    switch_at: int
    max_loads: int
    max_routes: int
    mechanic_signature: str
    case_id: str


@dataclass(frozen=True)
class State:
    left: int
    right: int
    orientation: int
    pending: int = 0
    ticks: int = 0
    loads: int = 0
    routes: int = 0
    post_delivery: bool = False
    committed: bool = False
    won: bool = False


def _digest(seed: int, component: str) -> bytes:
    return sha256(f"novel-router.v1:{seed}:{component}".encode()).digest()


def _rule(seed: int, phase: str) -> RoutingRule:
    data = _digest(seed, phase)
    return RoutingRule(data[0] % 2, data[1] % 4, data[2] % 3)


def _load_payload(state: State, rule: RoutingRule) -> int:
    return 1 + ((state.left + 2 * state.right + state.orientation +
                 rule.payload_shift) % 3)


def _deliver(state: State, rule: RoutingRule) -> tuple[int, int]:
    """Consume one delayed packet, potentially affecting *both* lanes."""
    port = (state.orientation + rule.lane_bias +
            ((state.left ^ state.right) & 1)) & 1
    routed = (state.pending + rule.coupling) % 4
    cross_talk = (state.pending & 1) * (rule.coupling & 1)
    if port == 0:
        return (state.left + routed) % 4, (state.right + cross_talk) % 4
    return (state.left + cross_talk) % 4, (state.right + routed) % 4


def make_case(seed: int) -> Case:
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError("seed must be a bounded exact int")
    before, after = _rule(seed, "before"), _rule(seed, "after")
    if (before.lane_bias, before.coupling, before.payload_shift) == (
            after.lane_bias, after.coupling, after.payload_shift):
        # Deterministically force a meaningful post-change rule instead of
        # accepting a trivial label switch.
        after = RoutingRule(1 - after.lane_bias, after.coupling,
                            after.payload_shift)
    data = _digest(seed, "case")
    left, right, orientation = data[0] % 4, data[1] % 4, data[2] % 2
    target_left, target_right = data[3] % 4, data[4] % 4
    switch_at = 2 + data[5] % 3
    max_loads, max_routes = 1 + data[6] % 3, 1 + data[7] % 2
    mechanism_bytes = bytes((*before.__dict__.values(), *after.__dict__.values(),
                             switch_at, max_loads, max_routes))
    mechanic = sha256(b"novel-router.mechanic.v1\0" + mechanism_bytes).hexdigest()
    case_id = sha256(("novel-router.case.v1:" + str(seed) + ":" + mechanic + ":" +
                      f"{left},{right},{orientation}:{target_left},{target_right}").encode()
                     ).hexdigest()
    return Case(seed, before, after, left, right, orientation, target_left,
                target_right, switch_at, max_loads, max_routes, mechanic, case_id)


def _valid_case(case: object) -> bool:
    if (type(case) is not Case or type(case.before) is not RoutingRule or
            type(case.after) is not RoutingRule or
            type(case.mechanic_signature) is not str or
            type(case.case_id) is not str or
            any(type(value) is not int for value in (
                case.seed, case.initial_left, case.initial_right,
                case.initial_orientation, case.target_left, case.target_right,
                case.switch_at, case.max_loads, case.max_routes,
                case.before.lane_bias, case.before.coupling,
                case.before.payload_shift, case.after.lane_bias,
                case.after.coupling, case.after.payload_shift))):
        return False
    try:
        canonical = make_case(case.seed)
    except (ValueError, TypeError):
        return False
    # Exact primitive fields only. Dataclass Case/Rule.__eq__ can be
    # monkeypatched, so an overloaded object equality is not a proof of
    # canonical source identity.
    return (
        case.seed, case.before.lane_bias, case.before.coupling,
        case.before.payload_shift, case.after.lane_bias, case.after.coupling,
        case.after.payload_shift, case.initial_left, case.initial_right,
        case.initial_orientation, case.target_left, case.target_right,
        case.switch_at, case.max_loads, case.max_routes,
        case.mechanic_signature, case.case_id,
    ) == (
        canonical.seed, canonical.before.lane_bias, canonical.before.coupling,
        canonical.before.payload_shift, canonical.after.lane_bias,
        canonical.after.coupling, canonical.after.payload_shift,
        canonical.initial_left, canonical.initial_right,
        canonical.initial_orientation, canonical.target_left, canonical.target_right,
        canonical.switch_at, canonical.max_loads, canonical.max_routes,
        canonical.mechanic_signature, canonical.case_id,
    )


def _state_key(state: State) -> tuple[int, int, int, int, int, int, int, bool, bool, bool]:
    """Search closure must never trust monkeypatchable dataclass equality."""
    if (type(state) is not State or
            any(type(x) is not int for x in (
                state.left, state.right, state.orientation, state.pending,
                state.ticks, state.loads, state.routes)) or
            any(type(flag) is not bool for flag in (
                state.post_delivery, state.committed, state.won))):
        raise ValueError("invalid finite router state")
    return (state.left, state.right, state.orientation, state.pending,
            state.ticks, state.loads, state.routes, state.post_delivery,
            state.committed, state.won)


def _independent_advance(case: Case, state: tuple[object, ...],
                         action: str) -> tuple[object, ...]:
    """Separate primitive-state physics: never call step, _deliver or _load_payload.

    Its different representation independently checks scorer and probe
    outputs against one faulty shared source helper in the normal DUT path.
    This remains source-level redundancy, not a separate privileged signer.
    """
    left, right, orientation, pending, ticks, loads, routes, post, committed, won = state
    if committed or ticks >= MAX_MOVES or action not in PLAN_ACTIONS:
        raise ValueError("independent invalid action")
    if action == "COMMIT":
        success = (pending == 0 and post and ticks >= case.switch_at and
                   (left, right) == (case.target_left, case.target_right))
        return (left, right, orientation, pending, ticks, loads, routes,
                post, True, success)
    law = case.after if ticks >= case.switch_at else case.before
    if action == "LOAD":
        if pending or loads >= case.max_loads:
            raise ValueError("independent illegal load")
        payload = (left + 2 * right + orientation + law.payload_shift) % 3 + 1
        return (left, right, orientation, payload, ticks + 1, loads + 1,
                routes, post, False, False)
    if action == "ROUTE":
        if routes >= case.max_routes:
            raise ValueError("independent illegal route")
        return (left, right, orientation ^ 1, pending, ticks + 1, loads,
                routes + 1, post, False, False)
    if action == "PURGE":
        if pending == 0:
            raise ValueError("independent illegal purge")
        return (left, right, orientation, 0, ticks + 1, loads,
                routes, post, False, False)
    if pending:
        tanks = [left, right]
        port = orientation ^ law.lane_bias ^ ((left ^ right) & 1)
        tanks[port] = (tanks[port] + pending + law.coupling) % 4
        if pending % 2 and law.coupling % 2:
            tanks[1 - port] = (tanks[1 - port] + 1) % 4
        left, right = tanks
    return (left, right, orientation, 0, ticks + 1, loads,
            routes, post or (pending > 0 and ticks >= case.switch_at),
            False, False)


def _independent_plan_verdict(case: Case, actions: object) -> str:
    if (type(actions) is not list or not 1 <= len(actions) <= MAX_MOVES or
            any(type(action) is not str or action not in PLAN_ACTIONS
                for action in actions)):
        return "VERIFIED_FAILURE"
    current = _state_key(initial_state(case))
    for action in actions:
        try:
            current = _independent_advance(case, current, action)
        except ValueError:
            return "VERIFIED_FAILURE"
    return "VERIFIED_WIN" if current[-2:] == (True, True) else "VERIFIED_FAILURE"


def _independent_closure_witness(case: Case) -> list[str] | None:
    """A different DFS algorithm checks any claimed finite UNSAT closure."""
    @lru_cache(maxsize=None)
    def extend(current: tuple[object, ...]):
        if current[4] >= MAX_MOVES:
            return None
        committed = _independent_advance(case, current, "COMMIT")
        if committed[-1]:
            return ("COMMIT",)
        if current[4] >= MAX_MOVES - 1:
            return None
        for action in ("TICK", "PURGE", "ROUTE", "LOAD"):
            try:
                evolved = _independent_advance(case, current, action)
            except ValueError:
                continue
            suffix = extend(evolved)
            if suffix is not None:
                return (action,) + suffix
        return None

    found = extend(_state_key(initial_state(case)))
    return list(found) if found is not None else None


def initial_state(case: Case) -> State:
    return State(case.initial_left, case.initial_right, case.initial_orientation)


def public_view(case: Case) -> dict[str, object]:
    """No private Rule values or seed/case fingerprints are exposed here."""
    return {
        "prompt": ("Two four-level reservoirs exchange delayed packets. LOAD queues a "
                   "hidden-rule-dependent packet without delivery; TICK moves an "
                   "in-flight packet through a phase-dependent route, sometimes "
                   "changing both reservoirs. ROUTE toggles the outlet orientation; "
                   "PURGE discards a pending packet. The routing law changes mid-run. "
                   "COMMIT requires both target levels and a post-change delivery. "
                   "Propose a legal path or an explicitly bounded uncertainty claim."),
        "initial_levels": [case.initial_left, case.initial_right],
        "initial_orientation": case.initial_orientation,
        "target_levels": [case.target_left, case.target_right],
        "switch_at": case.switch_at,
        "maximum_moves": MAX_MOVES,
        "maximum_loads": case.max_loads,
        "maximum_routes": case.max_routes,
        "maximum_probes": MAX_PROBES,
        "plan_actions": list(PLAN_ACTIONS),
        "probe_actions": list(PROBE_ACTIONS),
    }


def _legal_nonterminal(case: Case, state: State, action: str) -> bool:
    if state.committed or state.ticks >= MAX_MOVES or action not in PLAN_ACTIONS[:-1]:
        return False
    if action == "LOAD":
        return state.pending == 0 and state.loads < case.max_loads
    if action == "ROUTE":
        return state.routes < case.max_routes
    if action == "PURGE":
        return state.pending != 0
    return True  # TICK always advances time, including when no packet is in flight.


def step(case: Case, state: State, action: str) -> State:
    if type(case) is not Case or type(state) is not State or type(action) is not str:
        raise ValueError("invalid transition types")
    if state.committed or state.ticks >= MAX_MOVES or action not in PLAN_ACTIONS:
        raise ValueError("illegal terminal, exhausted or unknown action")
    if action == "COMMIT":
        won = (state.pending == 0 and state.post_delivery and
               state.ticks >= case.switch_at and state.left == case.target_left and
               state.right == case.target_right)
        return State(state.left, state.right, state.orientation, state.pending,
                     state.ticks, state.loads, state.routes, state.post_delivery,
                     True, won)
    if not _legal_nonterminal(case, state, action):
        raise ValueError("resource or packet precondition violated")
    rule = case.after if state.ticks >= case.switch_at else case.before
    if action == "LOAD":
        return State(state.left, state.right, state.orientation,
                     _load_payload(state, rule), state.ticks + 1, state.loads + 1,
                     state.routes, state.post_delivery)
    if action == "ROUTE":
        return State(state.left, state.right, 1 - state.orientation, state.pending,
                     state.ticks + 1, state.loads, state.routes + 1,
                     state.post_delivery)
    if action == "PURGE":
        return State(state.left, state.right, state.orientation, 0,
                     state.ticks + 1, state.loads, state.routes,
                     state.post_delivery)
    if state.pending:
        left, right = _deliver(state, rule)
        return State(left, right, state.orientation, 0, state.ticks + 1,
                     state.loads, state.routes,
                     state.post_delivery or state.ticks >= case.switch_at)
    return State(state.left, state.right, state.orientation, 0,
                 state.ticks + 1, state.loads, state.routes, state.post_delivery)


def verify_plan(case: Case, actions: object) -> dict[str, object]:
    if not _valid_case(case):
        raise ValueError("plan requires canonical generated case")
    if (type(actions) is not list or not 1 <= len(actions) <= MAX_MOVES or
            any(type(action) is not str or action not in PLAN_ACTIONS
                for action in actions)):
        return {"verdict": "VERIFIED_FAILURE", "reason": "INVALID_ACTION_PLAN"}
    state = initial_state(case)
    for index, action in enumerate(actions):
        if state.committed:
            return {"verdict": "VERIFIED_FAILURE", "reason": "ACTION_AFTER_COMMIT",
                    "index": index}
        try:
            state = step(case, state, action)
        except ValueError:
            return {"verdict": "VERIFIED_FAILURE", "reason": "ILLEGAL_ACTION",
                    "index": index}
    if not state.committed:
        return {"verdict": "VERIFIED_FAILURE", "reason": "NO_COMMIT"}
    return {"verdict": "VERIFIED_WIN" if state.won else "VERIFIED_FAILURE",
            "reason": "DECLARED_GOAL_CHECK"}


def search(case: Case, *, max_nodes: int = 100_000) -> dict[str, object]:
    """Exhaustive forward state closure, with honest capped-search UNKNOWN."""
    if not _valid_case(case) or type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("invalid finite search parameters")
    start = initial_state(case)
    frontier = deque([(start, ())])
    seen = {_state_key(start)}
    explored = 0
    while frontier:
        if explored >= max_nodes:
            return {"verdict": "NO_KNOWN_WIN", "complete": False,
                    "nodes_explored": explored, "witness": None}
        state, prefix = frontier.popleft()
        explored += 1
        if len(prefix) >= MAX_MOVES:
            continue
        candidate = prefix + ("COMMIT",)
        observed_verdict = verify_plan(case, list(candidate))["verdict"]
        independently_verified = _independent_plan_verdict(case, list(candidate))
        if observed_verdict != independently_verified:
            return {"verdict": "UNKNOWN", "complete": False,
                    "nodes_explored": explored, "witness": None,
                    "reason": "INDEPENDENT_SCORER_DISAGREEMENT"}
        if independently_verified == "VERIFIED_WIN":
            return {"verdict": "VERIFIED_WIN", "complete": True,
                    "nodes_explored": explored, "witness": list(candidate)}
        for action in PLAN_ACTIONS[:-1]:
            try:
                evolved = step(case, state, action)
            except ValueError:
                continue
            key = _state_key(evolved)
            if key not in seen:
                seen.add(key)
                frontier.append((evolved, prefix + (action,)))
    if _independent_closure_witness(case) is not None:
        return {"verdict": "UNKNOWN", "complete": False,
                "nodes_explored": explored, "witness": None,
                "reason": "INDEPENDENT_CLOSURE_DISAGREEMENT"}
    return {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "complete": True,
            "nodes_explored": explored, "witness": None}


def _canonical(data: dict[str, object]) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(",", ":"))


def adjudicate(case: Case, proposal: object) -> dict[str, object]:
    if not _valid_case(case) or type(proposal) is not dict:
        raise ValueError("expected canonical Case and JSON proposal")
    try:
        proposal = json.loads(_canonical(proposal))
    except (TypeError, ValueError):
        raise ValueError("invalid bounded JSON proposal") from None
    if len(_canonical(proposal).encode()) > 8192:
        raise ValueError("proposal exceeds bounded size")
    kind = proposal.get("kind")
    if kind == "PLAN" and set(proposal) == {"kind", "actions"}:
        claimed_verdict = verify_plan(case, proposal["actions"])["verdict"]
        independent_verdict = _independent_plan_verdict(case, proposal["actions"])
        verdict = (claimed_verdict if claimed_verdict == independent_verdict
                   else "UNKNOWN")
        reason = ("INDEPENDENT_PLAN_PARITY" if verdict != "UNKNOWN" else
                  "INDEPENDENT_SCORER_DISAGREEMENT")
        counterexample = None
    elif kind == "CLAIM_UNSAT" and set(proposal) == {"kind"}:
        proof = search(case)
        independent_witness = _independent_closure_witness(case)
        if independent_witness is not None:
            verdict, reason, counterexample = (
                "VERIFIED_FAILURE", "INDEPENDENT_WIN_COUNTEREXAMPLE",
                independent_witness)
        elif (proof["complete"] and
              proof["verdict"] == "PROVEN_UNSAT_UNDER_DECLARED_RULES"):
            verdict, reason, counterexample = (
                "PROVEN_UNSAT_UNDER_DECLARED_RULES", "TWO_FINITE_CLOSURES_AGREE", None)
        else:
            verdict, reason, counterexample = (
                "NO_KNOWN_WIN", "SOURCE_SEARCH_INCOMPLETE_OR_DISAGREES", None)
    elif kind == "NOVEL_RULE" and set(proposal) == {"kind", "description"} and (
            type(proposal["description"]) is str and
            0 < len(proposal["description"]) <= 2000):
        verdict, reason, counterexample = ("NOVEL_RULESET_CHALLENGE",
                                           "NOT_VERIFIED_UNDER_DECLARED_RULES", None)
    else:
        verdict, reason, counterexample = "UNKNOWN", "INVALID_OR_UNTESTED_PROPOSAL", None
    result = {
        "schema": "labyrinth.novel-router-receipt.v1",
        "evidence_class": "SOURCE_PUBLIC_SYNTHETIC",
        "source_commit": "UNBOUND",
        "private_custody": False,
        "neural_learning_verified": False,
        "graduation_eligible": False,
        "case_id": case.case_id,
        "seed": case.seed,
        "verdict": verdict,
        "verified_success": verdict == "VERIFIED_WIN",
        "reason": reason,
        "counterexample": counterexample,
        "proposal": proposal,
    }
    result["sha256"] = sha256(_canonical(result).encode()).hexdigest()
    return result


class ProbeSession:
    """In-process educational observations, never a private host boundary."""

    def __init__(self, case: Case):
        if not _valid_case(case):
            raise ValueError("noncanonical generated case")
        self._case = case
        self._state = initial_state(case)
        self._trace: list[dict[str, object]] = []
        self._finished = False
        self._receipt: dict[str, object] | None = None

    def view(self) -> dict[str, object]:
        view = public_view(self._case)
        view.update({
            "current_levels": [self._state.left, self._state.right],
            "orientation": self._state.orientation,
            "pending_packet": self._state.pending,
            "ticks": self._state.ticks,
            "loads_used": self._state.loads,
            "routes_used": self._state.routes,
            "post_delivery_observed": self._state.post_delivery,
            "remaining_probes": MAX_PROBES - len(self._trace),
            "observations": deepcopy(self._trace),
            "finished": self._finished,
        })
        return view

    def probe(self, action: object) -> dict[str, object]:
        if self._finished:
            return self.view()
        if len(self._trace) >= MAX_PROBES:
            self._finished = True
            self._receipt = {"verdict": "VERIFIED_FAILURE",
                             "reason": "PROBE_BUDGET_EXCEEDED", "verified_success": False}
            return self.view()
        if type(action) is not str or len(action) > 32:
            self._finished = True
            self._receipt = {"verdict": "UNKNOWN", "reason": "UNREPRESENTABLE_ACTION",
                             "verified_success": False}
            return self.view()
        before = [self._state.left, self._state.right]
        valid = action in PROBE_ACTIONS
        applied = False
        if valid:
            if action == "RESET":
                self._state = initial_state(self._case)
                applied = True
            else:
                prior = _state_key(self._state)
                try:
                    expected = _independent_advance(self._case, prior, action)
                except ValueError:
                    expected = None
                try:
                    evolved = step(self._case, self._state, action)
                except ValueError:
                    evolved = None
                if ((expected is None) != (evolved is None) or
                        (expected is not None and evolved is not None and
                         _state_key(evolved) != expected)):
                    # Both physical and independent source oracles must agree.
                    # A modified shared step must not inject a fabricated
                    # public observation or grant an uncharged retry.
                    self._trace.append({"action": action, "valid": True,
                                        "applied": False, "before": before,
                                        "after": None})
                    self._finished = True
                    self._receipt = {"verdict": "UNKNOWN",
                                     "reason": "INDEPENDENT_TRANSITION_DISAGREEMENT",
                                     "verified_success": False}
                    return self.view()
                if evolved is not None:
                    self._state = evolved
                    applied = True
        self._trace.append({"action": action, "valid": valid, "applied": applied,
                            "before": before,
                            "after": [self._state.left, self._state.right]})
        return self.view()

    def submit(self, proposal: object) -> dict[str, object]:
        if self._finished:
            return deepcopy(self._receipt) if self._receipt is not None else {
                "verdict": "UNKNOWN", "verified_success": False}
        self._finished = True
        core = adjudicate(self._case, proposal)
        result = {"schema": "labyrinth.novel-router-session.v1",
                  "evidence_class": "SOURCE_PUBLIC_SYNTHETIC",
                  "private_custody": False,
                  "neural_learning_verified": False,
                  "graduation_eligible": False,
                  "probe_count": len(self._trace), "probe_trace": deepcopy(self._trace),
                  "verdict": core["verdict"],
                  "verified_success": core["verified_success"],
                  "adjudication": core}
        result["sha256"] = sha256(_canonical(result).encode()).hexdigest()
        self._receipt = result
        return deepcopy(result)
