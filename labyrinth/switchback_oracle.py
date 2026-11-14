"""Separate finite state oracle for generated Switchback Circuit challenges."""

from __future__ import annotations

from collections import deque
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
import math
from threading import get_ident

from .switchback_circuit import Case, Rule, State, MAX_MOVES, MAX_PROBES


def _case_fields(case: Case) -> tuple[object, ...]:
    """Typed primitive fingerprint: never equate bool/float with integer 0/1."""
    fields = (type(case), type(case.before), type(case.after),
              case.seed, case.initial, case.target, case.switch_at, case.max_pulses,
              case.before.family, case.before.mask, case.before.bias,
              case.before.threshold, case.after.family, case.after.mask,
              case.after.bias, case.after.threshold,
              case.mechanic_signature, case.case_id)
    return tuple((type(value), value) for value in fields)


@dataclass
class _CanonicalParityScope:
    case: Case
    fields: tuple[object, ...]
    owner_thread: int
    active: bool = True


_PARITY_SCOPE: ContextVar[_CanonicalParityScope | None] = ContextVar(
    "labyrinth.switchback.parity_scope", default=None)


def oracle_case(seed: int) -> Case:
    """Independent genesis: no call to DUT make_case or DUT kernel."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("invalid independent seed")
    def digest(tag: str) -> bytes:
        return sha256(("switchback-circuit.v1:" + str(seed) + ":" + tag).encode()).digest()
    def make_rule(tag: str) -> Rule:
        data = digest(tag)
        return Rule(data[0] % 5, (data[1] % 7) + 1, data[2] % 8, (data[3] % 3) + 1)
    first, second = make_rule("pre"), make_rule("post")
    for _ in range(8):
        initial_truth = tuple(_rule_output(r, b, first) for r in range(8) for b in (0, 1))
        revision_truth = tuple(_rule_output(r, b, second) for r in range(8) for b in (0, 1))
        if initial_truth != revision_truth:
            break
        second = Rule(second.family, second.mask, (second.bias + 1) % 8, second.threshold)
    else:
        raise ValueError("independent rule revision identity failure")
    data = digest("case")
    initial, target = data[0] % 8, data[1] % 8
    switch_at, max_pulses = 2 + data[2] % 3, 1 + data[3] % 4
    truth = bytes(_rule_output(r, b, rule) for rule in (first, second)
                  for r in range(8) for b in (0, 1))
    mechanic = sha256(b"switchback.mechanic.v1\0" + bytes([switch_at, max_pulses])
                      + truth).hexdigest()
    identity = sha256(("switchback.case.v1:" + str(seed) + ":" + mechanic + ":" +
                       str(initial) + ":" + str(target)).encode()).hexdigest()
    return Case(seed, first, second, initial, target, switch_at, max_pulses,
                mechanic, identity)


def _valid_case(case: object) -> bool:
    if (type(case) is not Case or type(case.before) is not Rule or
            type(case.after) is not Rule or type(case.mechanic_signature) is not str or
            type(case.case_id) is not str or
            any(type(x) is not int for x in (
                case.seed, case.initial, case.target, case.switch_at, case.max_pulses,
                case.before.family, case.before.mask, case.before.bias, case.before.threshold,
                case.after.family, case.after.mask, case.after.bias, case.after.threshold))):
        return False
    try:
        canonical = oracle_case(case.seed)
    except (ValueError, TypeError):
        return False
    return case == canonical


def _rule_output(register: int, bit: int, rule: Rule) -> int:
    """Independent explicit evaluator, never calls simulator kernel/step."""
    count = sum((register >> k) & 1 for k in range(3))
    covered = sum(((register & rule.mask) >> k) & 1 for k in range(3))
    if rule.family == 0:
        feedback = covered % 2
        return ((2 * register + bit) ^ (feedback * 4) ^ rule.bias) % 8
    if rule.family == 1:
        rotation = ((register % 4) * 2 + register // 4) % 8
        return (rotation + bit + rule.bias) % 8
    if rule.family == 2:
        return ((2 * register + bit) ^ (rule.mask if count >= rule.threshold
                                         else rule.bias)) % 8
    if rule.family == 3:
        return (register + bit + covered + rule.bias) % 8
    if rule.family == 4:
        return register ^ (rule.mask if covered >= rule.threshold else 0) ^ bit ^ rule.bias
    raise ValueError("invalid family")


def oracle_step(case: Case, state: State, action: str) -> State:
    active = _PARITY_SCOPE.get()
    trusted_in_invocation = (active is not None and active.active and
                             active.owner_thread == get_ident() and
                             active.case is case and _case_fields(case) == active.fields)
    if ((not trusted_in_invocation and not _valid_case(case)) or
            type(state) is not State or type(action) is not str):
        raise ValueError("invalid oracle input")
    if state.committed or state.ticks >= MAX_MOVES or action not in ("0", "1", "WAIT", "COMMIT"):
        return state
    if action == "COMMIT":
        win = state.register == case.target and state.ticks >= case.switch_at and state.injected_after
        return State(state.register, state.ticks, state.pulses, state.injected_after,
                     True, win)
    if action == "WAIT":
        return State(state.register, state.ticks + 1, state.pulses, state.injected_after)
    if state.pulses >= case.max_pulses:
        return state
    revised = state.ticks >= case.switch_at
    value = _rule_output(state.register, int(action), case.after if revised else case.before)
    return State(value, state.ticks + 1, state.pulses + 1, state.injected_after or revised)


def verify_probe_trace(case: Case, history: object) -> bool:
    """Independently replay bounded candidate observations, not DUT step."""
    if not _valid_case(case) or type(history) not in (tuple, list) or len(history) > MAX_PROBES:
        return False
    state = State(case.initial)
    for record in history:
        if type(record) is not dict or set(record) != {
                "action", "valid", "applied", "before", "after"}:
            return False
        action = record["action"]
        if type(action) is not str or len(action) > 32 or action == "UNREPRESENTABLE":
            return False
        valid = action in ("0", "1", "WAIT", "RESET")
        applied = (action == "RESET" or
                   (valid and state.ticks < MAX_MOVES and
                    (action == "WAIT" or state.pulses < case.max_pulses)))
        if (record["valid"] is not valid or type(record["before"]) is not str
                or record["before"] != format(state.register, "03b")
                or record["applied"] is not applied):
            return False
        if action == "RESET":
            state = State(case.initial)
        elif valid:
            state = oracle_step(case, state, action)
        if type(record["after"]) is not str or record["after"] != format(state.register, "03b"):
            return False
    return True


def independent_probe_trace(case: Case, history: object) -> bool:
    """Separately replay exact public probes via DUT, not verify_probe_trace."""
    if (not _valid_case(case) or type(history) not in (tuple, list) or
            len(history) > MAX_PROBES):
        return False
    from . import switchback_circuit as dut
    try:
        state = State(case.initial)
        for record in history:
            if type(record) is not dict or set(record) != {
                    "action", "valid", "applied", "before", "after"}:
                return False
            action = record["action"]
            if (type(action) is not str or len(action) > 32 or
                    action == "UNREPRESENTABLE"):
                return False
            recognized = action in ("0", "1", "WAIT", "RESET")
            applied = (action == "RESET" or (recognized and
                       state.ticks < MAX_MOVES and (action == "WAIT" or
                       state.pulses < case.max_pulses)))
            if (record["valid"] is not recognized or
                    record["applied"] is not applied or
                    type(record["before"]) is not str or
                    record["before"] != format(state.register, "03b")):
                return False
            if action == "RESET":
                state = State(case.initial)
            elif recognized:
                state = dut.step(case, state, action)
                _state_key(state)
            if (type(record["after"]) is not str or
                    record["after"] != format(state.register, "03b")):
                return False
        return True
    except Exception:
        return False


def verify_plan(case: Case, actions: object) -> dict[str, object]:
    if (type(actions) is not list or not 1 <= len(actions) <= MAX_MOVES
            or any(type(a) is not str or a not in ("0", "1", "WAIT", "COMMIT")
                   for a in actions)):
        return {"verdict": "VERIFIED_FAILURE", "reason": "INVALID_ACTION_PLAN",
                "counterexample": None, "steps_checked": 0}
    state = State(case.initial)
    for index, action in enumerate(actions):
        if state.committed:
            return {"verdict": "VERIFIED_FAILURE", "reason": "ACTION_AFTER_COMMIT",
                    "counterexample": {"index": index, "action": action}, "steps_checked": index}
        if state.ticks >= MAX_MOVES:
            return {"verdict": "VERIFIED_FAILURE", "reason": "ATTEMPT_BUDGET_EXCEEDED",
                    "counterexample": {"index": index, "action": action}, "steps_checked": index}
        before = state
        state = oracle_step(case, state, action)
        if action in ("0", "1") and before.pulses >= case.max_pulses:
            return {"verdict": "VERIFIED_FAILURE", "reason": "PULSE_BUDGET_EXCEEDED",
                    "counterexample": {"index": index, "action": action}, "steps_checked": index + 1}
    if not state.committed:
        return {"verdict": "VERIFIED_FAILURE", "reason": "NO_COMMIT",
                "counterexample": None, "steps_checked": len(actions)}
    return {"verdict": "VERIFIED_WIN" if state.won else "VERIFIED_FAILURE",
            "reason": "EXACT_FINITE_WIN" if state.won else "FALSE_COMMIT",
            "counterexample": None if state.won else {
                "register": format(state.register, "03b"),
                "target": format(case.target, "03b"), "ticks": state.ticks,
                "post_change_input": state.injected_after},
            "steps_checked": len(actions)}


def search(case: Case, *, max_nodes: int = 100000) -> dict[str, object]:
    """Complete breadth-first finite-state reachability unless cap exhausted."""
    if not _valid_case(case) or type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("invalid case or search budget")
    initial = State(case.initial)
    queue = deque([(initial, ())])
    seen = {(initial, 0)}
    explored = 0
    while queue:
        if explored >= max_nodes:
            return {"verdict": "NO_KNOWN_WIN", "witness": None,
                    "nodes_explored": explored, "complete": False}
        state, path = queue.popleft()
        explored += 1
        if len(path) >= MAX_MOVES:
            continue
        for action in ("0", "1", "WAIT", "COMMIT"):
            candidate = (*path, action)
            next_state = oracle_step(case, state, action)
            if next_state.committed:
                if next_state.won:
                    return {"verdict": "VERIFIED_WIN", "witness": list(candidate),
                            "nodes_explored": explored, "complete": True}
                continue
            if (next_state, len(candidate)) not in seen and next_state != state:
                seen.add((next_state, len(candidate)))
                queue.append((next_state, candidate))
    return {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "witness": None,
            "nodes_explored": explored, "complete": True}


def _state_key(state: State) -> tuple[int, int, int, bool, bool, bool]:
    """Exact primitive state custody; never trust overloaded State equality."""
    if (type(state) is not State or type(state.register) is not int
            or type(state.ticks) is not int or type(state.pulses) is not int
            or any(type(flag) is not bool for flag in (
                state.injected_after, state.committed, state.won))):
        raise ValueError("ill-typed oracle or DUT state")
    return (state.register, state.ticks, state.pulses, state.injected_after,
            state.committed, state.won)


def independent_closure_witness(case: Case) -> list[str] | None:
    """Separate memoized depth-first finite closure, never call BFS search.

    Used to challenge its unsat verdict rather than trust `complete=True`.
    This shares declared oracle transition semantics, not BFS search logic.
    """
    if not _valid_case(case):
        raise ValueError("invalid closure case")

    @lru_cache(maxsize=None)
    def extend(state_key: tuple[int, int, int, bool, bool, bool],
               remaining: int) -> tuple[str, ...] | None:
        if remaining <= 0:
            return None
        state = State(*state_key)
        for action in ("0", "1", "WAIT", "COMMIT"):
            advanced = oracle_step(case, state, action)
            if advanced.committed:
                if advanced.won:
                    return (action,)
                continue
            advanced_key = _state_key(advanced)
            if advanced_key == state_key:
                continue
            suffix = extend(advanced_key, remaining - 1)
            if suffix is not None:
                return (action, *suffix)
        return None

    result = extend(_state_key(State(case.initial)), MAX_MOVES)
    return list(result) if result is not None else None


def independent_transition_parity(case: Case) -> bool:
    """Exhaust ALL finite DUT-reachable states/actions, compare both engines.

    Uses an independent DUT transition graph, not the oracle's graph; unlike
    spot checks it covers hidden changes at intermediate ticks/pulse counts.
    """
    if not _valid_case(case):
        return False
    from . import switchback_circuit as dut
    try:
        # The two physics engines share this State class. A corrupted genesis
        # constructor is therefore a genuine common-mode fault unless exact
        # primitive assignment is checked separately over the finite domain.
        for register in range(8):
            for clock in range(MAX_MOVES + 1):
                for pulses in range(case.max_pulses + 1):
                    for flags in product((False, True), repeat=3):
                        expected = (register, clock, pulses, *flags)
                        if _state_key(State(*expected)) != expected:
                            return False
        for rule in (case.before, case.after):
            for register in range(8):
                for bit in (0, 1):
                    if dut.kernel(register, bit, rule) != _rule_output(register, bit, rule):
                        return False
        # Earlier DUT construction/truth-table calls are not in the trusted
        # context. Revalidate immediately before capturing the exact fields;
        # a single-function mutator in dut.kernel cannot smuggle a changed
        # Case through an identity-only shortcut.
        if not _valid_case(case):
            return False
        scope = _CanonicalParityScope(case, _case_fields(case), get_ident())
        token = _PARITY_SCOPE.set(scope)
        try:
            initial = State(case.initial)
            frontier = deque([initial])
            visited = {_state_key(initial)}
            while frontier:
                state = frontier.popleft()
                for action in ("0", "1", "WAIT", "COMMIT"):
                    evolved = dut.step(case, state, action)
                    evolved_key = _state_key(evolved)
                    compared_key = _state_key(oracle_step(case, state, action))
                    if evolved_key != compared_key:
                        return False
                    if (not evolved.committed and evolved_key not in visited and
                            evolved_key != _state_key(state)):
                        visited.add(evolved_key)
                        frontier.append(evolved)
            return True
        finally:
            # A copied Context retains the record object; invalidating it
            # BEFORE reset prevents stale trusted scope in copied tasks.
            scope.active = False
            _PARITY_SCOPE.reset(token)
    except Exception:
        return False


def _canonical(value: dict[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _json_exact(value: object, depth: int = 0) -> bool:
    if depth > 8:
        return False
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return len(value) <= 128 and all(_json_exact(v, depth + 1) for v in value)
    if type(value) is dict:
        return len(value) <= 128 and all(type(k) is str and _json_exact(v, depth + 1)
                                         for k, v in value.items())
    return False


def independent_plan_won(case: Case, actions: object) -> bool:
    """Raw DUT replay, independent of the oracle's verify_plan scorer."""
    if (not _valid_case(case) or type(actions) is not list or
            not 1 <= len(actions) <= MAX_MOVES or any(
                type(a) is not str or a not in ("0", "1", "WAIT", "COMMIT")
                for a in actions)):
        return False
    from . import switchback_circuit as dut
    try:
        state = State(case.initial)
        for action in actions:
            if state.committed or state.ticks >= MAX_MOVES:
                return False
            if action in ("0", "1") and state.pulses >= case.max_pulses:
                return False
            state = dut.step(case, state, action)
            _state_key(state)
        return state.committed is True and state.won is True
    except Exception:
        return False


def adjudicate(case: Case, proposal: object) -> dict[str, object]:
    """Any valid solution wins, regardless of a designer's cached witness."""
    if not _valid_case(case):
        raise ValueError("forged case")
    if type(proposal) is not dict or not _json_exact(proposal) or len(
            _canonical({"proposal": proposal}).encode()) > 8192:
        raise ValueError("proposal must be bounded JSON dictionary")
    # Freeze exact caller-owned nested JSON before interpreting or retaining
    # it. Otherwise later mutation rewrites the finalized evidence in place.
    proposal = json.loads(_canonical(proposal))
    kind = proposal.get("kind")
    review: dict[str, object]
    if not independent_transition_parity(case):
        review = {"verdict": "UNKNOWN", "reason": "INDEPENDENT_TRANSITION_DISAGREEMENT",
                  "counterexample": None, "review_status": "BLOCKED_EVALUATOR_PARITY"}
    elif kind == "PLAN" and set(proposal) == {"kind", "actions"}:
        review = verify_plan(case, proposal["actions"])
        if (review.get("verdict") == "VERIFIED_WIN") != independent_plan_won(
                case, proposal["actions"]):
            review = {"verdict": "UNKNOWN", "reason": "INDEPENDENT_PLAN_DISAGREEMENT",
                      "counterexample": None, "review_status": "BLOCKED_SCORER_PARITY"}
    elif kind == "CLAIM_UNSAT" and set(proposal) == {"kind"}:
        outcome = search(case)
        closure = independent_closure_witness(case)
        complete = outcome.get("complete") is True
        witness = outcome.get("witness")
        valid_counterexample = (type(witness) is list and
                                verify_plan(case, witness)["verdict"] == "VERIFIED_WIN")
        if closure is not None:
            # An independently reconstructed valid path falsifies even a
            # syntactically complete but WRONG search certificate.
            witness = closure
            valid_counterexample = verify_plan(case, closure)["verdict"] == "VERIFIED_WIN"
        proven = (closure is None and complete and witness is None and outcome.get("verdict") ==
                  "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        review = {
            "verdict": ("PROVEN_UNSAT_UNDER_DECLARED_RULES" if proven else
                        "VERIFIED_FAILURE" if valid_counterexample else
                        "NO_KNOWN_WIN" if not complete else "UNKNOWN"),
            "reason": ("EXHAUSTIVE_FINITE_SEARCH" if proven else
                       "COUNTEREXAMPLE_WIN_EXISTS" if valid_counterexample else
                       "SEARCH_INCOMPLETE" if not complete else
                       "INCONSISTENT_SEARCH_EVIDENCE"),
            "counterexample": witness if valid_counterexample else None,
            "nodes_explored": outcome.get("nodes_explored"),
            "search_complete": complete,
            "independent_closure_checked": True,
        }
    elif kind == "NOVEL_RULE" and set(proposal) == {"kind", "description"} and (
            type(proposal["description"]) is str and 0 < len(proposal["description"]) <= 2000):
        review = {"verdict": "UNKNOWN", "reason": "UNTESTED_RULE_REVISION_PENDING",
                  "review_status": "PENDING_INDEPENDENT_REVIEW", "counterexample": None}
    else:
        review = {"verdict": "UNKNOWN", "reason": "UNTESTED_OR_INVALID_PROPOSAL",
                  "counterexample": None}
    verdict = review["verdict"]
    result: dict[str, object] = {
        "schema": "labyrinth.switchback-circuit-receipt.v1",
        "case_id": case.case_id, "seed": case.seed,
        "mechanic_signature": case.mechanic_signature,
        "source_commit": "UNBOUND", "evidence_class": "SYNTHETIC_SOURCE_TEST",
        "private_custody": False, "neural_learning_verified": False,
        "graduation_eligible": False, "semantic_family_independence": "UNQUALIFIED",
        "verdict": verdict, "verified_success": verdict == "VERIFIED_WIN",
        "proposal": proposal, "review": review,
    }
    result["sha256"] = sha256(_canonical(result).encode()).hexdigest()
    return result


def adjudicate_session(case: Case, history: object, proposal: object) -> dict[str, object]:
    """Bind candidate probes to an answer; no zero-probe history substitution."""
    if not verify_probe_trace(case, history) or not independent_probe_trace(case, history):
        raise ValueError("candidate probe trajectory failed independent replay")
    core = adjudicate(case, proposal)
    trace = deepcopy(list(history))
    payload: dict[str, object] = {
        "schema": "labyrinth.switchback-session-receipt.v1",
        "case_id": case.case_id, "seed": case.seed,
        "verdict": core["verdict"], "verified_success": core["verified_success"],
        "neural_learning_verified": False, "graduation_eligible": False,
        "evidence_class": "SYNTHETIC_SOURCE_TEST",
        "probe_count": len(trace), "probe_trace": trace,
        "probe_sha256": sha256(_canonical({"trace": trace}).encode()).hexdigest(),
        "adjudication": core,
    }
    payload["sha256"] = sha256(_canonical(payload).encode()).hexdigest()
    return payload


def _receipt_semantics_hold(case: Case, core: dict[str, object]) -> bool:
    """Independent terminal truth check before trusting ANY receipt generator.

    In particular, changing `adjudicate` itself must not become a shared
    emission/replay oracle that certifies an invalid WIN or UNSAT claim.
    """
    proposal = core.get("proposal")
    if type(proposal) is not dict or not independent_transition_parity(case):
        return False
    kind = proposal.get("kind")
    if kind == "PLAN" and set(proposal) == {"kind", "actions"}:
        semantic_verdict = ("VERIFIED_WIN" if independent_plan_won(
            case, proposal["actions"]) else "VERIFIED_FAILURE")
    elif kind == "CLAIM_UNSAT" and set(proposal) == {"kind"}:
        # Depth-first finite closure independent of BFS search and the
        # adjudicate receipt generator. Wrong complete=True cannot prove UNSAT.
        semantic_verdict = ("VERIFIED_FAILURE" if independent_closure_witness(case)
                            is not None else "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        if core.get("verdict") in ("UNKNOWN", "NO_KNOWN_WIN"):
            # A bounded evaluator may abstain even on truly closed rules.
            return core.get("verified_success") is False
    else:
        semantic_verdict = "UNKNOWN"
    return (core.get("verdict") == semantic_verdict and
            core.get("verified_success") is (semantic_verdict == "VERIFIED_WIN"))


def verify_receipt_replay(receipt: object) -> bool:
    if type(receipt) is not dict or type(receipt.get("sha256")) is not str:
        return False
    try:
        unsigned = {k: v for k, v in receipt.items() if k != "sha256"}
        if sha256(_canonical(unsigned).encode()).hexdigest() != receipt["sha256"]:
            return False
        if receipt.get("schema") == "labyrinth.switchback-session-receipt.v1":
            core = receipt["adjudication"]
            case = oracle_case(receipt["seed"])
            trace = receipt["probe_trace"]
            if (type(trace) is not list or not verify_probe_trace(case, trace)
                    or not independent_probe_trace(case, trace)
                    or type(receipt["probe_count"]) is not int
                    or receipt["probe_count"] != len(trace)
                    or receipt["probe_sha256"] != sha256(
                        _canonical({"trace": trace}).encode()).hexdigest()
                    or receipt["case_id"] != case.case_id
                    or receipt["neural_learning_verified"] is not False
                    or receipt["graduation_eligible"] is not False
                    or receipt["evidence_class"] != "SYNTHETIC_SOURCE_TEST"
                    or type(core) is not dict
                    or core.get("schema") != "labyrinth.switchback-circuit-receipt.v1"
                    or core.get("seed") != case.seed
                    or not _receipt_semantics_hold(case, core)
                    or not verify_receipt_replay(core)):
                return False
            # Independent receipt witness: NEVER call adjudicate_session to
            # verify a receipt made by adjudicate_session. Doing so lets a
            # single compromised generator self-certify a fabricated win.
            expected = {
                "schema": "labyrinth.switchback-session-receipt.v1",
                "case_id": case.case_id, "seed": case.seed,
                "verdict": core["verdict"], "verified_success": core["verified_success"],
                "neural_learning_verified": False, "graduation_eligible": False,
                "evidence_class": "SYNTHETIC_SOURCE_TEST",
                "probe_count": len(trace), "probe_trace": trace,
                "probe_sha256": sha256(_canonical({"trace": trace}).encode()).hexdigest(),
                "adjudication": core,
            }
            expected["sha256"] = sha256(_canonical(expected).encode()).hexdigest()
            return _canonical(expected) == _canonical(receipt)
        canonical = oracle_case(receipt["seed"])
        return (_receipt_semantics_hold(canonical, receipt) and
                _canonical(adjudicate(canonical, receipt["proposal"])) == _canonical(receipt))
    except Exception:
        return False
