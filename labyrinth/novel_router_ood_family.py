"""Source-public OOD family: irreversible Boolean causal-latch network.

This is NOT a seed variant of the three-bit register or delayed packet
router. The primitive law is a synchronous threshold dependency graph with
permanently sealed nodes and direct local interventions. All receipts are
unqualified synthetic source evidence, not sealed model evaluations.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math


NODES = 5
MASK = (1 << NODES) - 1
MAX_MOVES = 6
MAX_PROBES = 16
MAX_SEED = 2**31 - 1
TOGGLE_ACTIONS = tuple(f"TOGGLE_{node}" for node in range(NODES))
SEAL_ACTIONS = tuple(f"SEAL_{node}" for node in range(NODES))
PLAN_ACTIONS = TOGGLE_ACTIONS + SEAL_ACTIONS + ("PULSE", "COMMIT")
PROBE_ACTIONS = TOGGLE_ACTIONS + SEAL_ACTIONS + ("PULSE", "RESET")


@dataclass(frozen=True)
class Case:
    seed: int
    parents: tuple[int, ...]
    thresholds: tuple[int, ...]
    initial_bits: int
    target_bits: int
    target_sealed: int
    max_seals: int
    case_id: str
    mechanic_signature: str


@dataclass(frozen=True)
class State:
    bits: int
    sealed: int = 0
    ticks: int = 0
    seals_used: int = 0
    post_seal_pulse: bool = False
    committed: bool = False
    won: bool = False


def make_case(seed: int) -> Case:
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError("case seed must be bounded exact int")
    data = sha256(f"causal-latch-ood.v1:{seed}".encode()).digest()
    parents = []
    thresholds = []
    for node in range(NODES):
        # Mandatory directed ring connection prevents a disconnected dummy
        # node, while other influences vary per seed. The node never counts
        # itself, so an intervention and a neighbor response are distinct.
        neighbor_mask = ((data[node] & MASK) | (1 << ((node + 1) % NODES))) & ~(1 << node)
        parents.append(neighbor_mask)
        thresholds.append(1 + data[NODES + node] % neighbor_mask.bit_count())
    initial_bits = data[10] & MASK
    target_bits = data[11] & MASK
    first_seal = data[13] % NODES
    second_seal = data[14] % NODES
    if second_seal == first_seal:
        second_seal = (second_seal + 1) % NODES
    target_sealed = (1 << first_seal) | (1 << second_seal)
    # Requiring two irreversible commitments before a final causal pulse
    # prevents the trivial strategy "pulse once, toggle all five target bits".
    max_seals = 2
    mechanic = sha256(b"causal-latch.mechanic.v1\0" +
                      bytes((*parents, *thresholds, max_seals))).hexdigest()
    identity = sha256((f"causal-latch.case.v1:{seed}:{mechanic}:" +
                       f"{initial_bits}:{target_bits}:{target_sealed}").encode()).hexdigest()
    return Case(seed, tuple(parents), tuple(thresholds), initial_bits,
                target_bits, target_sealed, max_seals, identity, mechanic)


def _valid_case(case: object) -> bool:
    if (type(case) is not Case or type(case.parents) is not tuple or
            type(case.thresholds) is not tuple or
            len(case.parents) != NODES or len(case.thresholds) != NODES or
            any(type(number) is not int for number in
                (case.seed, case.initial_bits, case.target_bits,
                 case.target_sealed, case.max_seals,
                 *case.parents, *case.thresholds)) or
            type(case.case_id) is not str or
            type(case.mechanic_signature) is not str):
        return False
    try:
        canonical = make_case(case.seed)
    except (ValueError, TypeError):
        return False
    # No reliance on potentially monkeypatched Case.__eq__.
    return (case.seed, case.parents, case.thresholds, case.initial_bits,
            case.target_bits, case.target_sealed, case.max_seals, case.case_id,
            case.mechanic_signature) == (
                canonical.seed, canonical.parents, canonical.thresholds,
                canonical.initial_bits, canonical.target_bits,
                canonical.target_sealed, canonical.max_seals,
                canonical.case_id, canonical.mechanic_signature)


def initial_state(case: Case) -> State:
    return State(case.initial_bits)


def public_view(case: Case) -> dict[str, object]:
    return {
        "prompt": ("Five Boolean nodes evolve synchronously under an unknown "
                   "directed threshold-dependency graph. TOGGLE_i directly "
                   "intervenes on one unlocked node. SEAL_i irreversibly "
                   "freezes that node. PULSE simultaneously updates every "
                   "unsealed node from the previous complete network state. "
                   "Reach the target after sealing TWO distinct nodes and "
                   "executing a PULSE with both seals in place, then COMMIT "
                   "within the six-action limit."),
        "initial_bits": f"{case.initial_bits:05b}",
        "target_bits": f"{case.target_bits:05b}",
        "required_sealed_bits": f"{case.target_sealed:05b}",
        "max_seals": case.max_seals,
        "max_moves": MAX_MOVES,
        "max_probes": MAX_PROBES,
        "plan_actions": list(PLAN_ACTIONS),
        "probe_actions": list(PROBE_ACTIONS),
    }


def _state_key(state: State) -> tuple[int, int, int, int, bool, bool, bool]:
    if (type(state) is not State or
            any(type(x) is not int for x in (
                state.bits, state.sealed, state.ticks, state.seals_used)) or
            any(type(flag) is not bool for flag in (
                state.post_seal_pulse, state.committed, state.won))):
        raise ValueError("invalid state")
    return (state.bits, state.sealed, state.ticks, state.seals_used,
            state.post_seal_pulse, state.committed, state.won)


def _raw_initial(case: Case) -> tuple[int, int, int, int, bool]:
    """Primitive genesis never invokes the monkeypatchable State constructor."""
    return (case.initial_bits, 0, 0, 0, False)


def _trusted_state_from_raw(raw: tuple) -> State:
    """Quarantine state without invoking a potentially modified constructor."""
    fresh = object.__new__(State)
    for field, value in zip(("bits", "sealed", "ticks", "seals_used",
                             "post_seal_pulse", "committed", "won"),
                            (*raw, False, False)):
        object.__setattr__(fresh, field, value)
    return fresh


def _raw_next(case: Case, state: tuple, action: str) -> tuple | None:
    """Separate reference physics: no step/synchronous_pulse/State helpers."""
    bits, sealed, time, used_seals, after_seals = state
    if time >= MAX_MOVES:
        return None
    if action in TOGGLE_ACTIONS:
        node = TOGGLE_ACTIONS.index(action)
        if sealed & (1 << node):
            return None
        return (bits ^ (1 << node), sealed, time + 1, used_seals, after_seals)
    if action in SEAL_ACTIONS:
        node = SEAL_ACTIONS.index(action)
        if sealed & (1 << node) or used_seals >= case.max_seals:
            return None
        return (bits, sealed | (1 << node), time + 1, used_seals + 1,
                after_seals)
    if action == "PULSE":
        new_bits = 0
        for node in range(NODES):
            if sealed & (1 << node):
                bit_value = (bits >> node) & 1
            else:
                parents_on = sum((bits >> parent) & 1 for parent in range(NODES)
                                 if case.parents[node] & (1 << parent))
                bit_value = int(parents_on >= case.thresholds[node])
            new_bits |= bit_value << node
        return (new_bits, sealed, time + 1, used_seals,
                after_seals or used_seals == case.max_seals)
    raise ValueError("not a recognized raw action")


def _raw_goal(case: Case, state: tuple) -> bool:
    return (state[0] == case.target_bits and state[1] == case.target_sealed
            and state[3] == case.max_seals and state[4])


def _raw_plan_verdict(case: Case, actions: object) -> str:
    if (type(actions) is not list or not 1 <= len(actions) <= MAX_MOVES
            or any(type(action) is not str or action not in PLAN_ACTIONS
                   for action in actions)):
        return "VERIFIED_FAILURE"
    state = _raw_initial(case)
    for index, action in enumerate(actions):
        if action == "COMMIT":
            return ("VERIFIED_WIN" if index == len(actions) - 1 and
                    _raw_goal(case, state) else "VERIFIED_FAILURE")
        state = _raw_next(case, state, action)
        if state is None:
            return "VERIFIED_FAILURE"
    return "VERIFIED_FAILURE"


def _raw_closure_witness(case: Case) -> list[str] | None:
    @lru_cache(maxsize=None)
    def solve(state: tuple):
        if state[2] >= MAX_MOVES:
            return None
        if _raw_goal(case, state):
            return ("COMMIT",)
        if state[2] >= MAX_MOVES - 1:
            return None
        for action in ("PULSE", *reversed(SEAL_ACTIONS),
                       *reversed(TOGGLE_ACTIONS)):
            next_state = _raw_next(case, state, action)
            if next_state is None:
                continue
            continuation = solve(next_state)
            if continuation is not None:
                return (action, *continuation)
        return None

    found = solve(_raw_initial(case))
    return list(found) if found is not None else None


def synchronous_pulse(case: Case, bits: int, sealed: int) -> int:
    """Snapshot update: ALL unlocked nodes read the SAME old bitmask."""
    if any(type(x) is not int for x in (bits, sealed)) or not 0 <= bits <= MASK:
        raise ValueError("invalid pulse input")
    updated = bits
    for node, (neighbors, threshold) in enumerate(zip(case.parents, case.thresholds)):
        bit = 1 << node
        if sealed & bit:
            continue
        active = (bits & neighbors).bit_count()
        if active >= threshold:
            updated |= bit
        else:
            updated &= ~bit
    return updated & MASK


def step(case: Case, state: State, action: str) -> State:
    if type(case) is not Case or type(state) is not State or type(action) is not str:
        raise ValueError("invalid action types")
    if state.committed or state.ticks >= MAX_MOVES or action not in PLAN_ACTIONS:
        raise ValueError("unknown or exhausted action")
    if action == "COMMIT":
        won = (state.bits == case.target_bits and
               state.sealed == case.target_sealed and
               state.seals_used == case.max_seals and state.post_seal_pulse)
        return State(state.bits, state.sealed, state.ticks, state.seals_used,
                     state.post_seal_pulse, True, won)
    if action in TOGGLE_ACTIONS:
        node = TOGGLE_ACTIONS.index(action)
        if state.sealed & (1 << node):
            raise ValueError("sealed node cannot be changed")
        return State(state.bits ^ (1 << node), state.sealed,
                     state.ticks + 1, state.seals_used, state.post_seal_pulse)
    if action in SEAL_ACTIONS:
        node = SEAL_ACTIONS.index(action)
        if state.sealed & (1 << node) or state.seals_used >= case.max_seals:
            raise ValueError("irreversible seal capacity exceeded")
        return State(state.bits, state.sealed | (1 << node),
                     state.ticks + 1, state.seals_used + 1, state.post_seal_pulse)
    return State(synchronous_pulse(case, state.bits, state.sealed), state.sealed,
                 state.ticks + 1, state.seals_used,
                 state.post_seal_pulse or state.seals_used == case.max_seals)


def verify_plan(case: Case, actions: object) -> dict[str, object]:
    if not _valid_case(case):
        raise ValueError("uncustodied case")
    if (type(actions) is not list or not 1 <= len(actions) <= MAX_MOVES or
            any(type(action) is not str or action not in PLAN_ACTIONS
                for action in actions)):
        return {"verdict": "VERIFIED_FAILURE", "reason": "INVALID_PLAN"}
    state = initial_state(case)
    for i, action in enumerate(actions):
        try:
            state = step(case, state, action)
        except ValueError:
            return {"verdict": "VERIFIED_FAILURE", "reason": "ILLEGAL_ACTION", "index": i}
        if state.committed and i != len(actions) - 1:
            return {"verdict": "VERIFIED_FAILURE", "reason": "ACTION_AFTER_COMMIT"}
    return {"verdict": "VERIFIED_WIN" if state.committed and state.won else
            "VERIFIED_FAILURE", "reason": "DECLARED_FINITE_GOAL"}


def search(case: Case, *, max_nodes: int = 100_000) -> dict[str, object]:
    """Complete forward finite closure or explicit NO_KNOWN_WIN if capped."""
    if not _valid_case(case) or type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("invalid search input")
    root = initial_state(case)
    if (root.bits, root.sealed, root.ticks, root.seals_used,
            root.post_seal_pulse) != _raw_initial(case):
        return {"verdict": "UNKNOWN", "complete": False,
                "nodes_explored": 0, "witness": None,
                "reason": "INDEPENDENT_GENESIS_DISAGREEMENT"}
    frontier = deque([(root, ())])
    seen = {_state_key(root)}
    explored = 0
    while frontier:
        if explored >= max_nodes:
            return {"verdict": "NO_KNOWN_WIN", "complete": False,
                    "nodes_explored": explored, "witness": None}
        state, prefix = frontier.popleft()
        explored += 1
        if len(prefix) >= MAX_MOVES:
            continue
        proposal = prefix + ("COMMIT",)
        source_verdict = verify_plan(case, list(proposal))["verdict"]
        independent_verdict = _raw_plan_verdict(case, list(proposal))
        if source_verdict != independent_verdict:
            return {"verdict": "UNKNOWN", "complete": False,
                    "nodes_explored": explored, "witness": None,
                    "reason": "INDEPENDENT_SCORER_DISAGREEMENT"}
        if independent_verdict == "VERIFIED_WIN":
            return {"verdict": "VERIFIED_WIN", "complete": True,
                    "nodes_explored": explored, "witness": list(proposal)}
        for action in PLAN_ACTIONS[:-1]:
            try:
                changed = step(case, state, action)
            except ValueError:
                continue
            key = _state_key(changed)
            if key not in seen:
                seen.add(key)
                frontier.append((changed, prefix + (action,)))
    if _raw_closure_witness(case) is not None:
        return {"verdict": "UNKNOWN", "complete": False,
                "nodes_explored": explored, "witness": None,
                "reason": "INDEPENDENT_CLOSURE_DISAGREEMENT"}
    return {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "complete": True,
            "nodes_explored": explored, "witness": None}


def _canonical(value: dict[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _exact_json(value: object, depth: int = 0) -> bool:
    """Deny Python value subclasses that can lie about comparisons.

    A str subclass can serialize as `VERIFIED_WIN` while claiming to compare
    equal to `VERIFIED_FAILURE`.  The replay boundary accepts exact JSON
    primitives only, not arbitrary in-process Python objects.
    """
    if depth > 12:
        return False
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return len(value) <= 128 and all(
            _exact_json(item, depth + 1) for item in value)
    if type(value) is dict:
        return len(value) <= 128 and all(
            type(key) is str and _exact_json(item, depth + 1)
            for key, item in value.items())
    return False


def adjudicate(case: Case, proposal: object) -> dict[str, object]:
    if not _valid_case(case) or type(proposal) is not dict:
        raise ValueError("invalid input")
    proposal = json.loads(_canonical(proposal))
    if len(_canonical(proposal).encode()) > 8192:
        raise ValueError("oversized proposal")
    kind = proposal.get("kind")
    if kind == "PLAN" and set(proposal) == {"kind", "actions"}:
        source_verdict = verify_plan(case, proposal["actions"])["verdict"]
        independent_verdict = _raw_plan_verdict(case, proposal["actions"])
        verdict = (source_verdict if source_verdict == independent_verdict else
                   "UNKNOWN")
        reason = ("TWO_PLAN_CHECKS_AGREE" if verdict != "UNKNOWN" else
                  "INDEPENDENT_SCORER_DISAGREEMENT")
        counterexample = None
    elif kind == "CLAIM_UNSAT" and set(proposal) == {"kind"}:
        result = search(case)
        independent = _raw_closure_witness(case)
        if independent is not None:
            verdict, reason, counterexample = (
                "VERIFIED_FAILURE", "INDEPENDENT_WIN_COUNTEREXAMPLE", independent)
        elif result["complete"] and result["verdict"] == "PROVEN_UNSAT_UNDER_DECLARED_RULES":
            verdict, reason, counterexample = (
                "PROVEN_UNSAT_UNDER_DECLARED_RULES", "TWO_FINITE_CLOSURES_AGREE", None)
        else:
            verdict, reason, counterexample = (
                "NO_KNOWN_WIN", "INCOMPLETE_OR_DISAGREEING_SOURCE_SEARCH", None)
    elif kind == "NOVEL_RULE" and set(proposal) == {"kind", "description"} and (
            type(proposal["description"]) is str and
            0 < len(proposal["description"]) <= 2000):
        verdict, reason, counterexample = (
            "NOVEL_RULESET_CHALLENGE", "UNVERIFIED_EXTERNAL_RULE", None)
    else:
        verdict, reason, counterexample = "UNKNOWN", "INVALID_PROPOSAL", None
    output = {
        "schema": "labyrinth.causal-latch-ood-receipt.v1",
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
    output["sha256"] = sha256(_canonical(output).encode()).hexdigest()
    return output


def _independent_probe_trace(case: Case, trace: object) -> bool:
    """Replay every charged public observation without the source step."""
    if type(trace) is not list or len(trace) > MAX_PROBES:
        return False
    state = _raw_initial(case)
    for row in trace:
        if type(row) is not dict or set(row) != {
                "action", "valid", "applied", "before", "after"}:
            return False
        action = row["action"]
        if type(action) is not str or len(action) > 32:
            return False
        if type(row["before"]) is not str or type(row["after"]) is not str:
            return False
        recognized = action in PROBE_ACTIONS
        if action == "RESET":
            advanced = _raw_initial(case)
        elif recognized:
            advanced = _raw_next(case, state, action)
        else:
            advanced = None
        applied = advanced is not None
        if (row["valid"] is not recognized or
                row["applied"] is not applied or
                row["before"] != f"{state[0]:05b}"):
            return False
        if applied:
            state = advanced
        if row["after"] != f"{state[0]:05b}":
            return False
    return True


def _raw_core_is_valid(case: Case, proposal: object, core: object) -> bool:
    """Distinct receipt verifier; never regenerate by calling adjudicate."""
    if (type(core) is not dict or type(proposal) is not dict or
            not _exact_json(core) or not _exact_json(proposal)):
        return False
    fields = {"schema", "evidence_class", "source_commit", "private_custody",
              "neural_learning_verified", "graduation_eligible", "case_id", "seed",
              "verdict", "verified_success", "reason", "counterexample",
              "proposal", "sha256"}
    if set(core) != fields or type(core["sha256"]) is not str:
        return False
    try:
        unsigned = {key: value for key, value in core.items() if key != "sha256"}
        if sha256(_canonical(unsigned).encode()).hexdigest() != core["sha256"]:
            return False
        if _canonical(core["proposal"]) != _canonical(proposal):
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    if (core["schema"] != "labyrinth.causal-latch-ood-receipt.v1" or
            core["evidence_class"] != "SOURCE_PUBLIC_SYNTHETIC" or
            core["source_commit"] != "UNBOUND" or
            core["private_custody"] is not False or
            core["neural_learning_verified"] is not False or
            core["graduation_eligible"] is not False or
            type(core["seed"]) is not int or core["seed"] != case.seed or
            core["case_id"] != case.case_id or
            type(core["reason"]) is not str or
            core["verified_success"] is not (core["verdict"] == "VERIFIED_WIN")):
        return False
    kind = proposal.get("kind")
    if kind == "PLAN" and set(proposal) == {"kind", "actions"}:
        return (core["verdict"] == _raw_plan_verdict(case, proposal["actions"]) and
                core["counterexample"] is None and
                core["reason"] == "TWO_PLAN_CHECKS_AGREE")
    if kind == "CLAIM_UNSAT" and set(proposal) == {"kind"}:
        witness = _raw_closure_witness(case)
        if witness is None:
            expected_reason = {
                "PROVEN_UNSAT_UNDER_DECLARED_RULES": "TWO_FINITE_CLOSURES_AGREE",
                "NO_KNOWN_WIN": "INCOMPLETE_OR_DISAGREEING_SOURCE_SEARCH",
            }.get(core["verdict"])
            return (expected_reason is not None and
                    core["reason"] == expected_reason and
                    core["counterexample"] is None)
        return (core["verdict"] == "VERIFIED_FAILURE" and
                core["reason"] == "INDEPENDENT_WIN_COUNTEREXAMPLE" and
                _raw_plan_verdict(case, core["counterexample"]) == "VERIFIED_WIN")
    if kind == "NOVEL_RULE" and set(proposal) == {"kind", "description"} and (
            type(proposal["description"]) is str and
            0 < len(proposal["description"]) <= 2000):
        return (core["verdict"] == "NOVEL_RULESET_CHALLENGE" and
                core["reason"] == "UNVERIFIED_EXTERNAL_RULE" and
                core["counterexample"] is None)
    return (core["verdict"] == "UNKNOWN" and
            core["reason"] == "INVALID_PROPOSAL" and
            core["counterexample"] is None)


def verify_session_receipt(receipt: object) -> bool:
    """Source-public replay without the session emitter or scorer generator."""
    if (type(receipt) is not dict or not _exact_json(receipt) or
            type(receipt.get("sha256")) is not str):
        return False
    try:
        case = make_case(receipt["seed"])
        expected_keys = {"schema", "evidence_class", "private_custody",
                         "neural_learning_verified", "graduation_eligible", "case_id",
                         "seed", "probe_count", "probe_trace", "probe_sha256",
                         "verdict", "verified_success", "adjudication", "sha256"}
        if set(receipt) != expected_keys:
            return False
        unsigned = {key: value for key, value in receipt.items() if key != "sha256"}
        if sha256(_canonical(unsigned).encode()).hexdigest() != receipt["sha256"]:
            return False
        trace = receipt["probe_trace"]
        core = receipt["adjudication"]
        return (receipt["schema"] == "labyrinth.causal-latch-ood-session.v1" and
                receipt["evidence_class"] == "SOURCE_PUBLIC_SYNTHETIC" and
                receipt["private_custody"] is False and
                receipt["neural_learning_verified"] is False and
                receipt["graduation_eligible"] is False and
                receipt["case_id"] == case.case_id and
                type(receipt["seed"]) is int and receipt["seed"] == case.seed and
                type(receipt["probe_count"]) is int and
                receipt["probe_count"] == len(trace) and
                receipt["probe_sha256"] == sha256(
                    _canonical({"trace": trace}).encode()).hexdigest() and
                _independent_probe_trace(case, trace) and
                _raw_core_is_valid(case, core["proposal"], core) and
                receipt["verdict"] == core["verdict"] and
                receipt["verified_success"] is core["verified_success"])
    except (KeyError, ValueError, TypeError, OverflowError, RecursionError):
        return False


class ProbeSession:
    """Educational charged public observations, NOT a private host sandbox."""

    def __init__(self, case: Case):
        if not _valid_case(case):
            raise ValueError("invalid canonical case")
        self._case = case
        self._state = initial_state(case)
        if (self._state.bits, self._state.sealed, self._state.ticks,
                self._state.seals_used, self._state.post_seal_pulse) != _raw_initial(case):
            raise ValueError("independent probe genesis disagrees with source")
        self._history: list[dict[str, object]] = []
        self._finished = False
        self._result: dict[str, object] | None = None

    def view(self) -> dict[str, object]:
        view = public_view(self._case)
        view.update({"current_bits": f"{self._state.bits:05b}",
                     "sealed_bits": f"{self._state.sealed:05b}",
                     "ticks": self._state.ticks,
                     "seals_used": self._state.seals_used,
                     "post_seal_pulse": self._state.post_seal_pulse,
                     "remaining_probes": MAX_PROBES - len(self._history),
                     "observations": deepcopy(self._history),
                     "finished": self._finished})
        return view

    def _terminal_probe_fault(self, action: str, before: str,
                              reason: str) -> dict[str, object]:
        self._history.append({"action": action, "valid": True,
                              "applied": False, "before": before,
                              "after": None})
        self._finished = True
        self._result = {"verdict": "UNKNOWN", "reason": reason,
                        "verified_success": False}
        return self.view()

    def probe(self, action: object) -> dict[str, object]:
        if self._finished:
            return self.view()
        if len(self._history) >= MAX_PROBES:
            self._finished = True
            self._result = {"verdict": "VERIFIED_FAILURE",
                            "reason": "PROBE_BUDGET_EXCEEDED", "verified_success": False}
            return self.view()
        if type(action) is not str or len(action) > 32:
            self._finished = True
            self._result = {"verdict": "UNKNOWN", "reason": "UNREPRESENTABLE_ACTION",
                            "verified_success": False}
            return self.view()
        before = f"{self._state.bits:05b}"
        valid = action in PROBE_ACTIONS
        applied = False
        if valid:
            if action == "RESET":
                try:
                    candidate = initial_state(self._case)
                    acceptable = ((candidate.bits, candidate.sealed, candidate.ticks,
                                   candidate.seals_used, candidate.post_seal_pulse) ==
                                  _raw_initial(self._case))
                except Exception:
                    acceptable = False
                if not acceptable:
                    # A failed RESET must still leave a known safe canonical
                    # public state, without exposing corrupted constructor
                    # contents or granting another attempt.
                    self._state = _trusted_state_from_raw(_raw_initial(self._case))
                    return self._terminal_probe_fault(
                        action, before, "INDEPENDENT_GENESIS_DISAGREEMENT")
                self._state = candidate
                applied = True
            else:
                source_state = (self._state.bits, self._state.sealed,
                                self._state.ticks, self._state.seals_used,
                                self._state.post_seal_pulse)
                try:
                    expected = _raw_next(self._case, source_state, action)
                except Exception:
                    return self._terminal_probe_fault(
                        action, before, "INDEPENDENT_TRANSITION_DISAGREEMENT")
                source_fault = False
                try:
                    evolved = step(self._case, self._state, action)
                except ValueError:
                    evolved = None
                except Exception:
                    evolved = None
                    source_fault = True
                try:
                    source_fields = (None if evolved is None else
                                     (evolved.bits, evolved.sealed, evolved.ticks,
                                      evolved.seals_used, evolved.post_seal_pulse))
                except Exception:
                    source_fields = None
                    source_fault = True
                if (source_fault or (expected is None) != (evolved is None) or
                        (expected is not None and source_fields != expected)):
                    return self._terminal_probe_fault(
                        action, before, "INDEPENDENT_TRANSITION_DISAGREEMENT")
                if evolved is not None:
                    self._state = evolved
                    applied = True
        self._history.append({"action": action, "valid": valid,
                              "applied": applied, "before": before,
                              "after": f"{self._state.bits:05b}"})
        return self.view()

    def submit(self, proposal: object) -> dict[str, object]:
        if self._finished:
            return deepcopy(self._result) if self._result else {
                "verdict": "UNKNOWN", "verified_success": False}
        self._finished = True
        try:
            if not _independent_probe_trace(self._case, self._history):
                raise ValueError("inconsistent public probe trace")
            core = adjudicate(self._case, proposal)
            if not _raw_core_is_valid(self._case, proposal, core):
                raise ValueError("independent core verdict disagrees")
            trace = deepcopy(self._history)
            result = {"schema": "labyrinth.causal-latch-ood-session.v1",
                      "evidence_class": "SOURCE_PUBLIC_SYNTHETIC",
                      "private_custody": False, "neural_learning_verified": False,
                      "graduation_eligible": False,
                      "case_id": self._case.case_id, "seed": self._case.seed,
                      "probe_count": len(trace), "probe_trace": trace,
                      "probe_sha256": sha256(_canonical({"trace": trace}).encode()).hexdigest(),
                      "verdict": core["verdict"],
                      "verified_success": core["verified_success"],
                      "adjudication": core}
            result["sha256"] = sha256(_canonical(result).encode()).hexdigest()
            if not verify_session_receipt(result):
                raise ValueError("independent session replay disagrees")
        except Exception:
            result = {"verdict": "EVALUATOR_DISAGREEMENT",
                      "reason": "BLOCKED_SUBMISSION_EVALUATOR",
                      "verified_success": False}
        self._result = result
        return deepcopy(result)


def main(argv: list[str] | None = None) -> int:
    """Bounded $0 public diagnostic; oracle output is explicitly privileged."""
    import argparse

    parser = argparse.ArgumentParser(description="Public causal-latch OOD puzzle")
    parser.add_argument("--seed", type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", nargs="+", metavar="ACTION")
    mode.add_argument("--claim-unsat", action="store_true")
    mode.add_argument("--oracle", action="store_true")
    args = parser.parse_args(argv)
    case = make_case(args.seed)
    if args.plan is not None:
        response: dict[str, object] = adjudicate(case, {
            "kind": "PLAN", "actions": args.plan})
    elif args.claim_unsat:
        response = adjudicate(case, {"kind": "CLAIM_UNSAT"})
    elif args.oracle:
        response = {"evidence_class": "SOURCE_PUBLIC_SYNTHETIC",
                    "privileged_symbolic_upper_bound": search(case)}
    else:
        response = {"evidence_class": "SOURCE_PUBLIC_SYNTHETIC",
                    "public_view": public_view(case)}
    print(json.dumps(response, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
