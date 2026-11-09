"""Precommitted, source-public Novel Router behavioral trials; no neural learning.

The candidate receives a view/probe API and NEVER a seed, Case or solver.
Same-process Python introspection can still expose closures: this is not secure
custody, and the manifest SHA is not an independently witnessed commitment.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import marshal
import math
import time
from types import FunctionType, MappingProxyType

from labyrinth import switchback_novel_transfer as novel


SCHEMA = "labyrinth.novel-router-behavioral-protocol.v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _digest(value):
    return sha256(_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class Budget:
    max_probes: int = 20
    max_calls: int = 90
    max_seconds: float = 1.0

    def __post_init__(self):
        if (type(self.max_probes) is not int or not 0 <= self.max_probes <= 20 or
                type(self.max_calls) is not int or not 1 <= self.max_calls <= 200 or
                type(self.max_seconds) not in (int, float) or
                not math.isfinite(self.max_seconds) or
                not .01 <= self.max_seconds <= 30):
            raise ValueError("invalid budget")


@dataclass(frozen=True)
class CandidateChoice:
    kind: str
    actions: tuple[str, ...]
    solvable_probability: float


class PublicObservationPort:
    """Two visible methods, no direct private fields; NOT a Python sandbox."""
    __slots__ = ("view", "probe")

    def __init__(self, view, probe):
        self.view = view
        self.probe = probe


@dataclass(frozen=True)
class TrialManifest:
    pairs: tuple[tuple[int, int], ...]
    policies: tuple[tuple[str, str], ...]
    budget: Budget
    sha256: str


@dataclass(frozen=True)
class TrialRecord:
    split: str
    pair_index: int
    policy: str
    budget: Budget
    probes: int
    calls: int
    elapsed_ms: int
    terminal: str
    proposal_kind: str
    source_verdict: str
    solved_by_source: bool
    candidate_confidence: float
    confidence_brier: float
    abstention_penalty: float
    false_win_penalty: float
    loss: float
    receipt_json: str
    receipt_sha256: str
    witnessed_by_public_probes: bool


@dataclass(frozen=True)
class BehavioralReport:
    manifest_sha256: str
    trials: tuple[TrialRecord, ...]
    totals: object
    neural_learning_verified: bool = False
    generalization_qualified: bool = False
    private_custody: bool = False
    optimizer_updates: int = 0


def _policy_signature(policy):
    if type(policy) is not FunctionType or policy.__code__.co_freevars:
        raise ValueError("precommit requires closure-free Python policy")
    name = policy.__module__ + "." + policy.__qualname__
    content = marshal.dumps(policy.__code__) + repr(policy.__defaults__).encode()
    return name, sha256(content).hexdigest()


def precommit(pairs, policies, budget=Budget()):
    """Freeze case roster and policy bytecode before outcome access.

    No independent timestamp, signer or model artifact custody is claimed.
    """
    if (type(pairs) is not tuple or not 0 < len(pairs) <= 1000 or
            type(policies) is not tuple or not policies or type(budget) is not Budget):
        raise ValueError("invalid precommit roster")
    seen = set()
    for pair in pairs:
        if (type(pair) is not tuple or len(pair) != 2 or pair[0] == pair[1] or
                any(type(seed) is not int or not 0 <= seed <= novel.MAX_SEED
                    or seed in seen for seed in pair)):
            raise ValueError("duplicate or invalid train/heldout seed")
        seen.update(pair)
    signatures = tuple(_policy_signature(policy) for policy in policies)
    if len(set(signatures)) != len(signatures):
        raise ValueError("duplicate policy")
    if len({name for name, _ in signatures}) != len(signatures):
        # TrialRecord stores the policy name, not its code digest. Distinct
        # code bodies with one displayed name cannot be attributed per row.
        raise ValueError("duplicate policy name")
    digest = _digest({"schema": SCHEMA, "pairs": pairs, "policies": signatures,
                      "budget": asdict(budget)})
    return TrialManifest(pairs, signatures, budget, digest)


def _valid_manifest(manifest):
    """Re-admit an untrusted manifest at every consumption boundary.

    A SHA256 of self-declared fields is NOT a precommit receipt. In particular,
    an attacker must not turn a training seed into its own heldout twin.
    """
    if (type(manifest) is not TrialManifest or
            type(manifest.pairs) is not tuple or
            not 0 < len(manifest.pairs) <= 1000 or
            type(manifest.policies) is not tuple or not manifest.policies or
            type(manifest.budget) is not Budget or
            type(manifest.sha256) is not str):
        return False
    try:
        # Dataclass fields can be forcibly replaced after construction.
        Budget(manifest.budget.max_probes, manifest.budget.max_calls,
               manifest.budget.max_seconds)
        seen_seeds = set()
        for pair in manifest.pairs:
            if (type(pair) is not tuple or len(pair) != 2 or
                    any(type(seed) is not int or not 0 <= seed <= novel.MAX_SEED
                        or seed in seen_seeds for seed in pair) or
                    pair[0] == pair[1]):
                return False
            seen_seeds.update(pair)
        names = set()
        for item in manifest.policies:
            if type(item) is not tuple or len(item) != 2:
                return False
            name, digest = item
            if (type(name) is not str or not name or name in names or
                    type(digest) is not str or len(digest) != 64 or
                    any(c not in "0123456789abcdef" for c in digest)):
                return False
            names.add(name)
        return manifest.sha256 == _digest({
            "schema": SCHEMA, "pairs": manifest.pairs,
            "policies": manifest.policies, "budget": asdict(manifest.budget)})
    except Exception:
        return False


class _BudgetExceeded(Exception):
    pass


def _proposal(choice):
    if (type(choice) is not CandidateChoice or
            type(choice.kind) is not str or
            type(choice.solvable_probability) not in (float, int) or
            not math.isfinite(choice.solvable_probability) or
            not 0 <= choice.solvable_probability <= 1 or
            type(choice.actions) is not tuple):
        raise ValueError("invalid confidence or decision")
    if choice.kind == "PLAN" and 1 <= len(choice.actions) <= novel.MAX_MOVES and all(
            type(action) is str and action in novel.PLAN_ACTIONS
            for action in choice.actions):
        return {"kind": "PLAN", "actions": list(choice.actions)}
    if choice.kind in ("ABSTAIN", "CLAIM_UNSAT") and choice.actions == ():
        return {"kind": choice.kind}
    raise ValueError("invalid proposal")


def _witness(view, path):
    trace = view.get("observations")
    return (type(trace) is list and len(trace) == len(path) and
            all(type(row) is dict and row.get("action") == action and
                row.get("valid") is True and row.get("applied") is True
                for row, action in zip(trace, path)) and
            view.get("current_levels") == view.get("target_levels") and
            view.get("pending_packet") == 0 and
            view.get("post_delivery_observed") is True and
            view.get("ticks", -1) >= view.get("switch_at", 100))


def fixed_witness(port):
    """Constant legal public plan, only claim a WIN after observations."""
    initial = port.view()
    path = []
    for action in ["TICK"] * initial["switch_at"] + ["LOAD", "TICK"]:
        observed = port.probe(action)
        path.append(action)
        if observed.get("finished") or not observed["observations"][-1]["applied"]:
            return CandidateChoice("ABSTAIN", (), .5)
    return (CandidateChoice("PLAN", tuple(path + ["COMMIT"]), 1.)
            if _witness(observed, path) else CandidateChoice("ABSTAIN", (), .5))


def feedback_witness(port):
    """Deterministic, observed-level routing; never a private rule lookup."""
    initial = port.view()
    path = []
    for action in ["TICK"] * initial["switch_at"]:
        observed = port.probe(action)
        path.append(action)
        if observed.get("finished") or not observed["observations"][-1]["applied"]:
            return CandidateChoice("ABSTAIN", (), .5)
    for iteration in range(initial["maximum_loads"]):
        view = port.view()
        delta = ((view["target_levels"][0] - view["current_levels"][0]) % 4,
                 (view["target_levels"][1] - view["current_levels"][1]) % 4)
        if (iteration == 0 and delta[1] > delta[0] and
                view["routes_used"] < view["maximum_routes"] and
                len(path) + 4 <= 11):
            observed = port.probe("ROUTE")
            path.append("ROUTE")
            if not observed["observations"][-1]["applied"]:
                return CandidateChoice("ABSTAIN", (), .5)
        if len(path) + 3 > novel.MAX_MOVES:
            break
        for action in ("LOAD", "TICK"):
            observed = port.probe(action)
            path.append(action)
            if observed.get("finished") or not observed["observations"][-1]["applied"]:
                return CandidateChoice("ABSTAIN", (), .5)
        if _witness(observed, path):
            return CandidateChoice("PLAN", tuple(path + ["COMMIT"]), 1.)
    return CandidateChoice("ABSTAIN", (), .5)


def _one_trial(case, split, pair_index, policy, budget):
    session = novel.ProbeSession(case)
    calls, probes, actions = 0, 0, []
    start = time.monotonic()

    def charge(probing=False):
        nonlocal calls, probes
        if (time.monotonic() - start > budget.max_seconds or
                calls >= budget.max_calls or
                (probing and probes >= budget.max_probes)):
            raise _BudgetExceeded
        calls += 1
        if probing:
            probes += 1

    def view():
        charge()
        return deepcopy(session.view())

    def probe(action):
        charge(probing=True)
        if type(action) is not str:
            raise ValueError("invalid public action")
        actions.append(action)
        return deepcopy(session.probe(action))

    terminal, choice = "SUBMITTED", None
    try:
        choice = policy(PublicObservationPort(view, probe))
        proposal = _proposal(choice)
        if time.monotonic() - start > budget.max_seconds:
            raise _BudgetExceeded
    except _BudgetExceeded:
        terminal, proposal = "BUDGET_EXHAUSTED", {"kind": "ABSTAIN"}
    except (ValueError, TypeError, KeyError, IndexError, AttributeError,
            RecursionError, OverflowError):
        terminal, proposal = "INVALID_DECISION", {"kind": "ABSTAIN"}
    except Exception:
        terminal, proposal = "POLICY_ERROR", {"kind": "ABSTAIN"}
    # Candidate time only: exclude potentially costly host-side replay/closure.
    candidate_elapsed_ms = round((time.monotonic() - start) * 1000)
    observed = session.view()
    witnessed = bool(proposal["kind"] == "PLAN" and
                     choice is not None and len(choice.actions) > 0 and
                     list(choice.actions[:-1]) == actions and
                     choice.actions[-1] == "COMMIT" and _witness(observed, actions))
    receipt = session.submit(proposal)
    replay = novel.ProbeSession(case)
    for action in actions:
        replay.probe(action)
    if _json(receipt) != _json(replay.submit(deepcopy(proposal))):
        raise RuntimeError("source session replay disagreed")
    if proposal["kind"] == "PLAN" and receipt["verified_success"] and (
            novel._independent_plan_verdict(case, proposal["actions"]) !=
            "VERIFIED_WIN"):
        raise RuntimeError("independent plan scorer disagreed")
    closure = novel.search(case)  # host-side AFTER candidate terminal decision
    if (closure["verdict"] not in ("VERIFIED_WIN", "PROVEN_UNSAT_UNDER_DECLARED_RULES")
            or not closure["complete"]):
        raise RuntimeError("source cannot calibrate without complete closure")
    solvable = closure["verdict"] == "VERIFIED_WIN"
    confidence = (choice.solvable_probability if terminal == "SUBMITTED"
                  and choice is not None else .5)
    brier = (confidence - float(solvable)) ** 2
    abstention = .4 if proposal["kind"] == "ABSTAIN" else 0.
    false_win = 1. if proposal["kind"] == "PLAN" and receipt["verdict"] != "VERIFIED_WIN" else 0.
    false_unsat = 1. if proposal["kind"] == "CLAIM_UNSAT" and solvable else 0.
    serialized = _json(receipt)
    return TrialRecord(split, pair_index, _policy_signature(policy)[0], budget,
                       probes, calls, candidate_elapsed_ms,
                       terminal, proposal["kind"], receipt["verdict"], solvable,
                       confidence, round(brier, 8), round(abstention, 8),
                       round(false_win + false_unsat, 8),
                       round(brier + abstention + false_win + false_unsat, 8),
                       serialized, sha256(serialized.encode()).hexdigest(), witnessed)


def evaluate(manifest, policies):
    """Each declared policy gets identical limits for every frozen case."""
    if (not _valid_manifest(manifest) or type(policies) is not tuple or
            tuple(_policy_signature(p) for p in policies) != manifest.policies or
            manifest.sha256 != _digest({"schema": SCHEMA, "pairs": manifest.pairs,
                                       "policies": manifest.policies,
                                       "budget": asdict(manifest.budget)})):
        raise ValueError("precommit mismatch")
    trials = tuple(_one_trial(novel.make_case(seed), split, index, policy,
                              manifest.budget)
                   for index, pair in enumerate(manifest.pairs)
                   for split, seed in zip(("TRAIN", "HELDOUT"), pair)
                   for policy in policies)
    totals = {
        "count": len(trials),
        "verified_wins": sum(t.source_verdict == "VERIFIED_WIN" for t in trials),
        "failures": sum(t.source_verdict == "VERIFIED_FAILURE" for t in trials),
        "abstentions": sum(t.proposal_kind == "ABSTAIN" for t in trials),
        "other": sum(t.proposal_kind == "CLAIM_UNSAT" for t in trials),
        "false_wins": sum(t.proposal_kind == "PLAN" and
                          t.source_verdict != "VERIFIED_WIN" for t in trials),
        "false_unsat_claims": sum(t.proposal_kind == "CLAIM_UNSAT" and
                                  t.solved_by_source for t in trials),
        "total_loss": round(sum(t.loss for t in trials), 8),
        "abstention_penalty": round(sum(t.abstention_penalty for t in trials), 8),
        "total_probes": sum(t.probes for t in trials),
        "total_calls": sum(t.calls for t in trials),
        "optimizer_updates": 0,
    }
    return BehavioralReport(manifest.sha256, trials, MappingProxyType(totals))


def verify_local_record(manifest, record):
    """Recompute source receipt and derivable grading fields from frozen case.

    This does NOT attest that the candidate emitted the recorded confidence
    or plan, used the recorded calls/time, or lacked host-private access.
    """
    if not _valid_manifest(manifest) or type(record) is not TrialRecord:
        return False
    try:
        if (manifest.sha256 != _digest({
                "schema": SCHEMA, "pairs": manifest.pairs,
                "policies": manifest.policies, "budget": asdict(manifest.budget)}) or
                type(record.budget) is not Budget or
                record.budget != manifest.budget or
                type(record.pair_index) is not int or
                not 0 <= record.pair_index < len(manifest.pairs) or
                type(record.split) is not str or
                record.split not in ("TRAIN", "HELDOUT") or
                type(record.policy) is not str or
                record.policy not in {name for name, _ in manifest.policies} or
                type(record.source_verdict) is not str or
                type(record.probes) is not int or
                not 0 <= record.probes <= manifest.budget.max_probes or
                type(record.calls) is not int or
                not record.probes <= record.calls <= manifest.budget.max_calls or
                type(record.elapsed_ms) is not int or record.elapsed_ms < 0 or
                type(record.terminal) is not str or
                record.terminal not in ("SUBMITTED", "BUDGET_EXHAUSTED",
                                        "INVALID_DECISION", "POLICY_ERROR") or
                type(record.proposal_kind) is not str or
                type(record.solved_by_source) is not bool or
                type(record.witnessed_by_public_probes) is not bool or
                type(record.candidate_confidence) not in (float, int) or
                not math.isfinite(record.candidate_confidence) or
                not 0 <= record.candidate_confidence <= 1 or
                any(type(value) not in (float, int) or not math.isfinite(value)
                    for value in (record.confidence_brier,
                                  record.abstention_penalty,
                                  record.false_win_penalty, record.loss)) or
                type(record.receipt_json) is not str or
                type(record.receipt_sha256) is not str or
                record.receipt_sha256 != sha256(record.receipt_json.encode()).hexdigest()):
            return False
        # These reject only derivable temporal contradictions. Actual calls,
        # wall-time and terminal classification are not independently attested.
        budget_ms = manifest.budget.max_seconds * 1000
        if (record.terminal == "SUBMITTED" and
                record.elapsed_ms > budget_ms + 1):
            return False
        if (record.terminal == "BUDGET_EXHAUSTED" and
                record.probes < manifest.budget.max_probes and
                record.calls < manifest.budget.max_calls and
                record.elapsed_ms + 1 < budget_ms):
            return False
        # Reject noncanonical serializations, duplicate keys and altered traces.
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate receipt key")
                result[key] = value
            return result
        source = json.loads(record.receipt_json, object_pairs_hook=unique,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if record.receipt_json != _json(source):
            return False
        actions = [row["action"] for row in source["probe_trace"]]
        if record.probes != len(actions) or record.calls < record.probes:
            return False
        private_seed = manifest.pairs[record.pair_index][
            0 if record.split == "TRAIN" else 1]
        case = novel.make_case(private_seed)
        session = novel.ProbeSession(case)
        for action in actions:
            if type(action) is not str:
                return False
            session.probe(action)
        proposal = source["adjudication"]["proposal"]
        expected_witness = bool(
            proposal["kind"] == "PLAN" and
            type(proposal["actions"]) is list and
            proposal["actions"][-1:] == ["COMMIT"] and
            proposal["actions"][:-1] == actions and
            _witness(session.view(), actions))
        expected = session.submit(deepcopy(proposal))
        closure = novel.search(case)
        if (not closure["complete"] or
                closure["verdict"] not in ("VERIFIED_WIN",
                                           "PROVEN_UNSAT_UNDER_DECLARED_RULES")):
            return False
        solvable = closure["verdict"] == "VERIFIED_WIN"
        if (record.terminal != "SUBMITTED" and
                (proposal["kind"] != "ABSTAIN" or
                 record.candidate_confidence != .5)):
            return False
        brier = round((record.candidate_confidence - float(solvable)) ** 2, 8)
        abstention = .4 if proposal["kind"] == "ABSTAIN" else 0.
        false_win = (1. if proposal["kind"] == "PLAN" and
                     expected["verdict"] != "VERIFIED_WIN" else 0.)
        false_unsat = 1. if proposal["kind"] == "CLAIM_UNSAT" and solvable else 0.
        penalty = round(false_win + false_unsat, 8)
        return (_json(expected) == record.receipt_json and
                source["verdict"] == record.source_verdict and
                source["adjudication"]["seed"] == private_seed and
                record.proposal_kind == proposal["kind"] and
                record.solved_by_source is solvable and
                record.witnessed_by_public_probes is expected_witness and
                record.confidence_brier == brier and
                record.abstention_penalty == abstention and
                record.false_win_penalty == penalty and
                record.loss == round(brier + abstention + penalty, 8))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError,
            RecursionError, OverflowError, RuntimeError):
        return False


def verify_local_report(manifest, report):
    """Bind each source-replayed record to its frozen roster position.

    Unlike single-record replay this checks ordered policy attribution and
    aggregate arithmetic. Neither path authenticates who ran the policy or
    what the policy actually emitted; raw child custody would be required.
    """
    if (not _valid_manifest(manifest) or
            type(report) is not BehavioralReport or
            type(report.manifest_sha256) is not str or
            report.manifest_sha256 != manifest.sha256 or
            type(report.trials) is not tuple or
            type(report.totals) is not MappingProxyType or
            report.neural_learning_verified is not False or
            report.generalization_qualified is not False or
            report.private_custody is not False or
            type(report.optimizer_updates) is not int or
            report.optimizer_updates != 0):
        return False
    roles = tuple((index, split, policy)
                  for index, pair in enumerate(manifest.pairs)
                  for split, _ in zip(("TRAIN", "HELDOUT"), pair)
                  for policy, _ in manifest.policies)
    if len(report.trials) != len(roles):
        return False
    for record, (index, split, policy) in zip(report.trials, roles):
        if (type(record) is not TrialRecord or
                type(record.pair_index) is not int or
                type(record.split) is not str or
                type(record.policy) is not str or
                record.pair_index != index or record.split != split or
                record.policy != policy or
                not verify_local_record(manifest, record)):
            return False
    records = report.trials
    expected = {
        "count": len(records),
        "verified_wins": sum(r.source_verdict == "VERIFIED_WIN" for r in records),
        "failures": sum(r.source_verdict == "VERIFIED_FAILURE" for r in records),
        "abstentions": sum(r.proposal_kind == "ABSTAIN" for r in records),
        "other": sum(r.proposal_kind == "CLAIM_UNSAT" for r in records),
        "false_wins": sum(r.proposal_kind == "PLAN" and
                          r.source_verdict != "VERIFIED_WIN" for r in records),
        "false_unsat_claims": sum(r.proposal_kind == "CLAIM_UNSAT" and
                                  r.solved_by_source for r in records),
        "total_loss": round(sum(r.loss for r in records), 8),
        "abstention_penalty": round(sum(r.abstention_penalty for r in records), 8),
        "total_probes": sum(r.probes for r in records),
        "total_calls": sum(r.calls for r in records),
        "optimizer_updates": 0,
    }
    try:
        return (all(type(key) is str for key in report.totals) and
                set(report.totals) == set(expected) and
                all(type(report.totals[key]) is type(value) and
                    report.totals[key] == value for key, value in expected.items()))
    except Exception:
        # A read-only mapping does not make custom keys/values safe to hash,
        # compare, or retrieve. Invalid reports cannot abort a verifier batch.
        return False
