"""Temporal repair tribunal: order-sensitive scheduling with renewable energy.

Unlike path navigation, static exact cover, or causal world revision, success
depends on *when* a job is executed: release windows, deadlines, prerequisites,
finite machine energy, and paid WAIT actions. All transitions are declared and
finite. Three separately written checks constrain a finite UNSAT certificate;
none proves impossibility under undeclared affordances.

This module is source-public symbolic software with ZERO neural updates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import unicodedata


VERDICTS = frozenset({
    "VERIFIED_WIN", "VERIFIED_FAILURE", "NO_KNOWN_WIN",
    "PROVEN_UNSAT_UNDER_DECLARED_RULES", "UNKNOWN",
    "NOVEL_RULESET_CHALLENGE",
})


@dataclass(frozen=True)
class Job:
    name: str
    release: int
    deadline: int
    duration: int
    energy: int
    after: tuple[int, ...]


@dataclass(frozen=True)
class Scenario:
    jobs: tuple[Job, ...]
    capacity: int
    horizon: int
    action_budget: int


def check_scenario(s: Scenario) -> None:
    """Reject ambiguous callbacks/subclasses before reaching any state machine."""
    if type(s) is not Scenario or type(s.jobs) is not tuple:
        raise ValueError("invalid temporal scenario type")
    if not 3 <= len(s.jobs) <= 6:
        raise ValueError("temporal scenario needs 3..6 jobs")
    if (type(s.capacity) is not int or not 1 <= s.capacity <= 4
            or type(s.horizon) is not int or not 4 <= s.horizon <= 18
            or type(s.action_budget) is not int
            or not 1 <= s.action_budget <= 18):
        raise ValueError("invalid capacity, horizon, or action budget")
    names = set()
    for j in s.jobs:
        if type(j) is not Job:
            raise ValueError("job must be an exact Job record")
        if (type(j.name) is not str or not 1 <= len(j.name) <= 12
                or not j.name.isascii() or not j.name.isalnum()
                or j.name in names):
            raise ValueError("invalid or duplicate job identity")
        names.add(j.name)
        if (any(type(v) is not int for v in
                (j.release, j.deadline, j.duration, j.energy))
                or not 0 <= j.release < s.horizon
                or not 1 <= j.deadline <= s.horizon
                or not 1 <= j.duration <= 4
                or not 1 <= j.energy <= s.capacity):
            raise ValueError("invalid temporal/energy job bounds")
        if (type(j.after) is not tuple
                or any(type(v) is not int or not 0 <= v < len(s.jobs)
                       for v in j.after)
                or len(set(j.after)) != len(j.after)):
            raise ValueError("invalid prerequisite index")
    # Cyclic prerequisites and impossible release windows are LEGAL finite
    # dilemmas, not invalid data and not proof of global impossibility.


def fingerprint(s: Scenario) -> str:
    check_scenario(s)
    payload = {"protocol": "minotaur-temporal-repair.v1",
               "capacity": s.capacity, "horizon": s.horizon,
               "action_budget": s.action_budget,
               "jobs": [(j.name, j.release, j.deadline, j.duration,
                         j.energy, j.after) for j in s.jobs]}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def public_view(s: Scenario, *, variant: int = 0) -> dict:
    check_scenario(s)
    if type(variant) is not int or not 0 <= variant < 8:
        raise ValueError("invalid perspective")
    settings = (
        "A disaster-repair coordinator must sequence equipment repairs.",
        "An orbital station needs jobs finished before their windows close.",
        "A Minotaur controls a single repair machine and its battery.",
        "A caravan's one mechanic faces timed and interdependent failures.",
    )
    conclusions = (
        "A novel valid schedule must be independently considered.",
        "Unlisted equipment or rule changes require explicit adjudication.",
    )
    return {
        "prompt": f"{settings[variant % 4]} Every job runs on the ONE "
        "machine. A RUN costs its job's duration and energy. A charged WAIT "
        "consumes one tick and restores one energy to battery capacity. "
        "Jobs can start only after their release tick and their prerequisite "
        "jobs finish; they must FINISH by their deadline. Finish every job "
        f"within {s.horizon} ticks and {s.action_budget} actions. "
        + conclusions[variant // 4],
        "jobs": [{"name": j.name, "release": j.release,
                  "deadline": j.deadline, "duration": j.duration,
                  "energy": j.energy, "after": list(j.after)} for j in s.jobs],
        "capacity": s.capacity, "horizon": s.horizon,
        "action_budget": s.action_budget,
        "candidate_format": "tuple[int,...] of job indices, -1 for WAIT",
        "mechanics_fingerprint": fingerprint(s),
    }


def verify_schedule(s: Scenario, actions: object) -> dict:
    """Direct sequential replay; refuses malformed, incomplete, or late plans."""
    check_scenario(s)
    tag = fingerprint(s)
    if (type(actions) is not tuple or not 1 <= len(actions) <= s.action_budget
            or any(type(v) is not int or not -1 <= v < len(s.jobs)
                   for v in actions)):
        return {"verdict": "VERIFIED_FAILURE", "reason": "invalid_action_sequence",
                "rules_fingerprint": tag, "actions_charged": 0}
    t, energy, done = 0, s.capacity, set()
    for step, choice in enumerate(actions, start=1):
        if choice == -1:
            if len(done) == len(s.jobs):
                return {"verdict": "VERIFIED_FAILURE", "reason": "trailing_wait",
                        "rules_fingerprint": tag, "actions_charged": step}
            t += 1
            energy = min(s.capacity, energy + 1)
        else:
            j = s.jobs[choice]
            if (choice in done or t < j.release or energy < j.energy
                    or any(prereq not in done for prereq in j.after)):
                return {"verdict": "VERIFIED_FAILURE", "reason": "invalid_job_start",
                        "rules_fingerprint": tag, "actions_charged": step}
            t += j.duration
            if t > j.deadline or t > s.horizon:
                return {"verdict": "VERIFIED_FAILURE", "reason": "missed_deadline",
                        "rules_fingerprint": tag, "actions_charged": step}
            energy -= j.energy
            done.add(choice)
            if len(done) == len(s.jobs) and step != len(actions):
                return {"verdict": "VERIFIED_FAILURE", "reason": "trailing_actions",
                        "rules_fingerprint": tag, "actions_charged": step}
        if t > s.horizon:
            return {"verdict": "VERIFIED_FAILURE", "reason": "horizon_exceeded",
                    "rules_fingerprint": tag, "actions_charged": step}
    if len(done) != len(s.jobs) or actions[-1] == -1:
        return {"verdict": "VERIFIED_FAILURE", "reason": "unfinished_jobs",
                "rules_fingerprint": tag, "actions_charged": len(actions)}
    return {"verdict": "VERIFIED_WIN", "reason": "all_jobs_before_deadlines",
            "rules_fingerprint": tag, "actions_charged": len(actions),
            "elapsed_ticks": t, "energy_remaining": energy}


def independent_witness_check(s: Scenario, witness: object) -> bool:
    """Separate exact-witness interpreter, not the solver or direct replay."""
    check_scenario(s)
    if (type(witness) is not tuple or not 1 <= len(witness) <= s.action_budget
            or any(type(code) is not int or code < -1
                   or code >= len(s.jobs) for code in witness)):
        return False
    clock = 0
    battery = s.capacity
    finished = [False] * len(s.jobs)
    for pos, op in enumerate(witness):
        if op == -1:
            clock += 1
            battery += battery < s.capacity
            if all(finished):
                return False
            continue
        task = s.jobs[op]
        if finished[op] or clock < task.release or battery < task.energy:
            return False
        for needed in task.after:
            if not finished[needed]:
                return False
        end = clock + task.duration
        if end > s.horizon or end > task.deadline:
            return False
        clock = end
        battery -= task.energy
        finished[op] = True
        if all(finished) and pos != len(witness) - 1:
            return False
    return all(finished) and clock <= s.horizon and witness[-1] != -1


def independent_reachability(s: Scenario, *, state_cap: int = 300_000) -> bool | None:
    """Independent time-layer DP, not BFS state transitions; None=unknown."""
    check_scenario(s)
    if type(state_cap) is not int or not 0 <= state_cap <= 1_000_000:
        return None
    final = (1 << len(s.jobs)) - 1
    by_tick: list[set[tuple[int, int, int]]] = [set() for _ in range(s.horizon + 1)]
    by_tick[0].add((0, s.capacity, 0))
    found_states = 1
    for tick, entries in enumerate(by_tick):
        for mask, battery, steps in tuple(entries):
            if found_states > state_cap:
                return None
            if mask == final:
                return True
            if steps >= s.action_budget:
                continue
            if tick + 1 <= s.horizon:
                nxt = (mask, min(s.capacity, battery + 1), steps + 1)
                if nxt not in by_tick[tick + 1]:
                    by_tick[tick + 1].add(nxt)
                    found_states += 1
            for index, job in enumerate(s.jobs):
                bit = 1 << index
                if mask & bit or tick < job.release or battery < job.energy:
                    continue
                required = sum(1 << p for p in job.after)
                if required & mask != required:
                    continue
                end = tick + job.duration
                if end > job.deadline or end > s.horizon:
                    continue
                if mask | bit == final:
                    return True
                nxt = (mask | bit, battery - job.energy, steps + 1)
                if nxt not in by_tick[end]:
                    by_tick[end].add(nxt)
                    found_states += 1
    return False


def independent_depth_enumeration(s: Scenario, *, state_cap: int = 300_000) -> bool | None:
    """Third separate exhaustive DFS veto for claimed finite impossibility."""
    check_scenario(s)
    if type(state_cap) is not int or not 0 <= state_cap <= 1_000_000:
        return None
    total = len(s.jobs)
    explored = 0

    class Exhausted(Exception):
        pass

    @lru_cache(None)
    def can_finish(tick: int, fuel: int, used: int, count: int) -> bool:
        nonlocal explored
        if explored >= state_cap:
            raise Exhausted
        explored += 1
        if used.bit_count() == total:
            return True
        if count == s.action_budget or tick == s.horizon:
            return False
        if can_finish(tick + 1, min(s.capacity, fuel + 1), used, count + 1):
            return True
        for i, job in enumerate(s.jobs):
            if (used >> i & 1 or tick < job.release or fuel < job.energy
                    or tick + job.duration > job.deadline
                    or tick + job.duration > s.horizon):
                continue
            if any(not (used >> p & 1) for p in job.after):
                continue
            if can_finish(tick + job.duration, fuel - job.energy,
                          used | (1 << i), count + 1):
                return True
        return False

    try:
        return can_finish(0, s.capacity, 0, 0)
    except Exhausted:
        return None


def search_schedule(s: Scenario, *, node_limit: int = 100_000) -> dict:
    """BFS solver; exhaustion is NOT a proof unless independent checks agree."""
    check_scenario(s)
    if type(node_limit) is not int or not 0 <= node_limit <= 1_000_000:
        return {"verdict": "UNKNOWN", "reason": "invalid_node_limit"}
    tag = fingerprint(s)
    all_done = (1 << len(s.jobs)) - 1
    queue = deque([(0, s.capacity, 0, ())])
    visited = set()
    explored = 0
    while queue:
        tick, fuel, completed, path = queue.popleft()
        signature = (tick, fuel, completed)
        if signature in visited:
            continue
        if explored >= node_limit:
            return {"verdict": "NO_KNOWN_WIN", "exhaustive": False,
                    "searched_states": explored, "rules_fingerprint": tag}
        visited.add(signature)
        explored += 1
        if len(path) >= s.action_budget:
            continue
        for i, job in enumerate(s.jobs):
            bit = 1 << i
            if (completed & bit or tick < job.release or fuel < job.energy
                    or any(not (completed & (1 << p)) for p in job.after)):
                continue
            finished = tick + job.duration
            if finished > s.horizon or finished > job.deadline:
                continue
            candidate = path + (i,)
            newdone = completed | bit
            if newdone == all_done:
                replay = verify_schedule(s, candidate)
                witness_ok = independent_witness_check(s, candidate)
                existence = independent_reachability(s)
                if (type(replay) is not dict or type(replay.get("verdict")) is not str
                        or replay["verdict"] != "VERIFIED_WIN"
                        or not witness_ok or existence is not True):
                    return {"verdict": "UNKNOWN", "reason": "independent_witness_disagreement",
                            "rules_fingerprint": tag}
                return {"verdict": "VERIFIED_WIN", "schedule": candidate,
                        "exhaustive": False, "searched_states": explored,
                        "rules_fingerprint": tag}
            queue.append((finished, fuel - job.energy, newdone, candidate))
        if tick < s.horizon:
            queue.append((tick + 1, min(s.capacity, fuel + 1),
                          completed, path + (-1,)))
    # Separate reachability algorithms must reject correlated source faults.
    dp = independent_reachability(s)
    dfs = independent_depth_enumeration(s)
    if dp is not False or dfs is not False:
        return {"verdict": "UNKNOWN", "reason": "independent_exhaustive_disagreement",
                "rules_fingerprint": tag, "dp_has_win": dp,
                "dfs_has_win": dfs}
    return {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "exhaustive": True,
            "searched_states": explored, "rules_fingerprint": tag,
            "certificate_scope": "single_machine_temporal_energy_jobs_with_declared_horizon"}


def adjudicate_unexpected(s: Scenario, proposal: object,
                          prior_verdict: str) -> dict:
    if type(prior_verdict) is not str or prior_verdict not in VERDICTS:
        return {"verdict": "UNKNOWN", "reason": "invalid_prior_verdict"}
    tag = fingerprint(s)
    try:
        direct = verify_schedule(s, proposal)
        witness_ok = independent_witness_check(s, proposal)
    except Exception:
        return {"verdict": "UNKNOWN", "reason": "verifier_exception",
                "independently_replayed": False, "rules_fingerprint": tag}
    # The two checkers must agree about THIS proposed witness, not merely
    # agree that some winning schedule exists. A lying negative cannot veto
    # a valid discovery, and a lying positive cannot create a false win.
    if (type(direct) is not dict or type(direct.get("verdict")) is not str
            or type(direct.get("rules_fingerprint")) is not str
            or direct["rules_fingerprint"] != tag
            or direct["verdict"] not in ("VERIFIED_WIN", "VERIFIED_FAILURE")
            or type(witness_ok) is not bool
            or (direct["verdict"] == "VERIFIED_WIN") != witness_ok):
        return {"verdict": "UNKNOWN", "reason": "witness_verifiers_disagree",
                "independently_replayed": False, "rules_fingerprint": tag}
    if not witness_ok:
        return {"verdict": "VERIFIED_FAILURE", "independently_replayed": True,
                "overturned_prior": False, "rules_fingerprint": tag}
    return {"verdict": "VERIFIED_WIN", "independently_replayed": True,
            "overturned_prior": prior_verdict != "VERIFIED_WIN",
            "rules_fingerprint": tag}


def novel_rule_challenge(s: Scenario, proposal: object) -> dict:
    check_scenario(s)
    if (type(proposal) is not dict
            or any(type(k) is not str for k in proposal)
            or set(proposal) != {"new_action", "reason"}):
        return {"verdict": "UNKNOWN", "reason": "invalid_novel_proposal"}
    action, reason = proposal["new_action"], proposal["reason"]
    if (type(action) is not str or not 1 <= len(action) <= 64
            or type(reason) is not str or not 1 <= len(reason) <= 128):
        return {"verdict": "UNKNOWN", "reason": "invalid_novel_proposal"}
    # A blank/format-control 'action' is not a discoverable rule. Compatibility
    # spellings of RUN/WAIT must also not be laundered into novelty, even when
    # their original Unicode bytes differ from the declared ASCII opcodes.
    canonical_action = unicodedata.normalize("NFKC", action).strip()
    canonical_reason = unicodedata.normalize("NFKC", reason).strip()
    if (not canonical_action or not canonical_reason
            or any(unicodedata.category(ch).startswith("C")
                   for ch in canonical_action + canonical_reason)):
        return {"verdict": "UNKNOWN", "reason": "invalid_novel_proposal"}
    if canonical_action.upper() in {"RUN", "WAIT"}:
        return {"verdict": "UNKNOWN", "reason": "action_already_declared"}
    return {"verdict": "NOVEL_RULESET_CHALLENGE",
            "independent_verification": "NOT_PERFORMED",
            "proposed_new_action": action, "claimed_mechanism": reason,
            "rules_fingerprint": fingerprint(s)}


def families() -> dict[str, Scenario]:
    return {
        "recharge_window": Scenario((
            Job("alpha", 0, 2, 1, 2, ()),
            Job("beta", 2, 7, 2, 2, (0,)),
            Job("gamma", 0, 10, 1, 1, (1,))), 2, 10, 8),
        "release_pivot": Scenario((
            Job("alpha", 1, 4, 2, 1, ()),
            Job("beta", 0, 6, 1, 1, (0,)),
            Job("gamma", 3, 8, 1, 1, (1,))), 2, 9, 8),
        "dependency_cycle": Scenario((
            Job("alpha", 0, 10, 1, 1, (2,)),
            Job("beta", 0, 10, 1, 1, (0,)),
            Job("gamma", 0, 10, 1, 1, (1,))), 2, 10, 8),
        "deadline_collision": Scenario((
            Job("alpha", 0, 2, 2, 1, ()),
            Job("beta", 0, 2, 2, 1, ()),
            Job("gamma", 0, 2, 2, 1, ())), 3, 8, 8),
    }


def catalogue_control(unseen: Scenario) -> dict:
    check_scenario(unseen)
    table = {fingerprint(s): search_schedule(s)["verdict"]
             for s in families().values()}
    matched = sum(table[fingerprint(s)] == search_schedule(s)["verdict"]
                  for s in families().values())
    return {"public_fixtures": len(table), "exact_matches": matched,
            "unseen_verdict": table.get(fingerprint(unseen), "UNKNOWN"),
            "optimizer_updates": 0}


def fixed_script_control(script: tuple[int, ...] = (-1, 0, 1, 2)) -> dict:
    if (type(script) is not tuple or len(script) > 18
            or any(type(code) is not int for code in script)):
        raise ValueError("script must be exact bounded integer actions")
    cases = families()
    return {"public_cases": len(cases),
            "fixed_script_wins": sum(verify_schedule(s, script)["verdict"]
                                     == "VERIFIED_WIN" for s in cases.values()),
            "symbolic_wins": sum(search_schedule(s)["verdict"] == "VERIFIED_WIN"
                                 for s in cases.values()),
            "optimizer_updates": 0}
