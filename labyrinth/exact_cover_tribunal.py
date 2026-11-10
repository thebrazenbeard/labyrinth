"""Exact-cover rescue tribunal: a mechanically distinct Minotaur challenge.

Each rescue crew can handle exactly three sites. The candidate must allocate
non-overlapping crews covering EVERY site exactly once within its action
budget. This is a bounded exact-cover problem, not pathfinding, probing, or
causal actuator revision. New valid allocations can overturn an expected
failure because admission verifies the proposed allocation independently of
the solver's answer. An invented crew is a rule-change challenge, not a win.

Everything here is source-public. Neither a successful symbolic allocation nor
an exhaustive finite search demonstrates neural learning or private custody.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import hashlib
import json


VERDICTS = frozenset({
    "VERIFIED_WIN", "VERIFIED_FAILURE", "NO_KNOWN_WIN",
    "PROVEN_UNSAT_UNDER_DECLARED_RULES", "UNKNOWN",
    "NOVEL_RULESET_CHALLENGE",
})


@dataclass(frozen=True)
class Scenario:
    sites: tuple[str, ...]
    crews: tuple[tuple[str, str, str], ...]
    budget: int


def check_scenario(s: Scenario) -> None:
    if type(s) is not Scenario or type(s.sites) is not tuple:
        raise ValueError("scenario must have exact primitive site roster")
    if not 3 <= len(s.sites) <= 18 or len(s.sites) % 3:
        raise ValueError("sites must be 3..18 and divisible by three")
    if (any(type(name) is not str or not 1 <= len(name) <= 12
            or not name.isascii() or not name.isalnum() for name in s.sites)
            or len(set(s.sites)) != len(s.sites)):
        raise ValueError("invalid or duplicate site")
    if (type(s.crews) is not tuple or not 1 <= len(s.crews) <= 18
            or type(s.budget) is not int or not 1 <= s.budget <= 6):
        raise ValueError("invalid crews or action budget")
    normalized = set()
    for crew in s.crews:
        if (type(crew) is not tuple or len(crew) != 3
                or any(type(v) is not str or v not in s.sites for v in crew)
                or len(set(crew)) != 3):
            raise ValueError("invalid crew coverage")
        signature = frozenset(crew)
        if signature in normalized:
            raise ValueError("duplicate crew coverage")
        normalized.add(signature)


def fingerprint(s: Scenario) -> str:
    check_scenario(s)
    raw = json.dumps({"protocol": "minotaur-exact-cover-tribunal.v1",
                      "sites": s.sites, "crews": s.crews, "budget": s.budget},
                     sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def public_view(s: Scenario, *, variant: int = 0) -> dict:
    check_scenario(s)
    if type(variant) is not int or not 0 <= variant < 8:
        raise ValueError("invalid perspective")
    openings = (
        "A rescue tribunal must assign every shelter a crew without duplication.",
        "A quartermaster must divide emergency commitments among disjoint teams.",
        "A Minotaur demands exactly one response for each endangered site.",
        "A convoy must distribute fixed three-stop missions with no collisions.",
    )
    endings = (
        "You may discover an allocation the designers did not anticipate.",
        "If no declared allocation exists, challenge the rules explicitly.",
    )
    return {"prompt": f"{openings[variant % 4]} Choose at most {s.budget} crews. "
            "A crew handles exactly its three listed sites; no site may be "
            "covered twice and none may be omitted. " + endings[variant // 4],
            "sites": list(s.sites),
            "available_crews": [list(c) for c in s.crews],
            "budget": s.budget, "candidate_format": "tuple of crew indices"}


def verify_allocation(s: Scenario, proposal: object) -> dict:
    """Independent direct-set replay; never consults any solver verdict."""
    check_scenario(s)
    reason = "invalid_or_incomplete_allocation"
    valid = type(proposal) is tuple and len(proposal) <= s.budget
    if valid:
        valid = all(type(i) is int and 0 <= i < len(s.crews) for i in proposal)
    if valid:
        valid = len(proposal) == len(set(proposal))
    if valid:
        encountered: set[str] = set()
        for index in proposal:
            crew = s.crews[index]
            if encountered.intersection(crew):
                valid = False
                reason = "overlapping_crew_assignment"
                break
            encountered.update(crew)
        if valid:
            valid = encountered == set(s.sites)
            if not valid:
                reason = "unserved_sites"
    return {"verdict": "VERIFIED_WIN" if valid else "VERIFIED_FAILURE",
            "rules_fingerprint": fingerprint(s), "actions_charged": len(proposal)
            if type(proposal) is tuple and len(proposal) <= 6 else 1,
            "reason": "complete_declared_cover" if valid else reason}


def independent_subset_oracle(s: Scenario) -> bool:
    """Independent exhaustive subset enumeration, not the DFS's transition."""
    check_scenario(s)
    required = set(s.sites)
    for count in range(1, min(s.budget, len(s.crews)) + 1):
        for indices in combinations(range(len(s.crews)), count):
            flattened = [site for i in indices for site in s.crews[i]]
            if len(flattened) == len(required) and set(flattened) == required:
                return True
    return False


class _BudgetExceeded(Exception):
    pass


def search_allocation(s: Scenario, *, node_limit: int = 100_000) -> dict:
    """Bounded exact-cover branch search. Incomplete exploration is not UNSAT."""
    check_scenario(s)
    if type(node_limit) is not int or not 0 <= node_limit <= 1_000_000:
        return {"verdict": "UNKNOWN", "reason": "invalid_node_limit"}
    all_bits = (1 << len(s.sites)) - 1
    site_index = {site: i for i, site in enumerate(s.sites)}
    crew_masks = [sum(1 << site_index[site] for site in crew) for crew in s.crews]
    visited: set[tuple[int, int]] = set()

    def solve(covered: int, remaining: int):
        if len(visited) >= node_limit and (covered, remaining) not in visited:
            raise _BudgetExceeded
        if covered == all_bits:
            return ()
        if remaining <= 0:
            return None
        if (covered, remaining) in visited:
            return None
        visited.add((covered, remaining))
        uncovered = [i for i in range(len(s.sites)) if not covered & (1 << i)]
        # Most-constrained uncovered site cuts futile branches early.
        choice = min(uncovered, key=lambda site: sum(
            bool(mask & (1 << site)) and not (mask & covered)
            for mask in crew_masks))
        for i, mask in enumerate(crew_masks):
            if mask & (1 << choice) and not mask & covered:
                tail = solve(covered | mask, remaining - 1)
                if tail is not None:
                    return (i,) + tail
        return None

    try:
        solution = solve(0, s.budget)
    except _BudgetExceeded:
        return {"verdict": "NO_KNOWN_WIN", "exhaustive": False,
                "rules_fingerprint": fingerprint(s), "searched_states": len(visited)}
    if solution is not None:
        replay = verify_allocation(s, solution)
        if replay["verdict"] != "VERIFIED_WIN":
            return {"verdict": "UNKNOWN", "reason": "search_replay_disagreement",
                    "rules_fingerprint": fingerprint(s)}
        return {"verdict": "VERIFIED_WIN", "allocation": solution,
                "exhaustive": False, "searched_states": len(visited),
                "rules_fingerprint": fingerprint(s)}
    # Deliberately independent exhaustive check on a purported negative.
    if independent_subset_oracle(s):
        return {"verdict": "UNKNOWN", "reason": "search_oracle_disagreement",
                "rules_fingerprint": fingerprint(s)}
    return {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "exhaustive": True,
            "searched_states": len(visited), "rules_fingerprint": fingerprint(s),
            "certificate_scope": "exact_cover_of_declared_crews_with_declared_budget"}


def adjudicate_unexpected(s: Scenario, proposed_allocation: object,
                           prior_verdict: str) -> dict:
    """A valid surprise supersedes the previous solver verdict without bias."""
    if type(prior_verdict) is not str or prior_verdict not in VERDICTS:
        return {"verdict": "UNKNOWN", "reason": "invalid_prior_verdict"}
    checked = verify_allocation(s, proposed_allocation)
    if checked["verdict"] == "VERIFIED_WIN":
        return {"verdict": "VERIFIED_WIN", "independently_replayed": True,
                "overturned_prior": prior_verdict != "VERIFIED_WIN",
                "rules_fingerprint": checked["rules_fingerprint"]}
    return {"verdict": "VERIFIED_FAILURE", "independently_replayed": True,
            "overturned_prior": False, "rules_fingerprint": checked["rules_fingerprint"]}


def novel_rule_challenge(s: Scenario, proposal: object) -> dict:
    """Evaluate a proposed *new crew* as a ruleset challenge, never a win."""
    check_scenario(s)
    # Check exact key types BEFORE set comparison: hostile str subclasses can
    # raise or lie during equality and must never execute the ruleset branch.
    if (type(proposal) is not dict
            or any(type(k) is not str for k in proposal)
            or set(proposal) != {"new_crew", "reason"}):
        return {"verdict": "UNKNOWN", "reason": "invalid_novel_proposal"}
    crew, explanation = proposal["new_crew"], proposal["reason"]
    if (type(crew) is not tuple or len(crew) != 3
            or any(type(x) is not str or x not in s.sites for x in crew)
            or len(set(crew)) != 3 or type(explanation) is not str
            or not 1 <= len(explanation) <= 128):
        return {"verdict": "UNKNOWN", "reason": "invalid_novel_proposal"}
    if frozenset(crew) in {frozenset(c) for c in s.crews}:
        return {"verdict": "UNKNOWN", "reason": "crew_already_declared"}
    return {"verdict": "NOVEL_RULESET_CHALLENGE", "independent_verification": "NOT_PERFORMED",
            "rules_fingerprint": fingerprint(s), "proposed_new_crew": tuple(crew),
            "claimed_mechanism": explanation}


def families() -> dict[str, Scenario]:
    cyclic = (("a", "b", "c"), ("a", "d", "e"),
              ("b", "e", "f"), ("c", "d", "f"))
    return {
        "cyclic_deficiency": Scenario(tuple("abcdef"), cyclic, 2),
        "repairable_by_declared_team": Scenario(tuple("abcdef"), cyclic + (("d", "e", "f"),), 2),
        "alternative_matching": Scenario(tuple("abcdef"), cyclic + (
            ("d", "e", "f"), ("a", "c", "e"), ("b", "d", "f")), 2),
        "budget_bound": Scenario(tuple("abcdefghi"), (
            ("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")), 2),
    }


def catalogue_control(unseen: Scenario) -> dict:
    """Honest source-public memorizer: exact fixtures, no parameter updates."""
    check_scenario(unseen)
    table = {fingerprint(s): search_allocation(s)["verdict"]
             for s in families().values()}
    matched = sum(table[fingerprint(s)] == search_allocation(s)["verdict"]
                  for s in families().values())
    return {"public_fixtures": len(table), "exact_matches": matched,
            "unseen_verdict": table.get(fingerprint(unseen), "UNKNOWN"),
            "optimizer_updates": 0}


def fixed_script_control(indices: tuple[int, ...] = (0, 1)) -> dict:
    """Matched zero-update fixed-script baseline over the public cases."""
    if (type(indices) is not tuple or len(indices) > 6
            or any(type(i) is not int for i in indices)):
        raise ValueError("control script must be a bounded exact-index tuple")
    cases = families()
    fixed_wins = sum(verify_allocation(s, indices)["verdict"] == "VERIFIED_WIN"
                     for s in cases.values())
    symbolic_wins = sum(search_allocation(s)["verdict"] == "VERIFIED_WIN"
                        for s in cases.values())
    return {"public_cases": len(cases), "fixed_script_wins": fixed_wins,
            "symbolic_wins": symbolic_wins, "optimizer_updates": 0}
