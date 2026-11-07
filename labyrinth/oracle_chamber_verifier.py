"""Independent replay and universal minimax policy check for Oracle Chamber.

Different code path from oracle_chamber_exam._true_answer; same public source
tree and author, hence NOT independent external custody.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

from .oracle_chamber_exam import Case, MAX_QUERIES, PREDICATES, SECRETS


_TABLE = {
    "alpha": (0, 1, 0, 1),
    "beta": (0, 0, 1, 1),
    "parity": (0, 1, 1, 0),
}


@dataclass(frozen=True)
class PolicyReview:
    verdict: str
    cases_checked: int
    reason: str
    counterexample: dict[str, object] | None = None


def replay_answers(case: Case, attempted: object) -> tuple[int | None, ...] | None:
    """Independently reconstruct every answer, including charged invalids."""
    if type(attempted) not in (list, tuple) or len(attempted) > MAX_QUERIES:
        return None
    answers: list[int | None] = []
    for i, question in enumerate(attempted):
        if type(question) is not str or question not in _TABLE:
            answers.append(None)
        else:
            answers.append(_TABLE[question][case.secret] ^ int(i in case.lie_slots))
    return tuple(answers)


def _lie_patterns(max_lies: int) -> list[tuple[int, ...]]:
    return [pattern for count in range(max_lies + 1)
            for pattern in combinations(range(MAX_QUERIES), count)]


def _question(policy: dict[str, object], prefix: str, depth: int) -> str | None:
    schedule = policy.get("questions")
    if type(schedule) in (list, tuple) and len(schedule) == MAX_QUERIES:
        proposal = schedule[depth]
    else:
        tree = policy.get("query_tree")
        if type(tree) is not dict:
            return None
        proposal = tree.get(prefix)
    return proposal if type(proposal) is str and proposal in _TABLE else None


def _decode(policy: dict[str, object], bits: str) -> int | None:
    decoder = policy.get("decoder")
    if type(decoder) is dict:
        value = decoder.get(bits)
        return value if type(value) is int and value in SECRETS else None
    if type(decoder) is not str or decoder != "nearest_codeword":
        return None
    schedule = policy.get("questions")
    if type(schedule) not in (list, tuple) or len(schedule) != MAX_QUERIES:
        return None
    if any(type(q) is not str or q not in _TABLE for q in schedule):
        return None
    distances = []
    for secret in SECRETS:
        expected = [_TABLE[q][secret] for q in schedule]
        distances.append(sum(int(answer) != expected[i] for i, answer in enumerate(bits)))
    minimum = min(distances)
    winners = [secret for secret, distance in enumerate(distances) if distance == minimum]
    return winners[0] if len(winners) == 1 else None


def _declarative_policy(policy: object) -> bool:
    """Reject executable Python objects at the public strategy boundary.

    The examiner is intentionally declarative: arbitrary dictionary keys may
    execute custom equality/hash code and spy on verifier-local truth.
    """
    if type(policy) is not dict:
        return False
    if any(type(key) is not str for key in policy):
        return False
    keys = set(policy)
    if keys not in ({"questions", "decoder"}, {"query_tree", "decoder"}):
        return False
    if "questions" in policy:
        questions = policy["questions"]
        if (type(questions) not in (list, tuple) or len(questions) != MAX_QUERIES
                or any(type(q) is not str or q not in _TABLE for q in questions)):
            return False
    else:
        tree = policy["query_tree"]
        if type(tree) is not dict:
            return False
        for prefix, question in tree.items():
            if (type(prefix) is not str or len(prefix) >= MAX_QUERIES
                    or any(bit not in "01" for bit in prefix)
                    or type(question) is not str or question not in _TABLE):
                return False
    decoder = policy["decoder"]
    if type(decoder) is str:
        return decoder == "nearest_codeword" and "questions" in policy
    if type(decoder) is not dict:
        return False
    for transcript, secret in decoder.items():
        if (type(transcript) is not str or len(transcript) != MAX_QUERIES
                or any(bit not in "01" for bit in transcript)
                or (secret is not None and
                    (type(secret) is not int or secret not in SECRETS))):
            return False
    return True


def verify_policy(lie_budget: int, policy: object) -> PolicyReview:
    """Attack a declared strategy against EVERY secret and lie-placement choice.

    Nonadaptive schedule/decoder and adaptive binary-prefix policy trees are
    supported. This is a finite public toy policy checker, not a private exam.
    """
    if type(lie_budget) is not int or lie_budget not in (1, 2):
        return PolicyReview("UNKNOWN", 0, "unsupported lie-budget ruleset")
    if not _declarative_policy(policy):
        return PolicyReview("UNKNOWN", 0, "policy must contain exact declarative primitives")
    checked = 0
    for secret in SECRETS:
        for lies in _lie_patterns(lie_budget):
            transcript = ""
            questions: list[str] = []
            for depth in range(MAX_QUERIES):
                question = _question(policy, transcript, depth)
                if question is None:
                    return PolicyReview("UNKNOWN", checked,
                                        "query schedule/tree incomplete or invalid")
                questions.append(question)
                answer = _TABLE[question][secret] ^ int(depth in lies)
                transcript += str(answer)
            inferred = _decode(policy, transcript)
            checked += 1
            if inferred != secret:
                return PolicyReview("VERIFIED_FAILURE", checked,
                                    "strategy fails a permitted adversary",
                                    {"secret": secret, "lie_slots": list(lies),
                                     "questions": questions,
                                     "answers": transcript, "predicted": inferred})
    return PolicyReview("VERIFIED_WIN", checked,
                        "all secrets and lie patterns passed independent replay")


def verify_unsat_bound(case: Case, claimed: object) -> dict[str, object]:
    """Hamming-volume impossibility for *guaranteed* five-query identification.

    For fixed truth, each set of up to L lying positions yields a distinct
    transcript for any deterministic adaptive questioning strategy. The
    4*sum(C(5,i),i≤L) transcripts must occupy disjoint binary response paths
    if all secrets must be identified without error. When >2**5, impossible.
    """
    if type(claimed) is not dict:
        return {"verdict": "UNKNOWN", "certificate_verified": False,
                "reason": "missing declaration"}
    expected = {"secrets": 4, "binary_answers": 2,
                "max_queries": MAX_QUERIES, "lie_budget": case.lie_budget}
    supplied = claimed.get("declared_rules")
    if (type(supplied) is not dict or set(supplied) != set(expected)
            or any(type(supplied[key]) is not int or supplied[key] != expected[key]
                   for key in expected)):
        return {"verdict": "UNKNOWN", "certificate_verified": False,
                "reason": "declared rules do not match evaluated world"}
    volume = sum(comb(MAX_QUERIES, amount)
                 for amount in range(case.lie_budget + 1))
    paths = 2**MAX_QUERIES
    if 4 * volume <= paths:
        return {"verdict": "UNKNOWN", "certificate_verified": False,
                "reason": "counting bound is insufficient under declared rules",
                "required_paths": 4 * volume, "available_paths": paths}
    return {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES",
            "certificate_verified": True,
            "reason": "disjoint-response counting bound violated",
            "required_paths": 4 * volume, "available_paths": paths}


def adjudicate_claim(case: Case, claim: object, *, attempts: tuple[object, ...],
                     observations: tuple[int | None, ...]) -> dict[str, object]:
    if replay_answers(case, attempts) != observations:
        return {"verdict": "EVALUATOR_DISAGREEMENT", "verified_success": False,
                "instance_win_verified": False, "policy_verified": False,
                "certificate_verified": False}
    if type(claim) is not dict or type(claim.get("kind")) is not str:
        return {"verdict": "UNKNOWN", "verified_success": False,
                "instance_win_verified": False, "policy_verified": False,
                "certificate_verified": False}
    kind = claim["kind"]
    if kind == "COMMIT":
        selected = claim.get("secret")
        correct = type(selected) is int and selected == case.secret
        return {"verdict": "VERIFIED_WIN" if correct else "VERIFIED_FAILURE",
                "verified_success": False, "instance_win_verified": correct,
                "policy_verified": False, "certificate_verified": False,
                "reason": "single-case guess does not verify a universal strategy"}
    if kind == "POLICY":
        review = verify_policy(case.lie_budget, claim.get("policy"))
        success = review.verdict == "VERIFIED_WIN"
        # Task correctness is not evidence of model learning. A public
        # deterministic parser passes this family without parameter updates;
        # no generic graduation-success flag may be set by this prototype.
        return {"verdict": review.verdict, "verified_success": False,
                "instance_win_verified": False, "policy_verified": success,
                "certificate_verified": False,
                "cases_checked": review.cases_checked,
                "counterexample": review.counterexample, "reason": review.reason}
    if kind == "UNSAT_CERTIFICATE":
        review = verify_unsat_bound(case, claim)
        return {**review, "verified_success": False,
                "instance_win_verified": False, "policy_verified": False}
    if kind == "ABSTAIN":
        assessment = claim.get("assessment")
        if type(assessment) is not str or assessment not in ("UNKNOWN", "NO_KNOWN_WIN"):
            assessment = "UNKNOWN"
        return {"verdict": assessment, "verified_success": False,
                "instance_win_verified": False, "policy_verified": False,
                "certificate_verified": False,
                "reason": "calibrated abstention, no success credit"}
    if kind == "NOVEL_RULE":
        proposed = claim.get("rule_change")
        if type(proposed) is not str or not proposed.strip():
            return {"verdict": "UNKNOWN", "verified_success": False,
                    "instance_win_verified": False, "policy_verified": False,
                    "certificate_verified": False,
                    "reason": "nonempty rule proposal required"}
        return {"verdict": "NOVEL_RULESET_CHALLENGE",
                "verified_success": False, "instance_win_verified": False,
                "policy_verified": False, "certificate_verified": False,
                "proposal": proposed,
                "reason": "held for external rule review, not silently dismissed"}
    return {"verdict": "UNKNOWN", "verified_success": False,
            "instance_win_verified": False, "policy_verified": False,
            "certificate_verified": False, "reason": "unsupported claim"}


def hypothetical_attested_pair_query(case: Case) -> dict[str, object]:
    """Counterfactual: one NEW attested two-bit query reveals the whole secret.

    Such a query is absent from the declared binary, corruptible interface.
    This establishes only a conditional possibility under changed rules.
    """
    return {"verdict": "NOVEL_RULESET_CHALLENGE", "conditional_full_identification": True,
            "hypothetical_answer": case.secret,
            "reason": "new noiseless four-outcome oracle not established as an affordance"}
