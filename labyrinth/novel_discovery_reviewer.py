"""Bounded independent counterfactual reviewer for out-of-catalog hypotheses.

Never elevate a matching synthetic rule to a qualified fact. Review receipts
stay PENDING_INDEPENDENT_REVIEW until real external evidence is supplied.
"""

from __future__ import annotations

from hashlib import sha256
from itertools import product
import json
import math

from .novel_discovery_world import (CHANGE_AFTER, INPUTS, MAX_ATTEMPTS,
                                    RULESET, State, World, make_world)


FIELDS = ("x", "y", "prev_x", "prev2_x", "prev_y")
KNOWN_CATALOG = ("copy", "xor", "xnor", "and", "or", "edge", "lag")


def _advance(state: State, pair: object, *, control: bool) -> tuple[State, int | None]:
    """Independent explicit truth-set implementation, not world.transition."""
    if (type(pair) is not tuple or len(pair) != 2 or
            type(pair[0]) is not int or type(pair[1]) is not int or pair not in INPUTS):
        return state, None
    x, y = pair
    if state.interventions < CHANGE_AFTER:
        output = x
    elif control:
        output = int(y == 1 and x != state.prev_x)
    else:
        output = int((state.prev2_x, state.prev_x, x, y) == (1, 0, 1, 1))
    return State(state.interventions + 1, state.prev_x, x, y), output


def replay_world(world: World, attempts: object) -> tuple[State, tuple[int | None, ...]] | None:
    if type(attempts) not in (list, tuple):
        return None
    trace = tuple(attempts)
    if len(trace) > MAX_ATTEMPTS:
        return None
    state = State()
    outputs: list[int | None] = []
    for pair in trace:
        state, effect = _advance(state, pair, control=world.control)
        outputs.append(effect)
    return state, tuple(outputs)


def _exact_observations(expected: tuple[int | None, ...], actual: object) -> bool:
    if type(actual) not in (list, tuple) or len(expected) != len(actual):
        return False
    for truth, observed in zip(expected, actual):
        if truth is None:
            if observed is not None:
                return False
        elif type(truth) is not int or type(observed) is not int or truth != observed:
            return False
    return True


def exact_world_match(expected: object, current: object, effects: object) -> bool:
    """State-and-output parity without custom Python equality callbacks."""
    if expected is None or type(current) is not State:
        return False
    if (type(current.interventions) is not int or
            not 0 <= current.interventions <= MAX_ATTEMPTS or
            type(current.prev2_x) is not int or current.prev2_x not in (0, 1) or
            type(current.prev_x) is not int or current.prev_x not in (0, 1) or
            type(current.prev_y) is not int or current.prev_y not in (0, 1)):
        return False
    truth_state, truth_effects = expected
    return (truth_state.interventions == current.interventions and
            truth_state.prev2_x == current.prev2_x and
            truth_state.prev_x == current.prev_x and
            truth_state.prev_y == current.prev_y and
            _exact_observations(truth_effects, effects))


def normalize_proposal(candidate: object) -> tuple[dict[str, object] | None, str]:
    """A small finite DSL: AND of uniquely named typed Boolean literals.

    Natural-language proposals remain preservable as pending, untested
    hypotheses; unsupported executable Python is never run by the referee.
    """
    if type(candidate) is str and 0 < len(candidate.strip()) <= 2000:
        return {"form": "narrative", "description": candidate.strip()}, "UNTYPED_PENDING"
    if type(candidate) is not dict or set(candidate) != {"form", "literals"}:
        return None, "INVALID_SCHEMA"
    if type(candidate.get("form")) is not str or candidate["form"] != "and_literals":
        return None, "UNSUPPORTED_FORM"
    literals = candidate.get("literals")
    if type(literals) not in (list, tuple) or not 1 <= len(literals) <= len(FIELDS):
        return None, "INVALID_LITERAL_COUNT"
    normalized = []
    used = set()
    for literal in literals:
        if (type(literal) is not dict or set(literal) != {"field", "equals"}
                or type(literal.get("field")) is not str
                or literal["field"] not in FIELDS
                or type(literal.get("equals")) is not int
                or literal["equals"] not in (0, 1)
                or literal["field"] in used):
            return None, "INVALID_LITERAL"
        used.add(literal["field"])
        normalized.append({"field": literal["field"], "equals": literal["equals"]})
    return {"form": "and_literals",
            "literals": sorted(normalized, key=lambda item: item["field"])}, "EXECUTABLE"


def _candidate_effect(normalized: dict[str, object], context: dict[str, int]) -> int:
    return int(all(context[item["field"]] == item["equals"]
                   for item in normalized["literals"]))


def _known_effect(label: str, context: dict[str, int]) -> int:
    x, y, px, py = (context["x"], context["y"],
                    context["prev_x"], context["prev_y"])
    return {
        "copy": x, "xor": (x + y) % 2, "xnor": int(x == y),
        "and": x & y, "or": x | y, "edge": y & (x ^ px),
        "lag": x & py,
    }[label]


def exhaustive_rule_review(world: World, normalized: dict[str, object]) -> dict[str, object]:
    """Falsify candidate's truth function over all 32 binary causal contexts."""
    if normalized.get("form") != "and_literals":
        return {"fit_status": "UNKNOWN", "counterfactual_match": False,
                "contexts_checked": 0, "counterexample": None,
                "catalog_equivalent": None}
    contexts = [dict(zip(FIELDS, bits)) for bits in product((0, 1), repeat=5)]
    candidate_bits = [_candidate_effect(normalized, c) for c in contexts]
    equivalent = [label for label in KNOWN_CATALOG
                  if all(_known_effect(label, c) == predicted
                         for c, predicted in zip(contexts, candidate_bits))]
    checked = 0
    for context, proposed in zip(contexts, candidate_bits):
        checked += 1
        observed = (int(context["y"] == 1 and
                        context["x"] != context["prev_x"]) if world.control
                    else int((context["prev2_x"], context["prev_x"],
                              context["x"], context["y"]) == (1, 0, 1, 1)))
        if proposed != observed:
            return {"fit_status": "FALSIFIED", "counterfactual_match": False,
                    "contexts_checked": checked,
                    "counterexample": {"causal_context": context,
                                       "proposed": proposed, "observed": observed},
                    "catalog_equivalent": equivalent}
    return {"fit_status": "COUNTERFACTUAL_MATCH", "counterfactual_match": True,
            "contexts_checked": checked, "counterexample": None,
            "catalog_equivalent": equivalent}


def _classify_review(normalized: dict[str, object] | None,
                     result: dict[str, object]) -> tuple[str, str]:
    """Classify evidence before novelty; failed guesses are never discoveries."""
    if normalized is None:
        return "UNKNOWN", "INVALID_OR_UNSUPPORTED"
    if result["fit_status"] == "FALSIFIED":
        return "VERIFIED_FAILURE", "REFUTED_BY_COUNTEREXAMPLE"
    if result["fit_status"] != "COUNTERFACTUAL_MATCH":
        return "UNKNOWN", "UNTYPED_PENDING"
    if result["catalog_equivalent"]:
        return "UNKNOWN", "KNOWN_CATALOG_RULE"
    return "NOVEL_RULESET_CHALLENGE", "OUTSIDE_CATALOG"


def _canonical(data: dict[str, object]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _encode_scalar(value: object) -> dict[str, object] | None:
    if value is None:
        return {"type": "none", "value": None}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int and -(2**63) <= value < 2**63:
        return {"type": "int", "value": value}
    if type(value) is str and len(value) <= 256:
        return {"type": "str", "value": value}
    if type(value) is float and math.isfinite(value):
        return {"type": "float", "value": value}
    return None


def encode_attempts(attempts: object) -> list[dict[str, object]] | None:
    """Preserve each charged input with exact JSON-safe type and value."""
    if type(attempts) not in (list, tuple) or len(attempts) > MAX_ATTEMPTS:
        return None
    encoded = []
    for pair in attempts:
        if type(pair) is not tuple or len(pair) != 2:
            return None
        x, y = _encode_scalar(pair[0]), _encode_scalar(pair[1])
        if x is None or y is None:
            return None
        encoded.append({"x": x, "y": y})
    return encoded


def _decode_scalar(item: object) -> object:
    if type(item) is not dict or set(item) != {"type", "value"}:
        raise ValueError("malformed scalar record")
    label, value = item["type"], item["value"]
    if label == "none" and value is None:
        return None
    if label == "bool" and type(value) is bool:
        return value
    if label == "int" and type(value) is int and -(2**63) <= value < 2**63:
        return value
    if label == "str" and type(value) is str and len(value) <= 256:
        return value
    if label == "float" and type(value) is float and math.isfinite(value):
        return value
    raise ValueError("unsupported scalar record")


def decode_attempts(encoded: object) -> tuple[tuple[object, object], ...]:
    if type(encoded) is not list or len(encoded) > MAX_ATTEMPTS:
        raise ValueError("invalid encoded attempted action list")
    pairs = []
    for item in encoded:
        if type(item) is not dict or set(item) != {"x", "y"}:
            raise ValueError("invalid pair shape")
        pairs.append((_decode_scalar(item["x"]), _decode_scalar(item["y"])))
    return tuple(pairs)


def review_proposal(world: World, attempts: object, effects: object,
                    proposal: object, *, author: str = "candidate") -> dict[str, object]:
    """Preserve and attack a claimed novel causal rule; NEVER auto-admit it."""
    if type(author) is not str or not 1 <= len(author.strip()) <= 80:
        author = "UNVERIFIED_AUTHOR"
    snapshot = replay_world(world, attempts)
    if snapshot is None or not _exact_observations(snapshot[1], effects):
        return {"verdict": "EVALUATOR_DISAGREEMENT", "verified_success": False,
                "review_status": "BLOCKED_INTEGRITY"}
    encoded_attempts = encode_attempts(attempts)
    if encoded_attempts is None:
        return {"verdict": "UNKNOWN", "verified_success": False,
                "review_status": "BLOCKED_UNSERIALIZABLE_TRACE",
                "reason": "exact charged attempt values cannot be preserved"}
    normalized, proposal_status = normalize_proposal(proposal)
    result = (exhaustive_rule_review(world, normalized) if normalized is not None
              else {"fit_status": "UNKNOWN", "counterfactual_match": False,
                    "contexts_checked": 0, "counterexample": None,
                    "catalog_equivalent": None})
    verdict, result["novelty_status"] = _classify_review(normalized, result)
    payload: dict[str, object] = {
        "version": 1,
        "verdict": verdict,
        "verified_success": False,
        "review_status": "PENDING_INDEPENDENT_REVIEW",
        "evidence_class": "SYNTHETIC_TEST",
        "independent_custody": False,
        "subject": {"world_id": world.world_id, "ruleset": RULESET,
                    "seed": world.seed, "variant": world.variant,
                    "control": world.control, "source_commit": "UNBOUND"},
        "candidate": {"author": author.strip(), "proposal": normalized,
                      "proposal_status": proposal_status},
        "observed_attempt_count": len(attempts),
        "trace": {"attempts": encoded_attempts, "effects": list(effects)},
        "review": result,
    }
    payload["observed_trace_sha256"] = sha256(_canonical(payload["trace"]).encode()).hexdigest()
    payload["receipt_sha256"] = sha256(_canonical(payload).encode()).hexdigest()
    return payload


def verify_receipt_digest(receipt: object) -> bool:
    if type(receipt) is not dict or type(receipt.get("receipt_sha256")) is not str:
        return False
    supplied = receipt["receipt_sha256"]
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    try:
        return sha256(_canonical(unsigned).encode()).hexdigest() == supplied
    except (TypeError, ValueError):
        return False


def verify_receipt_replay(receipt: object) -> bool:
    """Reconstruct world, attempted inputs and review outcome from JSON alone.

    This is source-level integrity, NOT an authenticated external signature.
    A party controlling both source and receipt can still forge a story.
    """
    if not verify_receipt_digest(receipt):
        return False
    try:
        subject = receipt["subject"]
        world = make_world(subject["seed"], variant=subject["variant"],
                           control=subject["control"])
        if world.world_id != subject["world_id"] or subject["ruleset"] != RULESET:
            return False
        if (subject["source_commit"] != "UNBOUND" or
                receipt.get("verified_success") is not False or
                receipt.get("independent_custody") is not False or
                receipt.get("review_status") != "PENDING_INDEPENDENT_REVIEW" or
                receipt.get("evidence_class") != "SYNTHETIC_TEST"):
            return False
        trace = receipt["trace"]
        attempts = decode_attempts(trace["attempts"])
        outputs = trace["effects"]
        actual = replay_world(world, attempts)
        if actual is None or not _exact_observations(actual[1], outputs):
            return False
        if receipt["observed_attempt_count"] != len(attempts):
            return False
        if receipt["observed_trace_sha256"] != sha256(_canonical(trace).encode()).hexdigest():
            return False
        proposal = receipt["candidate"]["proposal"]
        proposal_status = receipt["candidate"]["proposal_status"]
        if proposal is not None:
            if type(proposal) is not dict:
                return False
            if proposal.get("form") == "narrative":
                if (proposal_status != "UNTYPED_PENDING" or
                        set(proposal) != {"form", "description"} or
                        type(proposal.get("description")) is not str or
                        not 0 < len(proposal["description"]) <= 2000):
                    return False
            else:
                re_normalized, re_status = normalize_proposal(proposal)
                if (re_status != "EXECUTABLE" or
                        proposal_status != re_status or re_normalized != proposal):
                    return False
        elif proposal_status not in ("INVALID_SCHEMA", "UNSUPPORTED_FORM",
                                     "INVALID_LITERAL_COUNT", "INVALID_LITERAL"):
            return False
        expected = (exhaustive_rule_review(world, proposal)
                    if proposal is not None else {
                        "fit_status": "UNKNOWN", "counterfactual_match": False,
                        "contexts_checked": 0, "counterexample": None,
                        "catalog_equivalent": None})
        expected_verdict, expected["novelty_status"] = _classify_review(proposal, expected)
        return expected == receipt["review"] and receipt["verdict"] == expected_verdict
    except (KeyError, TypeError, ValueError, IndexError):
        return False
