"""Independent finite replay and exhaustive truth-table verifier for Cipher Wheel.

This is an independently implemented source path, NOT a separately custodied
examiner, cryptographic authority, or neural learning measurement.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math

from .cipher_wheel_world import ACTIONS, LABELS, MAX_ATTEMPTS, World, make_world


def oracle_labels(seed: int) -> tuple[str, ...]:
    """Reconstruct seed relabeling without calling the simulator's generator."""
    available = list(LABELS)
    for count in (4, 3, 2):
        position = count - 1
        digest = sha256(("cipher-wheel.v1:" + str(seed) + ":" + str(position)).encode())
        swap = int(digest.hexdigest()[:16], 16) % count
        available[position], available[swap] = available[swap], available[position]
    return tuple(available)


def _oracle_step(state: int, action: str | None, *, heldout: bool) -> int:
    if action == "RESET":
        return 0
    if heldout:
        # A has a fixed state; B connects it to A's three-cycle.
        if action == "A":
            return (state + 1) % 3 if state < 3 else 3
        if action == "B":
            return 5 - state if state >= 2 else state
    else:
        # Four-cycle and reflection: non-commuting dihedral action.
        if action == "A":
            return (state + 1) % 4
        if action == "B":
            return (-state) % 4
    return state


def _exact_json_tree(value: object, depth: int = 0) -> bool:
    """Oracle-side independent bound against nested JSON coercion."""
    if depth > 8:
        return False
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return len(value) <= 256 and all(_exact_json_tree(item, depth + 1)
                                          for item in value)
    if type(value) is dict:
        return (len(value) <= 256 and all(type(key) is str for key in value)
                and all(_exact_json_tree(item, depth + 1)
                        for item in value.values()))
    return False


def valid_world(world: object) -> bool:
    if type(world) is not World:
        return False
    if (type(world.labels) is not tuple or len(world.labels) != 4
            or any(type(label) is not str for label in world.labels)
            or type(world.world_id) is not str):
        return False
    try:
        comparison = make_world(world.seed, heldout=world.heldout,
                                variant=world.variant)
    except (ValueError, TypeError):
        return False
    return (world.labels == oracle_labels(world.seed)
            and world.world_id == comparison.world_id)


def _decode_action(payload: object) -> tuple[bool, object]:
    """Validate exact serialized input types and canonical JSON values."""
    if type(payload) is not dict or set(payload) != {"type", "json"}:
        return False, None
    kind, raw = payload["type"], payload["json"]
    if (type(kind) is not str or kind not in ("str", "bool", "int", "float",
                                           "list", "dict", "none")
            or type(raw) is not str or len(raw.encode()) > 8192):
        return False, None
    try:
        value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if not _exact_json_tree(value):
            return False, None
        if json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False) != raw:
            return False, None
    except (TypeError, ValueError, OverflowError, RecursionError):
        return False, None
    expected_kind = "none" if value is None else type(value).__name__
    return expected_kind == kind, value


def replay_trace(world: World, trace: object) -> tuple[int, tuple[dict[str, object], ...]] | None:
    """Check full state/observable parity, charged invalid inputs and exact types."""
    if not valid_world(world) or type(trace) not in (list, tuple) or len(trace) > MAX_ATTEMPTS:
        return None
    state = 0
    expected = []
    for item in trace:
        if type(item) is not dict or set(item) != {"action", "valid", "before", "after"}:
            return None
        encoded = item["action"]
        safe, action = _decode_action(encoded)
        if not safe:
            return None
        valid = type(action) is str and action in ACTIONS
        if item["valid"] is not valid:
            return None
        before = oracle_labels(world.seed)[state]
        state = _oracle_step(state, action, heldout=world.heldout)
        after = oracle_labels(world.seed)[state]
        if (type(item["before"]) is not str or type(item["after"]) is not str
                or item["before"] != before or item["after"] != after):
            return None
        expected.append({"action": encoded, "valid": valid, "before": before, "after": after})
    return state, tuple(expected)


def normalize_table(candidate: object) -> dict[str, dict[str, str]] | None:
    if type(candidate) is not dict or set(candidate) != {"A", "B"}:
        return None
    normalized = {}
    for action in ("A", "B"):
        transitions = candidate[action]
        if (type(transitions) is not dict or set(transitions) != set(LABELS)
                or any(type(key) is not str or type(target) is not str
                       or target not in LABELS for key, target in transitions.items())):
            return None
        normalized[action] = {label: transitions[label] for label in LABELS}
    return normalized


def oracle_table(world: World) -> dict[str, dict[str, str]]:
    if not valid_world(world):
        raise ValueError("untrusted world identity")
    labels = oracle_labels(world.seed)
    return {action: {labels[state]: labels[_oracle_step(state, action, heldout=world.heldout)]
                     for state in range(4)} for action in ("A", "B")}


def verify_table(world: World, candidate: object) -> dict[str, object]:
    if not valid_world(world):
        return {"verdict": "EVALUATOR_DISAGREEMENT", "reason": "UNTRUSTED_WORLD",
                "counterexample": None, "full_rows_checked": 0}
    proposed = normalize_table(candidate)
    if proposed is None:
        return {"verdict": "VERIFIED_FAILURE", "reason": "INVALID_TABLE",
                "counterexample": None, "full_rows_checked": 0}
    expected = oracle_table(world)
    checked = 0
    for action in ("A", "B"):
        for label in LABELS:
            checked += 1
            if proposed[action][label] != expected[action][label]:
                return {"verdict": "VERIFIED_FAILURE", "reason": "FALSIFIED_TRANSITION",
                        "counterexample": {"action": action, "from": label,
                                           "proposed": proposed[action][label],
                                           "observed": expected[action][label]},
                        "full_rows_checked": checked}
    return {"verdict": "VERIFIED_WIN", "reason": "COMPLETE_FINITE_TABLE_MATCH",
            "counterexample": None, "full_rows_checked": checked}


def _canonical(value: dict[str, object]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False)


def bounded_json_candidate(candidate: object) -> object:
    """Preserve an exact JSON submission, including malformed table shapes."""
    if not _exact_json_tree(candidate):
        raise ValueError("submission contains non-JSON-safe nested Python types")
    try:
        encoded = _canonical({"candidate": candidate})
        if len(encoded.encode("utf-8")) > 8192:
            raise ValueError("oversized submission")
        return json.loads(encoded)["candidate"]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("submission must be bounded JSON data") from error


def adjudicate_receipt(world: World, trace: object, candidate: object,
                       reported_verdict: str, *, submitted: bool,
                       integrity: str) -> dict[str, object]:
    verified_trace = replay_trace(world, trace)
    normalized = normalize_table(candidate) if submitted else None
    table_result = verify_table(world, normalized) if normalized is not None else None
    if verified_trace is None or integrity != "OK":
        verdict, reason = ("EVALUATOR_DISAGREEMENT",
                           integrity if integrity != "OK" else "REPLAY_OR_STATE_DIVERGENCE")
    elif reported_verdict == "VERIFIED_FAILURE" and len(trace) == MAX_ATTEMPTS:
        verdict, reason = "VERIFIED_FAILURE", "ATTEMPT_BUDGET_EXHAUSTED"
    elif not submitted:
        verdict, reason = "UNKNOWN", "NO_TABLE_SUBMITTED"
    else:
        verdict, reason = (table_result["verdict"], table_result["reason"]) if table_result else (
            "VERIFIED_FAILURE", "INVALID_TABLE")
    if reported_verdict != verdict:
        verdict, reason = "EVALUATOR_DISAGREEMENT", "REPORTED_VERDICT_DISAGREES"
    receipt: dict[str, object] = {
        "schema": "labyrinth.cipher-wheel-receipt.v1",
        "world_id": world.world_id, "seed": world.seed, "heldout": world.heldout,
        "variant": world.variant, "source_commit": "UNBOUND",
        "evidence_class": "SYNTHETIC_SOURCE_TEST", "independent_custody": False,
        "neural_learning_verified": False, "graduation_eligible": False,
        "public_source_oracle": True,
        "semantic_family_id": "cipher-wheel.S4" if world.heldout else "cipher-wheel.D4",
        "effective_semantic_families_in_receipt": 1,
        "integrity_status": integrity,
        "verdict": verdict,
        "verified_success": verdict == "VERIFIED_WIN", "reason": reason,
        "attempts": [deepcopy(item) for item in trace],
        "submission_made": submitted,
        "submitted_raw": deepcopy(candidate) if submitted else None,
        "submitted_table": deepcopy(normalized), "table_review": deepcopy(table_result),
    }
    receipt["receipt_sha256"] = sha256(_canonical(receipt).encode()).hexdigest()
    return receipt


def verify_receipt_replay(receipt: object) -> bool:
    """Independent source readback; SHA-256 is *not* an authenticated signature."""
    if type(receipt) is not dict or type(receipt.get("receipt_sha256")) is not str:
        return False
    try:
        unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if sha256(_canonical(unsigned).encode()).hexdigest() != receipt["receipt_sha256"]:
            return False
        world = make_world(receipt["seed"], heldout=receipt["heldout"],
                           variant=receipt["variant"])
        if receipt["world_id"] != world.world_id:
            return False
        if (receipt["schema"] != "labyrinth.cipher-wheel-receipt.v1"
                or receipt["source_commit"] != "UNBOUND"
                or receipt["evidence_class"] != "SYNTHETIC_SOURCE_TEST"
                or receipt["independent_custody"] is not False
                or receipt["neural_learning_verified"] is not False
                or receipt["graduation_eligible"] is not False
                or receipt["public_source_oracle"] is not True
                or receipt["effective_semantic_families_in_receipt"] != 1
                or receipt["semantic_family_id"] != (
                    "cipher-wheel.S4" if world.heldout else "cipher-wheel.D4")
                or receipt["integrity_status"] != "OK"):
            return False
        trace = receipt["attempts"]
        if replay_trace(world, trace) is None:
            return False
        submitted = receipt["submission_made"]
        if type(submitted) is not bool:
            return False
        candidate = receipt["submitted_raw"] if submitted else None
        if submitted:
            if bounded_json_candidate(candidate) != candidate:
                return False
        elif receipt["submitted_raw"] is not None:
            return False
        if normalize_table(candidate) != receipt["submitted_table"]:
            return False
        if len(trace) == MAX_ATTEMPTS:
            reported = "VERIFIED_FAILURE"
        elif not submitted:
            reported = "UNKNOWN"
        else:
            reported = verify_table(world, candidate)["verdict"]
        expected = adjudicate_receipt(world, trace, candidate, reported,
                                      submitted=submitted, integrity="OK")
        return expected == receipt
    except (TypeError, ValueError, KeyError, IndexError):
        return False
