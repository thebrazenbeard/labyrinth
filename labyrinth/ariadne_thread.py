"""Hash-linked, replay-checkable *local* chronology for toy maze journeys.

The thread commits to public developmental metadata, not answers, prompts,
private exam contents, external custody, or real-time process persistence.
Anyone able to rewrite the whole report can recompute its hashes; the caller
must separately anchor ``ariadne_thread_head_sha256`` to detect rollback.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping


THREAD_SCHEMA = "labyrinth.ariadne-thread.v1"
_ZERO = "0" * 64
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_KINDS = {"entrance", "train_complete", "minotaur_verdict", "maze_regenerated", "exit"}
_FIELDS = frozenset({
    "schema", "sequence", "kind", "objective_id", "scope_fingerprint",
    "cycle_index", "checkpoint_before_sha256", "checkpoint_after_sha256",
    "status", "parent_sha256", "event_sha256",
})


def _digest_event(event: dict) -> str:
    payload = {key: value for key, value in event.items() if key != "event_sha256"}
    return hashlib.sha256(
        b"labyrinth-ariadne-thread-v1\x00" + json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def append_ariadne_event(
    thread: list[dict], *, kind: str, objective_id: str, scope_fingerprint: str,
    cycle_index: int, checkpoint_before_sha256: str,
    checkpoint_after_sha256: str, status: str,
) -> None:
    """Append a sealed event without including case IDs, prompts or gold labels."""
    if kind not in _KINDS or type(cycle_index) is not int or cycle_index < 0:
        raise ValueError("invalid Ariadne kind or cycle")
    if type(objective_id) is not str or not objective_id:
        raise ValueError("invalid Ariadne objective")
    for value in (scope_fingerprint, checkpoint_before_sha256, checkpoint_after_sha256):
        if type(value) is not str or _SHA256.fullmatch(value) is None:
            raise ValueError("invalid Ariadne hash identity")
    if type(status) is not str or not status:
        raise ValueError("invalid Ariadne status")
    event = {
        "schema": THREAD_SCHEMA,
        "sequence": len(thread),
        "kind": kind,
        "objective_id": objective_id,
        "scope_fingerprint": scope_fingerprint,
        "cycle_index": cycle_index,
        "checkpoint_before_sha256": checkpoint_before_sha256,
        "checkpoint_after_sha256": checkpoint_after_sha256,
        "status": status,
        "parent_sha256": thread[-1]["event_sha256"] if thread else _ZERO,
    }
    event["event_sha256"] = _digest_event(event)
    thread.append(event)


def verify_ariadne_thread(report: Mapping) -> bool:
    """Pure replay verifier; rejects changed events, order, ancestry and report drift.

    Returns ``True`` only when the source-level chronology is internally
    consistent. No signature, independent test result or durable write is
    established by this function.
    """
    if not isinstance(report, Mapping):
        raise ValueError("Ariadne report must be a mapping")
    events = report.get("ariadne_thread")
    cycles = report.get("cycles")
    if not isinstance(events, list) or not isinstance(cycles, list) or not events:
        raise ValueError("missing Ariadne chronology or journey cycles")
    if report.get("ariadne_thread_schema") != THREAD_SCHEMA:
        raise ValueError("invalid Ariadne chronology schema")
    if not cycles or report.get("attempts") != len(cycles):
        raise ValueError("Ariadne chronology requires complete toy attempts")
    objective = report.get("objective_id")
    scope = report.get("scope_fingerprint")
    baseline = report.get("baseline_sha256")
    final = report.get("final_checkpoint_sha256")
    if not isinstance(objective, str) or not objective:
        raise ValueError("missing Ariadne objective")
    for label, value in (("scope", scope), ("baseline", baseline), ("final", final)):
        if type(value) is not str or _SHA256.fullmatch(value) is None:
            raise ValueError(f"invalid Ariadne {label} identity")
    expected: list[dict] = []

    def add(kind: str, index: int, before: str, after: str, status: str) -> None:
        append_ariadne_event(expected, kind=kind, objective_id=objective,
                             scope_fingerprint=scope, cycle_index=index,
                             checkpoint_before_sha256=before,
                             checkpoint_after_sha256=after, status=status)

    add("entrance", 0, baseline, baseline, "TRAINING")
    predecessor = baseline
    for index, cycle in enumerate(cycles):
        if not isinstance(cycle, dict) or type(cycle.get("cycle_index")) is not int:
            raise ValueError("invalid Ariadne cycle")
        before = cycle.get("checkpoint_before_sha256")
        after = cycle.get("checkpoint_after_sha256")
        verdict = cycle.get("exam_status")
        if (cycle["cycle_index"] != index or before != predecessor or after == before
                or type(after) is not str or _SHA256.fullmatch(after) is None
                or verdict not in ("PASS", "FAIL", "INCONCLUSIVE")):
            raise ValueError("Ariadne checkpoint ancestry or Minotaur status mismatch")
        add("train_complete", index, before, after, "AWAITING_EXAM")
        add("minotaur_verdict", index, after, after, verdict)
        predecessor = after
        if index < len(cycles) - 1:
            if verdict != "FAIL":
                raise ValueError("only a failed Minotaur verdict regenerates a maze")
            add("maze_regenerated", index + 1, after, after, "TRAINING")
    if predecessor != final:
        raise ValueError("Ariadne final checkpoint mismatch")
    phase = report.get("final_phase")
    if phase not in ("EXHAUSTED", "development_passed_pending_private_qualification"):
        raise ValueError("unsupported Ariadne terminal phase")
    if (phase == "EXHAUSTED" and cycles[-1]["exam_status"] not in ("FAIL", "INCONCLUSIVE")):
        raise ValueError("Ariadne terminal status contradicts exam")
    if (phase == "development_passed_pending_private_qualification"
            and cycles[-1]["exam_status"] != "PASS"):
        raise ValueError("Ariadne terminal status contradicts exam")
    add("exit", len(cycles) - 1, final, final, phase)
    if len(events) != len(expected):
        raise ValueError("Ariadne event count mismatch")
    for actual, want in zip(events, expected):
        if (type(actual) is not dict or frozenset(actual) != _FIELDS
                or type(actual.get("sequence")) is not int
                or actual.get("event_sha256") != _digest_event(actual)
                or actual != want):
            raise ValueError("Ariadne event chain/content mismatch")
    if report.get("ariadne_thread_head_sha256") != expected[-1]["event_sha256"]:
        raise ValueError("Ariadne thread head mismatch")
    return True
