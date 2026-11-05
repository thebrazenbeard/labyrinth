"""Local hash-linked Ariadne trail for original model-directed saga sessions.

Each event binds public development measurements and actual local checkpoint
file references; no prompt, TRAIN gold, exam case identity or private content.
This is replayable *internal* integrity, not a signature or an external anchor.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re


SCHEMA = "labyrinth.model-saga-ariadne.v1"
ZERO = "0" * 64
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_FIELDS = frozenset((
    "schema", "sequence", "kind", "objective_id", "scope_fingerprint",
    "cycle_index", "checkpoint_before_sha256", "checkpoint_after_sha256",
    "summary", "parent_sha256", "event_sha256",
))
_SUMMARY_FIELDS = {
    "entrance": frozenset(("checkpoint_path", "optimizer_status", "optimizer_sha256",
                             "baseline_count", "baseline_correct")),
    "scene": frozenset(("skill", "room_index", "topology_sha256")),
    "train": frozenset(("skill", "room_index", "train_correct", "policy_before_sha256",
                          "policy_after_sha256", "checkpoint_path")),
    "verdict": frozenset(("status", "weak_skills", "exam_count", "candidate_score",
                            "baseline_score", "focus_skill", "pressure_difficulty",
                            "pressure_tactic")),
    "regeneration": frozenset(("weak_skills", "new_maze_plan_sha256")),
    "exit": frozenset(("phase",)),
}


def _is_local_absolute_checkpoint_path(value: str) -> bool:
    """Recognize local POSIX and drive-rooted Windows paths on either host OS.

    UNC shares are deliberately excluded: Ariadne receipts must not silently
    imply that a network path is an immutable local checkpoint.
    """
    if value.startswith(("\\\\", "//")):
        return False
    windows = PureWindowsPath(value)
    drive_local = bool(re.fullmatch(r"[A-Za-z]:", windows.drive))
    return PurePosixPath(value).is_absolute() or (
        drive_local and windows.is_absolute()
    )


def _hash(event: dict) -> str:
    value = {key: item for key, item in event.items() if key != "event_sha256"}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(b"labyrinth-model-saga-ariadne-v1\x00" + encoded).hexdigest()


def _validate_summary(kind: str, summary: dict) -> None:
    if kind not in _SUMMARY_FIELDS or type(summary) is not dict or frozenset(summary) != _SUMMARY_FIELDS[kind]:
        raise ValueError("invalid or incomplete Ariadne public event summary")
    for key, value in summary.items():
        if key.endswith("sha256"):
            if key == "optimizer_sha256" and value is None:
                continue
            if type(value) is not str or _SHA.fullmatch(value) is None:
                raise ValueError("invalid Ariadne digest")
        elif key in ("baseline_count", "baseline_correct", "room_index", "exam_count",
                     "pressure_difficulty"):
            if type(value) is not int or not 0 <= value <= 100_000:
                raise ValueError("invalid Ariadne public count")
        elif key == "train_correct":
            if type(value) is not bool:
                raise ValueError("invalid Ariadne training result")
        elif key in ("candidate_score", "baseline_score"):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 2048:
                raise ValueError("invalid Ariadne development score")
        elif key == "weak_skills":
            if (type(value) is not list or len(value) > 12
                    or any(type(skill) is not str or _ID.fullmatch(skill) is None for skill in value)):
                raise ValueError("invalid Ariadne weakness tags")
        elif key == "checkpoint_path":
            if (type(value) is not str or len(value) > 4096
                    or not _is_local_absolute_checkpoint_path(value)):
                raise ValueError("invalid Ariadne local checkpoint reference")
        else:
            if type(value) is not str or not 1 <= len(value) <= 128:
                raise ValueError("invalid Ariadne public label")
    if kind == "entrance":
        if (summary["baseline_count"] < 2
                or summary["baseline_correct"] > summary["baseline_count"]
                or summary["optimizer_status"] not in
                ("WEIGHTS_ONLY_CONTINUITY", "FULL_ARTIFACTS_PRESENT_UNQUALIFIED")):
            raise ValueError("inconsistent Ariadne entrance baseline")
        if (summary["optimizer_status"] == "WEIGHTS_ONLY_CONTINUITY") != (summary["optimizer_sha256"] is None):
            raise ValueError("Ariadne optimizer provenance does not match")
    if kind == "verdict":
        if (summary["status"] not in ("PASS", "FAIL", "INCONCLUSIVE")
                or summary["exam_count"] < 1
                or summary["candidate_score"] > summary["exam_count"]
                or summary["baseline_score"] > summary["exam_count"]
                or summary["pressure_difficulty"] not in range(1, 6)
                or summary["pressure_tactic"] not in ("holdout", "transfer", "mixed")):
            raise ValueError("inconsistent Ariadne Minotaur evidence")
    if kind == "exit" and summary["phase"] not in (
        "EXHAUSTED", "development_passed_pending_private_qualification", "ACTION_BUDGET_EXHAUSTED",
    ):
        raise ValueError("invalid Ariadne exit phase")


def append_saga_event(
    events: list[dict], *, kind: str, objective_id: str, scope_fingerprint: str,
    cycle_index: int, before: str, after: str, summary: dict,
) -> None:
    if (type(events) is not list or len(events) >= 256
            or type(objective_id) is not str or _ID.fullmatch(objective_id) is None
            or type(scope_fingerprint) is not str or _SHA.fullmatch(scope_fingerprint) is None
            or type(cycle_index) is not int or not 0 <= cycle_index <= 8
            or any(type(value) is not str or _SHA.fullmatch(value) is None
                   for value in (before, after))):
        raise ValueError("invalid Ariadne event identity or bounded budget")
    _validate_summary(kind, summary)
    event = {
        "schema": SCHEMA, "sequence": len(events), "kind": kind,
        "objective_id": objective_id, "scope_fingerprint": scope_fingerprint,
        "cycle_index": cycle_index, "checkpoint_before_sha256": before,
        "checkpoint_after_sha256": after, "summary": json.loads(json.dumps(summary)),
        "parent_sha256": events[-1]["event_sha256"] if events else ZERO,
    }
    event["event_sha256"] = _hash(event)
    events.append(event)


def verify_saga_ariadne(events: object, *, require_exit: bool = False) -> str:
    """Verify event hashes, ancestry, checkpoint chronology and basic phases.

    Returns the head digest. Anyone rewriting the complete record can compute
    a matching head; independent anchoring must be supplied separately.
    """
    if type(events) is not list or not 1 <= len(events) <= 256:
        raise ValueError("Ariadne journal requires bounded events")
    objective = scope = checkpoint = None
    previous_hash = ZERO
    previous_kind = ""
    cycle = 0
    for number, event in enumerate(events):
        if (type(event) is not dict or frozenset(event) != _FIELDS
                or event.get("schema") != SCHEMA
                or type(event.get("sequence")) is not int or event["sequence"] != number
                or event.get("parent_sha256") != previous_hash
                or event.get("event_sha256") != _hash(event)):
            raise ValueError("Ariadne event chain/content mismatch")
        kind = event["kind"]
        if (type(event["objective_id"]) is not str or _ID.fullmatch(event["objective_id"]) is None
                or type(event["scope_fingerprint"]) is not str
                or _SHA.fullmatch(event["scope_fingerprint"]) is None
                or type(event["cycle_index"]) is not int
                or not 0 <= event["cycle_index"] <= 8
                or any(type(event[field]) is not str or _SHA.fullmatch(event[field]) is None
                       for field in ("checkpoint_before_sha256", "checkpoint_after_sha256"))):
            raise ValueError("invalid Ariadne event provenance")
        _validate_summary(kind, event["summary"])
        if number == 0:
            if kind != "entrance" or event["cycle_index"] != 0:
                raise ValueError("Ariadne journal must start at the actual entrance")
            objective, scope = event["objective_id"], event["scope_fingerprint"]
            checkpoint = event["checkpoint_before_sha256"]
            if event["checkpoint_after_sha256"] != checkpoint:
                raise ValueError("Ariadne entrance cannot change model weights")
        else:
            if (event["objective_id"] != objective or event["scope_fingerprint"] != scope
                    or event["checkpoint_before_sha256"] != checkpoint
                    or type(event["cycle_index"]) is not int):
                raise ValueError("Ariadne objective, scope or checkpoint ancestry changed")
            if kind == "scene":
                if previous_kind not in ("entrance", "train", "regeneration") or event["cycle_index"] != cycle:
                    raise ValueError("Ariadne scene appeared without authorized corridor")
            elif kind == "train":
                if (previous_kind != "scene" or event["cycle_index"] != cycle
                        or event["checkpoint_after_sha256"] == checkpoint):
                    raise ValueError("Ariadne TRAIN lacks a changed model artifact or scene")
            elif kind == "verdict":
                if (previous_kind not in ("train", "verdict")
                        or event["cycle_index"] != cycle):
                    raise ValueError("Ariadne Minotaur verdict lacks completed TRAIN")
            elif kind == "regeneration":
                if (previous_kind != "verdict" or events[number - 1]["summary"]["status"] != "FAIL"
                        or event["cycle_index"] != cycle + 1):
                    raise ValueError("Ariadne regeneration requires a failed Minotaur")
                cycle += 1
            elif kind == "exit":
                if number != len(events) - 1 or event["cycle_index"] != cycle:
                    raise ValueError("Ariadne exit must be terminal")
                if event["summary"]["phase"] != "ACTION_BUDGET_EXHAUSTED":
                    if previous_kind != "verdict":
                        raise ValueError("graduation-like exit without development verdict")
                    status = events[number - 1]["summary"]["status"]
                    if ((event["summary"]["phase"] == "development_passed_pending_private_qualification")
                            != (status == "PASS")):
                        raise ValueError("Ariadne exit contradicts Minotaur verdict")
            else:
                raise ValueError("invalid Ariadne event order")
            if kind != "train" and event["checkpoint_after_sha256"] != checkpoint:
                raise ValueError("non-TRAIN event cannot claim a model weight update")
        if kind == "train":
            checkpoint = event["checkpoint_after_sha256"]
        previous_hash, previous_kind = event["event_sha256"], kind
    if require_exit and previous_kind != "exit":
        raise ValueError("Ariadne journey has no exit receipt")
    return previous_hash


__all__ = ["SCHEMA", "append_saga_event", "verify_saga_ariadne"]
