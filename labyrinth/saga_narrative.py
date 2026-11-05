"""Display a regenerative journey as a bounded, evidence-labelled chronicle.

This is a presentation of a *development* receipt, not a model memory, a
verified training trace, or an independent graduation decision. It deliberately
does not expose individual examination prompts or answer keys.
"""

from __future__ import annotations

import re

from .ariadne_thread import verify_ariadne_thread
from .journey import DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION, EXHAUSTED


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_EXAM_STATUSES = {"PASS", "FAIL", "INCONCLUSIVE"}
_MAX_TOY_ROOMS = 15  # One arithmetic skill: scale <= 5, specificity <= 5.


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a bounded identifier")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _count(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 1_000_000:
        raise ValueError(f"{label} must be a bounded integer >= {minimum}")
    return value


def render_journey_chronicle(report: object) -> str:
    """Render the toy journey's actual reported transitions, with claim limits.

    A fabricated but internally consistent report can still pass these shape
    checks. The caller must not treat this display as an authenticated audit.
    """
    if type(report) is not dict or report.get("schema") != "labyrinth.toy-journey.v1":
        raise ValueError("expected a toy-journey.v1 development report")
    if report.get("formal_graduation") is not False or report.get("qualification") != "NOT_ESTABLISHED":
        raise ValueError("a development chronicle cannot assert formal graduation")
    objective = _identifier(report.get("objective_id"), "objective_id")
    _digest(report.get("scope_fingerprint"), "scope_fingerprint")
    initial = _digest(report.get("baseline_sha256"), "baseline_sha256")
    final = _digest(report.get("final_checkpoint_sha256"), "final_checkpoint_sha256")
    cycles = report.get("cycles")
    if type(cycles) is not list or not 1 <= len(cycles) <= 8:
        raise ValueError("journey requires 1–8 recorded cycles")
    if _count(report.get("attempts"), "attempts") != len(cycles):
        raise ValueError("attempt count does not match recorded cycles")
    phase = report.get("final_phase")
    if phase not in (EXHAUSTED, DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION):
        raise ValueError("journey report has no recognized terminal phase")

    lines = [
        f"THE LABYRINTH — {objective}",
        "A development chronicle, not an authenticated training or graduation record.",
        f"Entrance: Ariadne's thread begins at reported model digest {initial[:12]}.",
    ]
    previous = initial
    for index, cycle in enumerate(cycles):
        if type(cycle) is not dict or _count(cycle.get("cycle_index"), "cycle_index") != index:
            raise ValueError("journey cycles are not consecutive")
        before = _digest(cycle.get("checkpoint_before_sha256"), "checkpoint_before_sha256")
        after = _digest(cycle.get("checkpoint_after_sha256"), "checkpoint_after_sha256")
        if before != previous or before == after:
            raise ValueError("Ariadne thread has a broken or unchanged checkpoint transition")
        rooms = _count(cycle.get("training_case_count"), "training_case_count", minimum=1)
        updates = _count(cycle.get("parameter_updates"), "parameter_updates", minimum=1)
        total = _count(cycle.get("exam_case_count"), "exam_case_count", minimum=1)
        epochs = _count(cycle.get("epochs"), "epochs", minimum=1)
        if (not 2 <= rooms <= _MAX_TOY_ROOMS or not 1 <= epochs <= 1000
                or updates > rooms * epochs or not 8 <= total <= 64):
            raise ValueError("impossible toy journey room, epoch, update or exam count")
        candidate = _count(cycle.get("exam_candidate_correct"), "exam_candidate_correct")
        frozen = _count(cycle.get("exam_frozen_baseline_correct"), "exam_frozen_baseline_correct")
        if candidate > total or frozen > total:
            raise ValueError("Minotaur score exceeds the examination size")
        status = cycle.get("exam_status")
        if type(status) is not str or status not in _EXAM_STATUSES:
            raise ValueError("invalid Minotaur development verdict")
        if index + 1 < len(cycles) and status != "FAIL":
            raise ValueError("only a failed development exam can regenerate the next toy maze")
        weak = cycle.get("weak_skills")
        if type(weak) is not list or len(weak) > 16:
            raise ValueError("invalid weak-skill feedback")
        weak = [_identifier(skill, "weak skill") for skill in weak]

        lines.extend([
            "",
            f"Cycle {index + 1} — The chambers",
            f"  {rooms} reported training encounters; {updates} parameter updates.",
            f"  Ariadne's thread: {before[:12]} → {after[:12]} (reported in-memory model digests).",
            f"  Minotaur: {status}; candidate {candidate}/{total}, frozen baseline {frozen}/{total}.",
        ])
        if status == "FAIL" and index + 1 < len(cycles):
            lines.append("  The maze regenerates; the learner's in-memory parameters continue.")
        if weak:
            lines.append("  Development focus: " + ", ".join(weak) + ".")
        previous = after
    if previous != final:
        raise ValueError("final checkpoint does not match Ariadne's thread")
    last_verdict = cycles[-1]["exam_status"]
    if ((phase == DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION and last_verdict != "PASS")
            or (phase == EXHAUSTED and last_verdict == "PASS")):
        raise ValueError("terminal journey phase contradicts the Minotaur verdict")
    thread_fields = ("ariadne_thread", "ariadne_thread_schema", "ariadne_thread_head_sha256")
    if any(field in report for field in thread_fields):
        verify_ariadne_thread(report)
        lines.append("  Ariadne's local event chain verified internally; not independently anchored.")
    else:
        lines.append("  Legacy development report: no Ariadne event chain; event integrity unverified.")
    lines.extend(["", "At the exit:"])
    if phase == EXHAUSTED:
        lines.append("  The authorized cycles are exhausted; no graduation is established.")
    else:
        lines.append("  Development criteria passed; protected graduation remains unestablished.")
    lines.append("  Final reported model digest: " + final[:12]
                 + "; no saved checkpoint is established by this report.")
    return "\n".join(lines)
