"""Public-rule finite-mechanics runtime: candidate verification + explicit fallback.

Reads only public mechanics and a separately supplied candidate response file.
An incorrect/unsupported model answer is replaced by an honest symbolic
fallback. Fallback success is NEVER evidence of neural reasoning/learning.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .blind_structural_exam import (ACTIONS, PLAN_RE, RESPONSE_SCHEMA,
                                     _oracle, _replay)
from .local_neural_observer import _public_trials, verify_local_observation
from .live_model_probe import (_canonical, _file_bytes, _load, _receipt,
                               _write_once, MAX_CASE_BYTES, SHA)

REPORT_SCHEMA = "labyrinth.public-mechanics-hybrid-runtime.v1"


def _sha(raw):
    return sha256(raw).hexdigest()


def _trace(world, answer):
    if answer == "UNSAT":
        return []
    actions = answer[6:].split(" ")
    state = world["start"]
    trace = []
    for turn, action in enumerate(actions):
        after = world["action_tables"][action][turn % 2][state]
        if after == -1:
            raise ValueError("selected symbolic action trace unexpectedly blocked")
        trace.append({
            "turn": turn, "action": action, "phase": turn % 2,
            "state_before": state, "state_after": after,
        })
        state = after
    if state != world["target"]:
        raise ValueError("selected symbolic action trace did not reach goal")
    return trace


def _candidate_valid(world, answer, oracle):
    if answer == "UNSAT":
        return oracle is None
    if not PLAN_RE.fullmatch(answer):
        return False
    actions = answer[6:].split(" ")
    return _replay(world, actions)


def run_public_mechanics(public_path, candidate_path, responses_out, report_out, *,
                         expected_public_sha256,
                         observation_path=None,
                         expected_observation_sha256=None):
    """Build runtime replies without loading a private exam or claiming learning.

    Only verified *public* finite mechanics are used to evaluate and correct
    candidate suggestions. Caller-supplied SHA pins remain self-custodied.
    """
    files = [Path(public_path), Path(candidate_path),
             Path(responses_out), Path(report_out)]
    if (len({str(p.absolute()) for p in files}) != len(files)
            or any(p.exists() for p in files[2:])):
        raise ValueError("hybrid requires distinct original inputs and new output paths")
    # Preflight both output destinations before producing a receipt or a
    # corrected roster. This is not an atomic two-file transaction.
    if not all(p.parent.is_dir() for p in files[2:]):
        raise FileNotFoundError("both hybrid output parent directories must exist")
    if ((observation_path is None) != (expected_observation_sha256 is None)):
        raise ValueError("original model observation and its pin are both required")
    if (type(expected_public_sha256) is not str
            or SHA.fullmatch(expected_public_sha256) is None):
        raise ValueError("original public source SHA pin is malformed")
    raw_public = _file_bytes(public_path, MAX_CASE_BYTES)
    if _sha(raw_public) != expected_public_sha256:
        raise ValueError("original public source SHA digest changed")
    public = _public_trials(raw_public)
    raw_candidate = _file_bytes(candidate_path, MAX_CASE_BYTES)
    candidate = _load(raw_candidate)
    if (type(candidate) is not dict
            or set(candidate) != {"schema", "exam_id", "responses"}
            or candidate["schema"] != RESPONSE_SCHEMA
            or candidate["exam_id"] != public["exam_id"]
            or type(candidate["responses"]) is not list
            or len(candidate["responses"]) != len(public["trials"])):
        raise ValueError("candidate response roster differs from public exam")
    for trial, row in zip(public["trials"], candidate["responses"]):
        if (type(row) is not dict
                or set(row) != {"trial_id", "response"}
                or row["trial_id"] != trial["trial_id"]
                or type(row["response"]) is not str
                or len(row["response"]) > 128):
            raise ValueError("candidate response roster is malformed or reordered")
    evidence = "NOT_SUPPLIED"
    observed_digest = None
    if observation_path is not None:
        observed = verify_local_observation(
            observation_path, expected_observation_sha256,
            raw_public, raw_candidate)
        evidence = observed["evidence"]
        observed_digest = observed["declared_model_digest"]
    rows, cases = [], []
    accepted = fallback = 0
    for public_trial, proposed in zip(public["trials"], candidate["responses"]):
        world = public_trial["world"]
        # Both finite-state solvers consume exactly this public transition table.
        # This source does not import or read the answer-key path/contents.
        symbolic_path = _oracle(world)
        proposed_answer = proposed["response"]
        candidate_ok = _candidate_valid(world, proposed_answer, symbolic_path)
        if candidate_ok:
            chosen = proposed_answer
            source = "CANDIDATE_PUBLICLY_VERIFIED"
            accepted += 1
        else:
            chosen = ("PLAN: " + " ".join(symbolic_path)
                      if symbolic_path is not None else "UNSAT")
            source = "DETERMINISTIC_PUBLIC_FALLBACK"
            fallback += 1
        if not _candidate_valid(world, chosen, symbolic_path):
            raise ValueError("selected runtime answer did not verify on public mechanics")
        rows.append({"trial_id": public_trial["trial_id"],
                     "response": chosen})
        cases.append({
            "trial_id": public_trial["trial_id"],
            "public_world_sha256": _sha(_canonical(world)),
            "candidate_response_sha256": _sha(proposed_answer.encode("utf-8")),
            "candidate_publicly_valid": candidate_ok,
            "decision_source": source,
            "selected_response": chosen,
            "selected_response_sha256": _sha(chosen.encode("utf-8")),
            "public_action_trace": _trace(world, chosen),
            "public_unsat_checked": chosen == "UNSAT" and symbolic_path is None,
        })
    selected = {
        "schema": RESPONSE_SCHEMA, "exam_id": public["exam_id"],
        "responses": rows,
    }
    selected_raw = _canonical(selected) + b"\n"
    report = _receipt({
        "schema": REPORT_SCHEMA,
        "status": "PUBLIC_RULE_HYBRID_NOT_NEURAL_LEARNING",
        "exam_id": public["exam_id"],
        "case_count": len(rows),
        "original_public_sha256": _sha(raw_public),
        "original_candidate_responses_sha256": _sha(raw_candidate),
        "original_observation_sha256": expected_observation_sha256,
        "candidate_invocation_evidence": evidence,
        "candidate_observed_model_digest": observed_digest,
        "selected_responses_file_sha256": _sha(selected_raw),
        "candidate_accepted_count": accepted,
        "symbolic_fallback_count": fallback,
        "case_results": cases,
        "neural_learning_verified": False,
        "independent_custody_verified": False,
        "served_weight_bytes_attested": False,
        "limitations": [
            "Only caller-provided public transition tables are evaluated.",
            "Every fallback is a deterministic zero-weight-update correction.",
            "Accepting a winning candidate plan does not prove neural learning.",
            "An original self-pinned HTTP receipt cannot authenticate loaded weights.",
            "No private exam keys, authenticated examiner or external custody are supplied.",
        ],
    })
    # Provenance must be published FIRST. A failure to write its receipt
    # must never leave a silently corrected answer file that looks neural.
    # The final response file may still fail; consumers must check BOTH.
    _write_once(report_out, report)
    _write_once(responses_out, selected)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expected-public-sha256", required=True)
    parser.add_argument("--responses-out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--observation")
    parser.add_argument("--expected-observation-sha256")
    args = parser.parse_args(argv)
    report = run_public_mechanics(
        args.public, args.candidate, args.responses_out, args.report_out,
        expected_public_sha256=args.expected_public_sha256,
        observation_path=args.observation,
        expected_observation_sha256=args.expected_observation_sha256)
    print(json.dumps({
        "candidate_accepted_count": report["candidate_accepted_count"],
        "symbolic_fallback_count": report["symbolic_fallback_count"],
        "candidate_invocation_evidence": report["candidate_invocation_evidence"],
        "neural_learning_verified": False,
        "receipt_sha256": report["receipt_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
