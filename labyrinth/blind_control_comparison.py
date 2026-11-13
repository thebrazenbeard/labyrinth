"""Matched-case controls for answer-withheld finite-transition exams.

All candidate and baseline responses are scored against the exact same frozen
exam. Exact discordance p values are descriptive conditional calculations,
not evidence of model training, independence or transfer.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from math import comb
import json
from pathlib import Path

from .blind_structural_exam import (_read_bundle, _response_correct, _sha,
                                    _write_once, _receipt, _file_bytes,
                                    _load, MAX_BYTES, RESPONSE_SCHEMA)
from .local_neural_observer import verify_local_observation

COMPARE_SCHEMA = "labyrinth.blind-mechanics-paired-controls.v1"


def _mcnemar_exact_two_sided(candidate_only, control_only):
    """Two-sided exact conditional discordance test; never one-sided by fiat."""
    if (type(candidate_only) is not int or type(control_only) is not int
            or not 0 <= candidate_only <= 16
            or not 0 <= control_only <= 16):
        raise ValueError("invalid paired discordance counts")
    n = candidate_only + control_only
    if n > 16:
        raise ValueError("paired discordance exceeds case budget")
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(candidate_only, control_only) + 1))
    return min(1.0, (2 * tail) / (2 ** n))


def _paired(candidate, control):
    if len(candidate) != len(control):
        raise ValueError("candidate and baseline case rosters differ")
    both = sum(x and y for x, y in zip(candidate, control))
    candidate_only = sum(x and not y for x, y in zip(candidate, control))
    control_only = sum(not x and y for x, y in zip(candidate, control))
    neither = len(candidate) - both - candidate_only - control_only
    return {
        "case_count": len(candidate),
        "candidate_only": candidate_only,
        "control_only": control_only,
        "both_correct": both,
        "both_wrong": neither,
        "paired_correct_delta": candidate_only - control_only,
        "mcnemar_exact_two_sided_p":
            _mcnemar_exact_two_sided(candidate_only, control_only),
        "test_evidence": "EXACT_CONDITIONAL_BINOMIAL_SOURCE_ONLY",
    }


def compare_with_controls(private_path, public_path, precommit_path,
                          responses_path, out_path, *, expected_precommit_sha256,
                          observation_path=None, expected_observation_sha256=None):
    """Score candidate and predeclared trivial controls on exactly one case set.

    A matched comparison is not a learning certificate. It cannot establish
    that the candidate actually ran, even with a self-pinned HTTP receipt.
    """
    if Path(out_path).exists():
        raise FileExistsError("paired comparison receipt is write-once")
    if (observation_path is None) != (expected_observation_sha256 is None):
        raise ValueError("local HTTP observation requires original paired pin")
    private, precommit = _read_bundle(
        private_path, public_path, precommit_path, expected_precommit_sha256)
    raw_public = _file_bytes(public_path, MAX_BYTES)
    if _sha(raw_public) != precommit["public_sha256"]:
        raise ValueError("public exam changed after original precommit replay")
    raw_responses = _file_bytes(responses_path, MAX_BYTES)
    observed = None
    if observation_path is not None:
        observed = verify_local_observation(
            observation_path, expected_observation_sha256,
            raw_public, raw_responses)
    data = _load(raw_responses)
    if (type(data) is not dict
            or set(data) != {"schema", "exam_id", "responses"}
            or data["schema"] != RESPONSE_SCHEMA
            or data["exam_id"] != private["exam_id"]
            or type(data["responses"]) is not list
            or len(data["responses"]) != len(private["trials"])):
        raise ValueError("candidate response roster differs from committed exam")
    candidate = []
    all_unsat = []
    fixed_abc = []
    rows = []
    for trial, item in zip(private["trials"], data["responses"]):
        if (type(item) is not dict or set(item) != {"trial_id", "response"}
                or type(item["trial_id"]) is not str
                or item["trial_id"] != trial["trial_id"]
                or type(item["response"]) is not str
                or len(item["response"]) > 128):
            raise ValueError("candidate response roster has malformed or duplicated trials")
        world = trial["world"]
        witness = trial["oracle_witness"]
        candidate_ok, _, _ = _response_correct(world, item["response"], witness)
        unsat_ok, _, _ = _response_correct(world, "UNSAT", witness)
        fixed_ok, _, _ = _response_correct(world, "PLAN: A B C", witness)
        candidate.append(candidate_ok)
        all_unsat.append(unsat_ok)
        fixed_abc.append(fixed_ok)
        rows.append({
            "trial_id": trial["trial_id"],
            "candidate_response_sha256": _sha(item["response"].encode("utf-8")),
            "candidate_correct": candidate_ok,
            "all_unsat_correct": unsat_ok,
            "fixed_abc_correct": fixed_ok,
        })
    payload = {
        "schema": COMPARE_SCHEMA,
        "status": "PAIRED_SOURCE_REPLAY_NOT_LEARNING",
        "claim_ceiling": "NO_NEURAL_OR_GENERALIZATION_CERTIFICATION",
        "exam_id": private["exam_id"],
        "case_count": len(candidate),
        "original_precommit_sha256": precommit["receipt_sha256"],
        "responses_file_sha256": _sha(raw_responses),
        "candidate_correct": sum(candidate),
        "all_unsat_correct": sum(all_unsat),
        "fixed_abc_correct": sum(fixed_abc),
        "all_unsat_paired": _paired(candidate, all_unsat),
        "fixed_abc_paired": _paired(candidate, fixed_abc),
        "invocation_evidence": (observed["evidence"] if observed else "NOT_SUPPLIED"),
        "observed_model_digest": (observed["declared_model_digest"] if observed else None),
        "observation_receipt_sha256": (observed["receipt_sha256"] if observed else None),
        "neural_learning_verified": False,
        "independent_custody_verified": False,
        "served_weight_bytes_attested": False,
        "case_results": rows,
        "limitations": [
            "Paired statistics refer only to this exact finite exam, not a population.",
            "Public fixed controls can be beaten or tied without any parameter learning.",
            "Two-sided exact McNemar is descriptive and does not establish independent samples.",
            "Self-pinned model invocation cannot authenticate actual served weights or custody.",
            "Without before/after lineage and protected external cases, transfer is UNKNOWN.",
        ],
    }
    return _write_once(out_path, _receipt(payload))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", required=True)
    parser.add_argument("--public", required=True)
    parser.add_argument("--precommit", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--expected-precommit-sha256", required=True)
    parser.add_argument("--observation")
    parser.add_argument("--expected-observation-sha256")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = compare_with_controls(
        args.private, args.public, args.precommit, args.responses, args.out,
        expected_precommit_sha256=args.expected_precommit_sha256,
        observation_path=args.observation,
        expected_observation_sha256=args.expected_observation_sha256)
    print(json.dumps({
        "candidate_correct": report["candidate_correct"],
        "all_unsat_correct": report["all_unsat_correct"],
        "fixed_abc_correct": report["fixed_abc_correct"],
        "case_count": report["case_count"],
        "invocation_evidence": report["invocation_evidence"],
        "receipt_sha256": report["receipt_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
