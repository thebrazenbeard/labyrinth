"""Read-only source-bound verification of typed V2 hybrid corrections.

A hybrid correction is not a raw model response. This verifier requires the
original public and candidate bytes, a separately supplied original hybrid
report SHA, and the original model HTTP observation and pin when applicable.
It deterministically replays the public-only hybrid computation in a temporary
directory and requires exact byte agreement for BOTH selected answers and
provenance. This does not establish independent examiner custody, loaded model
weights or neural learning.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .live_model_probe import SHA, MAX_CASE_BYTES, _canonical, _file_bytes, _load
from .public_mechanics_runtime import (
    REPORT_SCHEMA, SELECTED_SCHEMA, run_public_mechanics)


def _sha(data):
    return sha256(data).hexdigest()


def verify_hybrid_artifacts(public_path, candidate_path,
                            selected_path, report_path, *,
                            expected_public_sha256, expected_report_sha256,
                            observation_path=None,
                            expected_observation_sha256=None):
    """Fail closed unless the *original* V2 hybrid artifact reproduces exactly.

    The caller-held original report digest is a local source pin, not an
    independently trusted or authenticated signature. This function makes no
    writes to any user-provided path.
    """
    if ((observation_path is None) !=
            (expected_observation_sha256 is None)):
        raise ValueError("original model observation and its pin must be paired")
    if (type(expected_public_sha256) is not str
            or SHA.fullmatch(expected_public_sha256) is None
            or type(expected_report_sha256) is not str
            or SHA.fullmatch(expected_report_sha256) is None):
        raise ValueError("invalid original public or report receipt SHA pin")
    paths = [Path(public_path), Path(candidate_path),
             Path(selected_path), Path(report_path)]
    if len({str(p.absolute()) for p in paths}) != 4:
        raise ValueError("four distinct hybrid source and artifact paths required")
    public_raw = _file_bytes(public_path, MAX_CASE_BYTES)
    if _sha(public_raw) != expected_public_sha256:
        raise ValueError("original public source SHA differs from held pin")
    original = _file_bytes(candidate_path, MAX_CASE_BYTES)
    selected_raw = _file_bytes(selected_path, MAX_CASE_BYTES)
    report_raw = _file_bytes(report_path, MAX_CASE_BYTES)
    selected = _load(selected_raw)
    report = _load(report_raw)
    if (type(selected) is not dict
            or set(selected) != {"schema", "exam_id", "responses"}
            or selected["schema"] != SELECTED_SCHEMA
            or type(report) is not dict
            or report.get("schema") != REPORT_SCHEMA
            or report.get("status") != "PUBLIC_RULE_HYBRID_NOT_NEURAL_LEARNING"
            or report.get("receipt_sha256") != expected_report_sha256
            or _sha(_canonical({k: v for k, v in report.items()
                                if k != "receipt_sha256"}))
               != expected_report_sha256
            or report.get("original_public_sha256") != expected_public_sha256
            or report.get("original_candidate_responses_sha256") != _sha(original)
            or report.get("selected_responses_file_sha256") != _sha(selected_raw)
            or report.get("original_observation_sha256") !=
               expected_observation_sha256
            or report.get("neural_learning_verified") is not False
            or report.get("independent_custody_verified") is not False
            or report.get("served_weight_bytes_attested") is not False
            or selected.get("exam_id") != report.get("exam_id")):
        raise ValueError("hybrid selected answers, provenance, or source pin mismatch")
    # Requiring the actual canonical file bytes protects against a source
    # report being reformatted without changing its embedded receipt hash.
    if (report_raw != _canonical(report) + b"\n"
            or selected_raw != _canonical(selected) + b"\n"):
        raise ValueError("noncanonical or changed hybrid artifact file bytes")
    with TemporaryDirectory(prefix="labyrinth-hybrid-v2-verify-") as t:
        t = Path(t)
        regenerated = run_public_mechanics(
            public_path, candidate_path, t / "selected.json", t / "report.json",
            expected_public_sha256=expected_public_sha256,
            observation_path=observation_path,
            expected_observation_sha256=expected_observation_sha256)
        if (selected_raw != (t / "selected.json").read_bytes()
                or report_raw != (t / "report.json").read_bytes()
                or regenerated["receipt_sha256"] != expected_report_sha256):
            raise ValueError(
                "original hybrid answer/provenance diverges from public-only replay")
    if (type(report.get("candidate_accepted_count")) is not int
            or type(report.get("symbolic_fallback_count")) is not int
            or report["candidate_accepted_count"] < 0
            or report["symbolic_fallback_count"] < 0
            or report["candidate_accepted_count"]
               + report["symbolic_fallback_count"] != report.get("case_count")
            or not isinstance(selected["responses"], list)
            or len(selected["responses"]) != report["case_count"]):
        raise ValueError("hybrid candidate/fallback roster or source counts invalid")
    return {
        "status": "PUBLIC_HYBRID_SOURCE_VERIFIED_NOT_MODEL_ATTESTED",
        "exam_id": report["exam_id"],
        "case_count": report["case_count"],
        "selected_correct_count": report["case_count"],
        "candidate_accepted_count": report["candidate_accepted_count"],
        "symbolic_fallback_count": report["symbolic_fallback_count"],
        "candidate_invocation_evidence": report["candidate_invocation_evidence"],
        "original_model_observation_sha256":
            report["original_observation_sha256"],
        "report_receipt_sha256": expected_report_sha256,
        "selected_file_sha256": _sha(selected_raw),
        "neural_learning_verified": False,
        "independent_custody_verified": False,
        "served_weight_bytes_attested": False,
        "verification_class": "LOCAL_DETERMINISTIC_PUBLIC_ONLY_REPLAY_NOT_ATTESTED",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-public-sha256", required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--observation")
    parser.add_argument("--expected-observation-sha256")
    args = parser.parse_args(argv)
    result = verify_hybrid_artifacts(
        args.public, args.candidate, args.selected, args.report,
        expected_public_sha256=args.expected_public_sha256,
        expected_report_sha256=args.expected_report_sha256,
        observation_path=args.observation,
        expected_observation_sha256=args.expected_observation_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
