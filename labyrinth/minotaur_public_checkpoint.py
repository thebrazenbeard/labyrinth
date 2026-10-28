"""Source-bound checkpoint to public Minotaur evaluator (never graduation).

This is a bounded, local research diagnostic. Report/checkpoint hashes prove
internal consistency, not an independently trusted training or model identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any

from .minotaur_algebraic_eval_v5 import (
    AtomicEvaluationCase, ProgramRolloutCase,
    evaluate_atomic_predictions, evaluate_program_rollouts,
)

SCHEMA = "labyrinth.minotaur-public-checkpoint.v1"
CHALLENGE_SCHEMA = "labyrinth.minotaur-public-challenge.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _refusal(reason: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "disposition": "REFUSED", "reason": reason,
        "qualification": "NOT_ESTABLISHED", "graduation_authority": "none",
        "minotaur_graduated": False, "deployed": False,
    }


def _safe_path(path: Path, *, must_exist: bool) -> None:
    """Reject symlinked components and ambiguous traversal, not hostile races."""
    if ".." in path.parts:
        raise ValueError("ambiguous_path")
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink_path")
        if not current.exists():
            if must_exist or current != absolute:
                raise ValueError("missing_path")
    if must_exist and not absolute.exists():
        raise ValueError("missing_path")


def _read_regular(path: Path, maximum: int) -> bytes:
    _safe_path(path, must_exist=True)
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode) or not 0 < observed.st_size <= maximum:
        raise ValueError("invalid_file_size_or_type")
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0))
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
            raise ValueError("invalid_file_size_or_type")
        with os.fdopen(os.dup(fd), "rb") as stream:
            data = stream.read(maximum + 1)
        if len(data) != info.st_size:
            raise ValueError("file_changed_during_read")
        return data
    finally:
        os.close(fd)


def _strict_json(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    def constant(_: str) -> Any:
        raise ValueError("nonfinite_json_number")

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("nonfinite_json_number")
        return value

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=constant, parse_float=finite_float)


def _keys(value: Any, mandatory: set[str], optional: set[str] = frozenset()) -> None:
    if type(value) is not dict or not mandatory <= value.keys() or not value.keys() <= mandatory | optional:
        raise ValueError("invalid_public_case_schema")


def _parse_challenges(raw: bytes) -> tuple[list[AtomicEvaluationCase], list[ProgramRolloutCase]]:
    data = _strict_json(raw)
    _keys(data, {"schema", "atomic_cases", "program_cases"})
    if data["schema"] != CHALLENGE_SCHEMA:
        raise ValueError("invalid_challenge_version")
    atoms, programs = data["atomic_cases"], data["program_cases"]
    if (type(atoms) is not list or not 1 <= len(atoms) <= 4096
            or type(programs) is not list or len(programs) > 4096):
        raise ValueError("invalid_challenge_count")
    atomic_cases: list[AtomicEvaluationCase] = []
    for item in atoms:
        _keys(item, {"case_id", "op", "x", "y", "claimed_output"}, {"group_id"})
        atomic_cases.append(AtomicEvaluationCase(
            item["case_id"], item["op"], item["x"], item["y"],
            item["claimed_output"], item.get("group_id", "ungrouped"),
        ))
    program_cases: list[ProgramRolloutCase] = []
    for item in programs:
        _keys(item, {"case_id", "initial_state", "instructions", "observed_states", "claimed_first_error"},
              {"group_id"})
        if (type(item["initial_state"]) is not list
                or type(item["instructions"]) is not list
                or type(item["observed_states"]) is not list
                or any(type(v) is not list for v in item["instructions"])
                or any(type(v) is not list for v in item["observed_states"])):
            raise ValueError("invalid_program_container")
        program_cases.append(ProgramRolloutCase(
            item["case_id"], tuple(item["initial_state"]),
            tuple(tuple(v) for v in item["instructions"]),
            tuple(tuple(v) for v in item["observed_states"]),
            item["claimed_first_error"], item.get("group_id", "ungrouped"),
        ))
    return atomic_cases, program_cases


def _source_manifest() -> dict[str, str]:
    folder = Path(__file__).resolve().parent
    names = ("minotaur_public_checkpoint.py", "minotaur_public_evidence.py",
             "minotaur_algebraic_eval_v5.py")
    return {name: hashlib.sha256(_read_regular(folder / name, 128_000)).hexdigest()
            for name in names}


def evaluate_checkpoint_public(
    artifact_dir: Path, challenge_json: Path, *,
    expected_source_commit: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Run public evaluation only after source and artifact consistency checks."""
    try:
        import torch
        from safetensors.torch import load
        from .minotaur_algebraic_experiment import reload_experiment
        from .minotaur_algebraic_rules import build_algebraic_head, incomplete_training_atoms
        from .minotaur_public_evidence import evaluate_compact_public

        artifact_dir, challenge_json = Path(artifact_dir), Path(challenge_json)
        _safe_path(artifact_dir, must_exist=True)
        if not artifact_dir.is_dir():
            raise ValueError("invalid_artifact_directory")
        for filename, limit in (("report.json", 128_000), ("report.sha256", 100),
                                ("candidate.safetensors", 100_000)):
            _read_regular(artifact_dir / filename, limit)
        challenge_bytes = _read_regular(challenge_json, 1_000_000)
        atomic_cases, program_cases = _parse_challenges(challenge_bytes)
        atomic_preview = evaluate_atomic_predictions(
            atomic_cases, {case.case_id: None for case in atomic_cases},
        )
        if atomic_preview["disposition"] != "SCORED":
            return _refusal("invalid_public_atomic_claim")
        if program_cases:
            program_preview = evaluate_program_rollouts(
                program_cases, lambda _op, _x, _y: None,
            )
            if program_preview["disposition"] != "SCORED":
                return _refusal("invalid_public_program_claim")
    except Exception:
        return _refusal("invalid_artifact_or_challenge_input")

    try:
        replay = reload_experiment(artifact_dir, torch)
        if replay.get("matches_original") is not True:
            return _refusal("checkpoint_replay_mismatch")
        report_bytes = _read_regular(artifact_dir / "report.json", 128_000)
        checksum = _read_regular(artifact_dir / "report.sha256", 100).decode("ascii").strip()
        if hashlib.sha256(report_bytes).hexdigest() != checksum:
            return _refusal("training_report_changed_during_read")
        report = _strict_json(report_bytes)
        if type(report) is not dict:
            return _refusal("invalid_training_report")
        if (expected_source_commit is not None
                and (not _COMMIT.fullmatch(expected_source_commit)
                     or report.get("source_commit") != expected_source_commit)):
            return _refusal("expected_source_commit_mismatch")
        if (expected_checkpoint_sha256 is not None
                and (not _SHA256.fullmatch(expected_checkpoint_sha256)
                     or report.get("checkpoint_sha256") != expected_checkpoint_sha256)):
            return _refusal("expected_checkpoint_digest_mismatch")
        seed, count = report.get("seed"), report.get("train_examples_by_operation")
        training = incomplete_training_atoms(seed, count)
        digest = hashlib.sha256(json.dumps(training).encode()).hexdigest()
        if (digest != report.get("training_atoms_sha256")
                or len(training) != report.get("train_examples")):
            return _refusal("training_identity_digest_mismatch")
        checkpoint_bytes = _read_regular(artifact_dir / "candidate.safetensors", 100_000)
        if hashlib.sha256(checkpoint_bytes).hexdigest() != report.get("checkpoint_sha256"):
            return _refusal("checkpoint_changed_during_read")
        state = load(checkpoint_bytes)
        if (set(state) != {"coefficients"} or state["coefficients"].shape != (3, 4)
                or state["coefficients"].dtype != torch.float64
                or not bool(torch.isfinite(state["coefficients"]).all())):
            return _refusal("invalid_neural_parameter_state")
        head = build_algebraic_head(torch, seed=19)
        head.load_state_dict(state, strict=True)
        if sum(t.numel() for t in head.parameters()) != 12:
            return _refusal("unexpected_trainable_parameter_count")
        coefficient_bytes = head.coefficients.detach().cpu().contiguous().numpy().tobytes()
        if hashlib.sha256(coefficient_bytes).hexdigest() != report.get("candidate_coefficients_sha256"):
            return _refusal("coefficient_digest_mismatch")
        evaluated = evaluate_compact_public(head, atomic_cases, program_cases, training, torch)
        if evaluated.get("disposition") != "PUBLIC_DIAGNOSTIC":
            return _refusal("public_evaluator_refused")
        return {
            "schema": SCHEMA, "disposition": "PUBLIC_DIAGNOSTIC",
            "qualification": "NOT_ESTABLISHED", "graduation_authority": "none",
            "minotaur_graduated": False, "deployed": False,
            "source_commit": report["source_commit"],
            "training_source_manifest": report["source_manifest"],
            "public_evaluator_source_manifest": _source_manifest(),
            "training_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "training_atoms_sha256": digest,
            "challenge_sha256": hashlib.sha256(challenge_bytes).hexdigest(),
            "parameter_count": 12, "torch_version": str(torch.__version__),
            "process_diagnostic": {
                "numeric_pid_differs_from_training": replay["process_id_differs"],
                "independent_process_attestation": "NOT_ESTABLISHED",
            },
            "evaluation": evaluated,
            "limitations": [
                "Public source and artifact hashes are internally consistent, not trusted signatures",
                "Challenge cases and runtime are not under independent private custody",
                "Numeric PID comparison alone cannot prove or disprove distinct process execution",
            ],
        }
    except Exception:
        return _refusal("checkpoint_or_public_evaluation_failed")


def _write_report_exclusive(path: Path, report: dict[str, Any]) -> None:
    _safe_path(path, must_exist=False)
    if not path.parent.is_dir():
        raise ValueError("report_parent_missing")
    # Publish only a completely synced report; never replace an existing one.
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=".minotaur-public-", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded public Minotaur checkpoint diagnostic")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--challenge-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-checkpoint-sha256")
    args = parser.parse_args(argv)
    result = evaluate_checkpoint_public(
        args.artifact_dir, args.challenge_json,
        expected_source_commit=args.expected_source_commit,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    try:
        _write_report_exclusive(args.report, result)
    except (OSError, ValueError) as exc:
        print(f"minotaur-public-checkpoint: report not written ({type(exc).__name__})", file=sys.stderr)
        return 2
    print(json.dumps({"disposition": result["disposition"],
                      "qualification": "NOT_ESTABLISHED", "report": str(args.report)},
                     sort_keys=True))
    return 0 if result["disposition"] == "PUBLIC_DIAGNOSTIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
