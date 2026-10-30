"""Opt-in, host-ready two-cycle HF/PEFT adapter-continuation probe.

Default is a read-only dry run. --execute invokes the existing local-only HF
pilot and its fresh-process reload command. This probes *mechanics*, not
objective mastery, model origin, independent examination, or graduation.

Run from a Labyrinth checkout with optional HF dependencies installed and a
previously downloaded, genuinely pretrained local safetensors model.
No network/model download is performed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from labyrinth.hf_pilot import (  # noqa: E402 - script may run from an arbitrary cwd
    PilotConfig, SCHEMA as PILOT_SCHEMA, _checkpoint_manifest, _sha_file, validate_config,
)
from labyrinth.runner import _source_state, load_json  # noqa: E402
from labyrinth.training_scope import (  # noqa: E402
    MAX_SCOPE_JSON_BYTES, TrainingScopeSpec, parse_scope_json,
)


PLAN_SCHEMA = "labyrinth.hf-two-cycle-plan.v1"
REPORT_SCHEMA = "labyrinth.hf-two-cycle-probe.v1"
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
_AUTHORITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt",
                    "sentencepiece.bpe.model")
_MAX_CHILD_TIMEOUT_SECONDS = 14400


@dataclass(frozen=True)
class ProbePlan:
    scope: TrainingScopeSpec
    scope_path: Path
    config: PilotConfig
    model_path: Path
    run_root: Path
    model_manifest: dict[str, dict[str, str]]
    disk_free_bytes: int
    disk_required_bytes: int
    model_origin_claim: str
    child_timeout_seconds: int
    harness_sha256: str

    def public(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA, "execute": False, "cycle_count": 2,
            "qualification": "NOT_RUN",
            "scope_fingerprint": self.scope.fingerprint,
            "scope": self.scope.to_dict(),
            "model_path": str(self.model_path),
            "model_origin_claim": self.model_origin_claim,
            "harness_sha256": self.harness_sha256,
            "child_timeout_seconds": self.child_timeout_seconds,
            "base_model_origin_evidence": "OPERATOR_CLAIM_ONLY",
            "model_manifest": self.model_manifest,
            "run_root": str(self.run_root),
            "planned_cycle_dirs": [str(self.run_root / f"cycle-{i}") for i in range(2)],
            "config": {"seed": self.config.seed, "train_count": self.config.train_count,
                       "eval_count": self.config.eval_count, "steps": self.config.steps,
                       "device": self.config.device, "max_tokens": self.config.max_tokens,
                       "max_new_tokens": self.config.max_new_tokens,
                       "answer_only": self.config.answer_only},
            "disk_free_bytes": self.disk_free_bytes,
            "disk_required_bytes": self.disk_required_bytes,
            "device_resource_qualification": "NOT_RUN",
            "active_host_workloads": "NOT_INSPECTED_BY_THIS_SCRIPT",
            "independent_evaluation": "NOT_RUN",
            "formal_graduation": False,
            "limitations": [
                "A local model path and an operator origin claim do not prove pretrained provenance.",
                "Structural validity of model files does not establish they are loadable.",
                "No RAM/GPU capacity or workload coordination is established by dry-run.",
                "Cycle-one weakness selection is empty/operator-scoped, not Minotaur-authenticated.",
                "Public same-family diagnostics are not an independent transfer examination.",
            ],
        }


def preflight(args: argparse.Namespace) -> ProbePlan:
    """Read-only validation with exact base-model byte and disk-space checks."""
    if not _RUN_NAME.fullmatch(args.run_name):
        raise ValueError("run-name must be a bounded single directory name")
    if not args.model_origin_claim or len(args.model_origin_claim) > 256:
        raise ValueError("bounded operator model-origin-claim is required")
    if args.execute and (not args.authority_ref or not _AUTHORITY.fullmatch(args.authority_ref)):
        raise ValueError("--execute requires a bounded --authority-ref operator label")
    child_timeout = getattr(args, "child_timeout_seconds", 1800)
    if (type(child_timeout) is not int
            or not 1 <= child_timeout <= _MAX_CHILD_TIMEOUT_SECONDS):
        raise ValueError("child timeout must be 1-14400 seconds")
    if not (1 <= args.steps <= 16 and 2 <= args.train_count <= 16
            and 2 <= args.eval_count <= 16 and 32 <= args.max_tokens <= 512):
        raise ValueError("harness budget exceeds conservative steps/cases/token limits")
    if Path(args.model_path).expanduser().is_symlink():
        raise ValueError("model-path must not be a symlink")

    raw_scope = Path(args.scope_json).expanduser()
    if (raw_scope.is_symlink() or not raw_scope.is_file()
            or not 0 < raw_scope.stat().st_size <= MAX_SCOPE_JSON_BYTES):
        raise ValueError("scope-json must be a bounded regular non-symlink file")
    scope = parse_scope_json(raw_scope.read_bytes())
    if scope.max_cycles < 2 or scope.skills != ("arithmetic",):
        raise ValueError("two-cycle HF probe currently requires an arithmetic-only scope with max_cycles>=2")

    raw_parent = Path(args.output_parent).expanduser()
    if raw_parent.is_symlink():
        raise ValueError("output-parent must not be a symlink")
    parent = raw_parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("output-parent must be an existing directory")
    run_root = parent / args.run_name
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError(f"run root already exists: {run_root}")

    cfg = PilotConfig(
        model_path=args.model_path, output_dir=str(run_root),
        seed=args.seed, train_count=args.train_count, eval_count=args.eval_count,
        tiers=(1,), steps=args.steps, max_tokens=args.max_tokens,
        max_new_tokens=args.max_new_tokens, learning_rate=args.learning_rate,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        device=args.device, answer_only=args.answer_only,
    )
    model, _, weights = validate_config(cfg)
    if not any((model / name).is_file() and not (model / name).is_symlink()
               for name in _TOKENIZER_FILES):
        raise ValueError("local tokenizer data file is required")
    total_weight_bytes = sum(path.stat().st_size for path in weights)
    required = max(512 * 1024 * 1024, total_weight_bytes * 2)
    available = shutil.disk_usage(parent).free
    if available < required:
        raise ValueError("insufficient available disk for isolated two-cycle adapter outputs")
    return ProbePlan(scope, raw_scope.resolve(strict=True), cfg, model, run_root,
                     _checkpoint_manifest(model, weights), available, required,
                     args.model_origin_claim, child_timeout,
                     _sha_file(Path(__file__).resolve(strict=True)))


def _command(plan: ProbePlan, cycle: int, *, predecessor: dict[str, Any] | None,
             predecessor_report_path: Path | None, authority_ref: str) -> list[str]:
    cfg = plan.config
    output_dir = plan.run_root / f"cycle-{cycle}"
    command = [
        sys.executable, "-m", "labyrinth.hf_pilot", "train",
        "--model-path", str(plan.model_path), "--output-dir", str(output_dir),
        "--seed", str(cfg.seed), "--train-count", str(cfg.train_count),
        "--eval-count", str(cfg.eval_count), "--tiers", "1",
        "--steps", str(cfg.steps), "--max-tokens", str(cfg.max_tokens),
        "--max-new-tokens", str(cfg.max_new_tokens),
        "--learning-rate", str(cfg.learning_rate),
        "--lora-rank", str(cfg.lora_rank), "--lora-alpha", str(cfg.lora_alpha),
        "--device", cfg.device, "--scope-json", str(plan.scope_path),
        "--cycle-index", str(cycle),
    ]
    if cfg.answer_only:
        command.append("--answer-only")
    if predecessor is not None:
        if cycle != 1 or predecessor_report_path is None:
            raise ValueError("predecessor only valid for cycle one")
        value = predecessor.get("candidate_lora_tensor_value_sha256")
        if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("prior candidate LoRA digest missing or invalid")
        command.extend([
            "--predecessor-run-dir", str(predecessor_report_path.parent),
            "--predecessor-report-sha256", _sha_file(predecessor_report_path),
            "--predecessor-lora-sha256", value,
            "--resume-authority-ref", authority_ref,
        ])
    return command


def _read_report(path: Path, cycle: int, plan: ProbePlan) -> dict[str, Any]:
    report = load_json(path, strict=True)
    journey = report.get("resumed_journey")
    if (report.get("schema") != PILOT_SCHEMA or not isinstance(journey, dict)
            or journey.get("scope_fingerprint") != plan.scope.fingerprint
            or journey.get("cycle_index") != cycle
            or report.get("base_weights_sha256") != plan.model_manifest["weights"]
            or report.get("model_metadata_sha256") != plan.model_manifest["metadata"]
            or journey.get("same_initial_lora_for_all_controls") is not True):
        raise RuntimeError("HF report does not match pinned scope, model, cycle or controls")
    initial = report.get("initial_lora_parameter_sha256")
    candidate = report.get("candidate_lora_tensor_value_sha256")
    if type(initial) is not str or type(candidate) is not str or initial == candidate:
        raise RuntimeError("a cycle did not demonstrate a LoRA parameter change")
    return report


def _stop_owned_child(proc: subprocess.Popen) -> bool:
    """Stop only the subprocess we launched and, where supported, its children.

    The POSIX subprocess is a new session/process group. On Windows, taskkill
    targets the exact launched PID and its descendants; that host path still
    requires physical qualification. Never kill unrelated host workloads.
    """
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        # Descendants might remain after the group leader exits.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif proc.poll() is None:
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=6, check=False,
            )
            if result.returncode != 0:
                proc.kill()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
                return False  # Direct child stopped; descendants unconfirmed.
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            return False
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        return False
    return True


def _run_bounded_child(command: list[str], *, run_root: Path,
                       phase: str, timeout_seconds: float) -> None:
    """Run a single owned child with a wall clock cap and forensic receipts."""
    if not re.fullmatch(r"cycle-[01]-(train|reload)", phase):
        raise ValueError("invalid owned child phase")
    if (not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool)
            or not 0 < timeout_seconds <= _MAX_CHILD_TIMEOUT_SECONDS):
        raise ValueError("invalid child timeout")
    if (type(command) is not list or not command
            or any(type(part) is not str or not part for part in command)):
        raise ValueError("child command must be a nonempty argv list")
    if not run_root.is_dir() or run_root.is_symlink():
        raise ValueError("owned child requires an existing non-symlink run root")
    start_path = run_root / f"child-{phase}-start.json"
    outcome_path = run_root / f"child-{phase}-outcome.json"
    log_path = run_root / f"child-{phase}-output.log"
    if any(path.exists() or path.is_symlink()
           for path in (start_path, outcome_path, log_path)):
        raise FileExistsError("owned child receipts already exist")
    started = time.time()
    with log_path.open("xb") as log:
        kwargs: dict[str, Any] = {"cwd": REPO, "stdout": log, "stderr": subprocess.STDOUT}
        if os.name == "posix":
            kwargs["start_new_session"] = True
        elif os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            raise RuntimeError("process-tree timeout requires POSIX or Windows")
        proc = subprocess.Popen(command, **kwargs)
        try:
            _write_new(start_path, {
                "phase": phase, "pid": proc.pid, "started_unix": started,
                "timeout_seconds": timeout_seconds,
                "argv_sha256": hashlib.sha256("\x00".join(command).encode()).hexdigest(),
            })
        except BaseException:
            _stop_owned_child(proc)
            raise
        status, code, cleanup = "INTERRUPTED", None, None
        try:
            code = proc.wait(timeout=timeout_seconds)
            status = "SUCCESS" if code == 0 else "FAILED"
        except subprocess.TimeoutExpired:
            cleanup = _stop_owned_child(proc)
            status = "TIMEOUT" if cleanup else "TIMEOUT_CLEANUP_UNCONFIRMED"
        except BaseException:
            cleanup = _stop_owned_child(proc)
            raise
        finally:
            _write_new(outcome_path, {
                "phase": phase, "pid": proc.pid, "status": status,
                "exit_code": code, "cleanup_confirmed": cleanup,
                "elapsed_seconds": round(time.time() - started, 3),
            })
    if status.startswith("TIMEOUT"):
        raise TimeoutError(f"{phase} timed out; cleanup={cleanup}; see {outcome_path}")
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def _verify_plan_inputs(plan: ProbePlan) -> None:
    """Recheck mutable scope/model bytes before starting a training child.

    These hashes detect changes at the check boundary, not simultaneous writes
    during a model load. Physical runs still require a stable local checkpoint.
    """
    if (plan.scope_path.is_symlink() or not plan.scope_path.is_file()
            or not 0 < plan.scope_path.stat().st_size <= MAX_SCOPE_JSON_BYTES
            or parse_scope_json(plan.scope_path.read_bytes()) != plan.scope):
        raise ValueError("training scope changed since preflight")
    model, _, weights = validate_config(plan.config, output_must_be_new=False)
    if model != plan.model_path or _checkpoint_manifest(model, weights) != plan.model_manifest:
        raise ValueError("model manifest changed since preflight")


def execute(plan: ProbePlan, *, authority_ref: str) -> dict[str, Any]:
    """Run two bounded local training cycles and fresh-process reload checks.

    Creating the unique run root is intentional. On failure, partial artifacts
    remain for forensic review; this function never overwrites or deletes them.
    """
    # The command-line preflight also checks this label, but this callable is
    # independently importable. Keep the physical-execution boundary guarded
    # even when callers bypass argparse and construct a plan directly.
    if type(plan) is not ProbePlan:
        raise ValueError("execute requires a ProbePlan")
    if type(authority_ref) is not str or _AUTHORITY.fullmatch(authority_ref) is None:
        raise ValueError("execute requires a bounded --authority-ref operator label")
    if _sha_file(Path(__file__).resolve(strict=True)) != plan.harness_sha256:
        raise ValueError("harness digest changed since preflight")
    _verify_plan_inputs(plan)
    plan.run_root.mkdir(mode=0o700, exist_ok=False)
    _write_new(plan.run_root / "preflight.json", {**plan.public(), "execute": True,
                                                "operator_authority_ref": authority_ref,
                                                "source": _source_state()})
    cycles: list[dict[str, Any]] = []
    preceding: dict[str, Any] | None = None
    preceding_path: Path | None = None
    for cycle in range(2):
        _verify_plan_inputs(plan)
        run_dir = plan.run_root / f"cycle-{cycle}"
        command = _command(plan, cycle, predecessor=preceding,
                           predecessor_report_path=preceding_path,
                           authority_ref=authority_ref)
        _run_bounded_child(command, run_root=plan.run_root,
                           phase=f"cycle-{cycle}-train",
                           timeout_seconds=plan.child_timeout_seconds)
        report_path = run_dir / "pilot_report.json"
        report = _read_report(report_path, cycle, plan)
        reload_path = run_dir / "fresh_process_reload.json"
        _run_bounded_child([sys.executable, "-m", "labyrinth.hf_pilot", "eval",
                            "--run-dir", str(run_dir), "--report", str(reload_path)],
                           run_root=plan.run_root, phase=f"cycle-{cycle}-reload",
                           timeout_seconds=plan.child_timeout_seconds)
        reloaded = load_json(reload_path, strict=True)
        if (reloaded.get("process_id_differs") is not True
                or reloaded.get("matches_original_candidate_trials") is not True
                or reloaded.get("reloaded_lora_tensor_value_sha256")
                != report["candidate_lora_tensor_value_sha256"]):
            raise RuntimeError("fresh-process adapter reload did not qualify")
        if preceding is not None:
            receipt = report["resumed_journey"].get("predecessor_resume_receipt")
            if (type(receipt) is not dict
                    or receipt.get("previous_cycle_final_lora_sha256")
                    != preceding["candidate_lora_tensor_value_sha256"]
                    or receipt.get("this_cycle_initial_lora_sha256")
                    != report["initial_lora_parameter_sha256"]):
                raise RuntimeError("cycle-one initial LoRA values do not match cycle-zero final values")
        cycles.append({
            "cycle": cycle,
            "report_sha256": _sha_file(report_path),
            "adapter_manifest": report.get("candidate_adapter_sha256"),
            "initial_lora_sha256": report["initial_lora_parameter_sha256"],
            "final_lora_sha256": report["candidate_lora_tensor_value_sha256"],
            "parameter_change_observed": True,
            "fresh_process_reload": True,
            "public_diagnostic_correct": {
                arm: result.get("correct") for arm, result in report["evaluations"].items()
            },
        })
        preceding, preceding_path = report, report_path
    result = {
        "schema": REPORT_SCHEMA, "preflight": "preflight.json",
        "base_model_origin_evidence": "OPERATOR_CLAIM_ONLY",
        "model_origin_claim": plan.model_origin_claim,
        "harness_sha256": plan.harness_sha256,
        "scope_fingerprint": plan.scope.fingerprint,
        "cycles": cycles,
        "neural_parameter_continuation_mechanism": "OBSERVED_BY_LOCAL_HARNESS",
        "matched_controls": "PUBLIC_DIAGNOSTIC_RECORDS_ONLY",
        "independent_transfer": "NOT_RUN", "independent_graduation": "NOT_RUN",
        "host_qualification": "NOT_INDEPENDENTLY_ATTESTED",
        "learning_effectiveness": "UNKNOWN_REQUIRES_BLIND_EVALUATION",
        "qualification_gates": {
            "Q0_source": {"status": "UNKNOWN", "observation": "local source digests recorded; no independent custody"},
            "Q1_tests": {"status": "NOT_RUN", "observation": "host run does not execute source regression suite"},
            "Q2_integration": {"status": "UNKNOWN", "observation": "two-cycle HF pilot only; autonomous Minotaur integration not established"},
            "Q3_neural": {"status": "UNKNOWN", "observation": "two local subprocess training calls; pretrained origin independently unverified"},
            "Q4_persistence": {"status": "UNKNOWN", "observation": "self-checked LoRA digest continuity and separate-process reload"},
            "Q5_controls": {"status": "UNKNOWN", "observation": "public sham/corrupt arms measured, not blind causal qualification"},
            "Q6_transfer": {"status": "NOT_RUN", "observation": "no new structural transfer examination"},
            "Q7_independence": {"status": "NOT_RUN", "observation": "no independent evaluator custody"},
            "Q8_recovery": {"status": "UNKNOWN", "observation": "fresh-process reload only; interrupted optimizer recovery not tested"},
            "Q9_merge": {"status": "NOT_RUN", "observation": "not a release or merge gate"},
        },
        "limitations": plan.public()["limitations"] + [
            "This local process is not independent hostile review or formal attestation.",
            "Model optimizer moments are not retained between cycles.",
            "Any failure leaves partial artifacts; no automatic checkpoint repair or rerun.",
            "A successful parameter update is not evidence of better reasoning.",
        ],
    }
    _write_new(plan.run_root / "two_cycle_report.json", result)
    return result


def _write_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--scope-json", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-origin-claim", required=True,
                        help="Operator-provided model origin label, never independently authenticated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--eval-count", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run real local training; default is read-only")
    parser.add_argument("--child-timeout-seconds", type=int, default=1800,
                        help="Wall-time limit per owned training/reload child, 1-14400 seconds")
    parser.add_argument("--authority-ref", help="Opaque operator approval identifier, not a credential")
    args = parser.parse_args(argv)
    try:
        plan = preflight(args)
        result = execute(plan, authority_ref=args.authority_ref) if args.execute else plan.public()
    except (ValueError, FileNotFoundError, FileExistsError, OSError,
            subprocess.CalledProcessError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
