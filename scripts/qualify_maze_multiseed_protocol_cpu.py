"""Opt-in, bounded multi-seed CPU orchestration of the public Maze/MiniLM probe.

The backend currently provides correct-label, zero-*update* and inverted-label
arms. It does NOT implement a zero-learning-rate optimizer-step control or a
randomized-label arm; requests for those controls are rejected. Public
development and transfer sets are disclosed, and the same-skill cycle recheck
is not independently controlled prior-skill retention. No graduation claim.

The default dry run is read-only. --execute starts bounded, sequential child
experiments only under an existing caller-selected isolated output parent.
Physical resource reservations, independent custody and remote fencing are
NOT established by this source-level driver.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from labyrinth.training_scope import TrainingScopeSpec
from scripts.qualify_maze_conditioned_minilm_cpu import (
    SCHEMA as CHILD_SCHEMA, SKILLS, build_training_cycles, default_scope,
    select_training_batch,
)
from scripts.qualify_minilm_cpu import (
    ARMS, HELDOUT, MODEL_ID, TRANSFER, _sha_file, plan_output,
)


SCHEMA = "labyrinth.maze-minilm-multiseed-public-protocol.v1"
MIN_SEEDS = 3
MAX_SEEDS = 5
MAX_RUNS = 10
SUPPORTED_ARMS = tuple(ARMS)
PUBLIC_SPLITS = ("public_development", "public_transfer")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _valid_sha(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


@dataclass(frozen=True)
class ProtocolSpec:
    seeds: tuple[int, ...] = (1101, 2202, 3303)
    steps: tuple[int, ...] = (8, 16)
    weak_skill: str = "negative_polarity"
    required_arms: tuple[str, ...] = SUPPORTED_ARMS
    require_protected_retention: bool = False
    learning_rate: float = 0.002

    def __post_init__(self) -> None:
        if (type(self.seeds) is not tuple or not MIN_SEEDS <= len(self.seeds) <= MAX_SEEDS
                or any(type(seed) is not int or not 0 <= seed < 2**31
                       for seed in self.seeds)
                or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError("require 3..5 distinct bounded model/Maze seeds")
        if (type(self.steps) is not tuple or not 1 <= len(self.steps) <= 2
                or any(type(step) is not int or not 1 <= step <= 16
                       for step in self.steps)
                or len(set(self.steps)) != len(self.steps)
                or len(self.seeds) * len(self.steps) > MAX_RUNS):
            raise ValueError("require 1..2 distinct step budgets in [1,16], at most 10 runs")
        if type(self.weak_skill) is not str or self.weak_skill not in SKILLS:
            raise ValueError("weakness hint must name one approved TRAIN skill")
        if type(self.required_arms) is not tuple or self.required_arms != SUPPORTED_ARMS:
            raise ValueError(
                "backend lacks requested controls; supports correct, zero_update "
                "(skips optimizer.step), inverted ONLY; zero_lr and randomized "
                "labels are NOT implemented"
            )
        if type(self.require_protected_retention) is not bool:
            raise ValueError("require_protected_retention must be bool")
        if self.require_protected_retention:
            raise ValueError("backend has no independently held prior-skill retention")
        if (type(self.learning_rate) not in (int, float)
                or not 0 < self.learning_rate <= 0.01):
            raise ValueError("learning rate must be finite and in (0, .01]")


def build_plan(spec: ProtocolSpec) -> dict[str, Any]:
    spec.__post_init__()
    runner_source = Path(__file__).resolve().with_name(
        "qualify_maze_conditioned_minilm_cpu.py")
    if not runner_source.is_file():
        raise ValueError("approved Maze/MiniLM source is missing")
    jobs = []
    for seed in spec.seeds:
        scope = replace(default_scope(), seed=seed)
        cycles = build_training_cycles(scope, weak_cycle_one=(spec.weak_skill,))
        for step in spec.steps:
            jobs.append({
                "run_name": f"seed-{seed:010d}-steps-{step:02d}",
                "seed": seed, "steps_per_cycle": step,
                "scope": scope.to_dict(), "scope_fingerprint": scope.fingerprint,
                "cycle_room_ids": [
                    [row.room_id for row in cycle.examples] for cycle in cycles
                ],
                "cycle_exposure_ids": [
                    [row.exposure_id for row in cycle.examples] for cycle in cycles
                ],
            })
    rows = {
        "schema": SCHEMA,
        "model_id": MODEL_ID,
        "model_origin": "CACHED_PRETRAINED_MODEL_ID_NOT_PUBLISHER_ATTESTED",
        "runner_sha256": _sha_file(runner_source),
        "protocol_sha256": _sha_file(Path(__file__)),
        "seeds": list(spec.seeds), "step_budgets": list(spec.steps),
        "weakness_provenance": "OPERATOR_DECLARED_NOT_MEASURED",
        "weak_skill": spec.weak_skill,
        "learning_rate": float(spec.learning_rate),
        "arms": list(spec.required_arms),
        "control_semantics": {
            "correct": "TRAIN_LABELS",
            "zero_update": "MATCHED_FORWARD_BACKWARD_SKIPS_OPTIMIZER_STEP",
            "inverted": "DETERMINISTIC_BINARY_LABEL_INVERSION",
            "zero_lr_optimizer": "UNSUPPORTED_FAIL_CLOSED",
            "randomized_labels": "UNSUPPORTED_FAIL_CLOSED",
        },
        "public_fixture_sha256": {
            "development": _digest(HELDOUT),
            "transfer": _digest(TRANSFER),
        },
        "public_splits": list(PUBLIC_SPLITS),
        "prior_skill_retention": "NOT_TESTED_SAME_SKILL_PUBLIC_RECHECK_ONLY",
        "private_final_exam": "NOT_RUN",
        "independent_custody": "NOT_ESTABLISHED",
        "crash_recovery": "NOT_TESTED",
        "external_precommit_timestamp": "NONE_LOCAL_HASH_ONLY",
        "graduation_authorized": False,
        "jobs": jobs,
    }
    return {"manifest": rows, "manifest_sha256": _digest(rows)}


def child_command(job: dict[str, Any], root: Path, spec: ProtocolSpec) -> list[str]:
    spec.__post_init__()
    if (type(job) is not dict or job.get("seed") not in spec.seeds
            or job.get("steps_per_cycle") not in spec.steps
            or type(job.get("run_name")) is not str
            or job["run_name"] !=
            f"seed-{job['seed']:010d}-steps-{job['steps_per_cycle']:02d}"):
        raise ValueError("unbound multi-seed child job")
    return [
        sys.executable, "-m", "scripts.qualify_maze_conditioned_minilm_cpu",
        "--output-parent", str(root / "runs"),
        "--run-name", job["run_name"],
        "--scope-file", str(root / "scopes" / f"{job['seed']}.json"),
        "--weak-cycle-one", spec.weak_skill,
        "--steps-per-cycle", str(job["steps_per_cycle"]),
        "--learning-rate", str(spec.learning_rate), "--execute",
    ]


def _accuracy(value: Any) -> float:
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError("invalid public development score")
    return float(value)


def audit_child(job: dict[str, Any], plan: dict[str, Any], run_root: Path) -> dict[str, Any]:
    """Independent file-hash and source-report consistency audit; NOT an exam."""
    manifest = plan["manifest"]
    report_path = run_root / "report.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    if (type(data) is not dict or data.get("schema") != CHILD_SCHEMA
            or data.get("harness_sha256") != manifest["runner_sha256"]
            or data.get("model_id") != manifest["model_id"]
            or data.get("scope") != job["scope"]
            or data.get("scope_fingerprint") != job["scope_fingerprint"]
            or data.get("steps_per_cycle") != job["steps_per_cycle"]
            or data.get("weakness_feedback_provenance") !=
            "OPERATOR_DECLARED_NOT_MEASURED"
            or data.get("private_final_exam") != "NOT_RUN"
            or data.get("graduation_authorized") is not False
            or data.get("crash_recovery") != "NOT_TESTED"):
        raise ValueError("child report mismatches precommitted public source protocol")
    if (type(data.get("arms")) is not dict
            or set(data["arms"]) != set(SUPPORTED_ARMS)
            or not _valid_sha(data.get("pretrained_base_sha256"))):
        raise ValueError("missing matched neural arms or frozen-base commitment")
    scope = TrainingScopeSpec.from_dict(job["scope"])
    cycles = build_training_cycles(scope, weak_cycle_one=(manifest["weak_skill"],))
    for arm in SUPPORTED_ARMS:
        payload = data["arms"][arm]
        readback = payload.get("final_checkpoint_readback") if type(payload) is dict else None
        history = payload.get("cycles") if type(payload) is dict else None
        if (type(payload) is not dict
                or type(readback) is not dict
                or readback.get("status")
                != "SELF_REPORTED_FRESH_INSTANCE_PASS"
                or type(history) is not list or len(history) != 2):
            raise ValueError("missing fresh-instance readback or two training cycles")
        if (not _valid_sha(payload.get("initial_trainable_sha256"))
                or type(history[0]) is not dict or type(history[1]) is not dict
                or payload["initial_trainable_sha256"] !=
                history[0].get("trainable_sha256_before")):
            raise ValueError("initial trainable commitment does not bind cycle zero")
        if (not _valid_sha(payload.get("initial_lora_sha256"))
                or payload["initial_lora_sha256"] !=
                history[0].get("lora_sha256_before")):
            raise ValueError("initial LoRA commitment does not bind cycle zero")
        for index, cycle in enumerate(cycles):
            observed = history[index]
            if any(not _valid_sha(observed.get(key)) for key in (
                "pretrained_base_sha256", "frozen_peft_sha256_before",
                "frozen_peft_sha256_after", "trainable_sha256_before",
                "trainable_sha256_after", "lora_sha256_before",
                "lora_sha256_after",
            )):
                raise ValueError("invalid neural tensor commitment SHA256")
            expected_visits = [
                row.exposure_id
                for step in range(job["steps_per_cycle"])
                for row in select_training_batch(cycle, step)
            ]
            if (observed.get("cycle") != index
                    or observed.get("maze_room_ids") != job["cycle_room_ids"][index]
                    or observed.get("maze_exposure_ids") != job["cycle_exposure_ids"][index]
                    or observed.get("training_visit_exposure_ids") != expected_visits
                    or observed.get("training_visits") != len(expected_visits)
                    or observed.get("optimizer_updates") !=
                    (0 if arm == "zero_update" else job["steps_per_cycle"])
                    or observed.get("pretrained_base_sha256") !=
                    data["pretrained_base_sha256"]
                    or observed.get("frozen_peft_sha256_before") !=
                    observed.get("frozen_peft_sha256_after")):
                raise ValueError("neural arm is not bound to actual Maze exposures")
            for relpath, key in (
                ("adapter/adapter_model.safetensors", "adapter_file_sha256"),
                ("head.safetensors", "head_file_sha256"),
            ):
                expected_sha = observed.get(key)
                path = run_root / arm / f"cycle-{index}" / relpath
                if not _valid_sha(expected_sha) or _sha_file(path) != expected_sha:
                    raise ValueError("checkpoint file integrity mismatch")
            _accuracy(observed.get("public_development_accuracy"))
            _accuracy(observed.get("public_transfer_accuracy"))
        if (history[0].get("trainable_sha256_after") !=
                history[1].get("trainable_sha256_before")
                or history[0].get("frozen_peft_sha256_after") !=
                history[1].get("frozen_peft_sha256_before")):
            raise ValueError("regeneration lost learner or frozen-backbone state")
        if history[0]["lora_sha256_after"] != history[1]["lora_sha256_before"]:
            raise ValueError("regeneration lost LoRA adapter state")
        if arm == "zero_update":
            if any(row.get("trainable_sha256_before") !=
                   row.get("trainable_sha256_after") or
                   row.get("lora_sha256_before") !=
                   row.get("lora_sha256_after") for row in history):
                raise ValueError("zero-update changed parameters")
        elif any(row.get("trainable_sha256_before") ==
                 row.get("trainable_sha256_after") or
                 row.get("lora_sha256_before") ==
                 row.get("lora_sha256_after") for row in history):
            raise ValueError("updated neural arm retained identical LoRA or trainable parameters")
    for index in range(2):
        visits = {
            tuple(data["arms"][arm]["cycles"][index]["training_visit_exposure_ids"])
            for arm in SUPPORTED_ARMS
        }
        if len(visits) != 1:
            raise ValueError("control arms saw different training visits")
    initials = {data["arms"][arm]["initial_trainable_sha256"] for arm in SUPPORTED_ARMS}
    if len(initials) != 1:
        raise ValueError("control arms were not initialized identically")
    initial_lora = {data["arms"][arm]["initial_lora_sha256"] for arm in SUPPORTED_ARMS}
    if len(initial_lora) != 1:
        raise ValueError("control LoRA arms were not initialized identically")
    final = {
        arm: {
            "public_development": _accuracy(
                data["arms"][arm]["cycles"][-1]["public_development_accuracy"]),
            "public_transfer": _accuracy(
                data["arms"][arm]["cycles"][-1]["public_transfer_accuracy"]),
            "same_skill_public_cycle_delta": (
                _accuracy(data["arms"][arm]["cycles"][1]["public_development_accuracy"])
                - _accuracy(data["arms"][arm]["cycles"][0]["public_development_accuracy"])
            ),
        } for arm in SUPPORTED_ARMS
    }
    return {
        "run_name": job["run_name"], "seed": job["seed"],
        "steps_per_cycle": job["steps_per_cycle"],
        "pretrained_base_sha256": data["pretrained_base_sha256"],
        "report_sha256": _sha_file(report_path),
        "checkpoint_files_checked": 12,
        "physical_execution": "CHILD_SELF_REPORTED_UNATTESTED",
        "scores": final,
        "public_correct_advantage_both": {
            split: final["correct"][split] > final["zero_update"][split]
            and final["correct"][split] > final["inverted"][split]
            for split in PUBLIC_SPLITS
        },
        "independent_retention": "NOT_RUN",
        "graduation_authorized": False,
    }


def summarize_public_controls(
    rows: list[dict[str, Any]], spec: ProtocolSpec,
) -> dict[str, Any]:
    """Paired public-only summaries; no p-values or independent qualifications."""
    spec.__post_init__()
    if type(rows) is not list or len(rows) != len(spec.seeds) * len(spec.steps):
        raise ValueError("incomplete multiseed result matrix")
    expected = {(seed, steps) for seed in spec.seeds for steps in spec.steps}
    observed = {(row["seed"], row["steps_per_cycle"]) for row in rows}
    if len(observed) != len(rows) or observed != expected:
        raise ValueError("missing or duplicate seed/step arms")
    bases = {row.get("pretrained_base_sha256") for row in rows}
    if len(bases) != 1 or not all(_valid_sha(base) for base in bases):
        raise ValueError("pretrained base changed between seed experiments")
    result = {}
    for budget in spec.steps:
        group = [row for row in rows if row["steps_per_cycle"] == budget]
        split_summary = {}
        for split in PUBLIC_SPLITS:
            def value(row: dict[str, Any], arm: str) -> float:
                return _accuracy(row["scores"][arm][split])
            split_summary[split] = {
                "correct_mean": sum(value(row, "correct") for row in group) / len(group),
                "zero_update_mean": sum(value(row, "zero_update") for row in group) / len(group),
                "inverted_mean": sum(value(row, "inverted") for row in group) / len(group),
                "mean_correct_minus_zero": sum(
                    value(row, "correct") - value(row, "zero_update")
                    for row in group
                ) / len(group),
                "positive_seed_count": sum(
                    value(row, "correct") > value(row, "zero_update")
                    and value(row, "correct") > value(row, "inverted")
                    for row in group
                ),
                "seeds": len(group),
            }
        result[str(budget)] = split_summary
    return {
        "fixed_public_splits": result,
        "pretrained_base_sha256": next(iter(bases)),
        "qualification": "PUBLIC_DEVELOPMENT_AGGREGATES_NOT_INDEPENDENT_TRANSFER",
    }


def execute_plan(
    spec: ProtocolSpec, output_parent: str | Path, run_name: str,
    *, timeout_seconds_per_run: int = 900,
) -> dict[str, Any]:
    """Execute only with explicit opt-in. No cleanup/retry on partial failure."""
    if (type(timeout_seconds_per_run) is not int or
            not 60 <= timeout_seconds_per_run <= 1800):
        raise ValueError("bounded child timeout must be 60..1800 seconds")
    plan = build_plan(spec)
    root = plan_output(output_parent, run_name)
    # Exclusive creation; output is preserved after any failure for recovery.
    root.mkdir(mode=0o700)
    (root / "runs").mkdir()
    (root / "scopes").mkdir()
    (root / "logs").mkdir()
    (root / "plan.json").write_bytes(_canonical(plan))
    rows = []
    for job in plan["manifest"]["jobs"]:
        (root / "scopes" / f"{job['seed']}.json").write_bytes(
            _canonical(job["scope"]))
        command = child_command(job, root, spec)
        try:
            result = subprocess.run(
                command, cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=timeout_seconds_per_run, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            (root / "failure.json").write_bytes(_canonical({
                "run_name": job["run_name"], "reason": "CHILD_TIMEOUT",
                "graduation_authorized": False,
            }))
            raise RuntimeError("bounded neural child timed out; artifacts preserved") from exc
        (root / "logs" / f"{job['run_name']}.stdout.txt").write_text(
            result.stdout[-65536:], encoding="utf-8")
        (root / "logs" / f"{job['run_name']}.stderr.txt").write_text(
            result.stderr[-65536:], encoding="utf-8")
        if result.returncode != 0:
            (root / "failure.json").write_bytes(_canonical({
                "run_name": job["run_name"], "reason": "CHILD_FAILED",
                "exit_code": result.returncode, "graduation_authorized": False,
            }))
            raise RuntimeError("neural child failed; inspect isolated logs")
        try:
            rows.append(audit_child(job, plan, root / "runs" / job["run_name"]))
        except Exception as exc:
            # A successful process exit is not a successful neural audit. Preserve
            # a machine-readable, fail-closed receipt for malformed and missing
            # reports as well as integrity/lineage failures. Source-only custody.
            (root / "failure.json").write_bytes(_canonical({
                "run_name": job["run_name"],
                "reason": "CHILD_AUDIT_FAILED",
                "error_type": type(exc).__name__,
                "complete_jobs_before_failure": len(rows),
                "plan_sha256": _digest(plan),
                "graduation_authorized": False,
            }))
            raise RuntimeError("neural child audit failed; artifacts preserved") from exc
    result = {
        "schema": SCHEMA, "plan_sha256": _digest(plan),
        "qualification": "PUBLIC_MULTISEED_DEVELOPMENT_ONLY",
        "results": rows, "complete_jobs": len(rows),
        "public_aggregate": summarize_public_controls(rows, spec),
        "private_final_exam": "NOT_RUN", "prior_skill_retention": "NOT_RUN",
        "independent_model_and_evaluator_custody": "NOT_ESTABLISHED",
        "crash_resume": "NOT_SUPPORTED",
        "graduation_authorized": False,
    }
    (root / "result.json").write_bytes(_canonical(result))
    return result


def _csv_ints(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in raw.split(","))
    except ValueError as exc:
        raise ValueError("expected comma-delimited integers") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seeds", default="1101,2202,3303")
    parser.add_argument("--steps", default="8,16")
    parser.add_argument("--weak-skill", default="negative_polarity")
    parser.add_argument("--required-arms", default="correct,zero_update,inverted")
    parser.add_argument("--require-protected-retention", action="store_true")
    parser.add_argument("--timeout-seconds-per-run", type=int, default=900)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        spec = ProtocolSpec(
            seeds=_csv_ints(args.seeds), steps=_csv_ints(args.steps),
            weak_skill=args.weak_skill,
            required_arms=tuple(args.required_arms.split(",")),
            require_protected_retention=args.require_protected_retention,
        )
        plan = build_plan(spec)
        root = plan_output(args.output_parent, args.run_name)
        if not args.execute:
            print(json.dumps({
                "execute": False, "run_root": str(root), **plan,
            }, sort_keys=True))
            return 0
        result = execute_plan(
            spec, args.output_parent, args.run_name,
            timeout_seconds_per_run=args.timeout_seconds_per_run,
        )
        print(json.dumps({
            "execute": True, "run_root": str(root),
            "complete_jobs": result["complete_jobs"],
            "qualification": result["qualification"],
            "graduation_authorized": False,
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"preflight or execution failure: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
