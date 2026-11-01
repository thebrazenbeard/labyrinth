"""Bounded *development-only* five-arm MiniLM multi-seed campaign.

This driver uses the existing real PEFT runner in a fresh child process per
seed. It never downloads a model, installs dependencies, selects final-exam
examples, or authorizes graduation. Report and file checks are consistency
checks against self-reported artifacts, NOT operator/host attestation.

Dry-run from the repository root (no files, no neural imports)::

    python -m scripts.experiments.q5_multiseed_campaign --output-parent /tmp \
      --campaign-name q5-example --seeds 1,2,3,4,5,6,7,8

Execution requires an already-cached MiniLM and installed optional Torch,
Transformers, PEFT, and safetensors, as well as a separately authorized host
lease. Never point it at a shared training directory.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from labyrinth.maze_conditioned_minilm_cpu_experiment import (
    PREREG_ARMS, PREREG_SCHEMA, build_training_cycles, default_scope,
    preregistered_randomized_labels, select_training_batch,
    training_batch_labels, validate_cycles,
)
from labyrinth.minilm_cpu_experiment import MODEL_ID, _sha_file, plan_output


SCHEMA = "labyrinth.q5-multiseed-source-campaign.v1"
MAX_SEEDS = 32
SPLITS = ("public_development", "public_transfer")
_HEX = re.compile(r"[0-9a-f]{64}\Z")


class CampaignError(ValueError):
    """A campaign run or its claimed evidence violates a fixed source contract."""


def parse_seeds(raw: str) -> tuple[int, ...]:
    if type(raw) is not str or not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", raw):
        raise CampaignError("seeds must be an exact comma-separated integer list")
    values = tuple(int(token) for token in raw.split(","))
    if (not 8 <= len(values) <= MAX_SEEDS or len(set(values)) != len(values)
            or any(not 0 <= seed < 2**31 for seed in values)):
        raise CampaignError("8..32 distinct signed-31-bit seeds required")
    return values


def _digest(value: Any, field: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise CampaignError(f"invalid {field} digest")
    return value


def _ratio(value: Any, field: str) -> float:
    # Check the exact numeric range before math.isfinite converts integers to
    # floating point: attacker-supplied 10**309 otherwise raises OverflowError.
    if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
        raise CampaignError(f"invalid {field} accuracy")
    return float(value)


def verify_seed_report(report: Any, *, seed: int, steps: int) -> dict[str, Any]:
    """Recompute five-arm consistency from claimed receipts, never trust a PASS string."""
    if (type(report) is not dict or report.get("schema") != PREREG_SCHEMA
            or report.get("qualification") != "PUBLIC_DEVELOPMENT_ONLY"
            or report.get("graduation_authorized") is not False
            or report.get("private_final_exam") != "NOT_RUN"
            or report.get("control_arms") != list(PREREG_ARMS)
            or report.get("steps_per_cycle") != steps
            or report.get("model_id") != MODEL_ID):
        raise CampaignError("missing strict public five-arm report identity")
    expected_scope = replace(default_scope(), seed=seed)
    if (report.get("scope") != expected_scope.to_dict()
            or report.get("scope_fingerprint") != expected_scope.fingerprint
            or report.get("cycle_one_weak_skill_hint") != []
            or report.get("weakness_feedback_provenance") != "OPERATOR_DECLARED_NOT_MEASURED"):
        raise CampaignError("reported scope differs from predeclared seed")
    expected_cycles = build_training_cycles(expected_scope)
    base = _digest(report.get("pretrained_base_sha256"), "pretrained base")
    harness = _digest(report.get("harness_sha256"), "harness")
    arms = report.get("arms")
    if type(arms) is not dict or set(arms) != set(PREREG_ARMS):
        raise CampaignError("missing or excess five-arm outcome")
    first_trainable: set[str] = set()
    first_adapter: set[str] = set()
    exposure_rows: list[set[tuple[str, ...]]] = [set(), set()]
    scores = {split: {} for split in SPLITS}
    for arm in PREREG_ARMS:
        payload = arms[arm]
        if type(payload) is not dict or type(payload.get("cycles")) is not list:
            raise CampaignError("invalid arm checkpoint history")
        cycles = payload["cycles"]
        if len(cycles) != 2:
            raise CampaignError("regenerative run requires exactly two cycles")
        first_trainable.add(_digest(payload.get("initial_trainable_sha256"), "initial trainable"))
        first_adapter.add(_digest(payload.get("initial_lora_sha256"), "initial LoRA"))
        previous = None
        previous_lora = None
        for index, cycle in enumerate(cycles):
            if type(cycle) is not dict or cycle.get("cycle") != index:
                raise CampaignError("cycle identity or ordering changed")
            start = _digest(cycle.get("trainable_sha256_before"), "cycle before")
            after = _digest(cycle.get("trainable_sha256_after"), "cycle after")
            lora_start = _digest(cycle.get("lora_sha256_before"), "LoRA before")
            lora_after = _digest(cycle.get("lora_sha256_after"), "LoRA after")
            _digest(cycle.get("adapter_file_sha256"), "adapter file")
            _digest(cycle.get("head_file_sha256"), "head file")
            if (start != (previous if previous is not None
                          else payload["initial_trainable_sha256"]) or
                    lora_start != (previous_lora if previous_lora is not None
                                   else payload["initial_lora_sha256"]) or
                    cycle.get("pretrained_base_sha256") != base):
                raise CampaignError("pretrained base or learner checkpoint changed across regeneration")
            if (arm in ("zero_update", "zero_lr") and
                    (start != after or lora_start != lora_after)
                    or arm not in ("zero_update", "zero_lr") and start == after):
                raise CampaignError("trained and sham arms violate weight-update invariant")
            visits = cycle.get("training_visit_exposure_ids")
            actual_cycle = expected_cycles[index]
            expected_visits = [row.exposure_id for step in range(steps)
                               for row in select_training_batch(actual_cycle, step)]
            if (type(visits) is not list or len(visits) != 4 * steps
                    or any(type(value) is not str or not _HEX.fullmatch(value)
                           for value in visits) or visits != expected_visits):
                raise CampaignError("invalid or noncanonical ordered TRAIN exposures")
            randomized = (preregistered_randomized_labels(actual_cycle, seed)
                          if arm == "randomized_labels" else None)
            expected_labels = training_batch_labels(actual_cycle.examples, arm,
                                                    randomized)
            expected_assignment_sha = sha256(json.dumps({
                row.exposure_id: assigned for row, assigned in zip(
                    actual_cycle.examples, expected_labels)
            }, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
            if cycle.get("label_assignment_sha256") != expected_assignment_sha:
                raise CampaignError("control labels differ from canonical TRAIN assignments")
            exposure_rows[index].add(tuple(visits))
            if (cycle.get("forward_steps") != steps
                    or cycle.get("backward_steps") != steps
                    or cycle.get("optimizer_step_invocations")
                    != (0 if arm == "zero_update" else steps)
                    or cycle.get("training_visits") != 4 * steps):
                raise CampaignError("control compute/exposure mismatch")
            previous = after
            previous_lora = lora_after
        for split in SPLITS:
            scores[split][arm] = _ratio(
                cycles[-1].get(split + "_accuracy"), split + "/" + arm)
    if len(first_trainable) != 1 or len(first_adapter) != 1:
        raise CampaignError("five arms did not share exact initial state")
    if any(len(exposures) != 1 for exposures in exposure_rows):
        raise CampaignError("five arms received different ordered TRAIN examples")
    return {"seed": seed, "base_model_sha256": base, "harness_sha256": harness,
            "initial_trainable_sha256": next(iter(first_trainable)),
            "scores": scores, "scope_fingerprint": expected_scope.fingerprint}


def verify_checkpoint_files(report: dict[str, Any], run_dir: Path) -> None:
    """Read actual saved bytes; still not independent model execution custody."""
    for arm in PREREG_ARMS:
        for index, cycle in enumerate(report["arms"][arm]["cycles"]):
            root = run_dir / arm / f"cycle-{index}"
            for relative, field in (("adapter/adapter_model.safetensors", "adapter_file_sha256"),
                                    ("head.safetensors", "head_file_sha256")):
                path = root / relative
                if (not path.is_file() or any(part.is_symlink()
                                             for part in (run_dir, run_dir / arm, root,
                                                          path.parent, path))
                        or _sha_file(path) != cycle[field]):
                    raise CampaignError(f"{arm} cycle {index} saved {field} does not match bytes")


def read_report_json(path: Path) -> dict[str, Any]:
    """Reject duplicate fields and nonfinite claims before validation."""
    if (not path.is_file() or path.is_symlink()
            or not 0 < path.stat().st_size <= 8 * 1024 * 1024):
        raise CampaignError("missing, linked, empty or oversized report")

    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for key, value in pairs:
            if key in fields:
                raise CampaignError("duplicate JSON report field")
            fields[key] = value
        return fields

    def reject_constant(value: str) -> Any:
        raise CampaignError(f"nonfinite report number: {value}")

    try:
        report = json.loads(path.read_bytes().decode("utf-8"),
                            object_pairs_hook=unique_fields,
                            parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignError("invalid report JSON") from exc
    if type(report) is not dict:
        raise CampaignError("report must be a JSON object")
    return report


def aggregate(reports: list[dict[str, Any]], seeds: tuple[int, ...]) -> dict[str, Any]:
    if len(reports) != len(seeds) or len(set(seeds)) != len(seeds) or len(seeds) < 8:
        raise CampaignError("eight complete independent-seed source reports required")
    if tuple(row["seed"] for row in reports) != seeds:
        raise CampaignError("reordered/missing seed report")
    if (len({r["base_model_sha256"] for r in reports}) != 1
            or len({r["harness_sha256"] for r in reports}) != 1):
        raise CampaignError("source or base model changed between runs")
    comparisons = {}
    for split in SPLITS:
        differences = {
            arm: [round(r["scores"][split]["correct"] - r["scores"][split][arm], 8)
                  for r in reports]
            for arm in PREREG_ARMS if arm != "correct"
        }
        comparisons[split] = {
            arm: {"per_seed_correct_minus_control": deltas,
                  "positive_seed_count": sum(value > 0 for value in deltas),
                  "all_seed_deltas_at_least_0_10": all(value >= 0.10 for value in deltas)}
            for arm, deltas in differences.items()
        }
    return {
        "schema": SCHEMA, "seed_order": list(seeds), "seed_count": len(seeds),
        "base_model_sha256": reports[0]["base_model_sha256"],
        "harness_sha256": reports[0]["harness_sha256"],
        "comparisons": comparisons,
        "source_consistency": "SELF_REPORTED_ARTIFACTS_CHECKED_ONLY",
        "development_screen_only": True, "independent_final_exam": "NOT_RUN",
        "pretrained_origin_attested": False, "operator_custody_attested": False,
        "independent_learning_qualified": False, "graduation_authorized": False,
    }


def campaign(seeds: tuple[int, ...], *, output_parent: Path, name: str,
             steps: int, timeout: int, execute: bool) -> dict[str, Any]:
    if type(steps) is not int or not 4 <= steps <= 16:
        raise CampaignError("five-arm protocol requires 4..16 optimizer steps per cycle")
    if type(timeout) is not int or not 60 <= timeout <= 86400:
        raise CampaignError("bounded per-seed timeout must be 60..86400 seconds")
    root = plan_output(output_parent, name)
    plans = []
    for seed in seeds:
        scope = replace(default_scope(), seed=seed)
        validation = validate_cycles(build_training_cycles(scope))
        plans.append({"seed": seed, "scope_fingerprint": scope.fingerprint,
                      "generated_train_cases": validation["unique_training_prompts"]})
    if not execute:
        return {"schema": SCHEMA, "execute": False,
                "planned_root": str(root), "plans": plans,
                "model_id": MODEL_ID, "independent_learning_qualified": False}
    root.mkdir(mode=0o700)
    results = []
    traces = []
    for seed in seeds:
        scope = replace(default_scope(), seed=seed)
        scope_file = root / f"scope-{seed}.json"
        scope_file.write_text(json.dumps(scope.to_dict(), sort_keys=True) + "\n", encoding="ascii")
        run_name = f"seed-{seed}"
        args = [sys.executable, "-m", "labyrinth.maze_conditioned_minilm_cpu_experiment",
                "--output-parent", str(root), "--run-name", run_name,
                "--scope-file", str(scope_file), "--steps-per-cycle", str(steps),
                "--preregistered-controls", "--execute"]
        out_path, err_path = root / f"{run_name}.stdout.txt", root / f"{run_name}.stderr.txt"
        with out_path.open("xb") as stdout, err_path.open("xb") as stderr:
            try:
                completed = subprocess.run(args, stdout=stdout, stderr=stderr,
                                           timeout=timeout, check=False)
            except subprocess.TimeoutExpired as exc:
                traces.append({"seed": seed, "state": "TIMEOUT", "timeout_seconds": timeout})
                break
        if completed.returncode != 0:
            traces.append({"seed": seed, "state": "CHILD_FAILED",
                           "exit_code": completed.returncode})
            break
        path = root / run_name / "report.json"
        try:
            report = read_report_json(path)
            checked = verify_seed_report(report, seed=seed, steps=steps)
            verify_checkpoint_files(report, root / run_name)
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            traces.append({"seed": seed, "state": "REPORT_INVALID", "error": str(exc)[:240]})
            break
        traces.append({"seed": seed, "state": "SELF_REPORTED_ARTIFACTS_CHECKED",
                       "report_sha256": _sha_file(path)})
        results.append(checked)
    if len(results) == len(seeds):
        outcome = aggregate(results, seeds)
        outcome["state"] = "COMPLETE_SOURCE_CONSISTENCY_ONLY"
    else:
        outcome = {"schema": SCHEMA, "state": "INCOMPLETE", "seed_count": len(seeds),
                   "independent_learning_qualified": False,
                   "graduation_authorized": False}
    outcome.update({"run_root": str(root), "traces": traces,
                    "execute": True, "model_id": MODEL_ID,
                    "physical_source_claim": "UNATTESTED_LOCAL_OPERATOR"})
    (root / "campaign-report.json").write_text(
        json.dumps(outcome, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--campaign-name", required=True)
    parser.add_argument("--seeds", required=True, help="8..32 unique unsigned 31-bit seeds")
    parser.add_argument("--steps-per-cycle", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = campaign(parse_seeds(args.seeds), output_parent=Path(args.output_parent),
                          name=args.campaign_name, steps=args.steps_per_cycle,
                          timeout=args.timeout_seconds, execute=args.execute)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"campaign failed preflight: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("state") != "INCOMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
