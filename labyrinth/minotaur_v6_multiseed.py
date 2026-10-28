"""Source-first replicated spectral multiplier study; negative results stay negative.

This module records variation across genuine independent model seeds and
pair-grouped heldout splits. It does not use heldout metrics for training
or select/promote a model from these public tests.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


@dataclass(frozen=True)
class SeedSweepConfig:
    seeds: tuple[int, ...] = tuple(range(20261028, 20261038))
    rank: int = 192
    epochs: int = 160
    holdout_fraction: float = .2
    max_optimizer_seconds: float = 85.0

    def validate(self):
        if (not isinstance(self.seeds, tuple) or not 1 <= len(self.seeds) <= 10
                or len(self.seeds) != len(set(self.seeds))
                or any(type(x) is not int or not 0 <= x < 2**31 for x in self.seeds)
                or type(self.rank) is not int or not 1 <= self.rank <= 192
                or type(self.epochs) is not int or not 1 <= self.epochs <= 500
                or type(self.holdout_fraction) is not float
                or not .05 <= self.holdout_fraction <= .40
                or type(self.max_optimizer_seconds) is not float
                or not 1.0 <= self.max_optimizer_seconds <= 90.0):
            raise ValueError("Invalid bounded V6 seed sweep")
        return self


def summarize_runs(runs):
    if not isinstance(runs, (tuple, list)) or len(runs) != 10:
        raise ValueError("Exact 10 runs required for V6 research qualification")
    seen = set()
    scores = []
    records = []
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("config"), dict):
            raise ValueError("Missing source report")
        seed = run["config"].get("seed")
        if type(seed) is not int or seed in seen:
            raise ValueError("Duplicate or invalid seed")
        seen.add(seed)
        try:
            candidate = run["candidate"]["holdout"]
            constant = run["controls"]["constant"]["holdout"]
            frozen = run["controls"]["frozen_random"]["holdout"]
            train = run["candidate"]["train"]
            steps = run["training"]["optimizer_steps"]
            initial = run["training"]["initial_weight_sha256"]
            after = run["training"]["trained_weight_sha256"]
            held = run["split"]["holdout_count"]
            params = run["model"]["parameter_count"]
            reversal = run["split"]["reversals_grouped"]
        except (KeyError, TypeError):
            raise ValueError("Incomplete run evidence") from None
        accuracy = candidate.get("accuracy")
        if (not isinstance(accuracy, (float, int)) or isinstance(accuracy, bool)
                or not 0 <= accuracy <= 1
                or not isinstance(held, int) or held <= 0
                or candidate.get("count") != held or
                not reversal or type(steps) is not int or steps <= 0
                or not isinstance(params, int) or params <= 0 or params >= 150000
                or not initial or not after or initial == after):
            raise ValueError("Invalid trained holdout evidence")
        for control in (constant, frozen):
            val = control.get("accuracy")
            if (type(val) not in (float, int) or not 0 <= val <= 1
                    or accuracy < val):
                raise ValueError("Run underperforms a control or control invalid")
        records.append({"seed":seed,"heldout_cases":held,
                        "heldout_accuracy":float(accuracy),
                        "frozen_accuracy":float(frozen["accuracy"]),
                        "constant_accuracy":float(constant["accuracy"]),
                        "train_accuracy":float(train["accuracy"]),
                        "optimizer_steps":steps})
        scores.append(float(accuracy))
    records.sort(key=lambda row: row["seed"])
    worst = min(scores)
    passing = sum(x >= .95 for x in scores)
    return {"runs": records, "run_count":len(records),
            "mean_seed_accuracy":statistics.mean(scores),
            "median_seed_accuracy":statistics.median(scores),
            "worst_seed_accuracy":worst,
            "best_seed_accuracy":max(scores),
            "passing_seeds":passing,
            "criterion_95pct_on_9_of_10_seeds":passing >= 9 and worst >= .90,
            "promotion_recommendation":"DO_NOT_PROMOTE",
            "qualification":"NOT_ESTABLISHED",
            "grouped_case_confidence":"No IID inference; reversals in each unordered group share supervision."}


def run_sweep(config=SeedSweepConfig()):
    config.validate()
    from .minotaur_v6_spectral import SpectralConfig, run_experiment
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise RuntimeError("Refuse experiments on uncommitted source")
    commit = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    runs=[]
    for seed in config.seeds:
        report=run_experiment(SpectralConfig(
            seed=seed, rank=config.rank,epochs=config.epochs,
            holdout_fraction=config.holdout_fraction,
            max_optimizer_seconds=config.max_optimizer_seconds))
        runs.append(report)
        print("FINISHED_SEED",seed,"ACCURACY",report["candidate"]["holdout"]["accuracy"],flush=True)
    aggregate=summarize_runs(runs) if len(runs)==10 else {"status":"diagnostic only; not ten seeds","qualification":"NOT_ESTABLISHED"}
    return {"schema":"labyrinth.minotaur.v6-ten-seed-replication.v1",
            "source":{"commit":commit,"clean":True,
                      "module_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "config":asdict(config), "aggregate":aggregate,"individual_runs":runs,
            "qualification":"NOT_ESTABLISHED", "promotion_recommendation":"DO_NOT_PROMOTE",
            "limitations":["Public seeded grouped-pair holdouts are NOT a private independent exam.",
                           "No withheld-value, absent-output or modulus-101 transfer qualified by this sweep.",
                           "Training and independent external arithmetic scoring are separate pending states.",
                           "No model weights promoted or trainee graduated."]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--epochs",type=int,default=160)
    parser.add_argument("--rank",type=int,default=192)
    args=parser.parse_args()
    dest=Path(args.output)
    if dest.exists():parser.error("refusing to overwrite report")
    report=run_sweep(SeedSweepConfig(rank=args.rank,epochs=args.epochs))
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("RESULT",report["aggregate"]["mean_seed_accuracy"],"CRITERION",report["aggregate"]["criterion_95pct_on_9_of_10_seeds"],"SOURCE",report["source"]["commit"])


if __name__=="__main__":
    main()
