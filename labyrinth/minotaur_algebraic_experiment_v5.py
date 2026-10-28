"""Immutable-source V5 compact arithmetic generalization experiment.

A random-seeded low-rank neural predictor sees only 80% of atomic input pairs.
Validation is monitoring-only; the heldout evaluation occurs once after the
fixed training budget and never backpropagates or selects an epoch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .minotaur_algebraic_data_v5 import atomic_split
from .minotaur_algebraic_v5 import AlgebraicConfig, build_algebraic_model
from .minotaur_algebraic_training_v5 import train_atomic_operator
from .minotaur_algebraic_eval_v5 import (
    AtomicEvaluationCase, evaluate_atomic_predictions,
)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20261028
    split_seed: int = 8801
    train_fraction: float = .8
    challenge: str = "none"
    width: int = 48
    rank: int = 20
    steps: int = 800
    batch_size: int = 512
    learning_rate: float = .003
    time_budget: float = 85.0

    def validate(self):
        if (type(self.seed) is not int or type(self.split_seed) is not int
            or min(self.seed,self.split_seed) < 0
            or type(self.train_fraction) is not float
            or not .5 <= self.train_fraction <= .9
            or self.challenge not in ("none","value","diagonal")
            or type(self.steps) is not int or not 1 <= self.steps <= 20000
            or type(self.batch_size) is not int or not 32 <= self.batch_size <= 2048
            or type(self.learning_rate) is not float or not 0 < self.learning_rate <= .02
            or type(self.time_budget) is not float or not 1 <= self.time_budget <= 90):
            raise ValueError("invalid bounded V5 experiment")
        AlgebraicConfig(seed=self.seed,width=self.width,rank=self.rank).validate()
        return self


def checked_split_fingerprints(splits):
    groups={"train":splits.train,"val":splits.val,"holdout":splits.holdout}
    atom_keys={name:{row[:3] for row in rows} for name,rows in groups.items()}
    return {
        "atom_count":sum(len(rows) for rows in groups.values()),
        "counts":{name:len(rows) for name,rows in groups.items()},
        "sha256":dict(splits.commitments),
        "train_test_overlap":len(atom_keys["train"] & atom_keys["holdout"]),
        "dev_test_overlap":len(atom_keys["val"] & atom_keys["holdout"]),
        "train_dev_overlap":len(atom_keys["train"] & atom_keys["val"]),
        "reversal_pair_grouping":"strict",
        "challenge":splits.challenge,
        "withheld_value":splits.withheld_value,
    }


def _predictions(model,rows,torch,batch_size):
    model.eval()
    values=[]
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            chunk=rows[start:start+batch_size]
            op=torch.tensor([x[0] for x in chunk],dtype=torch.long)
            left=torch.tensor([x[1] for x in chunk],dtype=torch.long)
            right=torch.tensor([x[2] for x in chunk],dtype=torch.long)
            values.extend(int(x) for x in model(op,left,right).argmax(1).tolist())
    return values


def run_experiment(config=ExperimentConfig()):
    config.validate()
    import torch
    torch.set_num_threads(min(2,torch.get_num_threads()))
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    clean=subprocess.check_output(["git","status","--porcelain"],text=True).strip()
    if clean: raise RuntimeError("refuse training from uncommitted source")
    source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    splits=atomic_split(config.split_seed,config.train_fraction,challenge=config.challenge)
    split_receipt=checked_split_fingerprints(splits)
    if (split_receipt["atom_count"]!=28227 or
        any(split_receipt[x] for x in ("train_test_overlap","dev_test_overlap","train_dev_overlap"))):
        raise RuntimeError("dataset leakage or incomplete domain")
    model=build_algebraic_model(AlgebraicConfig(
        seed=config.seed,width=config.width,rank=config.rank))
    num_parameters=sum(p.numel() for p in model.parameters())
    if num_parameters>=150000:raise RuntimeError("compact parameter budget exceeded")

    training=train_atomic_operator(
        model,splits.train,splits.val,steps=config.steps,
        batch_size=config.batch_size,lr=config.learning_rate,
        time_budget=config.time_budget,torch=torch)
    preds=_predictions(model,splits.holdout,torch,config.batch_size)
    semantic={opaque:name for opaque,name in zip(splits.op_ids,("ADD","SUB","MUL"))}
    cases=[]
    predictions={}
    for i,(row,guess) in enumerate(zip(splits.holdout,preds)):
        op,x,y,label=row
        key=f"a{i:05d}"
        cases.append(AtomicEvaluationCase(key,semantic[op],x,y,label,
                                          f"r{min(x,y):02d}-{max(x,y):02d}"))
        predictions[key]=guess
    audit=evaluate_atomic_predictions(
        cases,predictions,
        training_atoms=tuple((semantic[op],x,y) for op,x,y,_ in splits.train))
    if (audit["disposition"]!="SCORED" or
        audit["structural_overlap_audit"]["exact_training_pair_overlap"]):
        raise RuntimeError("independent evaluator refused heldout results")
    trained_accuracy=audit["metrics"]["exact_accuracy"]
    by_op={k:v["exact_accuracy"] for k,v in audit["per_operation"].items()}
    qualified=all(value >= .95 for value in by_op.values()) and trained_accuracy>=.95
    return {
        "schema":"labyrinth.minotaur-v5-compact-algebraic-study.v1",
        "source":{"head":source,"clean":True,
                  "module_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "config":asdict(config),
        "model":{"architecture":"low-rank shared residue tensor + periodic features",
                 "parameters":num_parameters,"random_initialization":True,
                 "atomic_pair_table":False},
        "splits":split_receipt,
        "training":training,
        "heldout":audit,
        "predictions_sha256":hashlib.sha256(json.dumps(preds,separators=(",",":")).encode()).hexdigest(),
        "promotion_recommendation":"RESEARCH_ONLY_NO_PROMOTION",
        "criterion_95pct_each_op_met":qualified,
        "qualification":"NOT_ESTABLISHED",
        "graduation_authority":"none",
        "limitations":["Public, deterministic finite-domain labels, not independent private examination.",
                       "No test examples are used for optimizer updates or checkpoint selection.",
                       "Architecture and feature engineering supply modular periodic priors; success is not unassisted mathematical discovery.",
                       "No production model deployed or trainee graduated."],
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--seed",type=int,default=20261028)
    parser.add_argument("--challenge",choices=["none","value","diagonal"],default="none")
    parser.add_argument("--steps",type=int,default=800)
    parser.add_argument("--time-budget",type=float,default=85.)
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():parser.error("refuse overwrite evidence")
    report=run_experiment(ExperimentConfig(
        seed=args.seed,challenge=args.challenge,steps=args.steps,
        time_budget=float(args.time_budget)))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("SOURCE",report["source"]["head"])
    print("UPDATES",report["training"]["optimizer_updates"],"TIMED_OUT",report["training"]["timed_out"])
    print("HELDOUT",report["heldout"]["metrics"],"BY_OP",report["heldout"]["per_operation"])
    print("PARAMS",report["model"]["parameters"],"SUCCESS_CRITERION",report["criterion_95pct_each_op_met"])


if __name__=="__main__":
    main()
