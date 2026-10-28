"""Precommitted hybrid experiment: train ADD only, compose SUB/MUL algorithms.

This is *explicitly neuro-symbolic*: subtraction/multiplication algorithms and
modular negation are architectural priors, not rules discovered by the model.
The neural network learns an atomic addition predictor from incomplete pairs.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .minotaur_algebraic_data_v5 import atomic_split
from .minotaur_algebraic_training_v5 import train_atomic_operator
from .minotaur_algebraic_composer_v5 import predict_composed_operators
from .minotaur_algebraic_eval_v5 import AtomicEvaluationCase, evaluate_atomic_predictions


@dataclass(frozen=True)
class CompositionStudyConfig:
    seed: int = 20261028
    split_seed: int = 9911
    challenge: str = "none"
    steps: int = 1200
    batch_size: int = 384
    lr: float = .003
    time_budget: float = 85.
    width: int = 48
    rank: int = 20

    def validate(self):
        if (type(self.seed) is not int or self.seed<0
            or type(self.split_seed) is not int or self.split_seed<0
            or self.challenge not in ("none","value","diagonal")
            or type(self.steps) is not int or not 1 <= self.steps <= 5000
            or type(self.batch_size) is not int or not 32<=self.batch_size<=2048
            or type(self.lr) is not float or not 0<self.lr<=.02
            or type(self.time_budget) is not float or not 1<=self.time_budget<=90):
            raise ValueError("bounded hybrid composition config required")
        if (type(self.width) is not int or type(self.rank) is not int or
                self.width < 1 or self.rank < 1 or
                6*self.width*self.width+9*self.width*self.rank+202*self.width+98 >= 150000):
            raise ValueError('compact model parameter budget exceeded')
        return self


def _model_visible_rows(rows, op_add):
    """Retain one opcode; no heldout rows/answers cross into the trainer."""
    return tuple(row for row in rows if row[0] == op_add)


def _evaluate(model,rows,op_ids,torch,batch_size,train_rows):
    semantic={opaque:name for opaque,name in zip(op_ids,("ADD","SUB","MUL"))}
    predictions=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size]
            op=torch.tensor([r[0] for r in batch],dtype=torch.long)
            x=torch.tensor([r[1] for r in batch],dtype=torch.long)
            y=torch.tensor([r[2] for r in batch],dtype=torch.long)
            predictions.extend(predict_composed_operators(model,op,x,y,
                                             semantic_ids=op_ids).tolist())
    cases=[]
    mapped={}
    for i,(row,guess) in enumerate(zip(rows,predictions)):
        op,x,y,label=row
        identifier=f"hyb{i:05d}"
        cases.append(AtomicEvaluationCase(identifier,semantic[op],x,y,label,
                                          f"rev:{min(x,y):02d}-{max(x,y):02d}"))
        mapped[identifier]=int(guess)
    report=evaluate_atomic_predictions(cases,mapped,
        training_atoms=tuple((semantic[op],x,y) for op,x,y,_ in train_rows))
    if (report["disposition"]!="SCORED" or
        report["structural_overlap_audit"]["exact_training_pair_overlap"]):
        raise RuntimeError("independent evaluator rejected or detected pair leakage")
    return report,hashlib.sha256(json.dumps(predictions,separators=(",",":")).encode()).hexdigest()


def run_study(config=CompositionStudyConfig(),checkpoint_dir=None):
    config.validate()
    from .minotaur_algebraic_v5 import AlgebraicConfig, build_algebraic_model
    import torch
    from safetensors.torch import load_file,save_file
    torch.set_num_threads(min(2,torch.get_num_threads()))
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise RuntimeError("do not train from dirty source")
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    split=atomic_split(config.split_seed,.8,challenge=config.challenge)
    op_ids=tuple(split.op_ids)
    train_add=_model_visible_rows(split.train,op_ids[0])
    dev_add=_model_visible_rows(split.val,op_ids[0])
    if len(train_add)<5000 or set(r[:3] for r in train_add) & set(r[:3] for r in split.holdout):
        raise RuntimeError("invalid incomplete addition supervision")
    model_config=AlgebraicConfig(seed=config.seed,width=config.width,rank=config.rank)
    model=build_algebraic_model(model_config)
    trainer=train_atomic_operator(model,train_add,dev_add,
        steps=config.steps,batch_size=config.batch_size,lr=config.lr,
        time_budget=config.time_budget,torch=torch)
    observed, predictions_hash=_evaluate(model,split.holdout,op_ids,torch,
                                         config.batch_size,train_add)
    checkpoint=None
    if checkpoint_dir is not None:
        output=Path(checkpoint_dir)
        if output.exists():raise FileExistsError(output)
        output.mkdir(parents=True)
        saved=output/"model.safetensors"
        save_file({key:value.detach().cpu().contiguous() for key,value in model.state_dict().items()},str(saved))
        fresh=build_algebraic_model(model_config)
        fresh.load_state_dict(load_file(str(saved),device="cpu"),strict=True)
        reloaded,hash2=_evaluate(fresh,split.holdout,op_ids,torch,config.batch_size,train_add)
        if hash2!=predictions_hash or reloaded["metrics"]!=observed["metrics"]:
            raise RuntimeError("fresh process checkpoint changed predictions")
        checkpoint={"sha256":hashlib.sha256(saved.read_bytes()).hexdigest(),
                    "bytes":saved.stat().st_size,
                    "reloaded_predictions_identical":True}
    result={
        "schema":"labyrinth.minotaur-v5-neural-add-hybrid-composition.v1",
        "source":{"head":head,"clean":True,
                  "module_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "config":asdict(config),
        "architecture":{"neural_parameters":sum(x.numel() for x in model.parameters()),
                        "learned_rule":"ADD modulo97 from 80% of atomic addition pairs",
                        "provided_rules":["SUB(x,y) = ADD(x, inverse_mod97(y))",
                                         "MUL uses binary double-and-add via learned ADD"],
                        "all_atomic_pair_table":False},
        "supervision":{"ADD_train_pairs":len(train_add),
                       "ADD_validation_pairs":len(dev_add),
                       "test_not_used_for_optimizer":True,
                       "split_commitments":dict(split.commitments),
                       "semantic_op_ids":list(op_ids)},
        "training":trainer,
        "heldout":observed,
        "predictions_sha256":predictions_hash,
        "checkpoint":checkpoint,
        "qualification":"NOT_ESTABLISHED",
        "graduation_authority":"none",
        "promotion_recommendation":"RESEARCH_ONLY_NO_PRODUCTION",
        "limitations":["Operations SUB/MUL are composed by explicit algorithms; this does NOT show the network independently learned multiplication.",
                       "Public deterministic finite-domain tests, not a blind independent examiner.",
                       "No target labels enter neural forward; the learned ADD network remains fallible on unobserved pairs."],
    }
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--seed",type=int,default=20261028)
    parser.add_argument("--split-seed",type=int,default=9911)
    parser.add_argument("--challenge",choices=("none","value","diagonal"),default="none")
    parser.add_argument("--steps",type=int,default=1200)
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():parser.error("refuse overwrite")
    report=run_study(CompositionStudyConfig(seed=args.seed,split_seed=args.split_seed,
                                  challenge=args.challenge,steps=args.steps),args.checkpoint_dir)
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("SOURCE",report["source"]["head"],"PARAMS",report["architecture"]["neural_parameters"])
    print("TRAINING",report["training"]["optimizer_updates"])
    print("HELDOUT",report["heldout"]["metrics"])
    print("OPS",report["heldout"]["per_operation"])
    print("CHECKPOINT",report["checkpoint"])


if __name__=="__main__":
    main()
