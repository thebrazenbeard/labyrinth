"""Exhaustive finite-domain *learned* atomic transitions, then composition tests.

The trainable model is an embedding table of output logits. Arithmetic answers
occur ONLY in offline supervision, never in model.forward. This experiment
is a finite-domain memory/composition test, NOT extrapolation to unseen atoms.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

from .minotaur_transition_v4 import TransitionConfig, build_transition_model
from .minotaur_transition_curriculum import CurriculumConfig, build_transition_splits
from .minotaur_transition_study import _score


MOD=97
N_ATOMS=3*MOD*MOD


@dataclass(frozen=True)
class AtomicTrainingConfig:
    seed: int=20261027
    data_seed: int=7707
    lr: float=.25
    batch_size: int=1024
    dev_cases: int=128
    transfer_cases: int=128
    train_cases: int=256

    def validate(self):
        if (type(self.seed) is not int or not 0<=self.seed<2**31 or
            type(self.data_seed) is not int or not 0<=self.data_seed<2**31 or
            type(self.lr) is not float or not 0.001<=self.lr<=1.0 or
            type(self.batch_size) is not int or not 32<=self.batch_size<=4096 or
            any(type(n) is not int or not 64<=n<=1024 or n%2
                for n in (self.dev_cases,self.transfer_cases,self.train_cases))):
            raise ValueError("invalid atomic training configuration")
        return self


def atomic_supervision():
    """Exhaust all 28,227 distinct (op,target_value,operand_value) tuples."""
    output=[]
    for op in range(3):
        for x in range(MOD):
            for y in range(MOD):
                key=op*MOD*MOD+x*MOD+y
                truth=((x+y) if op==0 else (x-y) if op==1 else (x*y)) % MOD
                output.append((key,truth))
    return output


def atomic_train_holdout(seed:int):
    if type(seed) is not int or not 0<=seed<2**31:
        raise ValueError("invalid seed")
    examples=atomic_supervision()
    random.Random(seed).shuffle(examples)
    heldout=max(1,len(examples)//10)
    return examples[heldout:],examples[:heldout]


def _evaluate_atoms(model,entries,torch):
    keys=torch.tensor([key for key,_ in entries],dtype=torch.long)
    expected=torch.tensor([value for _,value in entries],dtype=torch.long)
    with torch.no_grad():
        guesses=model.atomic_logits(keys).argmax(-1)
    return {"correct":int((guesses==expected).sum().item()),
            "total":len(entries),
            "accuracy":float((guesses==expected).float().mean().item())}


def _train_atoms(model,entries,optimizer,torch):
    total_loss=0.0
    started=time.perf_counter()
    for i in range(0,len(entries),1024):
        batch=entries[i:i+1024]
        keys=torch.tensor([k for k,_ in batch],dtype=torch.long)
        expected=torch.tensor([v for _,v in batch],dtype=torch.long)
        logits=model.atomic_logits(keys)
        loss=torch.nn.functional.cross_entropy(logits,expected)
        if not bool(torch.isfinite(loss)): raise RuntimeError("nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss+=float(loss.detach().item())
    return {"updates":(len(entries)+1023)//1024,
            "mean_batch_loss":total_loss/((len(entries)+1023)//1024),
            "seconds":time.perf_counter()-started}


def _weights_digest(model,torch):
    weights=model.atomic_logits.weight.detach().cpu().contiguous()
    return hashlib.sha256(weights.numpy().tobytes()).hexdigest()


def run_atomic_training(config:AtomicTrainingConfig=AtomicTrainingConfig(),checkpoint_dir=None):
    config.validate()
    import torch
    from safetensors.torch import save_file,load_file
    torch.set_num_threads(min(2,torch.get_num_threads()))
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise RuntimeError("experiment must run from committed clean source")
    source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    model_config=TransitionConfig(seed=config.seed,architecture="learned_table")
    model=build_transition_model(model_config)
    first_digest=_weights_digest(model,torch)
    train,heldout=atomic_train_holdout(config.seed)
    splits=build_transition_splits(CurriculumConfig(
        seed=config.data_seed,train_cases=config.train_cases,
        dev_cases=config.dev_cases,transfer_cases=config.transfer_cases))
    train_structures=sorted({x.canonical_sequence for x in splits["train"].cases})
    frozen={name:_score(model,splits[name].cases,model_config,torch,train_structures)
            for name in ("dev","structure_disjoint_transfer")}
    opt=torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=0.0)
    stage1=_train_atoms(model,train,opt,torch)
    before_completion={"seen_atomic":_evaluate_atoms(model,train,torch),
                       "unseen_atomic":_evaluate_atoms(model,heldout,torch)}
    stage2=_train_atoms(model,heldout,opt,torch)
    all_atoms=_evaluate_atoms(model,atomic_supervision(),torch)
    after_digest=_weights_digest(model,torch)
    if after_digest==first_digest: raise RuntimeError("no neural weight update")
    trained={name:_score(model,splits[name].cases,model_config,torch,train_structures)
             for name in ("dev","structure_disjoint_transfer")}
    checkpoint=None
    if checkpoint_dir is not None:
        dest=Path(checkpoint_dir)
        if dest.exists(): raise FileExistsError(dest)
        dest.mkdir(parents=True)
        path=dest/"model.safetensors"
        save_file({"atomic_logits.weight":model.atomic_logits.weight.detach().cpu().contiguous()},str(path))
        fresh=build_transition_model(model_config)
        fresh.load_state_dict(load_file(str(path),device="cpu"),strict=True)
        if _weights_digest(fresh,torch)!=after_digest:
            raise RuntimeError("fresh reload changed neural parameters")
        replay={name:_score(fresh,splits[name].cases,model_config,torch,train_structures)
                for name in ("dev","structure_disjoint_transfer")}
        for name in replay:
            if replay[name]["metrics"]!=trained[name]["metrics"]:
                raise RuntimeError("fresh checkpoint replay mismatch")
        checkpoint={"safetensors_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                    "parameter_sha256":after_digest,"size_bytes":path.stat().st_size,
                    "fresh_reload_metrics_identical":True}
    return {"schema":"labyrinth.minotaur-v4-exhaustive-atom-composition.v1",
            "qualification":"NOT_ESTABLISHED","promotion_recommendation":"DO_NOT_PROMOTE",
            "source":{"git_head_at_run":source,"clean":True,
                      "training_module_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "config":asdict(config),
            "runtime":{"torch":torch.__version__,"device":"cpu"},
            "training":{"source":"exhaustive_ground_truth_atomic_supervision",
                        "initial_weights_sha256":first_digest,
                        "trained_weights_sha256":after_digest,
                        "stage1":stage1,"stage2":stage2,
                        "prior_to_full_coverage":before_completion,
                        "after_full_coverage":all_atoms,
                        "coverage_note":"100% of finite atomic operation-value pairs shown during training"},
            "splits":{name:{"cases":len(s.cases),
                             "unique_structures":len({c.canonical_sequence for c in s.cases}),
                             "commitment":s.commitment} for name,s in splits.items()},
            "frozen":frozen,"trained":trained,"checkpoint":checkpoint,
            "caveats":["Every possible atomic arithmetic operand pair in the 3-op modulo-97 grammar was supervised.",
                       "Success on unseen multi-step programs is *compositional* generalization within this closed finite domain, not general arithmetic extrapolation.",
                       "The untrained-atom control measures inability to extrapolate to unseen pairs.",
                       "No model release, independent private exam, or graduation authority."]}


def main():
    p=argparse.ArgumentParser(description="Finite-domain neural atomic-transition learning")
    p.add_argument("--output",required=True)
    p.add_argument("--checkpoint-dir")
    p.add_argument("--seed",type=int,default=20261027)
    args=p.parse_args()
    path=Path(args.output)
    if path.exists():p.error("refusing to overwrite report")
    report=run_atomic_training(AtomicTrainingConfig(seed=args.seed),args.checkpoint_dir)
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("SOURCE",report["source"]["git_head_at_run"])
    print("ATOM",report["training"]["after_full_coverage"])
    for name,r in report["trained"].items():
        m=r["metrics"]
        print("SPLIT",name,"exact",m["exact_first_error_accuracy"],
              "false_approval_rate",m["false_approval_rate"],"state",m["per_step_state_accuracy"])
    return 0

if __name__=="__main__":
    raise SystemExit(main())
