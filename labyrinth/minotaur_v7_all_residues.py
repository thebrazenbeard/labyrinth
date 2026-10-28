"""Public, source-frozen V7 neural stress: every output residue held out.

No training on heldout output labels or dev/test examples; no checkpoint promotion.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

@dataclass(frozen=True)
class StressConfig:
    seed: int = 20261028
    seeds: tuple[int,int] = (20261028,20261029)
    epochs: int = 6
    batch_size: int = 2048
    learning_rate: float = .003
    max_optimizer_seconds: float = 9.0
    width: int = 24
    depth: int = 2

    def validate(self):
        if (type(self.seed) is not int or self.seed<0 or
            type(self.seeds) is not tuple or len(self.seeds)!=2 or
            any(type(n) is not int or not 0<=n<2**31 for n in self.seeds) or
            len(set(self.seeds))!=2 or
            type(self.epochs) is not int or not 1<=self.epochs<=100 or
            type(self.batch_size) is not int or not 64<=self.batch_size<=4096 or
            type(self.learning_rate) is not float or not 0<self.learning_rate<=.02 or
            type(self.max_optimizer_seconds) is not float or not 1<=self.max_optimizer_seconds<=90 or
            type(self.width) is not int or not 8<=self.width<=48 or
            type(self.depth) is not int or not 1<=self.depth<=3):
            raise ValueError("invalid bounded V7 all-residue configuration")
        return self

def summarize_stress(runs,*,seeds,labels):
    if not runs or len(runs)!=len(seeds)*len(labels):
        raise ValueError("missing all-residue evidence")
    if any(type(s) is not int for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError("nonunique seed")
    if any(type(r) is not int or not 0<=r<97 for r in labels) or len(set(labels))!=len(labels):
        raise ValueError("nonunique output labels")
    received=set()
    accuracies=[]
    per_label={}
    for row in runs:
        if not isinstance(row,dict):
            raise ValueError("invalid run")
        key=(row.get("seed"),row.get("label"))
        if key in received or key[0] not in seeds or key[1] not in labels:
            raise ValueError("duplicate or unexpected run")
        received.add(key)
        count=row.get("heldout_count")
        correct=row.get("heldout_correct")
        accuracy=row.get("heldout_accuracy")
        if (type(count) is not int or count<=0 or
            type(correct) is not int or not 0<=correct<=count or
            type(accuracy) not in (int,float) or abs(accuracy-correct/count)>1e-9 or
            row.get("train_excludes_label") is not True or
            type(row.get("optimizer_updates")) is not int or row["optimizer_updates"]<=0 or
            row.get("weights_changed") is not True or
            type(row.get("frozen_accuracy")) not in (int,float) or
            type(row.get("constant_accuracy")) not in (int,float)):
            raise ValueError("incomplete or invalid negative-evidence row")
        accuracies.append(accuracy)
        per_label.setdefault(row["label"],[]).append(accuracy)
    full=(len(labels)==97 and set(labels)==set(range(97)) and len(seeds)>=2)
    return {
        "tested_label_count":len(labels),"seed_count":len(seeds),
        "total_runs":len(runs),"all_97_outputs_evaluated":full,
        "macro_accuracy":statistics.mean(accuracies),
        "median_accuracy":statistics.median(accuracies),
        "best_accuracy":max(accuracies),"worst_accuracy":min(accuracies),
        "perfect_output_labels":sum(all(v==1 for v in vals) for vals in per_label.values()),
        "labels_with_any_positive_prediction":sum(any(v>0 for v in vals) for vals in per_label.values()),
        "promotion_recommendation":"DO_NOT_PROMOTE","qualification":"NOT_ESTABLISHED",
        "row_independence":"Do not use directed reverse rows as independent statistical samples.",
    }

def _evaluate(model, rows, torch):
    inputs=torch.tensor([r[0] for r in rows],dtype=torch.long)
    second=torch.tensor([r[1] for r in rows],dtype=torch.long)
    correct=torch.tensor([r[2] for r in rows],dtype=torch.long)
    prediction=[]
    model.eval()
    with torch.no_grad():
        for i in range(0,len(rows),1024):
            a=inputs[i:i+1024];b=second[i:i+1024]
            prediction.extend(model(a,b,torch.full_like(a,97)).round().long().clamp(0,96).tolist())
    hits=sum(p==int(t) for p,t in zip(prediction,correct.tolist()))
    return hits,len(rows),hits/len(rows)

def run_stress(config=StressConfig(),*,labels=None,require_clean_source=True):
    config.validate()
    from .minotaur_v7_benchmark import build_mod97_dataset
    from .minotaur_v7_numeric import NumericConfig,build_numeric_model
    from .minotaur_v6_mul_eval import recompute_mul_label
    import torch
    torch.set_num_threads(min(2,torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    if require_clean_source and subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise RuntimeError("refuse training from uncommitted source")
    labels=tuple(range(97)) if labels is None else tuple(labels)
    if len(set(labels))!=len(labels) or not labels or any(type(x) is not int or not 0<=x<97 for x in labels):
        raise ValueError("invalid holdout labels")
    runs=[]
    for residue in labels:
        dataset=build_mod97_dataset(seed=config.seed,variant="absent_output",absent_label=residue)
        train=dataset.splits["train"]
        dev=dataset.splits["dev"]
        test=dataset.splits["test"]
        if (not test or any(t==residue for _,_,t in train+dev) or
            any(t!=residue for _,_,t in test) or
            any(recompute_mul_label(x,y)!=t for x,y,t in test)):
            raise RuntimeError("independent oracle or disjoint split failed")
        x=torch.tensor([a for a,b,t in train],dtype=torch.long)
        y=torch.tensor([b for a,b,t in train],dtype=torch.long)
        t=torch.tensor([z for a,b,z in train],dtype=torch.float32)
        train_majority=max(set(t.tolist()),key=lambda v:(int((t==v).sum()),-v))
        for seed in config.seeds:
            model=build_numeric_model(NumericConfig(
                seed=seed,seeds=config.seeds,width=config.width,depth=config.depth))
            before=hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
            frozen_hits,_,frozen=_evaluate(model,test,torch)
            optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
            gen=torch.Generator().manual_seed(seed+residue*100003)
            start=time.perf_counter();updates=0;timed_out=False
            model.train()
            for _ in range(config.epochs):
                indices=torch.randperm(len(train),generator=gen)
                for i in range(0,len(train),config.batch_size):
                    if time.perf_counter()-start>=config.max_optimizer_seconds:
                        timed_out=True;break
                    idx=indices[i:i+config.batch_size]
                    prediction=model(x[idx],y[idx],torch.full_like(x[idx],97))
                    loss=torch.nn.functional.smooth_l1_loss(prediction,t[idx])
                    if not torch.isfinite(loss):raise RuntimeError("nonfinite loss")
                    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                    updates+=1
                if timed_out: break
            trained_hits,n,accuracy=_evaluate(model,test,torch)
            after=hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
            runs.append({"seed":seed,"label":residue,"heldout_correct":trained_hits,
                         "heldout_count":n,"heldout_accuracy":accuracy,
                         "frozen_correct":frozen_hits,"frozen_accuracy":frozen,
                         "constant_accuracy":sum(z==train_majority for a,b,z in test)/n,
                         "train_excludes_label":True,"train_size":len(train),
                         "train_digest":dataset.digests["train"],"test_digest":dataset.digests["test"],
                         "optimizer_updates":updates,"timed_out":timed_out,
                         "weights_changed":before!=after,
                         "qualification":"NOT_ESTABLISHED"})
        if len(labels)>8 and residue%10==0:
            print("FINISHED_LABEL",residue,"OF",len(labels),flush=True)
    return {"schema":"labyrinth.minotaur.v7-all-outputs-stress.v1",
            "source":{"commit":source,"clean":require_clean_source,
                      "module_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "config":asdict(config),"labels":list(labels),"runs":runs,
            "aggregate":summarize_stress(runs,seeds=config.seeds,labels=labels),
            "limits":["Public deterministic labels; not an external blind examination.",
                      "Only 2 predeclared seeds and 6 epochs; not a new modulus experiment.",
                      "Numerical scalar model still uses supplied Fourier and symmetry priors."],
            "qualification":"NOT_ESTABLISHED","promotion_recommendation":"DO_NOT_PROMOTE"}

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():parser.error("refuse overwrite evidence")
    report=run_stress()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("SOURCE",report["source"]["commit"],"MACRO",report["aggregate"]["macro_accuracy"],"FULL",report["aggregate"]["all_97_outputs_evaluated"])
if __name__=="__main__":
    main()

