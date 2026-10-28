"""Source-first CPU study for V4 true per-step neural state prediction."""
from __future__ import annotations
import argparse, hashlib, json, random, subprocess, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from .minotaur_transition_v4 import (
    TransitionConfig, build_transition_model, encode_transition_cases, derive_first_mismatch,
)
from .minotaur_transition_curriculum import (
    CurriculumConfig, build_transition_splits,
)
from .minotaur_transition_eval import (
    TransitionEvaluationCase, TransitionPrediction, evaluate_transition_predictions,
)

@dataclass(frozen=True)
class TransitionStudyConfig:
    seed: int = 20261027
    data_seed: int = 7001
    train_cases: int = 512
    dev_cases: int = 128
    transfer_cases: int = 128
    epochs: int = 12
    batch_size: int = 64
    width: int = 64
    learning_rate: float = 0.002
    seconds_limit: float = 85.0

    def validate(self):
        if (type(self.seed) is not int or type(self.data_seed) is not int or
            any(type(x) is not int or x % 2 or not 32 <= x <= 1024
                for x in (self.train_cases,self.dev_cases,self.transfer_cases)) or
            type(self.epochs) is not int or not 1 <= self.epochs <= 50 or
            type(self.batch_size) is not int or not 8 <= self.batch_size <= 256 or
            type(self.width) is not int or not 16 <= self.width <= 128 or
            type(self.learning_rate) is not float or not 0.000001 <= self.learning_rate <= .02 or
            type(self.seconds_limit) is not float or not 1. <= self.seconds_limit <= 180.):
            raise ValueError("Invalid bounded transition study")
        return self


def _predict(model, cases, cfg, torch):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for i in range(0,len(cases),32):
            batch=cases[i:i+32]
            features, _=encode_transition_cases(batch,cfg,torch)
            states=model(features)["predicted_states"]
            for case, prediction in zip(batch,states.tolist()):
                points=tuple(tuple(x) for x in prediction[:len(case.steps)])
                guess=derive_first_mismatch(states[batch.index(case),:len(case.steps)],case.steps)
                predictions[case.case_id]=TransitionPrediction(points,0 if guess is None else guess,"model_inference")
    return predictions


def _score(model,cases,cfg,torch,train_structures):
    predictions=_predict(model,cases,cfg,torch)
    audit_cases=[TransitionEvaluationCase(c.case_id,c.question,c.steps,c.first_error_step,
                                         c.canonical_sequence,c.canonical_sequence)
                 for c in cases]
    result=evaluate_transition_predictions(audit_cases,predictions,training_structures=train_structures)
    if result["disposition"]!="SCORED":
        raise RuntimeError(f"independent evaluator refused: {result['audit']}")
    return result


def run_study(config=TransitionStudyConfig()):
    config.validate()
    import torch
    torch.set_num_threads(min(2,torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    source = Path(__file__).resolve()
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","status","--porcelain"],text=True).strip()
    if dirty: raise RuntimeError("refuse uncommitted experiment source")
    splits=build_transition_splits(CurriculumConfig(seed=config.data_seed,train_cases=config.train_cases,
                                           dev_cases=config.dev_cases,transfer_cases=config.transfer_cases))
    training=[case for case in splits["train"].cases if case.first_error_step is None]
    structures=sorted({c.canonical_sequence for c in splits["train"].cases})
    cfg=TransitionConfig(seed=config.seed,width=config.width)
    model=build_transition_model(cfg)
    initial_sha=hashlib.sha256(b"".join(p.detach().cpu().contiguous().numpy().tobytes()
                                     for p in model.parameters())).hexdigest()
    evaluated=("dev","structure_disjoint_transfer")
    frozen={key:_score(model,splits[key].cases,cfg,torch,structures) for key in evaluated}
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=.0)
    losses=[]
    started=time.perf_counter()
    batches=0
    rng=random.Random(config.seed)
    for _ in range(config.epochs):
        rng.shuffle(training)
        for i in range(0,len(training),config.batch_size):
            if time.perf_counter()-started>=config.seconds_limit:
                raise RuntimeError("optimizer wall budget exceeded")
            batch=training[i:i+config.batch_size]
            features,targets=encode_transition_cases(batch,cfg,torch)
            model.train()
            logits=model(features,teacher_states=targets)["state_logits"]
            selected=logits.gather(2, features.targets[:,:,None,None].expand(
                -1,-1,1,97)).squeeze(2)
            loss=torch.nn.functional.cross_entropy(selected[features.mask],targets.gather(
                2,features.targets[:,:,None]).squeeze(2)[features.mask])
            if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(norm): raise RuntimeError("nonfinite gradient")
            optimizer.step();batches+=1
            losses.append(float(loss.detach().item()))
    elapsed=time.perf_counter()-started
    if elapsed>config.seconds_limit: raise RuntimeError("time budget")
    final_sha=hashlib.sha256(b"".join(p.detach().cpu().contiguous().numpy().tobytes()
                                   for p in model.parameters())).hexdigest()
    if initial_sha==final_sha:raise RuntimeError("model did not train")
    trained={key:_score(model,splits[key].cases,cfg,torch,structures) for key in evaluated}
    return {"schema":"labyrinth.minotaur-transition-study.v1",
        "qualification":"NOT_ESTABLISHED","promotion_recommendation":"DO_NOT_PROMOTE",
        "source":{"head":head,"clean":True,"module_sha256":hashlib.sha256(source.read_bytes()).hexdigest()},
        "runtime":{"torch":torch.__version__,"device":"cpu"},
        "config":asdict(config),
        "splits":{key:{"cases":len(group.cases),
                       "structures":len({c.canonical_sequence for c in group.cases}),
                       "commitment":group.commitment}
                    for key,group in splits.items()},
        "training":{"steps":batches,"seconds":elapsed,
                    "loss_initial":losses[0],"loss_final":losses[-1],
                    "initial_weights_sha256":initial_sha,"trained_weights_sha256":final_sha,
                    "saved_model":False},
        "frozen":frozen,"trained":trained,
        "limits":"Structured copy prior is fixed; updated register is learned. This is public synthetic development, not a private independent exam."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    args=p.parse_args()
    path=Path(args.output)
    if path.exists():p.error("refuse overwrite")
    report=run_study()
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    for k,v in report["trained"].items():
        m=v["metrics"]
        print(k, "accuracy",m.get("exact_first_error_accuracy"),"false approvals",m.get("false_approvals"),"state accuracy",m["per_step_state_accuracy"])
    print("SOURCE",report["source"]["head"],"STEPS",report["training"]["steps"])
if __name__=="__main__":main()
