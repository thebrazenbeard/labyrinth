"""Executable partial-supervision Minotaur neural algebra study (NOT a gate).

No 28,227-row atomic table is used for the candidate. The forward rule has
12 trained coefficients and an explicit bilinear/modulus prior; all arithmetic
oracles and the memoization baseline exist only in offline evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any

from .minotaur_algebraic_rules import (
    MOD, all_atom_accuracy, build_algebraic_head, corrupt_labels_within_op,
    incomplete_training_atoms, train_rule_head,
)
from .minotaur_transition_v4 import (
    MAX_STEPS, TransitionConfig, TransitionFeatures, build_transition_model,
)

SCHEMA = "labyrinth.minotaur-algebraic-partial-study.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source() -> str:
    root=Path(__file__).resolve().parents[1]
    return subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],
                                   text=True).strip()


def evaluate_programs(
    head: Any, torch: Any, *, seed: int, count: int,
    training_atoms: list[tuple[int,int,int,int]] | None = None,
) -> dict[str,int]:
    """Evaluate multi-step free-running model, with gold computed *after* input."""
    if (type(seed) is not int or not 0 <= seed < 2**31 or
            type(count) is not int or not 1 <= count <= 1024):
        raise ValueError("bounded evaluation seed/count required")
    rng=random.Random(seed)
    examples=training_atoms or incomplete_training_atoms(47,12)
    seen={(op,x,y) for op,x,y,_ in examples}
    start,ops,targets,isreg,vals,masks,numbers,gold=[],[],[],[],[],[],[],[]
    seen_visits=0
    unseen_visits=0
    for _ in range(count):
        a,b=rng.randrange(MOD),rng.randrange(MOD)
        step_count=rng.randint(1,MAX_STEPS)
        start.append((a,b))
        commands=[]
        predicted_gold=[]
        states=[a,b]
        for j in range(step_count):
            op=rng.randrange(3)
            target=rng.randrange(2)
            reg=rng.randrange(2)==1
            value=rng.randrange(2) if reg else rng.randrange(MOD)
            operand=states[value] if reg else value
            old=states[target]
            if (op,old,operand) in seen:
                seen_visits+=1
            else:
                unseen_visits+=1
            # Independent offline gold evaluator, never passed to model.
            new=(old+operand if op==0 else old-operand if op==1 else old*operand)%MOD
            states[target]=new
            predicted_gold.append(tuple(states))
            commands.append((op,target,reg,value))
        fill=MAX_STEPS-step_count
        ops.append([c[0] for c in commands]+[0]*fill)
        targets.append([c[1] for c in commands]+[0]*fill)
        isreg.append([c[2] for c in commands]+[False]*fill)
        vals.append([c[3] for c in commands]+[0]*fill)
        masks.append([True]*step_count+[False]*fill)
        numbers.append(list(range(1,step_count+1))+[0]*fill)
        gold.append(predicted_gold)
    features=TransitionFeatures(
        initial_states=torch.tensor(start,dtype=torch.long),
        operations=torch.tensor(ops,dtype=torch.long),
        targets=torch.tensor(targets,dtype=torch.long),
        operand_is_register=torch.tensor(isreg,dtype=torch.bool),
        operand_values=torch.tensor(vals,dtype=torch.long),
        mask=torch.tensor(masks,dtype=torch.bool),
        step_numbers=torch.tensor(numbers,dtype=torch.long),
    )
    config=TransitionConfig(seed=19,architecture="algebraic_rule")
    model=build_transition_model(config)
    model.algebraic_head.load_state_dict(head.state_dict(),strict=True)
    model.eval()
    with torch.no_grad():
        predicted=model(features)["predicted_states"].tolist()
    exact=0
    correct_steps=0
    steps_total=0
    for actual,wanted in zip(predicted,gold):
        steps_total+=len(wanted)
        correct_steps+=sum(tuple(p)==g for p,g in zip(actual,wanted))
        exact+=int(all(tuple(p)==g for p,g in zip(actual,wanted)))
    return {"exact_programs":exact,"total_programs":count,
            "correct_states":correct_steps,"total_states":steps_total,
            "seen_atomic_visits":seen_visits,
            "unseen_atomic_visits":unseen_visits}


def run_experiment(
    output_dir: Path, torch: Any, *, seed: int = 47,
    training_per_op: int = 12, steps: int = 500,
) -> dict[str, Any]:
    """Train from deliberately incomplete examples and save a separate adapter."""
    output_dir=Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite algebraic training evidence")
    if not output_dir.parent.is_dir():
        raise ValueError("output parent must exist")
    if type(training_per_op) is not int or not 5 <= training_per_op <= 24:
        raise ValueError("invalid partial dataset size")
    examples=incomplete_training_atoms(seed,training_per_op)
    randomized=corrupt_labels_within_op(examples,seed+12)
    torch.set_num_threads(min(torch.get_num_threads(),2))
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(19)
    fresh=build_algebraic_head(torch,seed=19)
    frozen=all_atom_accuracy(fresh,torch)
    sham=build_algebraic_head(torch,seed=19)
    with torch.no_grad():
        initial=sham.coefficients.detach().clone()
    train_rule_head(sham,examples,torch,steps=steps,learning_rate=0.0)
    if not torch.equal(initial,sham.coefficients.detach()):
        raise RuntimeError("zero-learning-rate control changed parameters")
    sham_score=all_atom_accuracy(sham,torch)
    candidate=build_algebraic_head(torch,seed=19)
    train_rule_head(candidate,examples,torch,steps=steps,learning_rate=0.06)
    if torch.equal(initial,candidate.coefficients.detach()):
        raise RuntimeError("no neural parameter update")
    candidate_score=all_atom_accuracy(candidate,torch)
    # Explicitly exclude every supervised tuple when reporting transfer.
    with torch.no_grad():
        ops=torch.tensor([r[0] for r in examples],dtype=torch.long)
        x=torch.tensor([r[1] for r in examples],dtype=torch.long)
        y=torch.tensor([r[2] for r in examples],dtype=torch.long)
        supervised_predictions=candidate.predict(ops,x,y)
        supervised_truth=torch.tensor([r[3] for r in examples],dtype=torch.long)
        trained_correct=int((supervised_predictions==supervised_truth).sum().item())
    withheld_score={
        "correct":candidate_score["correct"]-trained_correct,
        "total":3*MOD*MOD-len(examples),
    }
    corrupted=build_algebraic_head(torch,seed=19)
    train_rule_head(corrupted,randomized,torch,steps=steps,learning_rate=0.06)
    corrupted_score=all_atom_accuracy(corrupted,torch)
    composition=evaluate_programs(
        candidate,torch,seed=808,count=128,training_atoms=examples,
    )
    # No learned lookup for unseen combinations. This control abstains.
    lookup_correct=len(examples)
    total=3*MOD*MOD
    with torch.no_grad():
        init_data=initial.cpu().contiguous().numpy().tobytes()
        trained_data=candidate.coefficients.cpu().detach().contiguous().numpy().tobytes()
    output_dir.mkdir(mode=0o700)
    from safetensors.torch import save_file
    ckpt=output_dir/"candidate.safetensors"
    save_file({key:value.detach().cpu().contiguous()
               for key,value in candidate.state_dict().items()},str(ckpt))
    report={
        "schema":SCHEMA,
        "source_commit":_source(),
        "train_process_id":os.getpid(),
        "donor":"scratch algebraically biased neural rule head, no pretrained model",
        "model_architecture":"learned_coefficients[3,4] over [1,x/8,y/8,xy/64]",
        "modulus":MOD,"seed":seed,"train_examples":len(examples),
        "train_examples_by_operation":training_per_op,
        "training_atoms_sha256":hashlib.sha256(json.dumps(examples).encode()).hexdigest(),
        "corrupted_atoms_sha256":hashlib.sha256(json.dumps(randomized).encode()).hexdigest(),
        "steps_each_control":steps,
        "parameter_count":sum(t.numel() for t in candidate.parameters()),
        "initial_coefficients_sha256":hashlib.sha256(init_data).hexdigest(),
        "candidate_coefficients_sha256":hashlib.sha256(trained_data).hexdigest(),
        "checkpoint_sha256":_sha(ckpt),
        "frozen":frozen,"sham":sham_score,"corrupted":corrupted_score,
        "candidate":candidate_score,"withheld_atomic_transfer":withheld_score,
        "lookup":{
            "covered":lookup_correct,"total":total,
            "unseen_abstentions":total-lookup_correct},
        "composition":composition,
        "learning_status":"PARAMETERS_UPDATED; TRANSFER_DIAGNOSTIC_ONLY",
        "minotaur_graduated":False,
        "deployed":False,
        "limitations":[
            "Bilinear feature x*y and modulo projection are supplied priors, not discovered",
            "Training is confined to operands 0..8 and non-wrapping targets",
            "All atom ground truths and program references are public offline evaluator results",
            "Full-domain evaluation is not a privately controlled independent exam",
            "No authenticated independent evaluator, Vera adapter installation or Minotaur graduation",
        ],
    }
    fp=output_dir/"report.json"
    with fp.open("x",encoding="utf8") as stream:
        json.dump(report,stream,sort_keys=True,indent=2)
        stream.write("\n")
    return report


def reload_experiment(output_dir: Path, torch: Any) -> dict[str, Any]:
    """Recompute stored candidate metrics from a new loaded model."""
    root=Path(output_dir)
    report_file=root/"report.json"
    checkpoint=root/"candidate.safetensors"
    if (not report_file.is_file() or report_file.is_symlink() or
            not checkpoint.is_file() or checkpoint.is_symlink()
            or checkpoint.stat().st_size > 100_000):
        raise ValueError("report/checkpoint must be bounded local regular files")
    report=json.loads(report_file.read_text(encoding="utf8"))
    if report.get("schema")!=SCHEMA or report.get("source_commit")!=_source():
        raise ValueError("report schema/source mismatch")
    if _sha(checkpoint)!=report.get("checkpoint_sha256"):
        raise ValueError("checkpoint changed after training")
    from safetensors.torch import load_file
    model=build_algebraic_head(torch,seed=19)
    model.load_state_dict(load_file(str(checkpoint),device="cpu"),strict=True)
    with torch.no_grad():
        digest=hashlib.sha256(
            model.coefficients.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
    if digest != report.get("candidate_coefficients_sha256"):
        raise ValueError("loaded tensor values differ from training")
    atom=all_atom_accuracy(model,torch)
    composition=evaluate_programs(
        model,torch,seed=808,count=128,
        training_atoms=incomplete_training_atoms(report["seed"],
                                                   report["train_examples_by_operation"]),
    )
    matches=atom==report["candidate"] and composition==report["composition"]
    return {"schema":"labyrinth.minotaur-algebraic-reload.v1",
            "matches_original":matches,
            "process_id_differs":os.getpid()!=report["train_process_id"],
            "atom_score":atom,"composition":composition,
            "qualification":"NOT_ESTABLISHED","minotaur_graduated":False}


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="Bounded compact neural algebra trial")
    parser.add_argument("mode",choices=("train","reload"))
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args(argv)
    import torch
    try:
        if args.mode=="train":
            result=run_experiment(Path(args.output_dir),torch)
            print(json.dumps({"candidate":result["candidate"],
                              "corrupted":result["corrupted"],
                              "composition":result["composition"],
                              "minotaur_graduated":False}))
        else:
            result=reload_experiment(Path(args.output_dir),torch)
            if result["matches_original"] is not True or result["process_id_differs"] is not True:
                raise RuntimeError("checkpoint reload failed fresh-process qualification")
            print(json.dumps(result,sort_keys=True))
    except (ValueError,RuntimeError,FileExistsError,OSError) as error:
        parser.error(str(error))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
