"""Strict SHA-pinned, advisory-only inference and independent public rollout."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

PINNED_SHA256="ae92b4e847a8c952d6672a14d9064d0e73df185822da2a042e98c3ce59f55c57"
_INITIAL=re.compile(r"^Initial registers: A=(\d{1,2}), B=(\d{1,2})$",re.MULTILINE)
_COMMAND=re.compile(r"^(\d+)\. (ADD|SUB|MUL) ([AB]) ([AB]|\d{1,2})$",re.MULTILINE)
_OBSERVED=re.compile(r"Step ([1-6]): A=(\d{1,2}), B=(\d{1,2})")


def load_pinned_neural_add(checkpoint):
    """Load only the exact reviewed 149-KB learned model; never use pickle."""
    path=Path(checkpoint)
    if not path.is_file() or path.is_symlink() or not 10000<=path.stat().st_size<=250000:
        raise ValueError("invalid checkpoint file type or size")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PINNED_SHA256:
        raise ValueError("unreviewed neural weights")
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("optional torch and Safetensors required") from exc
    from .minotaur_algebraic_v5 import AlgebraicConfig,build_algebraic_model
    root=Path(__file__).resolve().parents[1]
    report=json.loads((root/"models/minotaur_algebraic_v5_development/training_report.json").read_text())
    if report["checkpoint"]["sha256"]!=PINNED_SHA256:
        raise ValueError("checkpoint report mismatch")
    config=AlgebraicConfig(seed=20261028,width=48,rank=20)
    model=build_algebraic_model(config)
    model.load_state_dict(load_file(str(path),device="cpu"),strict=True)
    model.eval()
    return model,tuple(report["supervision"]["semantic_op_ids"])


def _program_case(case):
    from .minotaur_algebraic_eval_v5 import ProgramRolloutCase
    initial=_INITIAL.findall(case.question)
    matches=_COMMAND.findall(case.question)
    if len(initial)!=1 or len(matches)!=len(case.steps):
        raise ValueError("invalid printed program")
    operations=[]
    for index, (number,op,target,operand) in enumerate(matches,1):
        if int(number)!=index:
            raise ValueError("nonsequential program")
        operations.append((op,target,operand if operand in ("A","B") else int(operand)))
    observed=[]
    for index,line in enumerate(case.steps,1):
        row=_OBSERVED.fullmatch(line)
        if row is None or int(row[1])!=index:
            raise ValueError("malformed printed trace")
        observed.append((int(row[2]),int(row[3])))
    return ProgramRolloutCase(
        case_id=case.case_id, initial_state=tuple(map(int,initial[0])),
        instructions=tuple(operations),observed_states=tuple(observed),
        claimed_first_error=case.first_error_step,
        group_id=case.canonical_sequence[:128] or "ungrouped")


def evaluate_checkpoint_programs(model,semantic_op_ids,*,seed=818,cases=128):
    """Score complete independent held-out 1–6-step public programs."""
    if type(seed) is not int or not 0<=seed<2**31 or type(cases) is not int or not 10<=cases<=256 or cases%2:
        raise ValueError("bounded seed and paired case count")
    if type(semantic_op_ids) is not tuple or set(semantic_op_ids)!={0,1,2}:
        raise ValueError("invalid semantic op mapping")
    import torch
    from .minotaur_algebraic_data_v5 import program_composition_split
    from .minotaur_algebraic_composer_v5 import predict_composed_operators
    from .minotaur_algebraic_eval_v5 import evaluate_program_rollouts
    torch.set_num_threads(min(2,torch.get_num_threads()))
    split=program_composition_split(seed,train_cases=64,val_cases=32,holdout_cases=cases)
    examples=[_program_case(case) for case in split.holdout]
    op_mapping={name:id for name,id in zip(("ADD","SUB","MUL"),semantic_op_ids)}
    def callback(op,x,y):
        operation=torch.tensor([op_mapping[op]],dtype=torch.long)
        first=torch.tensor([x],dtype=torch.long)
        second=torch.tensor([y],dtype=torch.long)
        return int(predict_composed_operators(
            model,operation,first,second,semantic_ids=semantic_op_ids)[0].item())
    result=evaluate_program_rollouts(examples,callback)
    if result["disposition"]!="SCORED":
        raise RuntimeError("independent rollout evaluator refused cases")
    return {**result,"program_split_commitment":split.commitments["holdout"],
            "pair_structures":cases//2,
            "checkpoint_sha256":PINNED_SHA256,
            "model_provenance":"learned ADD + supplied binary multiplication and modular inversion",
            "graduation_authority":"none","qualification":"NOT_ESTABLISHED"}
