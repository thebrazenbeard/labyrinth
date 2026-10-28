"""SHA-pinned experimental Minotaur V4 advisory inference, no graduation rights."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

from .minotaur_transition_eval import independently_replay

PINNED_SHA256 = "717c1938ad776edb8a99a5a533a88f4b9509566ce01e525ecca9c0d8a1a4b7b9"
PARAMETER_SHA256 = "f279a5cea1a2e8c662f93bcbd5ff16d09f98782e5257ff18c52bf9b7f56ce418"


def _abstain(reason):
    return {"disposition":"ABSTAIN", "reason":reason,
            "qualification":"NOT_ESTABLISHED","graduation_authority":"none"}


def predict_guarded_transition(checkpoint,question,steps):
    """Compare actual scratch neural inference with independently replayed math."""
    independent=independently_replay(question,steps)
    if independent["disposition"]!="REPLAYED":
        return _abstain("unsupported_program_or_trace")
    checkpoint=Path(checkpoint)
    if (not checkpoint.is_file() or checkpoint.is_symlink()
            or checkpoint.stat().st_size>16_000_000):
        raise ValueError("checkpoint must be a bounded regular, non-symlink file")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=PINNED_SHA256:
        raise ValueError("refuse unpinned neural checkpoint")
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("optional CPU Torch and Safetensors are required") from error
    from .minotaur_transition_v4 import (
        TransitionConfig, build_transition_model,
        derive_first_mismatch, encode_transition_cases,
    )
    class InputTrace:
        def __init__(self,question,steps):
            self.question,self.steps=question,steps
    torch.set_num_threads(min(2,torch.get_num_threads()))
    config=TransitionConfig(seed=20261027,architecture="learned_table")
    features,_=encode_transition_cases([InputTrace(question,steps)],config,torch)
    model=build_transition_model(config)
    model.load_state_dict(load_file(str(checkpoint),device="cpu"),strict=True)
    digest=hashlib.sha256(model.atomic_logits.weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if digest!=PARAMETER_SHA256:
        raise ValueError("model parameters disagree with independently recorded digest")
    model.eval()
    with torch.no_grad():
        predicted=model(features)["predicted_states"][0,:len(steps)]
    first_error=derive_first_mismatch(predicted,steps)
    if first_error!=independent["first_error_step"]:
        return {**_abstain("neural_disagrees_with_independent_verifier"),
                "neural_first_error_step":first_error,
                "checker_first_error_step":independent["first_error_step"]}
    return {"disposition":"ALL_CORRECT" if first_error is None else "ERROR_STEP",
            "first_error_step":first_error,
            "neural_first_error_step":first_error,
            "checker_first_error_step":independent["first_error_step"],
            "qualification":"NOT_ESTABLISHED","graduation_authority":"none",
            "domain":"public_mod97_two_register_1_to_6_steps",
            "warning":"Independent symbolic checking ensures reliability; table saw every atomic pair in training."}


def main(argv=None):
    parser=argparse.ArgumentParser(description="Experimental V4 advisory only")
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--input-json",required=True)
    args=parser.parse_args(argv)
    try:
        path=Path(args.input_json)
        if not path.is_file() or path.is_symlink() or path.stat().st_size>16384:
            raise ValueError("JSON input must be bounded regular file")
        def reject_duplicates(pairs):
            seen={}
            for key,value in pairs:
                if key in seen: raise ValueError("duplicate JSON key")
                seen[key]=value
            return seen
        data=json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=reject_duplicates)
        if (type(data) is not dict or set(data)!={"question","steps"}
                or type(data["question"]) is not str or type(data["steps"]) is not list
                or any(type(step) is not str for step in data["steps"])):
            raise ValueError("require exact question and steps")
        result=predict_guarded_transition(args.checkpoint,data["question"],tuple(data["steps"]))
        print(json.dumps(result,sort_keys=True))
        return 0
    except (ValueError,TypeError,OSError,RuntimeError) as error:
        print("Minotaur V4 input/checkpoint refused: "+type(error).__name__,file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
