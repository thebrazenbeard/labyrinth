"""Source-independent public V8 examiner driving an actual learned model.

The examiner creates custody-committed canonical pairs and recomputes the labels.
Only input operands and modulus reach the neural candidate; no target in forward.
No private-examiner or model-promotion authority is conferred by this routine.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
from .minotaur_v8_examiner import build_exam_plan,make_prediction_submission,score_submission

@dataclass(frozen=True)
class QualificationConfig:
    seed:int=20261028
    seeds:tuple[int,int]=(20261028,20261029)
    epochs:int=28
    hidden_size:int=56
    batch_size:int=512
    max_optimizer_seconds:float=40.
    def validate(self):
        if (type(self.seed) is not int or self.seed<0 or
            type(self.seeds) is not tuple or len(self.seeds)!=2 or
            len(set(self.seeds))!=2 or any(type(s) is not int or s<0 for s in self.seeds)
            or type(self.epochs) is not int or not 1<=self.epochs<=160
            or type(self.hidden_size) is not int or not 16<=self.hidden_size<=96
            or type(self.batch_size) is not int or not 16<=self.batch_size<=2048
            or type(self.max_optimizer_seconds) is not float or
            not 1<=self.max_optimizer_seconds<=90):
            raise ValueError("invalid V8 bounded public qualification config")
        return self

def _digest(model):
    h=hashlib.sha256()
    for tensor in model.state_dict().values():
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def run_public_exam(config=QualificationConfig()):
    """Return 2 seeded public exam receipts, not a graduation decision."""
    config.validate()
    import torch
    from .minotaur_v8_structure import StructureConfig,build_structure_model,_fit,_batch
    torch.set_num_threads(min(2,torch.get_num_threads()))
    plan=build_exam_plan(seed=config.seed,holdout_fraction=.10)
    # Expands canonical pairs to both orientations, training labels strictly offline.
    ordered=[(x,y) for x,y in plan.training_members for x,y in
             (((x,y),(y,x)) if x!=y else ((x,y),))]
    train=_batch(ordered)
    if bool((train.target==53).any()) or bool(((train.x==53)|(train.y==53)).any()):
        raise RuntimeError("absent-residue data contamination")
    reports=[]
    for seed in config.seeds:
        settings=StructureConfig(seed=seed,seeds=config.seeds,
                                 hidden_size=config.hidden_size,
                                 epochs=config.epochs,batch_size=config.batch_size,
                                 max_optimizer_seconds=config.max_optimizer_seconds)
        model=build_structure_model(settings)
        initial=_digest(model)
        training=_fit(model,train,settings,seed)
        if training["optimizer_steps"]<1:
            raise RuntimeError("no optimizer steps")
        predictions={}
        model.eval()
        with torch.no_grad():
            for name,pairs in plan.evaluation_groups.items():
                modulus=plan.unseen_modulus if name=="modulus_101" else plan.modulus
                for start in range(0,len(pairs),1024):
                    items=pairs[start:start+1024]
                    x=torch.tensor([p[0] for p in items],dtype=torch.long)
                    y=torch.tensor([p[1] for p in items],dtype=torch.long)
                    mod=torch.full_like(x,modulus)
                    guesses=model(x,y,mod).round().long().clamp(0,modulus-1).tolist()
                    for pair,guess in zip(items,guesses):
                        predictions[pair]=int(guess)
        submission=make_prediction_submission(plan,predictions)
        exam=score_submission(plan,plan.training_members,submission)
        reports.append({"seed":seed,
            "training":{**training,"train_count":len(train),
                "input_53_train_rows":int(((train.x==53)|(train.y==53)).sum()),
                "target_53_train_rows":int((train.target==53).sum()),
                "initial_weight_sha256":initial,
                "trained_weight_sha256":_digest(model)},
            "exam":exam})
    return {"schema":"labyrinth.minotaur.v8-independent-public-exam.v1",
            "config":asdict(config),
            "plan":{"sha256":plan.custody_sha256,
                    "canonical_training_members":len(plan.training_members),
                    "evaluation_canonical_groups":{k:len(v) for k,v in plan.evaluation_groups.items()},
                    "modulus_101_scope":"only pairs containing residues 97-100; 97x97 overlapping pairs excluded"},
            "runs":reports,
            "graduation_authority":"none",
            "qualification":"NOT_ESTABLISHED",
            "promotion_recommendation":"DO_NOT_PROMOTE",
            "limitations":["Public synthetic test with known generating function, not private examiner",
                           "Training seed and public test family predeclared, no private custody",
                           "Output 53 and input 53 absent from train; new modulus 101 unseen",
                           "Mod101 exam intentionally restricts pairs to prevent key collision with mod97"]}

