"""Frozen, *public* paired neural representation study; no graduation authority.

The arithmetic domain and feature families have been inspected in development.
This is prospective only with respect to these predeclared seed-pair outcomes,
not an independent, private, or blinded generalization examination.
"""
from __future__ import annotations

import hashlib
import json
from math import comb
from pathlib import Path
import random
from typing import Any

from .minotaur_algebraic_rules import (
    ClosureBudgetExhausted, MOD, OPS, build_algebraic_head,
    corrupt_labels_within_op, incomplete_training_atoms, train_rule_head_lbfgs,
)

SCHEMA="labyrinth.minotaur-prospective-public-neural-study.v1"
PAIR_COUNT=100
PROGRAM_COUNT=128
PROGRAM_STEPS=6
MAX_ITER=500
MAX_EVAL=500
CLOSURE_BUDGET=500
# Development-exposed seed values; never replace an unsuccessful new pair.
EXPOSED_DATA_SEEDS=frozenset((0,3,7,11,12,31,42,47,89,91,314,521,1024,1234,2026,314159))
EXPOSED_INIT_SEEDS=frozenset((7,19,29,33,402,20261028))
EXPOSED_PROGRAM_SEEDS=frozenset((808,))
ARMS=("bilinear","no_xy")

PROTOCOL={
    "schema":SCHEMA,
    "paired_seed_count":PAIR_COUNT,
    "program_count_per_pair":PROGRAM_COUNT,
    "program_steps":PROGRAM_STEPS,
    "program_generation":"domain-separated SHA256 seeds; six steps; every gold atom outside small/small domain",
    "seed_generation":"SHA256 domain-separated index with collision/development-seed rejection",
    "training_examples_per_operation":12,
    "feature_basis_by_arm":{"bilinear":["1","x/8","y/8","xy/64"],
                            "no_xy":["1","x/8","y/8","x_squared/64"]},
    "parameter_count_per_arm":12,
    "optimizer":"torch.optim.LBFGS_strong_wolfe",
    "learning_rate":1.0,
    "max_iter":MAX_ITER,"max_eval_soft":MAX_EVAL,
    "closure_budget_hard":CLOSURE_BUDGET,
    "require_fit":False,
    "controls":["frozen_bilinear_initialization","within_op_deranged_bilinear_labels"],
    "primary_event":"10*(bilinear_exact_programs-no_xy_exact_programs)>128",
    "inferential_unit":"paired_dataset_and_initialization_seed",
    "failure_policy":"any incomplete pair or control failure is non-win; fixed denominator 100",
    "test":"one-sided exact binomial H0 P(pair_win)<=0.5, alpha=0.05; >=59 wins",
    "secondary":"withheld exact atoms by op, wrap atoms, control scores, fit/closure counts; descriptive only",
    "qualification":"NOT_ESTABLISHED",
}
FROZEN_PROTOCOL_SHA256="b37ac57d4f93fc4148b2bee5d18d6efb3dcb0d52646796cf35ae1142d69217ba"
FROZEN_SEED_MANIFEST_SHA256="664737a816fca6f8afeecb04051b3a0d692a0141b5fe127b6912f5e77d467c58"


def _canonical(data: Any) -> bytes:
    return json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf8")


def _digest(data: Any) -> str:
    return hashlib.sha256(_canonical(data)).hexdigest()


def _seed(domain: str, index: int, used: set[int], excluded: frozenset[int]) -> int:
    nonce=0
    while True:
        raw=f"{SCHEMA}|{domain}|{index}|{nonce}".encode("ascii")
        value=int.from_bytes(hashlib.sha256(raw).digest()[:4],"big") & 0x7fffffff
        if value not in used and value not in excluded:
            used.add(value)
            return value
        nonce+=1


def seed_manifest() -> list[dict[str,int]]:
    used={domain:set() for domain in ("dataset","initialization","program","corruption")}
    excluded={"dataset":EXPOSED_DATA_SEEDS,"initialization":EXPOSED_INIT_SEEDS,
              "program":EXPOSED_PROGRAM_SEEDS,"corruption":frozenset()}
    return [{domain+"_seed":_seed(domain,index,used[domain],excluded[domain])
             for domain in used} for index in range(PAIR_COUNT)]


def _oracle(op: int, x: int, y: int) -> int:
    return ((x+y) if op==0 else ((x-y) if op==1 else x*y)) % MOD


def make_programs(seed: int, count: int = PROGRAM_COUNT) -> list[Any]:
    """Generate unique six-step paths with no training-domain gold transition."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("invalid public program seed")
    if type(count) is not int or not 1 <= count <= PROGRAM_COUNT:
        raise ValueError("invalid public program count")
    rng=random.Random(seed)
    seen=set()
    programs=[]
    for _ in range(20000):
        start=(rng.randrange(9,MOD),rng.randrange(9,MOD))
        state=list(start)
        commands=[]
        for _step in range(PROGRAM_STEPS):
            op=rng.randrange(OPS)
            target=rng.randrange(2)
            is_register=bool(rng.randrange(2))
            value=rng.randrange(2) if is_register else rng.randrange(9,MOD)
            x=state[target]
            y=state[value] if is_register else value
            if x<=8 and y<=8:
                break
            state[target]=_oracle(op,x,y)
            commands.append((op,target,is_register,value))
        if len(commands)!=PROGRAM_STEPS:
            continue
        item=(start,tuple(commands))
        if item not in seen:
            seen.add(item)
            programs.append(item)
            if len(programs)==count:
                return programs
    raise RuntimeError("could not create disjoint program suite")


def _score_programs(head: Any, programs: list[Any], torch: Any) -> dict[str,int]:
    exact=states_correct=0
    with torch.no_grad():
        head.eval()
        for start,commands in programs:
            model_state=list(start)
            truth_state=list(start)
            complete=True
            for op,target,is_register,value in commands:
                model_operand=model_state[value] if is_register else value
                truth_operand=truth_state[value] if is_register else value
                output=head.predict(torch.tensor(op),torch.tensor(model_state[target]),
                                    torch.tensor(model_operand))
                if output.ndim!=0:
                    raise ValueError("non-scalar neural output")
                prediction=output.item()
                if type(prediction) is not int or not 0<=prediction<MOD:
                    raise ValueError("invalid neural residue")
                model_state[target]=prediction
                truth_state[target]=_oracle(op,truth_state[target],truth_operand)
                matches=model_state==truth_state
                states_correct+=int(matches)
                complete=complete and matches
            exact+=int(complete)
    return {"exact":exact,"total":len(programs),"states_correct":states_correct,
            "states_total":len(programs)*PROGRAM_STEPS}


def _withheld_atoms(head: Any, examples: list[tuple[int,int,int,int]], torch: Any) -> dict[str,Any]:
    a=torch.arange(MOD,dtype=torch.long)
    xs=a.repeat_interleave(MOD)
    ys=a.repeat(MOD)
    trained={(op,x,y) for op,x,y,_ in examples}
    counts=[]
    wraps=[]
    for op in range(OPS):
        truth=torch.tensor([_oracle(op,int(x),int(y)) for x,y in zip(xs,ys)],dtype=torch.long)
        with torch.no_grad():
            predicted=head.predict(torch.full_like(xs,op),xs,ys)
        if predicted.shape!=truth.shape or predicted.dtype!=torch.long:
            raise ValueError("malformed batch prediction")
        keep=torch.tensor([(op,int(x),int(y)) not in trained for x,y in zip(xs,ys)])
        wrapped=torch.tensor([((int(x)+int(y)) if op==0 else
                               (int(x)-int(y) if op==1 else int(x)*int(y))) not in range(MOD)
                              for x,y in zip(xs,ys)])
        correct=(predicted==truth)&keep
        counts.append({"correct":int(correct.sum()),"total":int(keep.sum())})
        wraps.append({"correct":int((correct&wrapped).sum()),
                      "total":int((keep&wrapped).sum())})
    return {"by_op":counts,"wrap_by_op":wraps,
            "correct":sum(item["correct"] for item in counts),
            "total":sum(item["total"] for item in counts)}


def _arm(head: Any, examples: list[Any], programs: list[Any], torch: Any,
         *, optimize: bool) -> dict[str,Any]:
    if optimize:
        stats=train_rule_head_lbfgs(head,examples,torch,max_iter=MAX_ITER,
                                    max_eval=MAX_EVAL,closure_budget=CLOSURE_BUDGET,
                                    require_fit=False)
    else:
        stats={"optimizer":"none","closure_evaluations":0,"training_fit":False}
    metrics=_withheld_atoms(head,examples,torch)
    return {"status":"SCORED","training":stats,
            "coefficients_sha256":hashlib.sha256(head.coefficients.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            "withheld":metrics,"programs":_score_programs(head,programs,torch)}


def _failed(exc: Exception) -> dict[str,Any]:
    result={"status":"FAILED","reason":type(exc).__name__}
    if isinstance(exc,ClosureBudgetExhausted):
        result["closure_evaluations"]=exc.closure_evaluations
        result["closure_budget"]=CLOSURE_BUDGET
        result["weights_rolled_back"]=True
    return result


def _pair(entry: dict[str,int], torch: Any) -> dict[str,Any]:
    record={**entry,"arms":{},"win":False,"status":"PROCESSED"}
    try:
        examples=incomplete_training_atoms(entry["dataset_seed"],12)
        programs=make_programs(entry["program_seed"])
    except (RuntimeError,ValueError,ArithmeticError,TimeoutError) as exc:
        record["status"]="SETUP_FAILED"
        record["setup_failure"]=_failed(exc)
        record["arms"]={name:{"status":"NOT_RUN"} for name in (*ARMS,"frozen","corrupted")}
        return record
    record["training_atom_identity_sha256"]=_digest(examples)
    record["program_suite_sha256"]=_digest(programs)
    for basis in ARMS:
        try:
            head=build_algebraic_head(torch,seed=entry["initialization_seed"],basis=basis)
            record["arms"][basis]=_arm(head,examples,programs,torch,optimize=True)
        except (RuntimeError,ValueError,ArithmeticError,TimeoutError) as exc:
            record["arms"][basis]=_failed(exc)
    try:
        frozen=build_algebraic_head(torch,seed=entry["initialization_seed"])
        record["arms"]["frozen"]=_arm(frozen,examples,programs,torch,optimize=False)
    except (RuntimeError,ValueError,ArithmeticError,TimeoutError) as exc:
        record["arms"]["frozen"]=_failed(exc)
    try:
        corrupted=corrupt_labels_within_op(examples,entry["corruption_seed"])
        corrupt_head=build_algebraic_head(torch,seed=entry["initialization_seed"])
        record["arms"]["corrupted"]=_arm(corrupt_head,corrupted,programs,torch,optimize=True)
    except (RuntimeError,ValueError,ArithmeticError,TimeoutError) as exc:
        record["arms"]["corrupted"]=_failed(exc)
    if all(record["arms"].get(name,{}).get("status")=="SCORED"
           for name in (*ARMS,"frozen","corrupted")):
        bilinear=record["arms"]["bilinear"]["programs"]["exact"]
        no_xy=record["arms"]["no_xy"]["programs"]["exact"]
        record["win"]=10*(bilinear-no_xy)>PROGRAM_COUNT
    return record


def _source_manifest() -> dict[str,str]:
    root=Path(__file__).resolve().parent
    files={"labyrinth/minotaur_prospective_study.py":root/"minotaur_prospective_study.py",
           "labyrinth/minotaur_algebraic_rules.py":root/"minotaur_algebraic_rules.py",
           "scripts/minotaur_prospective_public.py":root.parent/"scripts/minotaur_prospective_public.py"}
    return {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}


def _audit_records(records: list[dict[str,Any]], manifest: list[dict[str,int]]) -> int:
    """Reject malformed, reordered or unverified pair summaries before inference."""
    if len(records)!=len(manifest):
        raise ValueError("incomplete public study records")
    wins=0
    for record,entry in zip(records,manifest):
        if any(type(record.get(key)) is not int or record[key]!=value
               for key,value in entry.items()):
            raise ValueError("seed manifest record mismatch")
        arms=record.get("arms")
        if not isinstance(arms,dict) or set(arms)!={*ARMS,"frozen","corrupted"}:
            raise ValueError("incomplete arm record")
        if record.get("status")=="SETUP_FAILED":
            if any(arm.get("status")!="NOT_RUN" for arm in arms.values()):
                raise ValueError("malformed setup failure")
            if record.get("win") is not False or "setup_failure" not in record:
                raise ValueError("fabricated setup result")
            continue
        if record.get("status")!="PROCESSED":
            raise ValueError("unknown pair status")
        if any(not isinstance(record.get(field),str) or len(record[field])!=64
               for field in ("training_atom_identity_sha256","program_suite_sha256")):
            raise ValueError("missing split identity")
        for arm in arms.values():
            if not isinstance(arm,dict) or arm.get("status") not in ("SCORED","FAILED"):
                raise ValueError("malformed arm status")
            if arm["status"]=="SCORED":
                programs=arm.get("programs",{})
                atoms=arm.get("withheld",{})
                if (not isinstance(programs,dict) or not isinstance(atoms,dict)
                        or type(programs.get("exact")) is not int
                        or not 0<=programs["exact"]<=PROGRAM_COUNT
                        or programs.get("total")!=PROGRAM_COUNT
                        or type(programs.get("states_correct")) is not int
                        or not 0<=programs["states_correct"]<=PROGRAM_COUNT*PROGRAM_STEPS
                        or programs.get("states_total")!=PROGRAM_COUNT*PROGRAM_STEPS
                        or not PROGRAM_STEPS*programs["exact"]<=programs["states_correct"]
                        <=(PROGRAM_STEPS-1)*PROGRAM_COUNT+programs["exact"]
                        or atoms.get("total")!=OPS*MOD*MOD-OPS*12
                        or type(atoms.get("correct")) is not int
                        or not 0<=atoms["correct"]<=atoms["total"]
                        or not isinstance(arm.get("training"),dict)
                        or not isinstance(arm.get("coefficients_sha256"),str)
                        or len(arm["coefficients_sha256"])!=64):
                    raise ValueError("invalid scored arm metrics")
            elif not isinstance(arm.get("reason"),str) or not arm["reason"]:
                raise ValueError("missing arm failure reason")
        valid=all(arm["status"]=="SCORED" for arm in arms.values())
        expected_win=(10*(arms["bilinear"]["programs"]["exact"]-
                          arms["no_xy"]["programs"]["exact"])>PROGRAM_COUNT) if valid else False
        if record.get("win") is not expected_win:
            raise ValueError("fabricated primary outcome")
        wins+=int(expected_win)
    return wins


def run_study(torch: Any, *, expected_source_manifest_sha256: str) -> dict[str,Any]:
    """Only the full frozen manifest; a caller supplies an externally pinned source digest."""
    if (str(torch.__version__)!="2.5.1+cpu" or torch.version.cuda is not None
            or torch.get_num_threads()!=1
            or not torch.are_deterministic_algorithms_enabled()):
        raise RuntimeError("unverified public study numerical execution policy")
    manifest=seed_manifest()
    if (_digest(PROTOCOL)!=FROZEN_PROTOCOL_SHA256
            or _digest(manifest)!=FROZEN_SEED_MANIFEST_SHA256):
        raise RuntimeError("frozen study protocol or seed manifest mismatch")
    source=_source_manifest()
    source_digest=_digest(source)
    if type(expected_source_manifest_sha256) is not str or source_digest!=expected_source_manifest_sha256:
        raise RuntimeError("untrusted or drifted study source manifest")
    records=[_pair(entry,torch) for entry in manifest]
    if (_source_manifest()!=source or _digest(PROTOCOL)!=FROZEN_PROTOCOL_SHA256
            or _digest(seed_manifest())!=FROZEN_SEED_MANIFEST_SHA256):
        raise RuntimeError("study source or protocol drifted during execution")
    wins=_audit_records(records,manifest)
    p_value=sum(comb(PAIR_COUNT,k) for k in range(wins,PAIR_COUNT+1))/2**PAIR_COUNT
    return {"schema":SCHEMA,"protocol":PROTOCOL,"protocol_sha256":_digest(PROTOCOL),
            "seed_manifest_sha256":_digest(manifest),"source_manifest":source,
            "source_manifest_sha256":source_digest,
            "torch_version":str(torch.__version__),"torch_cuda":str(torch.version.cuda),
            "torch_num_threads":int(torch.get_num_threads()),
            "torch_deterministic_algorithms":bool(torch.are_deterministic_algorithms_enabled()),
            "pair_count":PAIR_COUNT,"required_pairs":PAIR_COUNT,
            "wins":wins,"exact_binomial_p_one_sided":p_value,
            "public_primary_gate":"PASS" if wins>=59 else "FAIL",
            "records":records,"qualification":"NOT_ESTABLISHED",
            "graduation_authority":"none","minotaur_graduated":False,
            "deployed":False,"independent_private_oracle":False,
            "limits":["Exposed public arithmetic and feature family; this is seed replication, not blind exam",
                      "A bilinear multiplication feature and modulo-97 projection are supplied priors",
                      "Published seed manifest and source digests do not independently attest execution"]}


def run_development_smoke(torch: Any) -> dict[str,Any]:
    """Exercise the pipeline on *previously exposed* seeds, never prospective pairs."""
    entry={"dataset_seed":47,"initialization_seed":19,"program_seed":808,
           "corruption_seed":59}
    return {"schema":SCHEMA,"phase":"DEVELOPMENT_SMOKE","record":_pair(entry,torch),
            "public_primary_gate":"INCOMPLETE","exact_binomial_p_one_sided":None,
            "qualification":"NOT_ESTABLISHED","graduation_authority":"none",
            "minotaur_graduated":False,"deployed":False}
