"""Deterministic public V7 multiplication challenges, with no neural imports.

Arithmetic ground truth is available ONLY in data generation and audit.
Training callers must pass model_features(rows), never full labeled rows.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import random

MOD97, MOD101, ABSENT_LABEL = 97, 101, 53

@dataclass(frozen=True)
class BenchmarkDataset:
    modulus: int
    splits: dict[str, tuple[tuple[int,int,int], ...]]
    digests: dict[str,str]
    counts: dict[str,dict[str,int]]
    absent_label: int|None
    variant: str
    seed: int

@dataclass(frozen=True)
class ExternalEvaluation:
    modulus: int
    domain: str
    rows: tuple[tuple[int,int,int], ...]
    digest: str
    seed: int

def data_oracle(x:int,y:int,modulus:int)->int:
    if (type(x) is not int or type(y) is not int or type(modulus) is not int
            or modulus not in (MOD97,MOD101) or not 0<=x<modulus or not 0<=y<modulus):
        raise ValueError("invalid modular arithmetic input")
    return (x*y)%modulus

def _digest(rows):
    return hashlib.sha256("\n".join(
        f"{x}\t{y}\t{z}" for x,y,z in rows).encode("ascii")).hexdigest()

def _groups(modulus):
    for x in range(modulus):
        for y in range(x,modulus):
            yield (x,y)

def _rows(groups,modulus):
    rows=[]
    for x,y in groups:
        rows.append((x,y,data_oracle(x,y,modulus)))
        if x!=y:
            rows.append((y,x,data_oracle(y,x,modulus)))
    return tuple(sorted(rows))

def _package(seed,variant,train_groups,dev_groups,test_groups,absent=None):
    pieces={"train":_rows(train_groups,MOD97),
            "dev":_rows(dev_groups,MOD97),
            "test":_rows(test_groups,MOD97)}
    if sum(len(v) for v in pieces.values())!=MOD97*MOD97:
        raise RuntimeError("incomplete source domain")
    return BenchmarkDataset(MOD97,pieces,{k:_digest(v) for k,v in pieces.items()},
        {k:{"rows":len(v),"groups":len({(min(x,y),max(x,y)) for x,y,_ in v})}
            for k,v in pieces.items()},absent,variant,seed)

def build_mod97_dataset(seed:int=20261028,variant:str="grouped",absent_label:int=ABSENT_LABEL):
    if type(seed) is not int or not 0<=seed<2**31:
        raise ValueError("seed must be bounded integer")
    if type(absent_label) is not int or not 0<=absent_label<MOD97:
        raise ValueError("invalid absent label")
    if variant not in ("grouped","absent_output","absent_operand_and_output"):
        raise ValueError("invalid challenge variant")
    shuffled=list(_groups(MOD97))
    random.Random(seed).shuffle(shuffled)
    if variant=="grouped":
        test_count=round(.10*len(shuffled))
        dev_count=round(.10*len(shuffled))
        test=shuffled[:test_count]
        dev=shuffled[test_count:test_count+dev_count]
        train=shuffled[test_count+dev_count:]
        absent=None
    else:
        test=[]
        available=[]
        for x,y in shuffled:
            target=data_oracle(x,y,MOD97)
            blocked=(target==absent_label or
                     variant=="absent_operand_and_output" and
                     (x==absent_label or y==absent_label))
            (test if blocked else available).append((x,y))
        dev_count=max(1,round(.10*len(available)))
        dev=available[:dev_count]
        train=available[dev_count:]
        absent=absent_label
    return _package(seed,variant,train,dev,test,absent)

def build_absent_output_stress_variants(seed:int=20261028):
    return {label:build_mod97_dataset(seed,"absent_output",label)
            for label in range(MOD97)}

def model_features(rows):
    return tuple((x,y) for x,y,_ in rows)

def build_mod101_evaluation(seed:int=20261028):
    if type(seed) is not int or seed<0 or seed>=2**31:
        raise ValueError("invalid seed")
    groups=list(_groups(MOD101))
    random.Random(seed).shuffle(groups)
    rows=_rows(groups,MOD101)
    return ExternalEvaluation(MOD101,"evaluation_only_mod101",rows,_digest(rows),seed)

def evaluate_baselines(dataset:BenchmarkDataset,seed:int=20261028):
    """Do not train on dev or test; compute only ground-truth comparisons."""
    if not isinstance(dataset,BenchmarkDataset) or not dataset.splits["train"]:
        raise ValueError("nonempty training set required")
    if type(seed) is not int or not 0<=seed<2**31:
        raise ValueError("invalid seed")
    constant=Counter(label for _,_,label in dataset.splits["train"]).most_common(1)[0][0]
    reports={}
    for name in ("zero","constant","copy_left","random"):
        runs={}
        for split,rows in dataset.splits.items():
            rng=random.Random(seed + {"train":0,"dev":1,"test":2}[split])
            predictions=[0 if name=="zero" else
                         constant if name=="constant" else
                         x if name=="copy_left" else
                         rng.randrange(dataset.modulus)
                         for x,y,label in rows]
            hits=sum(g==label for g,(_,_,label) in zip(predictions,rows))
            runs[split]={"accuracy":hits/len(rows) if rows else 0.0,
                         "correct":hits,"count":len(rows),
                         "fit_source":"train" if name=="constant" else None}
        reports[name]=runs
    return reports
