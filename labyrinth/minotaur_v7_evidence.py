"""Fail-closed audit of public, negative V7 neural evidence; no promotion authority."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

EXPECTED_REPORT_SHA="34e731a1c8699166eb78add7380053c30107c4ee587b7a61ea0ae22b64c6e27b"
EXPECTED_SOURCE="c52d56225e43612add05f181a5323522e7a6c627"
EXPECTED_REPORT_CANONICAL_SHA="6057ad7b3f43a2bb2ea10df2c6823b65fe09effddb5080570bf14ce8d99fd18d"
EXPECTED_SOURCE_CANONICAL_SHA={
    "minotaur_v7_numeric.py":"3bfda0695bbd508b1c4c5a961c20badfff253198791abe985d0dc13d7c028ec5",
    "minotaur_v7_benchmark.py":"681c84661d51693681d3aa73fe3bb4d55af8f95a16bfdbb3de418761727f3f1b",
}
EXPECTED_SOURCE_ORIGINAL_SHA={
    "minotaur_v7_numeric.py":"25ac065a25b72ce7d5a335b2f7dd05ebe906b282db979fec850efcbb2bba6772",
    "minotaur_v7_benchmark.py":"681c84661d51693681d3aa73fe3bb4d55af8f95a16bfdbb3de418761727f3f1b",
}
def canonical_sha256(raw:bytes)->str:
    """Canonical Git LF bytes; CRLF checkout does not alter evidence identity."""
    if b"\r" in raw.replace(b"\r\n",b""):
        raise ValueError("unsupported bare CR in source/evidence")
    return hashlib.sha256(raw.replace(b"\r\n",b"\n")).hexdigest()

EXPECTED_SEEDS=(20261028,20261029)
EXPECTED_COUNTS={"independent_holdout":910,"target_53":94,
                 "input_53":193,"modulus_101":10201}

def audit_v7_report(path, root):
    """Verify recorded evidence structure and local source, not model competence."""
    path=Path(path);root=Path(root)
    raw=path.read_bytes()
    digest=canonical_sha256(raw)
    if path.name=="MINOTAUR_V7_NUMERIC_2026-10-28.json" and digest!=EXPECTED_REPORT_CANONICAL_SHA:
        raise ValueError("frozen evidence hash changed")
    try:
        report=json.loads(raw)
        source=report["source"]
        runs=report["runs"]
    except (KeyError,ValueError,TypeError) as exc:
        raise ValueError("invalid evidence JSON") from exc
    if (report.get("schema")!="labyrinth.minotaur.v7-numeric.v1"
        or report.get("qualification")!="NOT_ESTABLISHED"
        or report.get("promotion_recommendation")!="DO_NOT_PROMOTE"
        or source.get("head")!=EXPECTED_SOURCE or source.get("clean") is not True):
        raise ValueError("unqualified or non-source-bound evidence")
    for name,key in (("minotaur_v7_numeric.py","model_sha256"),
                     ("minotaur_v7_benchmark.py","benchmark_sha256")):
        file=root/"labyrinth"/name
        if (source.get(key)!=EXPECTED_SOURCE_ORIGINAL_SHA[name]
                or canonical_sha256(file.read_bytes())!=EXPECTED_SOURCE_CANONICAL_SHA[name]):
            raise ValueError("source code no longer matches frozen training")
    if (type(runs) is not list or len(runs)!=2 or
        tuple(run.get("seed") for run in runs)!=EXPECTED_SEEDS):
        raise ValueError("wrong seed count or order")
    missed=[]
    for run in runs:
        split=run.get("split",{})
        if (split.get("target_53_train_rows")!=0 or
            split.get("input_53_train_rows")!=0 or
            split.get("train_count")!=8212 or
            split.get("independent_holdout_count")!=910):
            raise ValueError("missing class or operand leakage control")
        controls=run.get("controls",{})
        if set(controls)!={"constant","frozen_random"}:
            raise ValueError("missing neural baselines")
        train=run.get("training",{})
        if (type(train.get("optimizer_steps")) is not int or
            train["optimizer_steps"]<=0 or
            train.get("initial_sha256")==train.get("trained_sha256")):
            raise ValueError("training update not established")
        observations=run.get("evaluation",{})
        for name,count in EXPECTED_COUNTS.items():
            value=observations.get(name,{})
            if (value.get("count")!=count or
                type(value.get("correct")) is not int or
                not 0<=value["correct"]<=count or
                abs(value.get("accuracy",-1)-value["correct"]/count)>1e-12 or
                not all(name in controls[control] for control in controls)):
                raise ValueError("invalid evaluation or control count")
        missed.append(observations["target_53"]["correct"])
    return {"report_sha256":digest,"source":EXPECTED_SOURCE,"run_count":len(runs),
            "target_53_correct":missed,"no_promotion":True,
            "audit_scope":"source/hash/record integrity only; not independent neural inference"}
