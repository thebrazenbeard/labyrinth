"""Source-frozen independent replay of a numeric multi-head neural experiment.

The independent public evaluator never sees model weights or target labels at
inference.  The prediction snapshot contains (x,y,predicted) triples only.
This is public synthetic research, not a private examiner or graduation gate.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

def canonical_prediction_json(captured):
    if type(captured) is not dict:
        raise ValueError("expected seed-indexed prediction records")
    known={"unseen_group_pairs":97, "missing_output_53":97,
           "missing_input_53":97, "mod101":101}
    records=[]
    for seed,mapping in sorted(captured.items()):
        if type(seed) is not int or seed<0 or type(mapping) is not dict:
            raise ValueError("invalid prediction seed")
        challenges={}
        for name,predictions in sorted(mapping.items()):
            if name not in known or type(predictions) is not dict:
                raise ValueError("unknown challenge")
            limit=known[name]
            triples=[]
            for (x,y),value in sorted(predictions.items()):
                if (type(x) is not int or type(y) is not int
                    or not 0<=x<limit or not 0<=y<limit
                    or type(value) is not int or not 0<=value<limit):
                    raise ValueError("invalid model prediction")
                triples.append([x,y,value])
            challenges[name]=triples
        records.append({"seed":seed,"cases":challenges})
    return json.dumps({"schema":"labyrinth.minotaur.v7-neural-predictions.v1",
                       "records":records},sort_keys=True,separators=(",",":"))

def run_independent_replay(report_path,predictions_path):
    from .minotaur_v7_intermediate import (
        IntermediateConfig, run_intermediate_study, build_intermediate_split)
    from .minotaur_v7_blindish_eval import build_protocol, audit_predictions
    report_path=Path(report_path)
    predictions_path=Path(predictions_path)
    if report_path.exists() or predictions_path.exists():
        raise FileExistsError("refuse to overwrite existing evidence")
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise RuntimeError("refuse to assess uncommitted source")
    source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    config=IntermediateConfig()
    captures={}
    def sink(seed,name,values):
        if name in captures.setdefault(seed,{}):
            raise ValueError("duplicate model evidence")
        captures[seed][name]=values
    neural=run_intermediate_study(config,prediction_sink=sink)
    protocol=build_protocol(seed=config.seed,holdout_fraction=config.holdout_fraction)
    train=build_intermediate_split(seed=config.seed,holdout_fraction=config.holdout_fraction).train
    training_rows=tuple(zip(train.x.tolist(),train.y.tolist(),train.remainder.tolist()))
    if set(training_rows)!=set(protocol.train):
        raise RuntimeError("independent and model dataset membership disagree")
    verified={}
    names={"independent_holdout":"unseen_group_pairs",
           "absent_output_53":"missing_output_53",
           "absent_operand_53":"missing_input_53",
           "modulus_101":"mod101"}
    for run in neural["runs"]:
        seed=run["seed"]
        if set(captures.get(seed,{}))!=set(protocol.challenges):
            raise RuntimeError("incomplete prediction capture")
        checked=audit_predictions(protocol,captures[seed],training_rows=training_rows)
        if not checked["integrity"]["ok"]:
            raise RuntimeError("independent source mismatch")
        for ours,theirs in names.items():
            if checked["cases"][theirs]["correct"]!=run["evaluation"][ours]["correct"]:
                raise RuntimeError("neural and independently replayed scores disagree")
            if checked["cases"][theirs]["count"]!=run["evaluation"][ours]["count"]:
                raise RuntimeError("neural and independent counts disagree")
            if checked["cases"][theirs]["abstained"]!=0:
                raise RuntimeError("incomplete output disallowed in training replay")
        verified[str(seed)]={"cases":checked["cases"],
                             "protocol_digests":checked["protocol_digests"],
                             "protocol_counts":checked["protocol_counts"],
                             "integrity":checked["integrity"],
                             "qualification":checked["qualification"]}
    raw=canonical_prediction_json(captures)
    blob=gzip.compress(raw.encode("utf-8"),compresslevel=9,mtime=0)
    report={
        "schema":"labyrinth.minotaur.v7-intermediate-independent-replay.v1",
        "source":{"commit":source,"clean":True,
                  "model_sha256":hashlib.sha256(
                      Path("labyrinth/minotaur_v7_intermediate.py").read_bytes()).hexdigest(),
                  "examiner_sha256":hashlib.sha256(
                      Path("labyrinth/minotaur_v7_blindish_eval.py").read_bytes()).hexdigest()},
        "model":neural,
        "independent_public_replay":verified,
        "predictions":{"filename":predictions_path.name,
                       "json_sha256":hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                       "gzip_sha256":hashlib.sha256(blob).hexdigest(),
                       "bytes":len(blob)},
        "qualification":"NOT_ESTABLISHED",
        "promotion_recommendation":"DO_NOT_PROMOTE",
        "limitations":["Public deterministic challenges, not a private examiner.",
                       "Auxiliary quotient and product targets used offline during training.",
                       "No output-label-53 or modulus-101 qualification established."]}
    report_path.parent.mkdir(parents=True,exist_ok=True)
    predictions_path.parent.mkdir(parents=True,exist_ok=True)
    predictions_path.write_bytes(blob)
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--report",required=True)
    parser.add_argument("--predictions",required=True)
    args=parser.parse_args()
    report=run_independent_replay(args.report,args.predictions)
    for seed,audit in report["independent_public_replay"].items():
        print("AUDITED_SEED",seed,{k:(v["correct"],v["count"]) for k,v in audit["cases"].items()})
    print("SOURCE",report["source"]["commit"],
          "EVIDENCE_SHA256",report["predictions"]["gzip_sha256"])

if __name__=="__main__":
    main()

