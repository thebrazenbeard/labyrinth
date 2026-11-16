"""Bounded public synthetic Minotaur trial; NOT a held-out model assessment.

One real local model subprocess is compared with a zero-update fixed-action
control per public Timegate case. Private sealed/unsealed variants are labeled
only in host-side development reports. Neither matching public start nor model
identity self-label establishes semantic qualification or origin attestation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

from .adapters import OllamaAdapter
from .minotaur_candidate_bridge import (
    FixedActionAdapter, SubprocessCandidateAdapter, audit_receipt, run_candidate,
)
from .timegate_exam import CandidateSession, RULESET, candidate_view, make_case

_SCHEMA = "MINOTAUR_LOCAL_DEV_TRIAL_V1"
_CHILD = Path(__file__).resolve().parents[1] / "scripts" / "minotaur_ollama_child.py"


def _case_result(receipt: Any, public: dict[str, object], factory: Any) -> dict[str, Any]:
    return {
        "model_id_unattested": receipt.model_id_unattested,
        "adapter_provenance": receipt.adapter_provenance,
        "final_verdict": receipt.final_verdict,
        "verified_escape": receipt.verified_escape,
        "audit_replay": audit_receipt(receipt, public, factory),
        "receipt_self_sha256": receipt.receipt_sha256,
        "source_commit_unattested": receipt.source_commit_unattested,
        "turn_count": len(receipt.turns),
        "dispositions": [turn.disposition for turn in receipt.turns],
        "responses": [turn.response_text for turn in receipt.turns],
    }


def run_local_trial(
    model: str, *, endpoint: str = "http://127.0.0.1:11434",
    seeds: tuple[int, ...] = (7,),
    sealed_modes: tuple[bool, ...] = (False, True),
    max_calls: int = 8,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Execute public development cases using identical initial task input.

    A completed source-level replay checks *conditional* examiner actions,
    not model origin, independent private case custody or learning.
    """
    if (type(model) is not str or not model.strip() or len(model) > 128
            or any(ord(c) < 32 for c in model)):
        raise ValueError("local model name must be bounded printable text")
    if (type(seeds) is not tuple or not 1 <= len(seeds) <= 4
            or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be a bounded tuple of distinct 31-bit integers")
    if (type(sealed_modes) is not tuple or not 1 <= len(sealed_modes) <= 2
            or any(type(s) is not bool for s in sealed_modes)
            or len(set(sealed_modes)) != len(sealed_modes)):
        raise ValueError("sealed_modes must be a tuple of distinct booleans")
    if (type(max_calls) is not int or not 1 <= max_calls <= 16
            or len(seeds) * len(sealed_modes) * max_calls > 16):
        raise ValueError("maximum of 16 model calls across this development trial")
    if (type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or not 5 <= timeout_seconds <= 35):
        raise ValueError("child timeout must be between 5 and 35 seconds")
    # Validate locality before any session construction, subprocess, or HTTP.
    OllamaAdapter(model, base_url=endpoint, think=False)
    child = SubprocessCandidateAdapter(
        (sys.executable, str(_CHILD), model, endpoint),
        model_id=f"ollama:{model}:unattested", timeout_seconds=timeout_seconds)
    results: list[dict[str, Any]] = []
    for seed in seeds:
        for sealed in sealed_modes:
            # Same host-side case is reconstructed independently for the two
            # runs. No private case object is placed in the child request.
            case = make_case(seed, sealed=sealed)
            public = candidate_view(case)
            factory = lambda case=case: CandidateSession(case)
            begin = time.monotonic()
            model_run = run_candidate(
                public, factory, child, seed=seed,
                max_model_calls=max_calls, authorize_local_adapter=True,
                require_process_separation=True)
            model_elapsed = time.monotonic() - begin
            # Controls are stateful scripts; never carry their cursor into
            # another, even publicly identical, private case.
            control = FixedActionAdapter(
                tuple("forward" for _ in range(max_calls)),
                model_id="fixed:forward:zero-update")
            control_run = run_candidate(
                public, factory, control, seed=seed,
                max_model_calls=max_calls)
            public_json = json.dumps(public, sort_keys=True,
                                     ensure_ascii=False, separators=(",", ":"))
            results.append({
                "seed": seed,
                "sealed_for_development_only": sealed,
                "public_sha256": sha256(public_json.encode("utf-8")).hexdigest(),
                "candidate_elapsed_seconds": round(model_elapsed, 3),
                "candidate": _case_result(model_run, public, factory),
                "control": _case_result(control_run, public, factory),
            })
    return {
        "schema": _SCHEMA,
        "case_family": RULESET,
        "synthetic_public_development_only": True,
        "model_origin_attested": False,
        "comparison_qualified": False,
        "learning_qualified": False,
        "private_examiner_custody_qualified": False,
        "receipt_selfhash_only": True,
        "requested_model_unattested": model,
        "max_model_calls_per_case": max_calls,
        "cases": results,
    }


def write_trial_new(path: str | Path, trial: dict[str, Any]) -> None:
    """Exclusively persist a source-only report; never overwrite evidence."""
    if (type(trial) is not dict or trial.get("schema") != _SCHEMA
            or trial.get("comparison_qualified") is not False
            or trial.get("model_origin_attested") is not False
            or trial.get("learning_qualified") is not False
            or type(trial.get("cases")) is not list
            or len(trial["cases"]) > 8):
        raise ValueError("not an unqualified bounded local trial")
    encoded = (json.dumps(trial, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > 512_000:
        raise ValueError("trial record exceeds bounded evidence size")
    target = Path(path)
    if target.is_symlink():
        raise FileExistsError("refuse to follow evidence symlink")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    with os.fdopen(descriptor, "wb") as file:
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Public Timegate local-model trial; NOT a held-out test.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--sealed", choices=("both", "false", "true"), default="both")
    parser.add_argument("--max-calls", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--authorize-local-inference", action="store_true")
    args = parser.parse_args(argv)
    if not args.authorize_local_inference:
        parser.error("explicit --authorize-local-inference is required")
    try:
        seeds = tuple(int(s) for s in args.seeds.split(","))
    except ValueError:
        parser.error("seeds must be comma-separated 31-bit integers")
    sealed_modes = ((False, True) if args.sealed == "both"
                    else (args.sealed == "true",))
    report = run_local_trial(
        args.model, endpoint=args.endpoint, seeds=seeds,
        sealed_modes=sealed_modes, max_calls=args.max_calls,
        timeout_seconds=args.timeout_seconds)
    write_trial_new(args.output, report)
    print(json.dumps({
        "schema": _SCHEMA, "output": str(args.output),
        "cases": len(report["cases"]), "comparison_qualified": False,
        "learning_qualified": False,
        "results": [{
            "seed": case["seed"],
            "sealed_development_only": case["sealed_for_development_only"],
            "candidate_verdict": case["candidate"]["final_verdict"],
            "candidate_replay": case["candidate"]["audit_replay"],
            "candidate_escape": case["candidate"]["verified_escape"],
            "control_escape": case["control"]["verified_escape"],
        } for case in report["cases"]],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
