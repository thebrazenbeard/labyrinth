"""Conservative CI evidence summary; never promote a pilot to production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def verdict(train: dict[str, Any], reload: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    if audit.get("status") != "PASS":
        raise ValueError("the offline HF report audit was not complete and passing")
    if not reload.get("process_id_differs") or not reload.get("matches_original_candidate_trials"):
        raise ValueError("the fresh-process adapter reload did not reproduce the candidate")
    if reload.get("reloaded_lora_tensor_value_sha256") != train.get("candidate_lora_tensor_value_sha256"):
        raise ValueError("reloaded candidate tensor digest did not match training")
    if train.get("initial_lora_parameter_sha256") == train.get("candidate_lora_parameter_sha256"):
        raise ValueError("training did not change any LoRA parameters")
    scores = train.get("evaluations", {})
    if set(scores) != {"baseline", "sham", "corrupted", "candidate"}:
        raise ValueError("four controlled evaluation arms are required")
    counts: dict[str, int] = {}
    totals = set()
    for arm, result in scores.items():
        if type(result.get("correct")) is not int or type(result.get("total")) is not int:
            raise ValueError(f"invalid score for arm {arm}")
        if not (0 <= result["correct"] <= result["total"]):
            raise ValueError(f"out-of-range score for arm {arm}")
        totals.add(result["total"])
        counts[arm] = result["correct"]
    if len(totals) != 1 or min(totals) < 2:
        raise ValueError("all evaluation arms must share a nontrivial held-out set")
    candidate_wins = min(counts["candidate"] - counts[comp]
                         for comp in ("baseline", "sham", "corrupted"))
    return {
        "schema": "labyrinth.ci-pretrained-evidence.v1",
        "status": "EVIDENCE_ONLY_DO_NOT_PROMOTE",
        "fresh_process_reloaded": True,
        "internal_report_audit": audit["status"],
        "heldout_total_per_arm": totals.pop(),
        "semantic_correct": counts,
        "candidate_minimum_count_margin": candidate_wins,
        "demonstrated_semantic_advantage_over_all_controls": candidate_wins > 0,
        "graduation_eligible": False,
        "limitations": [
            "Public finite generator with shared task families is not a blind independent test.",
            "The unchanged adapter sham is not compute matched to 32 optimizer steps.",
            "Single seed, small sample size, same-owner report and runner; no independent examiner attestation.",
            "Semantic wins, if any, require a new sealed cross-task examination before promotion.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a real pretrained HF pilot without promotion")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--reload", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = verdict(*(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.train, args.reload, args.audit)
    ))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
