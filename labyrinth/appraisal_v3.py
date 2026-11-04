"""Prospective V3 synthetic appraisal experiment (execution gated).

Design-only lock: docs/APPRAISAL_V3_DESIGN_LOCK_PENDING.md, GitHub PR #119
head f412103a3b290ff1019ca4e044afb80e9fad66bb. This implementation
intentionally does not execute training or sample holdouts on import.

No LLM training, private data, or claim of subjective appraisal is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .appraisal_experiment import (
    AppraisalHead, _canonical, _expected, _train_probability, train,
)


PROTOCOL = "labyrinth.synthetic-appraisal-prospective.v3"
DESIGN_HEAD = "f412103a3b290ff1019ca4e044afb80e9fad66bb"
BASE_V2_HEAD = "a38ea3e3a679b88ce831791992b21201f0a088ee"
PYTHON_VERSION = "3.12.14"
SEEDS = (101, 307, 911, 2027, 4049)
TRAIN_STEPS = 3200
NULL_REPEATS = 250
EVAL_CASES = 1500
MAX_WALL_SECONDS = 900
NULL_P95_INDEX = 237  # nearest-rank ceil(0.95*250)-1, ZERO-indexed
EXPLORATION_EPSILON = 0.20
POLICY_TEMPERATURE = 0.15
H2_PROBES = (-1.0, -0.9, -0.75, -0.5, -0.25, -0.12, -0.01,
             0.0, 0.01, 0.12, 0.25, 0.5, 0.75, 0.9, 1.0)

# Integer tuple packing is injective for [0,2**31) seed and [0,1024)
# replicate. Domains occupy disjoint high bits. Never use unstable hash().
DOMAINS = {
    "null_shuffle": 1,
    "null_actions": 2,
    "context_C": 3,
    "context_D": 4,
    "eval_actions_C": 5,
    "eval_actions_D": 6,
    "eval_rewards_C": 7,
    "eval_rewards_D": 8,
}


def rng_seed(domain: str, seed: int, replicate: int = 0) -> int:
    if domain not in DOMAINS:
        raise ValueError("unregistered RNG domain")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed outside fixed packing bounds")
    if type(replicate) is not int or not 0 <= replicate < 1024:
        raise ValueError("replicate outside fixed packing bounds")
    return (DOMAINS[domain] << 42) | (seed << 10) | replicate


def protocol_manifest() -> dict[str, Any]:
    """Pure constants; never samples a V3 evaluation generator."""
    return {"protocol": PROTOCOL, "design_head": DESIGN_HEAD,
            "base_v2_head": BASE_V2_HEAD,
            "python_version_exact": PYTHON_VERSION,
            "python_implementation": "cpython",
            "seeds": list(SEEDS), "train_steps": TRAIN_STEPS,
            "null_repeats": NULL_REPEATS, "eval_cases_per_family": EVAL_CASES,
            "null_p95_index_zero_based": NULL_P95_INDEX,
            "null_p95_rule": "nearest_rank_ceil_0p95_n_minus_one",
            "rng_domains": DOMAINS,
            "context_digest_rule": "sha256_canonical_json_family_seed_float_hex_list",
            "h2_probe_context_hex": [v.hex() for v in H2_PROBES],
            "checkpoint_equality": "exact_float64_values_and_expected_payoff",
            "max_wall_seconds": MAX_WALL_SECONDS,
            "max_training_actions": len(SEEDS) * TRAIN_STEPS * (1 + NULL_REPEATS),
            "max_evaluation_context_policy_pairs": (
                len(SEEDS) * EVAL_CASES * 2 * (2 + NULL_REPEATS)),
            "eval_reward_draws": len(SEEDS) * EVAL_CASES * 2 * (2 + NULL_REPEATS),
            "null_trace_sha256_count": len(SEEDS) * NULL_REPEATS,
            "realized_outcomes_role": "secondary_descriptive_not_for_selection",
            "status": "DESIGN_ONLY_NOT_EXECUTED"}


def context_digest(family: str, seed: int, contexts: list[float]) -> str:
    if family not in ("C", "D") or seed not in SEEDS:
        raise ValueError("unexpected family or seed")
    if not isinstance(contexts, list) or len(contexts) != EVAL_CASES:
        raise ValueError("context count must equal frozen protocol")
    if any(type(x) is not float or not math.isfinite(x) or not -1 <= x <= 1
           for x in contexts):
        raise ValueError("nonfinite or out-of-domain context")
    return hashlib.sha256(_canonical({
        "family": family, "seed": seed,
        "contexts_float_hex": [x.hex() for x in contexts],
    })).hexdigest()


def _contexts(family: str, seed: int) -> list[float]:
    """MUST NOT BE CALLED before the externally approved run gate."""
    rng = random.Random(rng_seed("context_" + family, seed))
    cases: list[float] = []
    for _ in range(EVAL_CASES):
        if family == "C":
            sign = -1 if rng.random() < 0.5 else 1
            magnitude = rng.uniform(0.55, 1) if rng.random() < 0.8 else rng.uniform(0, 0.25)
            cases.append(float(sign * magnitude))
        elif family == "D":
            cases.append(float(rng.uniform(-0.12, 0.12)))
        else:
            raise ValueError("unexpected family")
    return cases


def _eval_softmax_expected(head: AppraisalHead, contexts: list[float]) -> float:
    total = 0.0
    for x in contexts:
        p0 = _train_probability(head, x)
        total += p0 * _expected(x, 0) + (1 - p0) * _expected(x, 1)
    return total / len(contexts)


def _eval_policy(head: AppraisalHead, contexts: list[float], *,
                 family: str, seed: int, arm: int) -> dict[str, float]:
    """Fixed-policy expected primary score; separate sampled descriptive outcome."""
    actions = random.Random(rng_seed("eval_actions_" + family, seed, arm))
    rewards = random.Random(rng_seed("eval_rewards_" + family, seed, arm))
    expected_total = 0.0
    realized_total = 0
    for x in contexts:
        p0 = _train_probability(head, x)
        expected_total += p0 * _expected(x, 0) + (1 - p0) * _expected(x, 1)
        action = 0 if actions.random() < p0 else 1
        realized_total += int(rewards.random() < _expected(x, action))
    return {"expected_payoff": expected_total / len(contexts),
            "realized_payoff": realized_total / len(contexts)}


def _null_training(seed: int, replicate: int, rewards: list[int]) -> tuple[AppraisalHead, str, int]:
    if len(rewards) != TRAIN_STEPS or any(type(x) is not int or x not in (0, 1)
                                         for x in rewards):
        raise ValueError("invalid candidate reward multiset")
    shuffled = list(rewards)
    random.Random(rng_seed("null_shuffle", seed, replicate)).shuffle(shuffled)
    contexts = random.Random(seed * 131 + 13)  # candidate's exact context tape
    actions = random.Random(rng_seed("null_actions", seed, replicate))
    head = AppraisalHead.zero()
    transcript = hashlib.sha256()
    for step, outcome in enumerate(shuffled):
        x = contexts.uniform(-1, 1)
        p0 = _train_probability(head, x)
        action = 0 if actions.random() < p0 else 1
        chosen_propensity = p0 if action == 0 else 1 - p0
        transcript.update(_canonical({"step": step, "context": x.hex(),
                                      "action": action, "propensity": chosen_propensity.hex(),
                                      "reward": outcome}) + b"\n")
        head.learn(x, action, outcome)
    return head, transcript.hexdigest(), TRAIN_STEPS


def _summary(scores: list[float], candidate: float) -> dict[str, float | int]:
    if len(scores) != NULL_REPEATS:
        raise ValueError("incomplete null cohort")
    ordered = sorted(scores)
    return {"count": len(scores), "p05": ordered[math.ceil(.05 * NULL_REPEATS) - 1],
            "median": (ordered[124] + ordered[125]) / 2,
            "p95": ordered[NULL_P95_INDEX], "max": ordered[-1],
            "within_0p06_or_above_fraction": (
                sum(v >= candidate - 0.06 for v in scores) / NULL_REPEATS),
            "at_or_above_candidate": sum(v >= candidate for v in scores)}


def _checkpoint_fresh_process(head: AppraisalHead, seed: int,
                              contexts_c: list[float]) -> dict[str, Any]:
    """A separate interpreter must reproduce exact values and C payoff."""
    with tempfile.TemporaryDirectory(prefix="labyrinth-v3-") as directory:
        path = Path(directory) / "head.json"
        sha = head.save_new(path)
        parent_values = [list(head.values(x)) for x in H2_PROBES]
        expected = _eval_softmax_expected(head, contexts_c)
        proc = subprocess.run([
            sys.executable, "-m", "labyrinth.appraisal_v3", "verify-checkpoint",
            "--checkpoint", str(path), "--seed", str(seed),
        ], input=_canonical({"contexts": [x.hex() for x in contexts_c]}),
            capture_output=True, check=True, timeout=30)
        record = json.loads(proc.stdout)
        if (record.get("pid") == os.getpid()
                or record.get("values") != parent_values
                or record.get("expected_payoff") != expected):
            raise RuntimeError("fresh-process head readback divergence")
        return {"checkpoint_sha256": sha, "h2_fresh_process": "PASS",
                "child_pid_distinct": True}


def _actual_run() -> dict[str, Any]:
    """All five fixed seeds; invoked only after source/owner guard in CLI."""
    started = time.monotonic()
    records = []
    null_beating_count = 0
    p95_margin_passes = 0
    candidate_frozen_passes = 0
    for seed in SEEDS:
        if time.monotonic() - started >= MAX_WALL_SECONDS:
            raise TimeoutError("V3 wall-clock execution budget exhausted: unqualified")
        candidate, trace = train(seed, steps=TRAIN_STEPS)
        labels = [row["reward"] for row in trace]
        contexts = {family: _contexts(family, seed) for family in ("C", "D")}
        manifests = {family: context_digest(family, seed, c)
                     for family, c in contexts.items()}
        eval_by_family: dict[str, dict[str, Any]] = {}
        for family, c in contexts.items():
            candidate_eval = _eval_policy(candidate, c, family=family,
                                          seed=seed, arm=0)
            frozen_eval = _eval_policy(AppraisalHead.zero(), c,
                                       family=family, seed=seed, arm=1)
            eval_by_family[family] = {
                "candidate": candidate_eval["expected_payoff"],
                "frozen": frozen_eval["expected_payoff"],
                "disabled": frozen_eval["expected_payoff"],
                "candidate_realized": candidate_eval["realized_payoff"],
                "frozen_realized": frozen_eval["realized_payoff"],
                "disabled_realized": frozen_eval["realized_payoff"],
                "null_realized_scores": [],
                "null_scores": [], "context_sha256": manifests[family],
            }
        trace_hashes = []
        for replicate in range(NULL_REPEATS):
            if time.monotonic() - started >= MAX_WALL_SECONDS:
                raise TimeoutError("V3 wall-clock execution budget exhausted: unqualified")
            null, audit_hash, steps = _null_training(seed, replicate, labels)
            if steps != TRAIN_STEPS:
                raise RuntimeError("null propensity trace incomplete")
            trace_hashes.append(audit_hash)
            for family, c in contexts.items():
                evaluation = _eval_policy(null, c, family=family,
                                          seed=seed, arm=replicate + 2)
                eval_by_family[family]["null_scores"].append(
                    evaluation["expected_payoff"])
                eval_by_family[family]["null_realized_scores"].append(
                    evaluation["realized_payoff"])
        for family, row in eval_by_family.items():
            row["nulls"] = _summary(row.pop("null_scores"), row["candidate"])
            row["null_realized_mean"] = (
                sum(row.pop("null_realized_scores")) / NULL_REPEATS)
            row["candidate_minus_frozen"] = row["candidate"] - row["frozen"]
            row["candidate_minus_null_p95"] = row["candidate"] - row["nulls"]["p95"]
        b = eval_by_family["C"]
        candidate_frozen_passes += int(b["candidate_minus_frozen"] > 0.08)
        p95_margin_passes += int(b["candidate_minus_null_p95"] > 0.03)
        null_beating_count += b["nulls"]["at_or_above_candidate"]
        records.append({"seed": seed, "evaluations": eval_by_family,
                        "null_pre_feedback_trace_sha256": trace_hashes,
                        "null_logged_propensities": len(trace_hashes) * TRAIN_STEPS,
                        "fresh_process": _checkpoint_fresh_process(
                            candidate, seed, contexts["C"]),
                        "candidate_logged_propensities": len(trace)})
    success = (candidate_frozen_passes == 5 and p95_margin_passes >= 4
               and null_beating_count <= 12)
    return {"protocol": PROTOCOL,
            "evaluation_python": sys.version.split()[0],
            "records": records,
            "nulls_ge_candidate_total": null_beating_count,
            "candidate_frozen_pass_count": candidate_frozen_passes,
            "candidate_null_p95_margin_pass_count": p95_margin_passes,
            "strong_criterion": "PASS" if success else "FAIL",
            "qualification": "SYNTHETIC_SOURCE_VISIBLE_IN_FAMILY_ONLY"}


def _qualified_source_head(expected: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("expected GitHub source head must be exact SHA-1")
    check = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=5, check=True)
    if check.stdout.strip() != expected:
        raise RuntimeError("source SHA diverged from reviewed execution head")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("protocol", help="print constants only; never sample")
    runner = modes.add_parser("run", help="requires separately approved source head")
    runner.add_argument("--approved-source-head", required=True)
    verifier = modes.add_parser("verify-checkpoint", help="isolated subprocess only")
    verifier.add_argument("--checkpoint", type=Path, required=True)
    verifier.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    if args.mode == "protocol":
        result = protocol_manifest()
    elif args.mode == "verify-checkpoint":
        # Strict private CLI called from verified parent through stdin; no
        # training and no generated evaluator data in this process.
        source = json.loads(sys.stdin.buffer.read(500_000))
        contexts = [float.fromhex(x) for x in source["contexts"]]
        context_digest("C", args.seed, contexts)
        head = AppraisalHead.load(args.checkpoint)
        result = {"pid": os.getpid(),
                  "values": [list(head.values(x)) for x in H2_PROBES],
                  "expected_payoff": _eval_softmax_expected(head, contexts)}
    else:
        if (sys.implementation.name != "cpython"
                or sys.version.split()[0] != PYTHON_VERSION):
            raise RuntimeError("unqualified Python version for prospective V3")
        _qualified_source_head(args.approved_source_head)
        result = _actual_run()
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
