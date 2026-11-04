"""Independent online shuffled-feedback negative controls for appraisal toy V1.

Source-visible synthetic mechanisms only. See
docs/APPRAISAL_SHUFFLED_ONLINE_NULL_V2.md (frozen before implementation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from typing import Any

from .appraisal_experiment import (
    AppraisalHead, _check_count, _expected, _train_probability, evaluate, train,
)


def train_online_null(
    seed: int, candidate_rewards: list[int], *, replicate: int,
    include_trace: bool = False,
) -> tuple[AppraisalHead, list[dict[str, Any]]]:
    """Train a null with OWN stochastic actions on an independent shuffled tape.

    Matching the candidate context sequence controls context exposure. It does
    not import the candidate's prior actions, propensities, or fitted weights.
    """
    _check_count(seed, name="seed", maximum=2**31 - 1)
    if type(replicate) is not int or not 0 <= replicate < 1000:
        raise ValueError("replicate index must be in [0,999]")
    _check_count(len(candidate_rewards), name="candidate reward count", maximum=100_000)
    if any(type(reward) is not int or reward not in (0, 1)
           for reward in candidate_rewards):
        raise ValueError("candidate reward tape must be binary")
    # Shuffling a fixed tape preserves the exact label multiset while removing
    # deliberate reward alignment with this null policy's own actions.
    tape = list(candidate_rewards)
    random.Random(seed * 1_299_709 + replicate * 1009 + 43).shuffle(tape)
    contexts = random.Random(seed * 131 + 13)
    actions = random.Random(seed * 1_000_003 + replicate * 31_337 + 31)
    head = AppraisalHead.zero()
    trace: list[dict[str, Any]] = []
    for i, reward in enumerate(tape):
        context = contexts.uniform(-1.0, 1.0)
        p0 = _train_probability(head, context)
        action = 0 if actions.random() < p0 else 1
        propensity = p0 if action == 0 else 1.0 - p0
        if include_trace:
            trace.append({"step": i, "context": context, "action": action,
                          "propensity": propensity, "reward": reward})
        head.learn(context, action, reward)
    return head, trace


def evaluate_triangular(head: AppraisalHead, seed: int, *, cases: int = 1200,
                        disabled: bool = False) -> dict[str, float | int]:
    """Fresh withheld sampler B; same toy latent reward relationship."""
    _check_count(seed, name="seed", maximum=2**31 - 1)
    _check_count(cases, name="eval_cases", maximum=100_000)
    contexts = random.Random(seed * 250_007 + 47)
    outcomes = random.Random(seed * 250_013 + 53)
    expected_total = 0.0
    realized_total = 0
    optimal_actions = 0
    for i in range(cases):
        context = (contexts.uniform(-1, 1) + contexts.uniform(-1, 1)) / 2
        r0 = int(outcomes.random() < _expected(context, 0))
        r1 = int(outcomes.random() < _expected(context, 1))
        v0, v1 = (0.0, 0.0) if disabled else head.values(context)
        action = 0 if v0 > v1 else 1 if v1 > v0 else i % 2
        expected_total += _expected(context, action)
        realized_total += r0 if action == 0 else r1
        optimal_actions += int((action == 0) == (context >= 0))
    return {"cases": cases, "expected_payoff": expected_total / cases,
            "realized_payoff": realized_total / cases,
            "optimal_action_rate": optimal_actions / cases}


def _null_summary(scores: list[float], candidate: float) -> dict[str, float | int]:
    ordered = sorted(scores)
    count = len(ordered)
    if count == 100:
        p05, p95 = ordered[4], ordered[94]
    else:
        p05, p95 = ordered[max(0, count * 5 // 100 - 1)], ordered[
            max(0, count * 95 // 100 - 1)]
    midpoint = (ordered[(count - 1) // 2] + ordered[count // 2]) / 2
    return {"count": count, "p05": p05, "median": midpoint, "p95": p95,
            "max": ordered[-1],
            "within_0p06_or_above_fraction": sum(
                score >= candidate - 0.06 for score in scores) / count,
            "beating_candidate_fraction": sum(score >= candidate for score in scores) / count}


def run_v2(seed: int, *, train_steps: int = 3200, eval_cases: int = 1200,
           null_repeats: int = 100) -> dict[str, Any]:
    _check_count(null_repeats, name="null_repeats", maximum=1000)
    _check_count(train_steps, name="train_steps", maximum=100_000)
    _check_count(eval_cases, name="eval_cases", maximum=100_000)
    candidate, trace = train(seed, steps=train_steps)
    labels = [row["reward"] for row in trace]
    frozen = AppraisalHead.zero()
    samplers = {
        "A_grid": lambda head: evaluate(head, seed, cases=eval_cases),
        "B_triangular": lambda head: evaluate_triangular(head, seed, cases=eval_cases),
    }
    output: dict[str, Any] = {}
    null_scores: dict[str, list[float]] = {name: [] for name in samplers}
    trace_digests: list[str] = []
    logged_probabilities = 0
    for name, evaluator in samplers.items():
        output[name] = {"candidate": evaluator(candidate),
                        "frozen": evaluator(frozen)}
    for replicate in range(null_repeats):
        head, null_trace = train_online_null(
            seed, labels, replicate=replicate, include_trace=True)
        logged_probabilities += len(null_trace)
        trace_digests.append(hashlib.sha256(json.dumps(
            null_trace, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest())
        for name, evaluator in samplers.items():
            null_scores[name].append(evaluator(head)["expected_payoff"])
    for name, scores in null_scores.items():
        output[name]["nulls"] = _null_summary(
            scores, output[name]["candidate"]["expected_payoff"])
    b = output["B_triangular"]
    fixed_protocol = (train_steps == 3200 and eval_cases == 1200
                      and null_repeats == 100 and seed in (11, 29, 71))
    criterion = (b["candidate"]["expected_payoff"]
                 - b["frozen"]["expected_payoff"] > 0.10
                 and b["candidate"]["expected_payoff"]
                 - b["nulls"]["p95"] >= 0.05)
    return {"protocol": "labyrinth.synthetic-appraisal-online-null.v2",
            "seed": seed, "training_steps": train_steps,
            "null_repeats": null_repeats,
            "null_logged_propensities": logged_probabilities,
            "null_trace_sha256_by_replicate": trace_digests,
            "evaluation": output,
            "strong_causal_criterion": (
                "PASS" if criterion else "FAIL") if fixed_protocol else "NOT_EVALUATED",
            "qualification": "SYNTHETIC_IN_FAMILY_ONLY"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-steps", type=int, default=3200)
    parser.add_argument("--eval-cases", type=int, default=1200)
    parser.add_argument("--null-repeats", type=int, default=100)
    args = parser.parse_args(argv)
    print(json.dumps(run_v2(args.seed, train_steps=args.train_steps,
                            eval_cases=args.eval_cases,
                            null_repeats=args.null_repeats), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
