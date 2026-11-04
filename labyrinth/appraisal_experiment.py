"""Tiny synthetic, persisted appraisal-head control experiment.

This is not LLM training or evidence of subjective reward. See
docs/APPRAISAL_HEAD_SYNTHETIC_PROTOCOL_V1.md for the frozen design.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VERSION = "labyrinth.synthetic-appraisal-head.v1"
SEEDS = (11, 29, 71)


def _canonical(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _check_count(value: int, *, name: str, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


@dataclass
class AppraisalHead:
    """Two independent [bias, context slope] action value estimators."""

    weights: list[list[float]]

    @classmethod
    def zero(cls) -> "AppraisalHead":
        return cls([[0.0, 0.0], [0.0, 0.0]])

    def values(self, context: float) -> tuple[float, float]:
        return tuple(w[0] + w[1] * context for w in self.weights)  # type: ignore[return-value]

    def learn(self, context: float, action: int, outcome: int,
              *, rate: float = 0.12) -> None:
        predicted = self.values(context)[action]
        residual = outcome - predicted
        self.weights[action][0] += rate * residual
        self.weights[action][1] += rate * residual * context

    def save_new(self, path: Path) -> str:
        _validate_weights(self.weights)
        payload = {"version": VERSION, "weights": self.weights}
        digest = hashlib.sha256(_canonical(payload)).hexdigest()
        encoded = _canonical({**payload, "sha256": digest}) + b"\n"
        with path.open("xb") as file:
            file.write(encoded)
        return digest

    @classmethod
    def load(cls, path: Path) -> "AppraisalHead":
        with path.open("rb") as file:
            raw = file.read(4097)
        if len(raw) > 4096:
            raise ValueError("checkpoint too large")
        try:
            record = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid JSON checkpoint") from exc
        if type(record) is not dict or set(record) != {"version", "weights", "sha256"}:
            raise ValueError("invalid checkpoint structure")
        if record["version"] != VERSION or type(record["version"]) is not str:
            raise ValueError("invalid checkpoint version")
        digest = hashlib.sha256(_canonical({
            "version": record["version"], "weights": record["weights"]
        })).hexdigest()
        if type(record["sha256"]) is not str or not hmac.compare_digest(
            digest, record["sha256"]
        ):
            raise ValueError("checkpoint SHA-256 mismatch")
        _validate_weights(record["weights"])
        return cls([[float(v) for v in row] for row in record["weights"]])


def _validate_weights(weights: object) -> None:
    if type(weights) is not list or len(weights) != 2:
        raise ValueError("weights must be two action rows")
    for row in weights:
        if type(row) is not list or len(row) != 2:
            raise ValueError("each action requires bias and slope")
        for value in row:
            if type(value) not in (float, int) or not math.isfinite(value) or abs(value) > 100:
                raise ValueError("checkpoint weights invalid or unbounded")


def _expected(context: float, action: int) -> float:
    # This hidden-from-policy *synthetic evaluator* is not a confidential oracle.
    return 0.5 + (0.4 * context if action == 0 else -0.4 * context)


def _train_probability(head: AppraisalHead, context: float) -> float:
    v0, v1 = head.values(context)
    difference = max(-60.0, min(60.0, (v0 - v1) / 0.15))
    return 0.1 + 0.8 / (1.0 + math.exp(-difference))


def train(seed: int, *, steps: int = 3200) -> tuple[AppraisalHead, list[dict[str, Any]]]:
    _check_count(seed, name="seed", maximum=2**31 - 1)
    _check_count(steps, name="steps", maximum=100_000)
    head = AppraisalHead.zero()
    contexts = random.Random(seed * 131 + 13)
    actions = random.Random(seed * 157 + 17)
    outcomes = random.Random(seed * 173 + 19)
    trace = []
    for i in range(steps):
        context = contexts.uniform(-1.0, 1.0)
        p0 = _train_probability(head, context)
        action = 0 if actions.random() < p0 else 1
        propensity = p0 if action == 0 else 1 - p0
        outcome = int(outcomes.random() < _expected(context, action))
        trace.append({"step": i, "context": context, "action": action,
                      "propensity": propensity, "reward": outcome})
        head.learn(context, action, outcome)
    return head, trace


def shuffled_feedback(seed: int, trace: list[dict[str, Any]]) -> AppraisalHead:
    rewards = [entry["reward"] for entry in trace]
    shuffled = list(rewards)
    random.Random(seed * 193 + 23).shuffle(shuffled)
    if shuffled == rewards:
        raise ValueError("shuffled control did not change feedback assignment")
    head = AppraisalHead.zero()
    for entry, reward in zip(trace, shuffled):
        head.learn(entry["context"], entry["action"], reward)
    return head


def _evaluation_cases(seed: int, count: int) -> list[tuple[float, int, int]]:
    _check_count(count, name="eval_cases", maximum=100_000)
    # Different sampler and seeds from the continuous uniform train generator.
    contexts = random.Random(seed * 211 + 29)
    outcomes = random.Random(seed * 223 + 31)
    cases = []
    for _ in range(count):
        context = max(-1.0, min(1.0, contexts.randrange(-20, 21) / 20.0
                                     + contexts.uniform(-0.01, 0.01)))
        r0 = int(outcomes.random() < _expected(context, 0))
        r1 = int(outcomes.random() < _expected(context, 1))
        cases.append((context, r0, r1))
    return cases


def evaluate(head: AppraisalHead, seed: int, *, cases: int = 1200,
             disabled: bool = False) -> dict[str, float | int]:
    records = _evaluation_cases(seed, cases)
    expected_sum = 0.0
    sampled_sum = 0
    correct_action_count = 0
    for index, (context, r0, r1) in enumerate(records):
        v0, v1 = (0.0, 0.0) if disabled else head.values(context)
        action = 0 if v0 > v1 else 1 if v1 > v0 else index % 2
        expected_sum += _expected(context, action)
        sampled_sum += r0 if action == 0 else r1
        correct_action_count += int((action == 0) == (context >= 0))
    return {"cases": len(records), "expected_payoff": expected_sum / len(records),
            "realized_payoff": sampled_sum / len(records),
            "optimal_action_rate": correct_action_count / len(records)}


def run(seed: int, *, train_steps: int = 3200,
        eval_cases: int = 1200) -> tuple[dict[str, Any], AppraisalHead, list[dict[str, Any]]]:
    candidate, trace = train(seed, steps=train_steps)
    shuffled = shuffled_feedback(seed, trace)
    frozen = AppraisalHead.zero()
    arms = {
        "candidate": evaluate(candidate, seed, cases=eval_cases),
        "frozen": evaluate(frozen, seed, cases=eval_cases),
        "shuffled_feedback": evaluate(shuffled, seed, cases=eval_cases),
        "disabled_head": evaluate(candidate, seed, cases=eval_cases,
                                  disabled=True),
    }
    # Explicitly POST-HOC, not a pre-registered decision rule. A conservative
    # Hoeffding bound assumes independent held-out draws *conditional on the
    # fixed fitted heads*, and correct synthetic reward expectations. Three
    # fixed seeds do not establish external generalization or LLM learning.
    comparisons = ("frozen", "shuffled_feedback", "disabled_head")
    deltas = {other: (arms["candidate"]["expected_payoff"]
                      - arms[other]["expected_payoff"])
              for other in comparisons}
    radius = 1.6 * math.sqrt(math.log(2 * 9 / 0.05) / (2 * eval_cases))
    return ({"protocol": VERSION, "seed": seed,
             "training_steps": train_steps,
             "propensities_recorded": len(trace),
             "arms": arms,
             "posthoc_uncertainty": {
                 "method": "conditional_Hoeffding_familywise_9_95pct",
                 "preregistered": False,
                 "radius_per_delta": radius,
                 "candidate_minus_controls_expected_payoff": deltas,
             }}, candidate, trace)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_subparsers(dest="mode", required=True)
    training = mode.add_parser("run", help="synthetic training and matched arms")
    training.add_argument("--seed", type=int, default=SEEDS[0])
    training.add_argument("--train-steps", type=int, default=3200)
    training.add_argument("--eval-cases", type=int, default=1200)
    training.add_argument("--checkpoint", type=Path)
    training.add_argument("--trace", type=Path)
    verify = mode.add_parser("evaluate", help="fresh-process checkpoint readback")
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify.add_argument("--seed", type=int, default=SEEDS[0])
    verify.add_argument("--eval-cases", type=int, default=1200)
    args = parser.parse_args(argv)
    if args.mode == "evaluate":
        result = evaluate(AppraisalHead.load(args.checkpoint), args.seed,
                          cases=args.eval_cases)
    else:
        result, head, trace = run(args.seed, train_steps=args.train_steps,
                                  eval_cases=args.eval_cases)
        if args.checkpoint is not None:
            result["checkpoint_sha256"] = head.save_new(args.checkpoint)
        if args.trace is not None:
            with args.trace.open("x", encoding="utf-8") as file:
                for entry in trace:
                    file.write(json.dumps(entry, sort_keys=True,
                                          allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
