"""Optional *development-only* Reasoning Gym dataset scoring.

The actual `reasoning_gym` dataset supplies questions and the authoritative
`score_answer` method. This bridge does not train model weights and cannot
certify independent evaluation: dataset labels and scoring run in this same
process. Never pass sealed final-set datasets through this development API.

`reasoning_gym` is deliberately a lazy optional import. Core Labyrinth tests
and commands must continue to work without downloading or installing it.
"""

from __future__ import annotations

import hashlib
import importlib
from importlib import metadata
import math
import re
from typing import Any

from .adapters import ModelAdapter


_DATASET_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,79}\Z", re.ASCII)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def benchmark_reasoning_gym(
    adapter: ModelAdapter,
    dataset_name: str,
    *,
    size: int = 16,
    seed: int = 2026,
) -> dict[str, Any]:
    """Evaluate questions with their source dataset's real scoring contract.

    `score_answer` returns a float in [0, 1], not necessarily binary exact
    match. No expected answer, metadata or untrusted score function is sent to
    the model. Score feedback is exposed in this report, hence development use
    only; it must never serve as the blind final-evaluation interface.
    """
    if not isinstance(dataset_name, str) or _DATASET_RE.fullmatch(dataset_name) is None:
        raise ValueError("dataset_name must be a registered short dataset name")
    if type(size) is not int or not 1 <= size <= 128:
        raise ValueError("size must be an integer from 1 through 128")
    if type(seed) is not int or not 0 <= seed or seed + size > 2**31 - 1:
        raise ValueError("seed plus case count must fit a nonnegative 31-bit model seed")
    model_id = getattr(adapter, "model_id", None)
    if not isinstance(model_id, str) or not 1 <= len(model_id) <= 256:
        raise ValueError("adapter must supply a bounded model_id")

    try:
        reasoning_gym = importlib.import_module("reasoning_gym")
    except ModuleNotFoundError as exc:
        raise RuntimeError("optional reasoning-gym package is not installed") from exc

    # The installed 0.1.25 distribution has exposed a stale 0.1.19 module
    # __version__. Use distribution metadata rather than that module label.
    # Absence of metadata is explicit, not silently converted to a made-up
    # version or taken as proof that the imported module is trustworthy.
    try:
        distribution_version: str | None = metadata.version("reasoning-gym")
    except metadata.PackageNotFoundError:
        distribution_version = None

    # Exact upstream scorer semantics are delegated to this dataset object.
    # Do not synthesize a supposedly equivalent oracle in Labyrinth.
    dataset = reasoning_gym.create_dataset(dataset_name, size=size, seed=seed)
    score_fn = getattr(dataset, "score_answer", None)
    if not callable(score_fn):
        raise ValueError("Reasoning Gym dataset has no score_answer method")

    scores: list[float] = []
    prompt_digests: list[str] = []
    response_digests: list[str] = []
    for index in range(size):
        case = dataset[index]
        if not isinstance(case, dict) or not isinstance(case.get("question"), str):
            raise ValueError("Reasoning Gym dataset returned a malformed question")
        question = case["question"]
        if not 1 <= len(question) <= 64_000:
            raise ValueError("Reasoning Gym question violates length limits")
        # Deliberately pass only the public question. The object holding gold
        # answers/metadata stays on the scoring side of the call boundary.
        response = adapter.generate(question, seed=seed + index)
        if not isinstance(response, str) or len(response) > 64_000:
            raise ValueError("adapter returned an invalid or oversized response")
        score = score_fn(answer=response, entry=case)
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Reasoning Gym scorer returned an invalid score")
        scores.append(float(score))
        prompt_digests.append(_digest(question))
        response_digests.append(_digest(response))

    return {
        "schema": "labyrinth.reasoning-gym-development.v1",
        "status": "development_only",
        "evidence_limit": "self-reported model identity; no sealed evaluator custody or training evidence",
        "dataset_name": dataset_name,
        "reasoning_gym_distribution_version": distribution_version,
        "generator_seed": seed,
        "case_count": size,
        "model_id": model_id,
        "question_sha256": prompt_digests,
        "response_sha256": response_digests,
        "per_case_scores": scores,
        "correctness_mean": sum(scores) / len(scores),
    }
