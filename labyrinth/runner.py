"""Deterministic, model-agnostic challenge runner and evaluation controls.

All prompts sent to a model contain only public challenge text. Training answer
keys appear only in intentionally constructed few-shot demonstrations, never
in held-out evaluation prompts. No model output is executed as code.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Protocol

from .adapters import OllamaAdapter, ReplayAdapter
from .evaluation import score_response
from .maze import Challenge, generate_maze


REPORT_SCHEMA = "labyrinth.report.v1"
ARTIFACT_SCHEMA = "labyrinth.demonstrations.v1"
_MAX_DEMONSTRATIONS = 64
# Bound CLI-supplied replay mappings and training artifacts before JSON parsing.
# An 8 MiB input is ample for the maximum 64 bounded demonstrations/trajectories.
_MAX_JSON_BYTES = 8 * 1024 * 1024
CONTROL_DESCRIPTIONS = {
    "baseline": "No examples supplied",
    "candidate": "In-context examples from disjoint training split",
    "placebo": "Text-length-matched neutral passage (not token matched)",
    "shuffled": "Training questions with label-multiset-preserving, complete answer derangement",
}


class Model(Protocol):
    model_id: str

    def generate(self, prompt: str, *, seed: int | None = None) -> str: ...


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _source_state() -> dict[str, Any]:
    """Best-effort provenance; never turn a missing Git client into a run failure."""
    root = Path(__file__).resolve().parent.parent
    # This digest identifies working source even while the branch has no commit
    # or has uncommitted changes. It does not authenticate the author/source.
    source_files = sorted((root / "labyrinth").glob("*.py")) + [root / "pyproject.toml"]
    source_digest = digest([
        {"path": p.relative_to(root).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in source_files if p.is_file()
    ])
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True, timeout=3,
        ).stdout.strip()
        modified = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True,
            text=True, check=True, timeout=3,
        ).stdout
        return {"commit": head, "clean": not bool(modified), "source_tree_sha256": source_digest}
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"commit": None, "clean": None, "source_tree_sha256": source_digest}


def _model_seed(seed: int, challenge_id: str) -> int:
    # A 31-bit value works with common local inference backends.
    raw = f"{seed}|{challenge_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") & 0x7FFFFFFF


def _challenge_commitment(challenges: list[Challenge]) -> str:
    # Gold answers are committed to the report hash, but never disclosed there.
    return digest([{"id": c.id, "prompt": c.prompt, "answer": c.answer} for c in challenges])


def _normalized_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt).strip().casefold()


def _adapter_provenance(adapter: Model) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "id": adapter.model_id,
        "class": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "weight_digest": None,  # No model-file verification is performed.
    }
    if isinstance(adapter, ReplayAdapter):
        meta["replay_fixture_sha256"] = digest(dict(adapter.responses))
        meta["replay_default"] = adapter.default
    if isinstance(adapter, OllamaAdapter):
        meta["generation_options"] = {
            "temperature": 0,
            "max_output_tokens": adapter.max_output_tokens,
            "timeout_seconds": adapter.timeout_seconds,
            "base_url": adapter.base_url,
        }
    return meta


def _evaluate_arm(
    adapter: Model, challenges: list[Challenge], seed: int, arm: str,
    prefix: str = "",
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    for challenge in challenges:
        model_prompt = f"{prefix}{challenge.prompt}" if prefix else challenge.prompt
        model_seed = _model_seed(seed, challenge.id)
        response = adapter.generate(model_prompt, seed=model_seed)
        score = score_response(challenge, response)
        trials.append({
            "challenge_id": challenge.id,
            "model_seed": model_seed,
            "model_prompt_sha256": hashlib.sha256(model_prompt.encode("utf-8")).hexdigest(),
            "tier": challenge.tier,
            "kind": challenge.kind,
            "correct": score.correct,
            "points": score.points,
            "scoring_reason": score.reason,
            # Bound untrusted report content; preserve its hash and byte length.
            "response_preview": response[:4096] if isinstance(response, str) else "",
            "response_length": len(response) if isinstance(response, str) else None,
            "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest()
            if isinstance(response, str) else None,
        })
    points = sum(trial["points"] for trial in trials)
    formatted = sum(trial["scoring_reason"] in ("correct", "incorrect") for trial in trials)
    return {
        "arm": arm,
        "total": len(trials),
        "points": points,
        "accuracy": points / len(trials) if trials else 0.0,
        "format_valid": formatted,
        "format_adherence": formatted / len(trials) if trials else 0.0,
        "trials": trials,
    }


def run_maze(
    adapter: Model, *, seed: int, count: int,
    tiers: tuple[int, ...] = (1, 2, 3), split: str = "eval",
) -> dict[str, Any]:
    challenges = generate_maze(seed=seed, count=count, tiers=tiers, split=split)
    report = {
        "schema": REPORT_SCHEMA,
        "mode": "maze",
        "source": _source_state(),
        "model_id": adapter.model_id,
        "adapter": _adapter_provenance(adapter),
        "parameters": {"seed": seed, "count": count, "tiers": list(tiers), "split": split},
        "challenge_commitment_sha256": _challenge_commitment(challenges),
        "evaluation": _evaluate_arm(adapter, challenges, seed, "baseline"),
        "limitations": ["A score is not evidence of persistent learning or weight updates."],
    }
    return report


def build_training_artifact(
    *, seed: int, count: int, tiers: tuple[int, ...] = (1, 2, 3),
    adapter: Model | None = None,
) -> dict[str, Any]:
    """Persist a few-shot memory of *scored* training cases and feedback.

    If ``adapter`` is omitted, training observations are explicitly absent:
    useful for fixture generation, but not evidence of an adaptation run.
    """
    if not (1 <= count and 2 <= count * len(tiers) <= _MAX_DEMONSTRATIONS):
        raise ValueError("train count times number of tiers must be between 2 and 64")
    challenges = generate_maze(seed=seed, count=count, tiers=tiers, split="train")
    examples = [{"challenge_id": c.id, "prompt": c.prompt, "answer": c.answer} for c in challenges]
    trajectory = []
    if adapter is not None:
        for c in challenges:
            response = adapter.generate(c.prompt, seed=_model_seed(seed, c.id))
            if not isinstance(response, str) or len(response) > 8192:
                raise ValueError("training response exceeds the bounded scoring contract")
            result = score_response(c, response)
            trajectory.append({
                "challenge_id": c.id, "model_response": response,
                "correct": result.correct, "points": result.points,
                "scoring_reason": result.reason,
                "feedback_answer": c.answer,
            })
    payload = {
        "schema": ARTIFACT_SCHEMA,
        "source": _source_state(),
        "training_parameters": {"seed": seed, "count": count, "tiers": list(tiers)},
        "examples": examples,
        "training_trajectory": trajectory,
        "adaptation_kind": "scored-exemplar-memory" if adapter is not None else "unobserved-exemplar-fixture",
        "training_model_id": adapter.model_id if adapter is not None else None,
    }
    return {**payload, "integrity_sha256": digest(payload)}


def validate_training_artifact(artifact: dict[str, Any]) -> list[dict[str, str]]:
    if not isinstance(artifact, dict) or artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported training artifact schema")
    expected = artifact.get("integrity_sha256")
    payload = {k: v for k, v in artifact.items() if k != "integrity_sha256"}
    if not isinstance(expected, str) or digest(payload) != expected:
        raise ValueError("training artifact integrity mismatch")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not (2 <= len(examples) <= _MAX_DEMONSTRATIONS):
        raise ValueError("training artifact needs 2–64 examples")
    for ex in examples:
        if not isinstance(ex, dict) or set(ex) != {"challenge_id", "prompt", "answer"}:
            raise ValueError("invalid training example fields")
        if any(not isinstance(ex[k], str) or len(ex[k]) > 10000 for k in ex):
            raise ValueError("invalid training example values")
    params = artifact.get("training_parameters")
    if not isinstance(params, dict) or set(params) != {"seed", "count", "tiers"}:
        raise ValueError("invalid training parameters")
    seed, count, tiers = params["seed"], params["count"], params["tiers"]
    if type(seed) is not int or type(count) is not int or not isinstance(tiers, list):
        raise ValueError("invalid training seed/count/tiers")
    if not (1 <= count and 2 <= count * len(tiers) <= _MAX_DEMONSTRATIONS):
        raise ValueError("invalid training count or tier cardinality")
    expected_cases = generate_maze(seed=seed, count=count, tiers=tuple(tiers), split="train")
    expected_examples = [
        {"challenge_id": c.id, "prompt": c.prompt, "answer": c.answer} for c in expected_cases
    ]
    if examples != expected_examples:
        raise ValueError("training examples do not match canonical generator outputs")
    source = artifact.get("source")
    if not isinstance(source, dict) or source.get("source_tree_sha256") != _source_state()["source_tree_sha256"]:
        raise ValueError("training artifact source tree does not match current code")
    trajectory = artifact.get("training_trajectory")
    if not isinstance(trajectory, list) or len(trajectory) not in (0, len(examples)):
        raise ValueError("invalid training trajectory length")
    for i, (item, ex) in enumerate(zip(trajectory, examples)):
        if not isinstance(item, dict) or item.get("challenge_id") != ex["challenge_id"]:
            raise ValueError("invalid training trajectory challenge")
        if item.get("feedback_answer") != ex["answer"]:
            raise ValueError("invalid training trajectory feedback")
        response = item.get("model_response")
        if not isinstance(response, str):
            raise ValueError("invalid training trajectory response")
        expected_score = score_response(expected_cases[i], response)
        if item.get("points") != expected_score.points or item.get("correct") != expected_score.correct or item.get("scoring_reason") != expected_score.reason:
            raise ValueError("training trajectory scoring does not match evaluator")
    return examples


def _examples_context(examples: list[dict[str, str]]) -> str:
    lines = [
        "Examples from a separate TRAINING split are supplied below. "
        "Solve the NEW challenge using only its stated rules and end with ANSWER: <value>.",
    ]
    for index, ex in enumerate(examples, 1):
        lines.append(f"TRAINING EXAMPLE {index}\n{ex['prompt']}\nANSWER: {ex['answer']}")
    lines.append("NEW HELD-OUT CHALLENGE\n")
    return "\n\n".join(lines)


def _deranged_training_labels(examples: list[dict[str, str]]) -> list[dict[str, str]]:
    """Preserve the label multiset while changing every label, or fail closed.

    A cyclic shift of examples is insufficient when neighboring gold answers
    coincide: it can make the purported negative control identical to the
    candidate. A complete derangement exists only if no value occupies more
    than half the positions.
    """
    labels = [item["answer"] for item in examples]
    n = len(labels)
    max_frequency = max(Counter(labels).values())
    if max_frequency * 2 > n:
        raise ValueError("shuffled control invalid: training labels cannot be fully deranged")
    indices = sorted(range(n), key=lambda i: (labels[i], i))
    replacements = [""] * n
    for position, original_index in enumerate(indices):
        replacement_index = indices[(position + max_frequency) % n]
        replacements[original_index] = labels[replacement_index]
    if any(original == replacement for original, replacement in zip(labels, replacements)):
        raise ValueError("shuffled control invalid: a repeated label remained unchanged")
    return [{**example, "answer": replacement} for example, replacement in zip(examples, replacements)]


def _placebo_context(target_length: int) -> str:
    # Text-length matched, deliberately *not* token-count matched. Disclosed in report.
    prefix = "Neutral calibration passage (not task examples).\n"
    filler = "Blue hills under clear skies. "
    return (prefix + filler * ((max(target_length - len(prefix), 0) // len(filler)) + 1))[:target_length]


def run_experiment(
    adapter_factory: Callable[[], Model], *, seed: int, eval_count: int,
    tiers: tuple[int, ...] = (1, 2, 3),
    artifact: dict[str, Any],
) -> dict[str, Any]:
    examples = validate_training_artifact(artifact)
    eval_seed = seed ^ 0x5C18A7
    challenges = generate_maze(seed=eval_seed, count=eval_count, tiers=tiers, split="eval")
    train_ids = {ex["challenge_id"] for ex in examples}
    train_prompts = {_normalized_prompt(ex["prompt"]) for ex in examples}
    if any(c.id in train_ids or _normalized_prompt(c.prompt) in train_prompts for c in challenges):
        raise ValueError("training/evaluation split contamination: repeated challenge ID or prompt")

    candidate_prefix = _examples_context(examples)
    shuffled = _deranged_training_labels(examples)
    label_collisions = sum(a["answer"] == b["answer"] for a, b in zip(examples, shuffled))

    # Fresh Python adapter instances are essential: adapters may otherwise
    # accumulate state or history and bias later arms. An external model server
    # could still have hidden mutable state; disclose that residual risk.
    arms: dict[str, dict[str, Any]] = {}
    instances: list[Model] = []
    arm_prefixes = {
        "baseline": "", "candidate": candidate_prefix,
        "placebo": _placebo_context(len(candidate_prefix)),
        "shuffled": _examples_context(shuffled),
    }
    for arm, prefix in arm_prefixes.items():
        adapter = adapter_factory()
        if any(adapter is old for old in instances):
            raise ValueError("adapter factory reused the same instance across control arms")
        if instances and adapter.model_id != instances[0].model_id:
            raise ValueError("adapter factory returned different model identities")
        instances.append(adapter)
        arms[arm] = _evaluate_arm(adapter, challenges, eval_seed, arm, prefix)
    return {
        "schema": REPORT_SCHEMA,
        "mode": "controlled-few-shot",
        "source": _source_state(),
        "model_id": instances[0].model_id,
        "adapter": _adapter_provenance(instances[0]),
        "parameters": {
            "eval_seed": eval_seed, "eval_count": eval_count, "tiers": list(tiers),
            "original_seed": seed,
        },
        "training_artifact_sha256": artifact["integrity_sha256"],
        "training_model_id": artifact.get("training_model_id"),
        "adaptation_kind": artifact.get("adaptation_kind"),
        "training_count": len(examples),
        "challenge_commitment_sha256": _challenge_commitment(challenges),
        "shuffled_label_collisions": label_collisions,
        "controls": dict(CONTROL_DESCRIPTIONS),
        "arms": arms,
        "limitations": [
            "In-context conditioning is not parameter training or proof of persistent learning.",
            "Model identity and seed support vary by adapter/runtime; no inference determinism is assumed.",
            "Fresh adapters isolate Python object state but not necessarily external inference server state.",
            "Shuffled control fails closed when the answer-label distribution cannot be completely deranged.",
        ],
    }


def save_json_new(path: str | Path, value: Any) -> None:
    """Write-once to avoid quietly replacing existing experimental evidence."""
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"nonfinite JSON token forbidden: {value}")


def load_json(path: str | Path, *, strict: bool = False) -> Any:
    # Read at most max+1 bytes even if a local file grows between stat/open;
    # never hand unbounded input to a JSON decoder.
    with Path(path).open("rb") as handle:
        raw = handle.read(_MAX_JSON_BYTES + 1)
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError(f"JSON input exceeds {_MAX_JSON_BYTES} bytes")
    try:
        return json.loads(raw.decode("utf-8"),
                          object_pairs_hook=_unique_json_pairs if strict else None,
                          parse_constant=_reject_nonfinite_json if strict else None)
    except RecursionError as exc:
        raise ValueError("JSON input nesting exceeds parser limit") from exc
