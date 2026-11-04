"""Opt-in pretrained-feature neural Minotaur development probe on CPU.

The taskmaster observes a textual skill/performance summary and proposes one
of four training tactics/difficulty bands. A cached pretrained MiniLM encoder
is FROZEN; a small PyTorch policy head actually receives parameter updates.
Correct-label, no-update, and balanced corrupted-label arms see exactly the
same public synthetic training examples through two regenerated corridors.

No private exam, authentic learning objective, model-host integration, or
graduation is established. This script cannot mutate a Journey or grant
Minotaur evaluation/graduation authority. Default CLI is a read-only plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any


SCHEMA = "labyrinth.neural-taskmaster-public-development.v1"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
ARMS = ("correct", "zero_update", "corrupted")
TACTICS = ("practice-easy", "contrast-medium", "transfer-hard", "review-easy")
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")

# Exposed synthetic development data, not examiner-controlled final cases.
# The label NEVER enters the encoder input. These are deliberately small,
# same-task fixtures and cannot establish independent semantic transfer.
_SKILLS = {
    "cycle0": ("arithmetic", "program chains", "source hierarchy", "logic puzzles"),
    "cycle1": ("algebra", "dependency tracing", "evidence ranking", "number patterns"),
    "heldout": ("geometry", "code review", "document checks", "planning tasks"),
    "transfer": ("workshop diagnosis", "inventory forecasting", "story inference", "scheduling"),
}
_PATTERNS = {
    "cycle0": (
        "With {skill}, eight of ten recent attempts failed and mistakes are increasing; the learner misses fundamental steps.",
        "With {skill}, half the answers are wrong because two similar rules are repeatedly confused.",
        "With {skill}, nine of ten familiar cases are correct and accuracy keeps rising; unfamiliar applications are next.",
        "With {skill}, a capability that was previously reliable deteriorated after a new topic and errors returned.",
    ),
    "cycle1": (
        "For {skill}, basic steps remain incorrect on most exercises and performance is falling again.",
        "For {skill}, the learner repeatedly swaps closely related methods, scoring near one half correctly.",
        "For {skill}, almost every practiced example is correct and progress is steady across familiar formats.",
        "For {skill}, earlier proficiency slipped after a change of subject and past mistakes have reappeared.",
    ),
    "heldout": (
        "While studying {skill}, the pupil now misses nearly all introductory questions and performance trends downward.",
        "While studying {skill}, similar-looking alternatives cause persistent mix-ups despite partial mastery.",
        "While studying {skill}, routine questions are consistently accurate and the trend is strongly positive.",
        "While studying {skill}, yesterday's secure ability has declined following newer material.",
    ),
    "transfer": (
        "In the unfamiliar area of {skill}, the trainee cannot perform the foundational moves and errors keep climbing.",
        "In the unfamiliar area of {skill}, closely competing principles are confused about half of the time.",
        "In the unfamiliar area of {skill}, the trainee reliably solves routine examples and performance has improved.",
        "In the unfamiliar area of {skill}, an old mastered method is being forgotten after new practice.",
    ),
}


def fixtures() -> dict[str, tuple[tuple[str, int], ...]]:
    return {
        split: tuple((template.format(skill=skill), label)
                     for skill in skills for label, template in enumerate(_PATTERNS[split]))
        for split, skills in _SKILLS.items()
    }


def validate_fixtures() -> dict[str, int]:
    rows = fixtures()
    seen: set[str] = set()
    result = {}
    for split in ("cycle0", "cycle1", "heldout", "transfer"):
        cases = rows[split]
        if len(cases) != 16 or tuple(sum(y == i for _, y in cases) for i in range(4)) != (4, 4, 4, 4):
            raise ValueError("unbalanced taskmaster fixture")
        result[split] = len(cases)
        for prompt, label in cases:
            if type(prompt) is not str or not 20 < len(prompt) < 500 or type(label) is not int:
                raise ValueError("invalid taskmaster case")
            commitment = " ".join(prompt.casefold().split())
            if commitment in seen:
                raise ValueError("training/development overlap")
            seen.add(commitment)
    return result


def plan_output(parent: str | Path, run_name: str) -> Path:
    if type(run_name) is not str or not RUN_NAME.fullmatch(run_name):
        raise ValueError("invalid bounded run name")
    base = Path(parent)
    if (not base.is_dir() or base.is_symlink()
            or (hasattr(base, "is_junction") and base.is_junction())):
        raise ValueError("output parent must be an existing nonlinked directory")
    result = base.resolve(strict=True) / run_name
    if result.exists() or result.is_symlink():
        raise FileExistsError("taskmaster run already exists")
    return result


def training_indices(steps: int, length: int = 16) -> tuple[tuple[int, ...], ...]:
    if type(steps) is not int or not 1 <= steps <= 128 or length != 16:
        raise ValueError("invalid bounded training steps/fixture length")
    return tuple(tuple((step * 7 + offset) % length for offset in range(8))
                 for step in range(steps))


def compare_arms(scores: dict[str, float]) -> dict[str, Any]:
    if (type(scores) is not dict or set(scores) != set(ARMS)
            or any(type(x) not in (int, float) or not math.isfinite(x)
                   or not 0 <= x <= 1 for x in scores.values())):
        raise ValueError("all finite matched-control accuracies required")
    return {
        "correct_minus_zero": scores["correct"] - scores["zero_update"],
        "correct_minus_corrupted": scores["correct"] - scores["corrupted"],
        "correct_outperforms_both": (scores["correct"] > scores["zero_update"]
                                      and scores["correct"] > scores["corrupted"]),
    }


def _tensor_digest(named: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named:
        array = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def train_head_cycle(head: Any, encoded: Any, labels: Any, *, arm: str,
                     steps: int, learning_rate: float) -> dict[str, Any]:
    """Run real backward passes and audited optimizer updates (torch required)."""
    import torch

    if arm not in ARMS or not 1 <= steps <= 128 or not 0 < learning_rate <= 0.1:
        raise ValueError("invalid neural arm/optimization budget")
    if (encoded.ndim != 2 or len(encoded) != 16 or labels.shape != (16,)
            or not encoded.is_floating_point() or labels.dtype != torch.int64):
        raise ValueError("invalid frozen features/labels")
    names = sorted(head.named_parameters())
    before = _tensor_digest(names)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    visits = []
    head.train()
    for batch_indices in training_indices(steps):
        index = torch.tensor(batch_indices, dtype=torch.long)
        target = labels[index]
        if arm == "corrupted":
            target = (target + 2) % len(TACTICS)
        logits = head(encoded[index])
        loss = torch.nn.functional.cross_entropy(logits, target)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite taskmaster loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if arm != "zero_update":
            optimizer.step()
        visits.extend(batch_indices)
    after = _tensor_digest(names)
    if (arm == "zero_update" and after != before) or (arm != "zero_update" and after == before):
        raise RuntimeError("unexpected neural parameter update behavior")
    return {"before_sha256": before, "after_sha256": after,
            "visits": visits, "forward_backward_steps": steps,
            "optimizer_updates": 0 if arm == "zero_update" else steps}


def _hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def execute(root: Path, *, model_id: str, seed: int,
            steps_per_cycle: int, learning_rate: float) -> dict[str, Any]:
    # Reassert inside execution, including if imported directly by a caller.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_id, local_files_only=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    rows = fixtures()
    encoded = {}
    with torch.no_grad():
        for split, cases in rows.items():
            tokens = tokenizer([text for text, _ in cases], padding=True,
                               truncation=True, max_length=96, return_tensors="pt")
            result = encoder(**tokens)
            mask = tokens["attention_mask"].unsqueeze(-1)
            encoded[split] = ((result.last_hidden_state * mask).sum(dim=1)
                              / mask.sum(dim=1)).detach().contiguous()
    width = encoder.config.hidden_size
    base_tensor_sha = _tensor_digest(sorted(encoder.named_parameters()))
    report: dict[str, Any] = {
        "schema": SCHEMA, "model_id": model_id,
        "model_origin": "CACHED_PRETRAINED_ID_NOT_PUBLISHER_ATTESTED",
        "qualification": "PUBLIC_SYNTHETIC_DEVELOPMENT_ONLY",
        "frozen_encoder_sha256": base_tensor_sha,
        "script_sha256": _hash_file(Path(__file__)),
        "seed": seed, "steps_per_cycle": steps_per_cycle,
        "learning_rate": learning_rate, "fixture_sizes": validate_fixtures(),
        "tactics": list(TACTICS), "arms": {}, "torch": str(torch.__version__),
        "device": "cpu", "graduation_authorized": False,
        "independent_exam": "NOT_RUN", "model_execution_authenticated": False,
        "zero_update_control": "MATCHED_FORWARD_BACKWARD_NO_OPTIMIZER_STEP",
        "regeneration": "DISTINCT_PUBLIC_TRAIN_FIXTURE_PERSISTED_POLICY_HEAD",
    }
    root.mkdir(mode=0o700)
    for arm in ARMS:
        arm_root = root / arm
        arm_root.mkdir()
        torch.manual_seed(seed)
        head = torch.nn.Linear(width, len(TACTICS))
        initial = _tensor_digest(sorted(head.named_parameters()))
        previous = None
        cycles = []
        for cycle, split in enumerate(("cycle0", "cycle1")):
            if previous is not None:
                reloaded = torch.nn.Linear(width, len(TACTICS))
                reloaded.load_state_dict(load_file(str(previous)))
                head = reloaded
                if _tensor_digest(sorted(head.named_parameters())) != cycles[-1]["after_sha256"]:
                    raise RuntimeError("regeneration reset taskmaster learned head")
            labels = torch.tensor([label for _, label in rows[split]], dtype=torch.int64)
            before_scores = {}
            head.eval()
            with torch.no_grad():
                for partition in ("heldout", "transfer"):
                    pred = head(encoded[partition]).argmax(dim=-1)
                    gold = torch.tensor([label for _, label in rows[partition]])
                    before_scores[partition] = float((pred == gold).float().mean().item())
            update = train_head_cycle(head, encoded[split], labels, arm=arm,
                                      steps=steps_per_cycle, learning_rate=learning_rate)
            checkpoint = arm_root / f"cycle{cycle}.safetensors"
            save_file({name: value.detach().cpu().contiguous()
                       for name, value in head.state_dict().items()}, str(checkpoint))
            head.eval()
            with torch.no_grad():
                after_scores = {}
                for partition in ("heldout", "transfer"):
                    pred = head(encoded[partition]).argmax(dim=-1)
                    gold = torch.tensor([label for _, label in rows[partition]])
                    after_scores[partition] = float((pred == gold).float().mean().item())
            cycles.append({"cycle": cycle, **update,
                           "heldout_before": before_scores["heldout"],
                           "heldout_after": after_scores["heldout"],
                           "transfer_before": before_scores["transfer"],
                           "transfer_after": after_scores["transfer"],
                           "checkpoint_sha256": _hash_file(checkpoint),
                           "prompt_visits_sha256": [hashlib.sha256(rows[split][i][0].encode()).hexdigest()
                                                    for i in update["visits"]]})
            previous = checkpoint
        report["arms"][arm] = {"initial_head_sha256": initial, "cycles": cycles}
    if len({report["arms"][arm]["initial_head_sha256"] for arm in ARMS}) != 1:
        raise RuntimeError("matched arms did not start from identical weights")
    for cycle in (0, 1):
        if len({tuple(report["arms"][arm]["cycles"][cycle]["prompt_visits_sha256"])
                for arm in ARMS}) != 1:
            raise RuntimeError("matched arms did not share training visits")
    report["heldout"] = compare_arms({arm: report["arms"][arm]["cycles"][-1]["heldout_after"]
                                      for arm in ARMS})
    report["transfer"] = compare_arms({arm: report["arms"][arm]["cycles"][-1]["transfer_after"]
                                       for arm in ARMS})
    _write_json(root / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--steps-per-cycle", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.steps_per_cycle <= 128 or not 0 < args.learning_rate <= 0.1:
        parser.error("invalid bounded steps or learning rate")
    if type(args.seed) is not int or not 0 <= args.seed < 2**32:
        parser.error("invalid nonnegative seed")
    try:
        sizes = validate_fixtures()
        root = plan_output(args.output_parent, args.run_name)
        if not args.execute:
            print(json.dumps({"schema": SCHEMA, "execute": False,
                              "run_root": str(root), "model_id": args.model_id,
                              "fixture_sizes": sizes,
                              "graduation_authorized": False}, sort_keys=True))
            return 0
        report = execute(root, model_id=args.model_id, seed=args.seed,
                         steps_per_cycle=args.steps_per_cycle,
                         learning_rate=args.learning_rate)
        print(json.dumps({"run_root": str(root), "report_sha256": _hash_file(root / "report.json"),
                          "heldout": report["heldout"], "transfer": report["transfer"],
                          "graduation_authorized": False}, sort_keys=True))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"taskmaster preflight or execution failure: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
