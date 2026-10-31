"""Opt-in local Hugging Face/PEFT LoRA pilot: *actual* LM parameter updates.

This intentionally uses an already-present, small local safetensors causal-LM
checkpoint. It does not download a model, install dependencies, execute model
output, or attempt to fit multi-billion-parameter weights onto a 4 GiB GPU.

The generated mazes are public finite templates, so even a successful update
does not establish broad reasoning transfer. Running this module is not part of
the dependency-free Labyrinth test suite.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any

from .evaluation import score_response
from .maze import Challenge, generate_maze
from .runner import _source_state, load_json


LEGACY_SCHEMA = "labyrinth.hf-lora-pilot.v2"
SCHEMA = "labyrinth.hf-lora-pilot.v3"
EVAL_SCHEMA = "labyrinth.hf-lora-reload-eval.v2"
EVAL_SEED_XOR = 0x5C18A7
# Bounded deterministic attempts only; if a public heldout split cannot be
# separated from TRAIN, refuse to run instead of quietly dropping examples.
EVAL_SPLIT_MAX_ATTEMPTS = 64
# No quantization/CPU offload in v1: bound pre-existing weights to a small size.
MAX_LOCAL_WEIGHTS = 1_073_741_824
SIGNED_DEVELOPMENT_RESUME = "SIGNED_DEVELOPMENT_FAIL_BOUND"
RESEARCH_ONLY_RESUME = "UNAUTHENTICATED_RESEARCH_ONLY"
# This is an in-process misuse guard, not isolation from arbitrary code with
# execution rights inside the trainer. Durable replay and evaluator-key custody
# belong to the trusted host invoking hf_development_transition.
_authorized_development_resume: ContextVar[tuple[Any, ...] | None] = ContextVar(
    "labyrinth_hf_authorized_development_resume", default=None,
)


@dataclass(frozen=True)
class PilotConfig:
    model_path: str
    output_dir: str
    seed: int = 0
    train_count: int = 8
    eval_count: int = 8
    tiers: tuple[int, ...] = (1,)
    steps: int = 2
    max_tokens: int = 256
    max_new_tokens: int = 16
    learning_rate: float = 0.0002
    lora_rank: int = 4
    lora_alpha: int = 8
    device: str = "cpu"
    answer_only: bool = False


def _check_int(value: object, label: str, lower: int, upper: int) -> None:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{label} must be an integer in [{lower}, {upper}]")


def _paths(config: PilotConfig, *, output_must_be_new: bool) -> tuple[Path, Path, list[Path]]:
    if not isinstance(config.model_path, str) or not isinstance(config.output_dir, str):
        raise ValueError("model_path and output_dir must be strings")
    model = Path(config.model_path).expanduser().resolve(strict=True)
    if not model.is_dir() or not (model / "config.json").is_file() or (model / "config.json").is_symlink():
        raise ValueError("model_path must be an existing local HF model directory")
    weights = sorted(model.glob("*.safetensors"))
    if not weights or any(not path.is_file() or path.is_symlink() for path in weights):
        raise ValueError("local safetensors weights required; pickle/.bin-only models refused")
    total = sum(path.stat().st_size for path in weights)
    if not 0 < total <= MAX_LOCAL_WEIGHTS:
        raise ValueError("checkpoint exceeds the 1 GiB no-quantization pilot limit")
    _validate_sharded_index(model, weights)
    raw_output = Path(config.output_dir).expanduser()
    if raw_output.is_symlink():
        raise ValueError("output path must not be a symlink")
    output = raw_output.resolve(strict=False)
    if output == model or model in output.parents or output in model.parents:
        raise ValueError("output directory must be separate from local model directory")
    if output_must_be_new and (output.exists() or output.is_symlink()):
        raise FileExistsError(f"refusing to overwrite: {output}")
    if output_must_be_new and not output.parent.is_dir():
        raise ValueError("output parent must already exist")
    return model, output, weights


def _validate_sharded_index(model: Path, weights: list[Path]) -> None:
    """Fail closed if a sharded checkpoint index escapes the pinned local files."""
    index = model / "model.safetensors.index.json"
    if not index.exists() and not index.is_symlink():
        return
    if not index.is_file() or index.is_symlink():
        raise ValueError("sharded index must be a regular non-symlink JSON file")
    try:
        parsed = load_json(index, strict=True)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid sharded index: {exc}") from exc
    mapping = parsed.get("weight_map") if isinstance(parsed, dict) else None
    filenames = {p.name for p in weights}
    if (not isinstance(mapping, dict) or not mapping
            or any(type(name) is not str or not name or type(target) is not str
                   or target not in filenames for name, target in mapping.items())
            or set(mapping.values()) != filenames):
        raise ValueError("sharded index refers to unknown, unused, or nonlocal weight files")


def validate_config(config: PilotConfig, *, output_must_be_new: bool = True) -> tuple[Path, Path, list[Path]]:
    """Reject unsupported resource scales before importing heavyweight libraries."""
    if not isinstance(config, PilotConfig):
        raise TypeError("expected PilotConfig")
    _check_int(config.seed, "seed", 0, 2**31 - 1)
    _check_int(config.train_count, "train_count", 2, 32)
    _check_int(config.eval_count, "eval_count", 2, 32)
    _check_int(config.steps, "steps", 1, 128)
    _check_int(config.max_tokens, "max_tokens", 32, 1024)
    _check_int(config.max_new_tokens, "max_new_tokens", 1, 32)
    _check_int(config.lora_rank, "lora_rank", 1, 16)
    _check_int(config.lora_alpha, "lora_alpha", 1, 64)
    if type(config.learning_rate) is not float or not math.isfinite(config.learning_rate) or not 1e-7 <= config.learning_rate <= 0.001:
        raise ValueError("learning_rate must be a finite float in [1e-7, 0.001]")
    if type(config.tiers) is not tuple or not config.tiers or any(type(t) is not int for t in config.tiers):
        raise ValueError("tiers must be a non-empty tuple of integers")
    # Invoke the generator's source-of-truth validation; no eval/gold read here.
    generate_maze(seed=config.seed, count=1, tiers=config.tiers, split="train")
    if len(config.tiers) * config.train_count > 64:
        raise ValueError("training set limited to 64 examples")
    if config.steps < len(config.tiers):
        raise ValueError("steps insufficient for each training tier")
    if config.device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if type(config.answer_only) is not bool:
        raise ValueError("answer_only must be a boolean")
    resolved = _paths(config, output_must_be_new=output_must_be_new)
    _metadata_digest(resolved[0])
    return resolved


def _arithmetic_equivalence_key(challenge: Challenge) -> tuple[str, int, int] | None:
    """A narrow known-template semantic key for *public* tier-one arithmetic.

    Operand order is irrelevant for add/multiply, but not subtraction. Equal
    numerical answers to different expressions do not imply duplication.
    Unrecognized arithmetic templates fail closed; this is not a general
    mathematical equivalence checker or a private-exam isolation guarantee.
    """
    if challenge.kind != "arithmetic":
        return None
    if challenge.tier != 1:
        raise ValueError("unexpected arithmetic challenge tier")
    matched = re.fullmatch(
        r"Calculate (\d+) ([+*\-]) (\d+)\.\n"
        r"Respond with exactly one line: ANSWER: <integer>\.",
        challenge.prompt,
    )
    if matched is None:
        raise ValueError("unrecognized arithmetic prompt; cannot establish split separation")
    left, operation, right = int(matched[1]), matched[2], int(matched[3])
    if operation in ("+", "*"):
        left, right = sorted((left, right))
    return operation, left, right


def _arithmetic_semantic_commitments(cases: list[Challenge]) -> set[str]:
    """Seal known arithmetic equivalence without exposing public answer keys.

    These commitments are only for the bounded tier-one expression grammar.
    A different task family requires its own independently reviewed semantic
    equivalence relation; IDs/prompt hashes alone cannot supply one.
    """
    commitments: set[str] = set()
    for case in cases:
        key = _arithmetic_equivalence_key(case)
        if key is not None:
            operation, left, right = key
            payload = f"labyrinth.hf-public-arithmetic.v1\x00{operation}\x00{left}\x00{right}"
            commitments.add(hashlib.sha256(payload.encode("ascii")).hexdigest())
    return commitments


def _historical_arithmetic_case_count(case_ids: list[str]) -> int:
    """Count tier-one arithmetic identities in the replayable public history."""
    return sum(re.fullmatch(r"(?:train|eval)-t1-\d+-[0-9a-f]{16}", case_id) is not None
               for case_id in case_ids)


def _arithmetic_history_witnesses(cases: list[Challenge]) -> list[dict[str, str]]:
    """Retain answerless public prompts needed to verify older semantic seals.

    A digest list alone is not self-authenticating: deleting a historic
    semantic digest can conceal a future equivalent prompt. These witnesses
    bind each semantic key to the separately recorded prompt SHA and case ID.
    Independent predecessor custody is still required for *authenticity*.
    """
    return sorted(
        ({"case_id": case.id, "prompt": case.prompt}
         for case in cases if case.kind == "arithmetic"),
        key=lambda entry: entry["case_id"],
    )


def _arithmetic_case_binding_records(witnesses: list[dict[str, str]]) -> list[dict[str, str]]:
    """Bind each old case ID to its exact public prompt and semantic key.

    These hashes detect partial field substitution inside an operator-pinned
    report; they are not signatures and cannot authenticate a wholly forged
    report supplied *before* the operator establishes custody.
    """
    records: list[dict[str, str]] = []
    for witness in witnesses:
        case_id, prompt = witness["case_id"], witness["prompt"]
        case = Challenge(
            id=case_id, kind="arithmetic", tier=1,
            split="eval" if case_id.startswith("eval-") else "train",
            prompt=prompt, answer="",
        )
        semantics = _arithmetic_semantic_commitments([case])
        if len(semantics) != 1:
            raise ValueError("arithmetic witness cannot produce a unique semantic key")
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        semantic_digest = next(iter(semantics))
        raw = (f"labyrinth.hf-arithmetic-case-binding.v1\x00{case_id}\x00"
               f"{prompt_digest}\x00{semantic_digest}")
        records.append({
            "case_id": case_id,
            "prompt_sha256": prompt_digest,
            "semantic_sha256": semantic_digest,
            "binding_sha256": hashlib.sha256(raw.encode("ascii")).hexdigest(),
        })
    return sorted(records, key=lambda row: row["case_id"])


def _arithmetic_case_binding_root(records: list[dict[str, str]]) -> str:
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(("labyrinth.hf-arithmetic-bindings-root.v1\x00" + canonical)
                           .encode("ascii")).hexdigest()


def _verified_arithmetic_history_witnesses(
    witnesses: Any, historical_ids: list[str], historical_hashes: list[str],
    historical_semantics: list[str],
    historical_bindings: Any = None, historical_binding_root: Any = None,
) -> set[str]:
    """Recompute arithmetic equivalence from prompt text, never reported keys.

    This checks internal consistency of a report, not who produced it. A
    complete replacement of an unpinned report remains a custody problem.
    """
    if type(witnesses) is not list or len(witnesses) > 4096:
        raise ValueError("predecessor arithmetic witnesses missing or malformed")
    expected_ids = {case_id for case_id in historical_ids
                    if re.fullmatch(r"(?:train|eval)-t1-\d+-[0-9a-f]{16}", case_id)}
    known_hashes = set(historical_hashes)
    actual_ids: set[str] = set()
    actual_hashes: set[str] = set()
    semantic: set[str] = set()
    for witness in witnesses:
        if (type(witness) is not dict or set(witness) != {"case_id", "prompt"}
                or type(witness.get("case_id")) is not str
                or type(witness.get("prompt")) is not str
                or not 1 <= len(witness["prompt"]) <= 512
                or witness["case_id"] not in expected_ids
                or witness["case_id"] in actual_ids):
            raise ValueError("predecessor arithmetic witness identity is invalid")
        digest = hashlib.sha256(witness["prompt"].encode("utf-8")).hexdigest()
        if digest not in known_hashes or digest in actual_hashes:
            raise ValueError("predecessor arithmetic witness prompt is not committed")
        observed = Challenge(
            id=witness["case_id"], kind="arithmetic", tier=1,
            split="eval" if witness["case_id"].startswith("eval-") else "train",
            prompt=witness["prompt"], answer="",
        )
        # This rejects non-template prompts rather than guessing equivalence.
        key = _arithmetic_equivalence_key(observed)
        if key is None:
            raise ValueError("predecessor arithmetic witness lacks semantic identity")
        operation, left, right = key
        payload = f"labyrinth.hf-public-arithmetic.v1\x00{operation}\x00{left}\x00{right}"
        commitment = hashlib.sha256(payload.encode("ascii")).hexdigest()
        if commitment in semantic:
            raise ValueError("predecessor arithmetic history repeats semantic content")
        actual_ids.add(witness["case_id"])
        actual_hashes.add(digest)
        semantic.add(commitment)
    if actual_ids != expected_ids or semantic != set(historical_semantics):
        raise ValueError("predecessor semantic arithmetic history disagrees with prompt witnesses")
    if historical_bindings is not None or historical_binding_root is not None:
        expected_bindings = _arithmetic_case_binding_records(witnesses)
        if (type(historical_bindings) is not list
                or historical_bindings != expected_bindings
                or historical_binding_root != _arithmetic_case_binding_root(expected_bindings)):
            raise ValueError("predecessor arithmetic case-to-prompt binding is invalid")
    return semantic


def training_cases(config: PilotConfig) -> tuple[list[Challenge], list[Challenge]]:
    """Keep supervision in TRAIN; reroll public eval on known arithmetic overlap.

    This improves the finite generated arithmetic split without proving
    semantic independence for all tasks or any signed private examination.
    """
    train = generate_maze(seed=config.seed, count=config.train_count, tiers=config.tiers, split="train")
    train_ids = {case.id for case in train}
    train_prompts = {case.prompt for case in train}
    train_arithmetic = {_arithmetic_equivalence_key(case) for case in train
                        if case.kind == "arithmetic"}
    for attempt in range(EVAL_SPLIT_MAX_ATTEMPTS):
        heldout = generate_maze(
            seed=config.seed ^ EVAL_SEED_XOR ^ attempt, count=config.eval_count,
            tiers=config.tiers, split="eval",
        )
        heldout_arithmetic = {_arithmetic_equivalence_key(case) for case in heldout
                              if case.kind == "arithmetic"}
        if (not train_ids.intersection(case.id for case in heldout)
                and not train_prompts.intersection(case.prompt for case in heldout)
                and not train_arithmetic.intersection(heldout_arithmetic)):
            return train, heldout
    raise ValueError("public heldout could not avoid semantic arithmetic TRAIN overlap")


def _scoped_public_heldout(config: PilotConfig, scope: Any, cycle_index: int,
                           train: list[Challenge]) -> list[Challenge]:
    """Deterministic, cycle-specific *public development* diagnostic cases.

    The seed is committed to the declared scope and cycle, never influenced by
    candidate answers or training labels. This is not an unseen/private final
    examination. Unscoped v2/v3 pilots retain their historical fixed split.
    """
    from .training_scope import TrainingScopeSpec

    if type(scope) is not TrainingScopeSpec:
        raise ValueError("scoped diagnostic requires a validated TrainingScopeSpec")
    TrainingScopeSpec.__post_init__(scope)
    if type(cycle_index) is not int or not 0 <= cycle_index < scope.max_cycles:
        raise ValueError("diagnostic cycle outside declared training scope")
    digest = hashlib.sha256(
        f"labyrinth.hf-public-diagnostic.v1:{scope.fingerprint}:{config.seed}:{cycle_index}"
        .encode("ascii")
    ).digest()
    seed = int.from_bytes(digest[:4], "big") & (2**31 - 1)
    train_ids = {case.id for case in train}
    train_prompts = {case.prompt for case in train}
    train_arithmetic = {_arithmetic_equivalence_key(case) for case in train
                        if case.kind == "arithmetic"}
    for attempt in range(EVAL_SPLIT_MAX_ATTEMPTS):
        heldout = generate_maze(
            seed=seed ^ attempt, count=config.eval_count,
            tiers=config.tiers, split="eval",
        )
        heldout_arithmetic = {_arithmetic_equivalence_key(case) for case in heldout
                              if case.kind == "arithmetic"}
        if (train_ids.isdisjoint(case.id for case in heldout)
                and train_prompts.isdisjoint(case.prompt for case in heldout)
                and train_arithmetic.isdisjoint(heldout_arithmetic)):
            return heldout
    raise ValueError("scoped public heldout could not avoid semantic arithmetic TRAIN overlap")


def _scoped_training_cases(config: PilotConfig, scope: Any, cycle_index: int,
                           weak_skills: tuple[str, ...]
                           ) -> tuple[list[Challenge], list[Challenge], Any]:
    """Materialize a scoped TRAIN maze; never use exam cases for SGD.

    Public diagnostic cases change with each cycle and are shared by all
    matched arms *within* that cycle. They are not a private final exam. A
    nonexistent skill provider fails closed instead of pretending a declared
    goal was taught.
    """
    from .journey_challenges import prepare_training_rooms
    from .regenerative_maze import generate_maze_plan
    from .training_scope import TrainingScopeSpec

    if type(scope) is not TrainingScopeSpec:
        raise ValueError("resume requires a validated TrainingScopeSpec")
    TrainingScopeSpec.__post_init__(scope)
    # A scoped HF diagnostic must be backed by a genuine matching public
    # tier. maze.py now routes tier 4 to the bounded register-program generator:
    # TRAIN uses 3/4-step structures, public eval uses distinct 5/6-step
    # structures. Neither is a private graduation examination. All other
    # declared skills still fail closed.
    supported = {"arithmetic", "operation_chain", "source_priority", "program_reasoning"}
    if any(skill not in supported for skill in scope.skills):
        raise ValueError("v3 HF pilot has no matching heldout diagnostic for a declared skill")
    if type(weak_skills) is not tuple:
        raise ValueError("weak_skills must be a tuple")
    plan = generate_maze_plan(scope, cycle_index, weak_skills=weak_skills)
    prepared = prepare_training_rooms(plan, semantic_scope=scope)
    train = [room.challenge for room in prepared]
    if not 2 <= len(train) <= 64:
        raise ValueError("resumed training maze must contain 2..64 challenges")
    if tuple(sorted({challenge.tier for challenge in train})) != tuple(sorted(config.tiers)):
        raise ValueError("resumed maze tiers must exactly match pilot diagnostic tiers")
    if config.steps < len({challenge.tier for challenge in train}):
        raise ValueError("steps insufficient to visit each regenerated training tier")
    heldout = _scoped_public_heldout(config, scope, cycle_index, train)
    if ({challenge.id for challenge in train} & {challenge.id for challenge in heldout}
            or {challenge.prompt for challenge in train} & {challenge.prompt for challenge in heldout}):
        raise ValueError("scoped training maze overlaps public heldout diagnostic cases")
    return train, heldout, plan


def _resume_training_cases(config: PilotConfig, scope: Any, cycle_index: int,
                           weak_skills: tuple[str, ...], predecessor_report: dict[str, Any],
                           *, allow_unscoped_predecessor: bool = False,
                           ) -> tuple[list[Challenge], list[Challenge], Any]:
    train, heldout, plan = _scoped_training_cases(config, scope, cycle_index, weak_skills)
    from .journey_challenges import prepare_training_rooms
    from .regenerative_maze import generate_maze_plan

    prior_train_ids = predecessor_report.get("training_case_ids")
    prior_heldout_ids = predecessor_report.get("heldout_case_ids")
    if (type(prior_train_ids) is not list or type(prior_heldout_ids) is not list
            or any(type(item) is not str for item in prior_train_ids + prior_heldout_ids)):
        raise ValueError("predecessor report lacks recorded training/heldout case identities")
    prior_conf = predecessor_report.get("config")
    if type(prior_conf) is not dict:
        raise ValueError("predecessor report lacks a valid config")
    prior_training = PilotConfig(**{**prior_conf, "tiers": tuple(prior_conf["tiers"])})
    prior_cycle = predecessor_report.get("resumed_journey")
    if prior_cycle is None:
        if not allow_unscoped_predecessor or cycle_index != 1:
            raise ValueError("unscoped predecessor requires explicit cycle-1 legacy bootstrap authorization")
        # Legacy v3 had no scope declaration; this is operator-authorized
        # transfer, not a claim that the previous run learned this objective.
        prior_train, prior_eval = training_cases(prior_training)
    elif (type(prior_cycle) is not dict
          or prior_cycle.get("scope_fingerprint") != scope.fingerprint
          or prior_cycle.get("cycle_index") != cycle_index - 1):
        raise ValueError("predecessor objective/cycle identity does not match requested continuation")
    else:
        prior_weak_skills = prior_cycle.get("weak_skills")
        if type(prior_weak_skills) is not list or any(type(x) is not str for x in prior_weak_skills):
            raise ValueError("predecessor maze lacks re-playable weakness declarations")
        prior_plan = generate_maze_plan(scope, cycle_index - 1, weak_skills=prior_weak_skills)
        prior_train = [room.challenge for room in prepare_training_rooms(
            prior_plan, semantic_scope=scope)]
        prior_eval = _scoped_public_heldout(prior_training, scope, cycle_index - 1, prior_train)
    if [challenge.id for challenge in prior_train] != prior_train_ids or [challenge.id for challenge in prior_eval] != prior_heldout_ids:
        raise ValueError("predecessor training/evaluation generation disagrees with report")
    earlier = prior_train + prior_eval
    old_ids = {challenge.id for challenge in earlier}
    old_prompt_hashes = {hashlib.sha256(challenge.prompt.encode("utf-8")).hexdigest()
                         for challenge in earlier}
    old_arithmetic = _arithmetic_semantic_commitments(earlier)
    if prior_cycle is not None:
        history_ids = prior_cycle.get("historical_case_ids")
        history_hashes = prior_cycle.get("historical_prompt_sha256")
        if (type(history_ids) is not list or type(history_hashes) is not list
                or len(set(history_ids)) != len(history_ids)
                or len(set(history_hashes)) != len(history_hashes)
                or any(type(x) is not str or not x for x in history_ids)
                or any(type(x) is not str or len(x) != 64
                       or any(char not in "0123456789abcdef" for char in x)
                       for x in history_hashes)
                or not old_ids.issubset(history_ids)
                or not old_prompt_hashes.issubset(history_hashes)):
            raise ValueError("predecessor history does not seal all immediate public cases")
        if (cycle_index == 1
                and (set(history_ids) != old_ids
                     or set(history_hashes) != old_prompt_hashes)):
            raise ValueError("initial predecessor has unexplained public case history")
        old_ids.update(history_ids)
        old_prompt_hashes.update(history_hashes)
        history_arithmetic = prior_cycle.get("historical_arithmetic_sha256")
        if history_arithmetic is None and cycle_index == 1:
            # Only cycle zero is fully reconstructible from this predecessor.
            # Existing first-cycle v3 artifacts can make this single bounded
            # transition, and the successor will emit the complete new seal.
            pass
        elif (type(history_arithmetic) is not list
              or len(history_arithmetic) > 4096
              or any(type(item) is not str or len(item) != 64
                     or any(char not in "0123456789abcdef" for char in item)
                     for item in history_arithmetic)
              or len(set(history_arithmetic)) != len(history_arithmetic)
              or not old_arithmetic.issubset(history_arithmetic)
              or len(history_arithmetic) != _historical_arithmetic_case_count(history_ids)):
            raise ValueError("predecessor semantic arithmetic history is missing or invalid")
        if history_arithmetic is not None:
            old_arithmetic.update(history_arithmetic)
        witnesses = prior_cycle.get("historical_arithmetic_witnesses_v1")
        history_bindings = prior_cycle.get("historical_arithmetic_case_bindings_v1")
        history_binding_root = prior_cycle.get("historical_arithmetic_binding_root_v1")
        if witnesses is None:
            if cycle_index != 1:
                raise ValueError("predecessor lacks replayable arithmetic history witnesses")
        else:
            if history_arithmetic is None:
                raise ValueError("predecessor arithmetic witnesses lack a semantic commitment")
            if ((history_bindings is None or history_binding_root is None)
                    and cycle_index != 1):
                raise ValueError("predecessor arithmetic case-to-prompt binding is missing")
            verified = _verified_arithmetic_history_witnesses(
                witnesses, history_ids, history_hashes, history_arithmetic,
                history_bindings, history_binding_root,
            )
            immediate = {case.id: case.prompt for case in earlier
                         if case.kind == "arithmetic"}
            present = {witness["case_id"]: witness["prompt"] for witness in witnesses}
            if any(present.get(case_id) != prompt
                   for case_id, prompt in immediate.items()):
                raise ValueError("predecessor immediate arithmetic witness was altered")
            old_arithmetic.update(verified)
    new_ids = {challenge.id for challenge in train + heldout}
    new_prompt_hashes = {hashlib.sha256(challenge.prompt.encode("utf-8")).hexdigest()
                         for challenge in train + heldout}
    new_arithmetic = _arithmetic_semantic_commitments(train + heldout)
    if len(new_arithmetic) != sum(case.kind == "arithmetic" for case in train + heldout):
        raise ValueError("regenerated maze repeats semantic arithmetic cases internally")
    if new_arithmetic & old_arithmetic:
        raise ValueError("regenerated training maze overlaps prior semantic arithmetic content")
    if (new_ids & old_ids or new_prompt_hashes & old_prompt_hashes
            or {challenge.id for challenge in train} & {challenge.id for challenge in heldout}
            or {challenge.prompt for challenge in train} & {challenge.prompt for challenge in heldout}):
        raise ValueError("regenerated training maze overlaps prior or current evaluation/training content")
    return train, heldout, plan


def derange_answers(cases: list[Challenge]) -> list[str]:
    """Permutation control preserving the answer multiset, with zero collisions."""
    labels = [case.answer for case in cases]
    if len(labels) < 2:
        raise ValueError("at least two training cases required")
    frequency = max(Counter(labels).values())
    if frequency * 2 > len(labels):
        raise ValueError("cannot fully derange training answers")
    indices = sorted(range(len(labels)), key=lambda index: (labels[index], index))
    changed = list(labels)
    for position, destination in enumerate(indices):
        changed[destination] = labels[indices[(position + frequency) % len(indices)]]
    if any(a == b for a, b in zip(labels, changed)):
        raise ValueError("answer derangement left unchanged supervision")
    return changed


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weights_digest(weights: list[Path]) -> dict[str, str]:
    return {path.name: _sha_file(path) for path in weights}


def _metadata_digest(model_path: Path) -> dict[str, str]:
    """Bind tokenizer/config bytes alongside model weights for replay."""
    names = {"config.json", "generation_config.json", "special_tokens_map.json",
             "model.safetensors.index.json",
             "tokenizer_config.json", "tokenizer.json", "added_tokens.json",
             "vocab.json", "vocab.txt", "merges.txt", "sentencepiece.bpe.model",
             "tokenizer.model"}
    matches = sorted(path for path in model_path.iterdir()
                     if path.name in names or path.name.startswith("tokenizer"))
    if any(not p.is_file() or p.is_symlink() or p.stat().st_size > 268_435_456
           for p in matches):
        raise ValueError("local model metadata must be regular non-symlink files")
    return {p.name: _sha_file(p) for p in matches}


def _checkpoint_manifest(model_path: Path, weights: list[Path]) -> dict[str, dict[str, str]]:
    """Digest declared local model bytes before a load or training operation."""
    return {"weights": _weights_digest(weights), "metadata": _metadata_digest(model_path)}


def _assert_checkpoint_unchanged(model_path: Path, weights: list[Path], expected: dict[str, Any]) -> None:
    """Detect changes across load/train windows (not an atomic filesystem snapshot)."""
    current_weights = sorted(model_path.glob("*.safetensors"))
    if (current_weights != weights
            or any(not p.is_file() or p.is_symlink() for p in current_weights)
            or _checkpoint_manifest(model_path, weights) != expected):
        raise RuntimeError("base model checkpoint files changed during load or evaluation")


def _adapter_manifest(adapter_dir: Path) -> dict[str, str]:
    """Pin all local adapter file bytes, including PEFT metadata and scaling."""
    if not adapter_dir.is_dir() or adapter_dir.is_symlink():
        raise RuntimeError("adapter checkpoint root must be a non-symlink directory")
    entries = sorted(adapter_dir.rglob("*"))
    if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in entries):
        raise RuntimeError("adapter checkpoint contains a symlink or nonregular entry")
    files = [path for path in entries if path.is_file()]
    if not any(path.suffix == ".safetensors" for path in files):
        raise RuntimeError("PEFT adapter safetensors checkpoint is missing")
    return {path.relative_to(adapter_dir).as_posix(): _sha_file(path) for path in files}


def _assert_adapter_unchanged(adapter_dir: Path, expected: dict[str, str]) -> None:
    """Detect persistent adapter swaps across a load/eval window, not atomic reads."""
    try:
        current = _adapter_manifest(adapter_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("adapter checkpoint files changed during load or evaluation") from exc
    if current != expected:
        raise RuntimeError("adapter checkpoint files changed during load or evaluation")


def _prompt_commitment(cases: list[Challenge]) -> str:
    payload = [{"id": case.id, "prompt": case.prompt} for case in cases]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _selected_parameter_digest(model: Any, torch: Any, *, include: Any) -> str:
    """Hash selected tensor *values*, names, shapes and dtypes, not just keys."""
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if not include(name, parameter):
            continue
        count += 1
        raw = parameter.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
        metadata = json.dumps([name, list(parameter.shape), str(parameter.dtype)], separators=(",", ":"))
        digest.update(metadata.encode("utf-8") + b"\x00" + raw)
    if count == 0:
        raise RuntimeError("PEFT model exposed no selected adapter parameters")
    return digest.hexdigest()


def _parameter_digest(model: Any, torch: Any) -> str:
    """Hash trainable tensors before/after optimization."""
    return _selected_parameter_digest(model, torch, include=lambda name, p: p.requires_grad)


def _frozen_parameter_digest(model: Any, torch: Any) -> str:
    """Hash all frozen neural parameters rather than only trainable LoRA tensors."""
    return _selected_parameter_digest(
        model, torch, include=lambda _name, parameter: not parameter.requires_grad,
    )


def _clip_trainable_gradients(torch: Any, trainable: Any) -> None:
    """Reject finite-loss/nonfinite-gradient cases before any optimizer step."""
    torch.nn.utils.clip_grad_norm_(
        trainable, 1.0, error_if_nonfinite=True,
    )


def _assert_frozen_state(
    model: Any, torch: Any, frozen_digest: str, buffer_digest: str, phase: str,
) -> None:
    if _frozen_parameter_digest(model, torch) != frozen_digest:
        raise RuntimeError(f"{phase}: frozen neural parameters changed")
    if _buffer_digest(model, torch) != buffer_digest:
        raise RuntimeError(f"{phase}: persistent model buffers changed")


def _buffer_digest(model: Any, torch: Any) -> str:
    """Value-hash persistent model buffers, including non-trainable state.

    Optimizer LR=0 does not guarantee training forwards preserve mutable
    running-statistics buffers. Reject changed state rather than treating
    a parameter-only digest as proof of identical starting conditions.
    """
    digest = hashlib.sha256()
    for name, buffer in sorted(model.named_buffers()):
        metadata = json.dumps([name, list(buffer.shape), str(buffer.dtype)],
                              separators=(",", ":")).encode("utf-8")
        raw = buffer.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
        digest.update(metadata + b"\x00" + raw)
    return digest.hexdigest()


def _lora_value_digest(model: Any, torch: Any) -> str:
    """Hash LoRA tensor values regardless of frozen `requires_grad` on reload."""
    return _selected_parameter_digest(model, torch, include=lambda name, p: ".lora_" in name)


def _encode_pair(tokenizer: Any, prompt: str, answer: str, max_tokens: int, *,
                 answer_only: bool = False) -> tuple[list[int], list[int]]:
    # Pre-fill the constant ANSWER: scaffold, supervise only numerical output.
    task = f"Task:\n{prompt}\nAssistant:\n" + ("ANSWER:" if answer_only else "")
    completion = (f" {answer}" if answer_only else f"ANSWER: {answer}") + (tokenizer.eos_token or "")
    prefix = tokenizer(task, add_special_tokens=True)["input_ids"]
    suffix = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not suffix or not prefix or len(prefix) + len(suffix) > max_tokens:
        raise ValueError("example exceeds token budget or tokenizer produced empty tokens")
    return prefix + suffix, [-100] * len(prefix) + suffix


def _score_model(model: Any, tokenizer: Any, torch: Any, cases: list[Challenge],
                 *, device: str, max_tokens: int, max_new_tokens: int,
                 answer_only: bool = False) -> dict[str, Any]:
    model.eval()
    trials: list[dict[str, Any]] = []
    with torch.no_grad():
        for challenge in cases:
            prompt = f"Task:\n{challenge.prompt}\nAssistant:\n" + ("ANSWER:" if answer_only else "")
            encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
            ids = encoded["input_ids"]
            mask = encoded.get("attention_mask")
            if mask is None or mask.shape != ids.shape:
                raise ValueError("tokenizer must supply an aligned attention mask")
            if ids.shape[1] + max_new_tokens > max_tokens:
                raise ValueError("evaluation prompt exceeds context budget")
            ids = ids.to(device)
            output = model.generate(input_ids=ids, attention_mask=mask.to(device), do_sample=False, max_new_tokens=max_new_tokens,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()
            if answer_only:
                # Evaluation does not reward generating the supplied scaffold.
                response = f"ANSWER: {response}"
            result = score_response(challenge, response)
            trials.append({
                "id": challenge.id, "correct": result.correct,
                "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "response_preview": response[:128], "score_reason": result.reason,
            })
    return _summarize_trials(trials)


def _summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate strict-format adherence from answer correctness."""
    if not trials:
        raise ValueError("cannot summarize an empty evaluation")
    correct = sum(bool(t["correct"]) for t in trials)
    format_valid = sum(t["score_reason"] in ("correct", "incorrect") for t in trials)
    return {"correct": correct, "total": len(trials), "accuracy": correct / len(trials),
            "format_valid": format_valid, "format_rate": format_valid / len(trials),
            "trials": trials}


def _training_visit_indices(train: list[Challenge], *, seed: int, steps: int) -> list[int]:
    """Deterministic per-epoch, per-tier interleaving without silent tier omission.

    A short run can skip some examples but never an entire requested tier.
    Candidate, zero-LR sham and corrupted-label controls use identical visits.
    """
    if not train or type(seed) is not int or type(steps) is not int or steps < 1:
        raise ValueError("training visitation requires cases, seed and positive steps")
    tiers = tuple(dict.fromkeys(case.tier for case in train))
    if steps < len(tiers):
        raise ValueError("training steps insufficient for every tier")
    indices: list[int] = []
    epoch = 0
    while len(indices) < steps:
        pools: dict[int, list[int]] = {
            tier: [i for i, case in enumerate(train) if case.tier == tier]
            for tier in tiers
        }
        for tier in tiers:
            random.Random(f"{seed}:epoch:{epoch}:tier:{tier}").shuffle(pools[tier])
        for position in range(max(len(group) for group in pools.values())):
            for tier in tiers:
                if position < len(pools[tier]):
                    indices.append(pools[tier][position])
                    if len(indices) == steps:
                        return indices
        epoch += 1
    return indices


def _train_steps(model: Any, tokenizer: Any, torch: Any, train: list[Challenge],
                 labels: list[str], config: PilotConfig, *,
                 learning_rate: float | None = None,
                 optimizer: Any | None = None,
                 rng_policy: str = "seed_for_control",
                 scheduler: Any | None = None,
                 visit_cursor: int | None = None,
                 expected_visit_exposures: tuple[tuple[str, str], ...] | None = None
                 ) -> dict[str, Any]:
    """Train an arm or advance an externally restored CPU candidate.

    The default starts a fresh optimizer and reproducibly reseeds each arm.
    ``optimizer`` plus ``rng_policy='preserve'`` instead leaves the caller's
    AdamW moments and process RNG untouched. It requires a cursor and exact
    scheduled (case ID, public prompt digest) exposure slice, preventing a
    resumed segment from replaying the schedule's first examples. The caller
    must independently pin the prior cursor, catalog, source and RNG to its
    journal and keep control branches separate; this private seam alone is
    not full-state resume authority.
    """
    if len(train) != len(labels) or not train:
        raise ValueError("training supervision must match every training case")
    if scheduler is not None:
        raise ValueError("scheduler-dependent continuation is unsupported")
    external_optimizer = optimizer is not None
    if not external_optimizer:
        if rng_policy != "seed_for_control":
            raise ValueError("preserved RNG requires an externally restored optimizer")
        if visit_cursor is not None or expected_visit_exposures is not None:
            raise ValueError("exposure cursor is only for externally restored candidate training")
    else:
        if rng_policy != "preserve":
            raise ValueError("externally restored optimizer requires preserved RNG")
        if config.device != "cpu":
            raise ValueError("externally restored optimizer is CPU-only")
        if learning_rate is not None:
            raise ValueError("externally restored optimizer does not admit learning-rate controls")
        if labels != [case.answer for case in train]:
            raise ValueError("externally restored optimizer is only for correct-label candidate training")
        if (type(visit_cursor) is not int or visit_cursor < 0
                or visit_cursor + config.steps > 100000):
            raise ValueError("externally restored exposure cursor is missing or out of bounds")
        if (type(expected_visit_exposures) is not tuple
                or len(expected_visit_exposures) != config.steps
                or any(type(pair) is not tuple or len(pair) != 2
                       or type(pair[0]) is not str or type(pair[1]) is not str
                       or not re.fullmatch(r"[0-9a-f]{64}", pair[1])
                       for pair in expected_visit_exposures)):
            raise ValueError("externally restored exposure slice is missing or malformed")
        if any(case.split != "train" for case in train) or len({case.id for case in train}) != len(train):
            raise ValueError("externally restored training catalog is invalid")
        if not isinstance(optimizer, torch.optim.AdamW):
            raise ValueError("externally restored optimizer must be AdamW")
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("externally restored optimizer requires trainable model parameters")
        if any(getattr(getattr(parameter, "device", None), "type", None) != "cpu"
               for parameter in trainable):
            raise ValueError("externally restored optimizer requires CPU model parameters")
        try:
            no_buffer = object()
            first_buffer = next(iter(model.named_buffers()), no_buffer)
        except (AttributeError, TypeError) as exc:
            raise ValueError("externally restored model buffer inventory is unavailable") from exc
        if first_buffer is not no_buffer:
            raise ValueError("registered model buffers are not captured by full-neural checkpoints")
        try:
            optimizer_params = [parameter for group in optimizer.param_groups
                                for parameter in group["params"]]
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError("externally restored AdamW parameter groups are malformed") from exc
        if (len(optimizer_params) != len(trainable)
                or {id(parameter) for parameter in optimizer_params}
                != {id(parameter) for parameter in trainable}):
            raise ValueError("externally restored optimizer does not own the model's trainable parameters")
    if optimizer is None:
        indices = _training_visit_indices(train, seed=config.seed, steps=config.steps)
    else:
        full_indices = _training_visit_indices(train, seed=config.seed,
                                               steps=visit_cursor + config.steps)
        indices = full_indices[visit_cursor:]
        actual_exposures = tuple(
            (train[index].id, hashlib.sha256(train[index].prompt.encode("utf-8")).hexdigest())
            for index in indices
        )
        if actual_exposures != expected_visit_exposures:
            raise ValueError("externally restored exposure schedule differs from pinned slice")
    prepared = [_encode_pair(tokenizer, case.prompt, answer, config.max_tokens,
                             answer_only=config.answer_only)
                for case, answer in zip(train, labels)]
    if learning_rate is not None and learning_rate != 0.0:
        raise ValueError("control override only admits exact zero learning rate")
    if optimizer is None:
        # Ordinary candidate and corrupted control arms start at a common RNG
        # seed. Never reseed after a separately verified full-state restore.
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if config.device == "cuda":
            torch.cuda.manual_seed_all(config.seed)
    model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=config.learning_rate if learning_rate is None else learning_rate)
    token_total = 0
    losses: list[float] = []
    for step in range(config.steps):
        inputs, target = prepared[indices[step]]
        ids = torch.tensor([inputs], dtype=torch.long, device=config.device)
        targets = torch.tensor([target], dtype=torch.long, device=config.device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=targets).loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite SFT loss")
        loss.backward()
        _clip_trainable_gradients(
            torch, (p for p in model.parameters() if p.requires_grad)
        )
        optimizer.step()
        for parameter in model.parameters():
            if parameter.requires_grad and not bool(torch.isfinite(parameter).all()):
                raise RuntimeError("non-finite trained LoRA parameter")
        value = float(loss.detach().cpu().item())
        losses.append(value)
        token_total += len(inputs)
    result = {"steps": config.steps, "example_visits": config.steps,
              "visited_case_ids": [train[index].id for index in indices],
              "input_tokens_seen": token_total, "losses": losses}
    if external_optimizer:
        result["visit_cursor_before"] = visit_cursor
        result["visit_cursor_after"] = visit_cursor + config.steps
        result["visited_exposures"] = list(actual_exposures)
    return result


def _deps(device: str) -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise RuntimeError("optional torch, transformers and peft are required; nothing was installed") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training unavailable in this Python environment; no CPU fallback")
    return torch, AutoModelForCausalLM, AutoTokenizer, (LoraConfig, PeftModel, get_peft_model), device


def _device_headroom(torch: Any, device: str, weights: list[Path]) -> dict[str, int] | None:
    """Conservative CUDA preflight; not a memory-use guarantee for any model."""
    if device != "cuda":
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    # File size is not in-memory size. Float32 expansion, transient copies,
    # activations and CUDA allocator overhead can exceed this guard estimate.
    estimated_minimum = 4 * sum(path.stat().st_size for path in weights) + 1_073_741_824
    if estimated_minimum >= free_bytes:
        raise RuntimeError("insufficient free CUDA memory for conservative fp32 LoRA pilot preflight")
    return {"free_before_bytes": int(free_bytes), "total_bytes": int(total_bytes),
            "estimated_minimum_bytes": int(estimated_minimum)}


def _load_base(model_path: Path, device: str, *, torch: Any,
               auto_model: Any, auto_tokenizer: Any) -> tuple[Any, Any]:
    tokenizer = auto_tokenizer.from_pretrained(str(model_path), local_files_only=True,
                                               trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("checkpoint tokenizer must have an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = auto_model.from_pretrained(str(model_path), local_files_only=True,
                                       trust_remote_code=False, use_safetensors=True,
                                       dtype=torch.float32)
    model.to(device)
    model.config.use_cache = False
    return model, tokenizer


def run_pilot(config: PilotConfig, *, scope: Any = None, cycle_index: int | None = None,
              predecessor_run_dir: str | Path | None = None, resume_context: Any = None,
              weak_skills: tuple[str, ...] = (),
              allow_unscoped_predecessor: bool = False,
              unauthenticated_research_resume: bool = False) -> dict[str, Any]:
    """Run fixed-budget LoRA-SFT and controls, save a new adapter and report.

    For ordinary runs, baseline and sham share the base checkpoint. With an
    explicitly declared scope and resume context, all candidate/sham/corrupted
    arms start from identical, verified previous LoRA *tensor values*. No
    optimizer state is retained. Cycle zero can start a new scoped journey.
    A separate process must run `reload_evaluate` to qualify persistence.
    """
    if type(unauthenticated_research_resume) is not bool:
        raise ValueError("research-only resume flag must be a boolean")
    development_resume_authority = None
    # Enforce the authorization distinction before touching a model or
    # loading heavyweight dependencies. An operator-signed FAIL plus atomic
    # event reservation is required for the *bound* path; a deliberately
    # untrusted, labeled research probe can still test mechanical adaptation.
    if type(cycle_index) is int and cycle_index > 0:
        from .hf_resume_bridge import ResumeContext
        from .training_scope import TrainingScopeSpec

        if (type(scope) is not TrainingScopeSpec or type(resume_context) is not ResumeContext
                or predecessor_run_dir is None):
            raise ValueError("resumed scoped journey needs a valid scope, predecessor and resume context")
        expected_authorization = (
            config, scope.fingerprint, cycle_index,
            str(Path(predecessor_run_dir).expanduser().resolve(strict=False)),
            resume_context, weak_skills,
        )
        if _authorized_development_resume.get() == expected_authorization:
            if unauthenticated_research_resume:
                raise ValueError("signed development successor cannot be marked untrusted research")
            development_resume_authority = SIGNED_DEVELOPMENT_RESUME
        elif unauthenticated_research_resume:
            development_resume_authority = RESEARCH_ONLY_RESUME
        else:
            raise ValueError(
                "scoped successor requires an evaluator-signed development FAIL "
                "and durable reservation; use run_exam_bound_successor "
                "or explicitly label an unqualified research-only resume"
            )
    elif unauthenticated_research_resume:
        raise ValueError("research-only resume flag requires a scoped successor cycle")
    model_path, output_dir, weights = validate_config(config)
    initial_manifest = _checkpoint_manifest(model_path, weights)
    resume_source = None
    resume_receipt = None
    plan = None
    if scope is None:
        if (cycle_index is not None or predecessor_run_dir is not None or resume_context is not None
                or weak_skills or allow_unscoped_predecessor):
            raise ValueError("journey options require an explicit training scope")
        train, heldout = training_cases(config)
    else:
        from .hf_resume_bridge import ResumeContext, preflight_resume
        from .training_scope import TrainingScopeSpec

        if type(scope) is not TrainingScopeSpec:
            raise ValueError("scope must be a validated TrainingScopeSpec")
        TrainingScopeSpec.__post_init__(scope)
        if type(cycle_index) is not int or not 0 <= cycle_index < scope.max_cycles:
            raise ValueError("cycle_index outside declared training scope")
        if cycle_index == 0:
            if predecessor_run_dir is not None or resume_context is not None or allow_unscoped_predecessor:
                raise ValueError("journey cycle zero must start from fresh LoRA initialization")
            train, heldout, plan = _scoped_training_cases(config, scope, cycle_index, weak_skills)
        else:
            if type(resume_context) is not ResumeContext or predecessor_run_dir is None:
                raise ValueError("resumed journey requires explicit predecessor and ResumeContext")
            if (resume_context.scope_fingerprint != scope.fingerprint
                    or resume_context.cycle_index != cycle_index):
                raise ValueError("resume context differs from declared objective/cycle")
            if resume_context.allow_legacy_unscoped is not allow_unscoped_predecessor:
                raise ValueError("legacy bootstrap authorization differs between resume context and training request")
            resume_source = preflight_resume(config, predecessor_run_dir, resume_context)
            prior_report_path = resume_source.predecessor_run / "pilot_report.json"
            prior_report = load_json(prior_report_path, strict=True)
            if _sha_file(prior_report_path) != resume_context.predecessor_report_sha256:
                raise RuntimeError("predecessor report changed after pinned resume preflight")
            train, heldout, plan = _resume_training_cases(
                config, scope, cycle_index, weak_skills, prior_report,
                allow_unscoped_predecessor=allow_unscoped_predecessor,
            )
    corrupted_answers = derange_answers(train)
    torch, auto_model, auto_tokenizer, peft_parts, device = _deps(config.device)
    cuda_headroom = _device_headroom(torch, device, weights)
    lora_config, _, get_peft_model = peft_parts
    random.seed(config.seed)
    if device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    # Deterministic algorithms fail closed rather than silently switching to
    # nondeterministic kernels. Full cross-hardware determinism is not claimed.
    torch.use_deterministic_algorithms(True)
    if resume_source is not None:
        # The bridge repeats the pinned on-disk preflight and verifies actual
        # trainable LoRA parameter values against prior adapter values.
        from .hf_resume_bridge import load_trainable_resume

        resumed = load_trainable_resume(config, predecessor_run_dir, resume_context)
        model, tokenizer = resumed.model, resumed.tokenizer
        resume_receipt = resumed.receipt
        if (resume_receipt["previous_cycle_final_lora_sha256"]
                != resume_receipt["this_cycle_initial_lora_sha256"]):
            raise RuntimeError("resumed LoRA tensor values differ from previous cycle")
    else:
        base, tokenizer = _load_base(model_path, device, torch=torch,
                                     auto_model=auto_model, auto_tokenizer=auto_tokenizer)
    _assert_checkpoint_unchanged(model_path, weights, initial_manifest)
    # Validate all token budgets before creating an output directory.
    for case in train:
        _encode_pair(tokenizer, case.prompt, case.answer, config.max_tokens,
                     answer_only=config.answer_only)
    for case, answer in zip(train, corrupted_answers):
        _encode_pair(tokenizer, case.prompt, answer, config.max_tokens,
                     answer_only=config.answer_only)
    baseline = _score_model(model if resume_source is not None else base,
                            tokenizer, torch, heldout, device=device,
                            max_tokens=config.max_tokens, max_new_tokens=config.max_new_tokens,
                            answer_only=config.answer_only)
    if resume_source is None:
        model = get_peft_model(base, lora_config(r=config.lora_rank, lora_alpha=config.lora_alpha,
                                                 lora_dropout=0, bias="none", task_type="CAUSAL_LM",
                                                 target_modules="all-linear"))
    initial_digest = _parameter_digest(model, torch)
    initial_frozen_digest = _frozen_parameter_digest(model, torch)
    initial_buffer_digest = _buffer_digest(model, torch)
    if resume_receipt is not None:
        if (initial_digest != resume_receipt["this_cycle_initial_lora_sha256"]
                or initial_frozen_digest != resume_receipt["frozen_parameter_sha256"]
                or initial_buffer_digest != resume_receipt["persistent_buffers_sha256"]):
            raise RuntimeError("resumed adapter values or frozen model state changed before controls")
    initial_state = {name: parameter.detach().clone()
                     for name, parameter in model.named_parameters() if parameter.requires_grad}
    # Compute/exposure-matched *zero-effective-update* control: same supervised
    # examples and optimiser steps, but LR=0. Verify all trainable tensors are
    # unchanged, then start the candidate from the same initialization.
    sham_training = _train_steps(
        model, tokenizer, torch, train, [c.answer for c in train], config,
        learning_rate=0.0,
    )
    if _parameter_digest(model, torch) != initial_digest:
        raise RuntimeError("zero-LR placebo changed trainable adapter parameters")
    _assert_frozen_state(
        model, torch, initial_frozen_digest, initial_buffer_digest,
        "zero-LR placebo",
    )
    sham = _score_model(model, tokenizer, torch, heldout, device=device,
                        max_tokens=config.max_tokens, max_new_tokens=config.max_new_tokens,
                            answer_only=config.answer_only)
    _assert_frozen_state(
        model, torch, initial_frozen_digest, initial_buffer_digest,
        "zero-LR placebo evaluation",
    )
    if _parameter_digest(model, torch) != initial_digest:
        raise RuntimeError("zero-LR placebo evaluation changed initial LoRA tensor values")
    correct_training = _train_steps(model, tokenizer, torch, train, [c.answer for c in train], config)
    trained_digest = _parameter_digest(model, torch)
    if trained_digest == initial_digest:
        raise RuntimeError("no trainable LoRA parameter changed; refusing learning claim")
    trained_saved_digest = _lora_value_digest(model, torch)
    if trained_saved_digest != trained_digest:
        raise RuntimeError("trainable parameter selection differs from LoRA tensor selection")
    _assert_frozen_state(
        model, torch, initial_frozen_digest, initial_buffer_digest,
        "correct-label training",
    )
    candidate = _score_model(model, tokenizer, torch, heldout, device=device,
                             max_tokens=config.max_tokens, max_new_tokens=config.max_new_tokens,
                            answer_only=config.answer_only)
    _assert_frozen_state(
        model, torch, initial_frozen_digest, initial_buffer_digest,
        "correct-label evaluation",
    )
    # A clean, newly created output directory is used; no existing run changes.
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    candidate_dir = output_dir / "candidate_adapter"
    if candidate_dir.is_symlink():
        raise RuntimeError("candidate adapter root must not be a symlink")
    model.save_pretrained(str(candidate_dir), safe_serialization=True)
    if candidate_dir.is_symlink():
        raise RuntimeError("candidate adapter root was replaced by a symlink")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in initial_state:
                parameter.copy_(initial_state[name])
    if _parameter_digest(model, torch) != initial_digest:
        raise RuntimeError("failed to restore LoRA initialization for corrupted-label control")
    _assert_frozen_state(
        model, torch, initial_frozen_digest, initial_buffer_digest,
        "restoration before corrupted-label training",
    )
    corrupted_training = _train_steps(model, tokenizer, torch, train, corrupted_answers, config)
    corrupted_digest = _parameter_digest(model, torch)
    _assert_frozen_state(
        model, torch, initial_frozen_digest, initial_buffer_digest,
        "corrupted-label training",
    )
    corrupted = _score_model(model, tokenizer, torch, heldout, device=device,
                             max_tokens=config.max_tokens, max_new_tokens=config.max_new_tokens,
                            answer_only=config.answer_only)
    _assert_frozen_state(
        model, torch, initial_frozen_digest, initial_buffer_digest,
        "corrupted-label evaluation",
    )
    _assert_checkpoint_unchanged(model_path, weights, initial_manifest)
    candidate_adapter_manifest = _adapter_manifest(candidate_dir)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "adaptation_kind": "local HF causal-LM PEFT LoRA supervised parameter update",
        "source": _source_state(),
        "training_process_id": os.getpid(),
        "config": {**asdict(config), "model_path": str(model_path), "output_dir": str(output_dir)},
        "base_weights_sha256": initial_manifest["weights"],
        "model_metadata_sha256": initial_manifest["metadata"],
        "runtime_versions": {"python": sys.version.split()[0], "torch": package_version("torch"),
                             "transformers": package_version("transformers"),
                             "peft": package_version("peft")},
        "candidate_adapter_sha256": candidate_adapter_manifest,
        "training_case_ids": [c.id for c in train],
        "heldout_case_ids": [c.id for c in heldout],
        "heldout_public_prompt_sha256": _prompt_commitment(heldout),
        "initial_lora_parameter_sha256": initial_digest,
        "initial_persistent_buffers_sha256": initial_buffer_digest,
        "initial_frozen_parameter_sha256": initial_frozen_digest,
        "candidate_lora_parameter_sha256": trained_digest,
        "candidate_lora_tensor_value_sha256": trained_saved_digest,
        "corrupted_lora_parameter_sha256": corrupted_digest,
        "control_data": {"corrupted_label_collisions": sum(a.answer == b for a, b in zip(train, corrupted_answers)),
                         "sham_updates": 0, "sham_optimizer_steps": config.steps,
                         "sham_train_case_visits": config.steps,
                         "sham_evaluation_before_updates": False,
                         "sham_is_compute_matched": True,
                         "answer_scaffold_supplied": config.answer_only},
        "cuda_preflight": cuda_headroom,
        "training": {"sham": sham_training, "candidate": correct_training,
                     "corrupted": corrupted_training},
        "evaluations": {"baseline": baseline, "sham": sham, "candidate": candidate, "corrupted": corrupted},
        "fresh_process_reload": "NOT RUN: execute `python -m labyrinth.hf_pilot eval` separately",
        "limitations": ["Finite public templates are not evidence of general reasoning transfer.",
                        "One-seed and small-N pilot; do not generalize accuracy deltas.",
                        "The zero-LR sham matches visits and optimizer steps but not guaranteed hardware timing.",
                        "Only the adapter was persisted; base weights remain unchanged.",
                        "Same-process scores do not qualify independent checkpoint reload."],
    }
    if plan is not None:
        historical_ids = {case.id for case in train + heldout}
        historical_prompts = {hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()
                              for case in train + heldout}
        historical_arithmetic = _arithmetic_semantic_commitments(train + heldout)
        arithmetic_witness_map = {
            witness["case_id"]: witness["prompt"]
            for witness in _arithmetic_history_witnesses(train + heldout)
        }
        if resume_source is not None:
            previous_path = resume_source.predecessor_run / "pilot_report.json"
            previous = load_json(previous_path, strict=True)
            if _sha_file(previous_path) != resume_context.predecessor_report_sha256:
                raise RuntimeError("predecessor report changed before successor receipt was persisted")
            previous_cycle = previous.get("resumed_journey")
            if previous_cycle is not None:
                historical_ids.update(previous_cycle["historical_case_ids"])
                historical_prompts.update(previous_cycle["historical_prompt_sha256"])
                prior_semantic = previous_cycle.get("historical_arithmetic_sha256")
                prior_witnesses = previous_cycle.get("historical_arithmetic_witnesses_v1")
                if prior_semantic is None:
                    if previous_cycle.get("cycle_index") != 0:
                        raise ValueError("predecessor semantic arithmetic history is missing")
                    if prior_witnesses is not None:
                        raise ValueError("predecessor witnesses have no semantic commitment")
                if prior_witnesses is None:
                    if previous_cycle.get("cycle_index") != 0:
                        raise ValueError("predecessor replayable arithmetic witnesses are missing")
                    old_config = PilotConfig(**{**previous["config"],
                                                "tiers": tuple(previous["config"]["tiers"])})
                    old_train, old_eval, _ = _scoped_training_cases(
                        old_config, scope, 0, tuple(previous_cycle["weak_skills"]),
                    )
                    prior_witnesses = _arithmetic_history_witnesses(old_train + old_eval)
                    if prior_semantic is not None:
                        _verified_arithmetic_history_witnesses(
                            prior_witnesses, previous_cycle["historical_case_ids"],
                            previous_cycle["historical_prompt_sha256"], prior_semantic,
                        )
                else:
                    if (previous_cycle.get("cycle_index") != 0
                            and (previous_cycle.get("historical_arithmetic_case_bindings_v1") is None
                                 or previous_cycle.get("historical_arithmetic_binding_root_v1") is None)):
                        raise ValueError("predecessor arithmetic case-to-prompt binding is missing")
                    _verified_arithmetic_history_witnesses(
                        prior_witnesses, previous_cycle["historical_case_ids"],
                        previous_cycle["historical_prompt_sha256"], prior_semantic,
                        previous_cycle.get("historical_arithmetic_case_bindings_v1"),
                        previous_cycle.get("historical_arithmetic_binding_root_v1"),
                    )
                if prior_semantic is None:
                    historical_arithmetic.update(_arithmetic_semantic_commitments(
                        old_train + old_eval))
                else:
                    historical_arithmetic.update(prior_semantic)
            else:
                old_config = PilotConfig(**{**previous["config"], "tiers": tuple(previous["config"]["tiers"])})
                old_train, old_eval = training_cases(old_config)
                historical_ids.update(case.id for case in old_train + old_eval)
                historical_prompts.update(
                    hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()
                    for case in old_train + old_eval
                )
                historical_arithmetic.update(
                    _arithmetic_semantic_commitments(old_train + old_eval)
                )
                prior_witnesses = _arithmetic_history_witnesses(old_train + old_eval)
            for witness in prior_witnesses:
                case_id, prompt = witness["case_id"], witness["prompt"]
                if case_id in arithmetic_witness_map and arithmetic_witness_map[case_id] != prompt:
                    raise ValueError("predecessor arithmetic witness conflicts with current content")
                arithmetic_witness_map[case_id] = prompt
        complete_witnesses = [
            {"case_id": case_id, "prompt": prompt}
            for case_id, prompt in sorted(arithmetic_witness_map.items())
        ]
        case_bindings = _arithmetic_case_binding_records(complete_witnesses)
        result["resumed_journey"] = {
            "scope": scope.to_dict(), "scope_fingerprint": scope.fingerprint,
            "cycle_index": cycle_index, "weak_skills": list(weak_skills),
            "development_resume_authority": development_resume_authority,
            "training_room_ids": [room.room_id for room in plan.rooms],
            "historical_case_ids": sorted(historical_ids),
            "historical_prompt_sha256": sorted(historical_prompts),
            "historical_arithmetic_sha256": sorted(historical_arithmetic),
            "historical_arithmetic_witnesses_v1": complete_witnesses,
            "historical_arithmetic_case_bindings_v1": case_bindings,
            "historical_arithmetic_binding_root_v1": _arithmetic_case_binding_root(case_bindings),
            "predecessor_resume_receipt": resume_receipt,
            "predecessor_scope_status": (
                "LEGACY_UNSCOPED_OPERATOR_BOOTSTRAP_NOT_SAME_OBJECTIVE_PROOF"
                if allow_unscoped_predecessor else
                "SAME_DECLARED_SCOPE_RECORDED" if resume_receipt is not None else
                "NEW_SCOPED_JOURNEY_INITIALIZATION"
            ),
            "same_initial_lora_for_all_controls": True,
            "optimizer_state_carried": False,
            "minotaur_private_graduation": False,
            "private_exam_history_coverage": "NOT_ESTABLISHED",
            "qualification": "NOT_ESTABLISHED",
        }
        result["limitations"].append(
            "Scoped public diagnostics and LoRA changes do not establish Minotaur private graduation."
        )
        if resume_receipt is not None:
            result["limitations"].append(
                "Resumed only adapter tensor values, not AdamW optimizer moments or any subjective experience."
            )
    _save_new(output_dir / "pilot_report.json", result)
    return result


def _save_new(path: Path, result: dict[str, Any]) -> None:
    """Publish one complete report without ever replacing an existing report.

    A same-directory temporary file is fully written and synced before a
    hard-link atomically claims the final path. Unlike ``os.replace``, linking
    fails if another run already owns the destination. Filesystems that do not
    support hard links fail closed rather than publishing partially written
    evidence. An interrupted process can leave a hidden staging file, but the
    final report is either absent or complete.
    """
    fd, staged_name = tempfile.mkstemp(prefix=f".{path.name}.stage-", dir=path.parent)
    staged = Path(staged_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Same-filesystem hard-link creation is atomic and never overwrites.
        # If unsupported (e.g. some network filesystems), leave no final file.
        os.link(staged, path)
        # On POSIX, also sync the directory entry for power-loss durability.
        # Windows Python cannot open directories with O_DIRECTORY; the staged
        # file has still been flushed before the atomic hard-link operation.
        if os.name != "nt" and hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        staged.unlink(missing_ok=True)


def reload_evaluate(run_dir: str | Path, report_path: str | Path) -> dict[str, Any]:
    """Separate-process command: load adapter from disk and score fixed holdouts.

    This function cannot by itself attest process separation. The module CLI
    writes its own PID and a different-run CLI invocation is required.
    """
    root = Path(run_dir).expanduser().resolve(strict=True)
    source = root / "pilot_report.json"
    report = load_json(source)
    if report.get("schema") not in (SCHEMA, LEGACY_SCHEMA):
        raise ValueError("unknown training pilot report schema")
    source_state = report.get("source")
    if not isinstance(source_state, dict) or source_state.get("source_tree_sha256") != _source_state()["source_tree_sha256"]:
        raise ValueError("source revision changed since training")
    source_config = report.get("config")
    if not isinstance(source_config, dict):
        raise ValueError("invalid training pilot config")
    config = PilotConfig(**{**source_config, "tiers": tuple(source_config["tiers"])})
    model_path, output_dir, weights = validate_config(config, output_must_be_new=False)
    expected_manifest = {"weights": report.get("base_weights_sha256"),
                         "metadata": report.get("model_metadata_sha256")}
    if output_dir.resolve(strict=True) != root:
        raise ValueError("report output path does not match run directory")
    if _weights_digest(weights) != report.get("base_weights_sha256"):
        raise ValueError("base model files changed after training")
    if _metadata_digest(model_path) != report.get("model_metadata_sha256"):
        raise ValueError("local model tokenizer/config changed after training")
    adapter_dir = root / "candidate_adapter"
    if adapter_dir.is_symlink():
        raise ValueError("candidate adapter root must not be a symlink")
    actual_adapters = _adapter_manifest(adapter_dir)
    if actual_adapters != report.get("candidate_adapter_sha256"):
        raise ValueError("candidate adapter changed after training")
    cycle_receipt = report.get("resumed_journey")
    if cycle_receipt is None:
        _, heldout = training_cases(config)
    else:
        from .training_scope import TrainingScopeSpec

        if type(cycle_receipt) is not dict:
            raise ValueError("malformed scoped diagnostic receipt")
        scoped = TrainingScopeSpec.from_dict(cycle_receipt.get("scope"))
        if scoped.fingerprint != cycle_receipt.get("scope_fingerprint"):
            raise ValueError("scoped diagnostic scope fingerprint changed")
        weak = cycle_receipt.get("weak_skills", [])
        if (type(weak) is not list
                or any(type(skill) is not str for skill in weak)):
            raise ValueError("malformed scoped weakness receipt")
        _, heldout, _ = _scoped_training_cases(
            config, scoped, cycle_receipt.get("cycle_index"), tuple(weak),
        )
    if [c.id for c in heldout] != report.get("heldout_case_ids"):
        raise ValueError("heldout generation differs from training report")
    if _prompt_commitment(heldout) != report.get("heldout_public_prompt_sha256"):
        raise ValueError("heldout prompt text changed since training")
    torch, auto_model, auto_tokenizer, peft_parts, device = _deps(config.device)
    _, peft_model, _ = peft_parts
    base, tokenizer = _load_base(model_path, device, torch=torch,
                                 auto_model=auto_model, auto_tokenizer=auto_tokenizer)
    _assert_checkpoint_unchanged(model_path, weights, expected_manifest)
    model = peft_model.from_pretrained(base, str(adapter_dir), is_trainable=False,
                                      local_files_only=True)
    _assert_adapter_unchanged(adapter_dir, actual_adapters)
    observed_adapter_digest = _lora_value_digest(model, torch)
    if observed_adapter_digest != report.get("candidate_lora_tensor_value_sha256"):
        raise ValueError("reloaded LoRA tensor values differ from trained candidate")
    observed = _score_model(model, tokenizer, torch, heldout, device=device,
                            max_tokens=config.max_tokens, max_new_tokens=config.max_new_tokens,
                            answer_only=config.answer_only)
    _assert_adapter_unchanged(adapter_dir, actual_adapters)
    _assert_checkpoint_unchanged(model_path, weights, expected_manifest)
    value = {"schema": EVAL_SCHEMA, "training_report_sha256": _sha_file(source),
             "adapter_sha256": actual_adapters, "process_id": os.getpid(),
             "reloaded_lora_tensor_value_sha256": observed_adapter_digest,
             "evaluation": observed, "claimed_fresh_process": None,
             "process_id_differs": os.getpid() != report.get("training_process_id"),
             "matches_original_candidate_trials": observed["trials"] == report["evaluations"]["candidate"]["trials"],
             "limitations": ["Different process must be independently established; PID alone is not proof.",
                             "Determinism depends on compatible model kernels/hardware."]}
    _save_new(Path(report_path).expanduser(), value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Opt-in local HF LoRA pilot (no downloads)")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train tiny local safetensors checkpoint")
    train.add_argument("--model-path", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--train-count", type=int, default=8)
    train.add_argument("--eval-count", type=int, default=8)
    train.add_argument("--tiers", default="1")
    train.add_argument("--steps", type=int, default=2)
    train.add_argument("--max-tokens", type=int, default=256)
    train.add_argument("--max-new-tokens", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=0.0002)
    train.add_argument("--lora-rank", type=int, default=4)
    train.add_argument("--lora-alpha", type=int, default=8)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--answer-only", action="store_true",
                       help="Pre-fill ANSWER: and supervise only numerical completion + EOS")
    # All journey flags are opt-in. No previous adapter or scope manifest is
    # silently discovered; a new cycle always uses a distinct output root.
    train.add_argument("--scope-json", help="Local predeclared TrainingScopeSpec JSON (no downloads)")
    train.add_argument("--cycle-index", type=int)
    train.add_argument("--weak-skills", default="", help="Comma-separated skills from the declared scope")
    train.add_argument("--predecessor-run-dir", help="Exact previous v3 HF pilot output directory")
    train.add_argument("--predecessor-report-sha256", help="Operator-pinned previous pilot report SHA-256")
    train.add_argument("--predecessor-lora-sha256", help="Operator-pinned previous LoRA tensor SHA-256")
    train.add_argument("--resume-authority-ref", help="Opaque operator approval record identifier")
    train.add_argument("--allow-legacy-unscoped-predecessor", action="store_true",
                       help="Explicitly bootstrap from an unscoped v3 run (no same-objective lineage claim)")
    train.add_argument("--unauthenticated-research-resume", action="store_true",
                       help="Explicitly permit an unqualified unsigned successor for mechanical research only")
    evaluate = commands.add_parser("eval", help="Reload saved adapter in a new Python process")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            journey_keys = {"scope_json", "cycle_index", "weak_skills", "predecessor_run_dir",
                            "predecessor_report_sha256", "predecessor_lora_sha256",
                            "resume_authority_ref", "allow_legacy_unscoped_predecessor",
                            "unauthenticated_research_resume"}
            config_dict = {key: value for key, value in vars(args).items()
                           if key != "command" and key not in journey_keys}
            config_dict["tiers"] = tuple(int(t) for t in args.tiers.split(",")) if args.tiers else ()
            journey_kwargs: dict[str, Any] = {}
            if args.scope_json is not None:
                from .training_scope import MAX_SCOPE_JSON_BYTES, parse_scope_json
                scope_path = Path(args.scope_json).expanduser()
                if (scope_path.is_symlink() or not scope_path.is_file()
                        or not 0 < scope_path.stat().st_size <= MAX_SCOPE_JSON_BYTES):
                    raise ValueError("scope JSON must be a bounded regular non-symlink local file")
                scope = parse_scope_json(scope_path.read_bytes())
                if args.cycle_index is None:
                    raise ValueError("a scoped journey requires --cycle-index")
                weak = tuple(args.weak_skills.split(",")) if args.weak_skills else ()
                journey_kwargs = {"scope": scope, "cycle_index": args.cycle_index,
                                  "weak_skills": weak,
                                  "allow_unscoped_predecessor": args.allow_legacy_unscoped_predecessor,
                                  "unauthenticated_research_resume": args.unauthenticated_research_resume}
                if args.cycle_index > 0:
                    if not all((args.predecessor_run_dir, args.predecessor_report_sha256,
                                args.predecessor_lora_sha256, args.resume_authority_ref)):
                        raise ValueError("resume requires all explicitly pinned predecessor flags")
                    from .hf_resume_bridge import ResumeContext
                    journey_kwargs["predecessor_run_dir"] = args.predecessor_run_dir
                    journey_kwargs["resume_context"] = ResumeContext(
                        scope_fingerprint=scope.fingerprint, cycle_index=args.cycle_index,
                        predecessor_report_sha256=args.predecessor_report_sha256,
                        predecessor_lora_value_sha256=args.predecessor_lora_sha256,
                        authority_ref=args.resume_authority_ref,
                        allow_legacy_unscoped=args.allow_legacy_unscoped_predecessor,
                    )
                elif any((args.predecessor_run_dir, args.predecessor_report_sha256,
                          args.predecessor_lora_sha256, args.resume_authority_ref)):
                    raise ValueError("cycle zero cannot load a predecessor adapter")
            elif (args.cycle_index is not None or args.weak_skills
                  or args.predecessor_run_dir or args.predecessor_report_sha256
                  or args.predecessor_lora_sha256 or args.resume_authority_ref
                  or args.allow_legacy_unscoped_predecessor
                  or args.unauthenticated_research_resume):
                raise ValueError("journey options require a local --scope-json manifest")
            result = run_pilot(PilotConfig(**config_dict), **journey_kwargs)
            print(json.dumps({"schema": result["schema"], "output_dir": result["config"]["output_dir"],
                              "changed": result["initial_lora_parameter_sha256"] != result["candidate_lora_parameter_sha256"]}))
        else:
            result = reload_evaluate(args.run_dir, args.report)
            # The diagnostic report was persisted by reload_evaluate even on
            # mismatch; the CLI must not represent mismatch as successful
            # qualification by returning zero status.
            if result["matches_original_candidate_trials"] is not True:
                raise RuntimeError("reloaded candidate did not reproduce training trials")
            if result["process_id_differs"] is not True:
                raise RuntimeError("reload was not observed in a different process")
            print(json.dumps({"schema": result["schema"],
                              "matches_original_candidate_trials": True,
                              "process_id_differs": True,
                              "independent_executor_attestation": False}))
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
