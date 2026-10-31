"""Evaluator-custodied, *public-development-grade* structural transfer probe.

Prepare 7/8-instruction register programs outside the training generator's
3/4-instruction and public 5/6-instruction templates. The evaluator retains
the private key file and provides only answerless prompts to inference workers.
Scoring accepts predictions as data; it does not authenticate their origin,
execute a model, provide a private graduation exam, or establish independence
merely because two output paths are named differently.

Example:
  python scripts/evaluate_sealed_transfer.py prepare \
      --private-dir EVALUATOR_ONLY_DIR --public-dir ANSWERLESS_PROMPTS_DIR
  # Supply only ANSWERLESS_PROMPTS_DIR/prompts.jsonl to inference workers.
  python scripts/evaluate_sealed_transfer.py score \
      --private-dir EVALUATOR_ONLY_DIR --public-dir ANSWERLESS_PROMPTS_DIR \
      --baseline BASE.jsonl --candidate CAND.jsonl --sham SHAM.jsonl \
      --corrupted CORR.jsonl --output NEW_RESULT.json

Keep EVALUATOR_ONLY_DIR (including keys and random seed) off trainer paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import secrets
import sys
from typing import Any


SCHEMA = "labyrinth.evaluator-structural-transfer.v1"
ARM_NAMES = ("baseline", "candidate", "sham", "corrupted")
ANSWER = re.compile(r"\s*ANSWER:\s*([+-]?[0-9]{1,12})\s*\Z", re.I | re.ASCII)
COUNT = 64
INSTRUCTIONS = ("ADD", "SUB", "MUL")
PROGRAM_HEADER = (
    "Execute this register program in order. ADD, SUB, and MUL update "
    "the named target register using its current value and the operand. "
    "An operand is either a literal integer or the current value of the "
    "named register. After each instruction, replace the updated value "
    "with its nonnegative remainder modulo 97. Other registers do not change."
)
PROGRAM_FOOTER = (
    "What is the final value of A? Respond with exactly one line: ANSWER: <integer>."
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def _create_new(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON object key: {key}")
        parsed[key] = value
    return parsed


def _read_jsonl(path: Path, *, max_bytes: int = 2_000_000) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= max_bytes:
        raise ValueError(f"invalid or oversized JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_bytes().splitlines():
        if not line or len(line) > 16_384:
            raise ValueError("empty or oversized JSONL row")
        row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        if not isinstance(row, dict):
            raise ValueError("JSONL rows must be objects")
        rows.append(row)
        if len(rows) > COUNT:
            raise ValueError("too many evaluation rows")
    return rows


def _all_steps_reach_final_a(
    commands: list[tuple[str, str, str, str]],
) -> bool:
    """Versioned backward dataflow: reject dead instruction padding."""
    latest: dict[str, str] = {"A": "A0", "B": "B0"}
    dependencies: dict[str, tuple[str, ...]] = {}
    for position, (_, target, operand_kind, _) in enumerate(commands, 1):
        node = str(position)
        incoming = (latest[target],)
        if operand_kind != "C":
            incoming += (latest[operand_kind],)
        dependencies[node] = incoming
        latest[target] = node
    seen: set[str] = set()
    stack = [latest["A"]]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(dependencies.get(node, ()))
    required = {str(position) for position in range(1, len(commands) + 1)}
    return required.issubset(seen) and "B0" in seen


def _execute_commands(
    commands: list[tuple[str, str, str, str]], initial_a: int, initial_b: int,
    *, intervene_after: int | None = None, perturb: int = 1,
) -> int:
    registers = {"A": initial_a, "B": initial_b}
    for position, (operation, target, operand_kind, printed_operand) in enumerate(commands, 1):
        operand = int(printed_operand) if operand_kind == "C" else registers[operand_kind]
        previous = registers[target]
        if operation == "ADD":
            result = previous + operand
        elif operation == "SUB":
            result = previous - operand
        else:
            result = previous * operand
        registers[target] = result % 97
        if position == intervene_after:
            registers[target] = (registers[target] + perturb) % 97
    return registers["A"]


def _case(secret_seed: str, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write an independent arithmetic oracle while formatting each instruction."""
    rng = random.Random(int(_digest(f"{secret_seed}:{index}".encode()), 16))
    steps = 7 if index < COUNT // 2 else 8
    # Deterministically resample until the probe actually tests cross-register
    # state, updates both registers, and takes a modular wraparound.
    for _ in range(1024):
        regs = {"A": rng.randrange(97), "B": rng.randrange(97)}
        initial = dict(regs)
        structures: list[str] = []
        lines: list[str] = []
        commands: list[tuple[str, str, str, str]] = []
        targets: set[str] = set()
        cross_reads = 0
        cross_directions: set[tuple[str, str]] = set()
        wrapped = False
        for step in range(1, steps + 1):
            operation = rng.choice(INSTRUCTIONS)
            target = rng.choice(("A", "B"))
            operand_kind = rng.choice(("A", "B", "C"))
            operand = str(rng.randrange(2, 10)) if operand_kind == "C" else operand_kind
            right = int(operand) if operand_kind == "C" else regs[operand_kind]
            before = regs[target]
            if operation == "ADD":
                raw = before + right
            elif operation == "SUB":
                raw = before - right
            else:
                raw = before * right
            regs[target] = raw % 97
            wrapped |= not 0 <= raw < 97
            cross_reads += int(operand_kind in ("A", "B") and operand_kind != target)
            if operand_kind in ("A", "B") and operand_kind != target:
                cross_directions.add((operand_kind, target))
            targets.add(target)
            commands.append((operation, target, operand_kind, operand))
            structures.append(f"{operation} {target} {operand_kind}")
            lines.append(f"{step}. {operation} {target} {operand}")
        structurally_live = (targets == {"A", "B"} and cross_reads >= 2 and wrapped
                and cross_directions == {("A", "B"), ("B", "A")}
                and _all_steps_reach_final_a(commands))
        if structurally_live:
            sensitive_to_b = any(
                _execute_commands(commands, initial["A"], (initial["B"] + delta) % 97)
                != regs["A"] for delta in (1, 2, 3)
            )
            sensitive_to_a = any(
                _execute_commands(commands, (initial["A"] + delta) % 97, initial["B"])
                != regs["A"] for delta in (1, 2, 3)
            )
            every_step_effective = all(
                any(_execute_commands(commands, initial["A"], initial["B"],
                                      intervene_after=position, perturb=delta)
                    != regs["A"] for delta in (1, 2, 3))
                for position in range(1, steps + 1)
            )
            if sensitive_to_a and sensitive_to_b and every_step_effective:
                break
    else:
        raise RuntimeError("could not generate an all-step-effective, B-dependent transfer case")
    structure = ";".join(structures)
    prompt = (
        PROGRAM_HEADER + "\n"
        f"Initial registers: A={initial['A']}, B={initial['B']}\n"
        + "\n".join(lines)
        + "\n" + PROGRAM_FOOTER
    )
    case_id = f"evaluator-v1-{steps}-{index:03d}-{_digest(prompt.encode())[:12]}"
    public = {"id": case_id, "steps": steps, "prompt": prompt,
              "prompt_sha256": _digest(prompt.encode()),
              "structure_sha256": _digest(structure.encode())}
    private = {"id": case_id, "answer": str(regs["A"]),
               "steps": steps, "prompt_sha256": public["prompt_sha256"]}
    return public, private


def make_cases(seed: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not re.fullmatch(r"[0-9a-f]{32}", seed):
        raise ValueError("evaluator seed must be a 128-bit lowercase hex value")
    records = [_case(seed, index) for index in range(COUNT)]
    public = [record[0] for record in records]
    private = [record[1] for record in records]
    if len({row["structure_sha256"] for row in public}) != COUNT:
        raise RuntimeError("novel structural transfer programs were not distinct")
    if len({row["prompt_sha256"] for row in public}) != COUNT:
        raise RuntimeError("duplicate transfer prompts")
    return public, private


def as_custodied_transfer_cases(
    public: list[dict[str, Any]], private: list[dict[str, Any]],
) -> tuple[Any, ...]:
    """Adapt verified transfer cases to the independent holdout protocol.

    Call only inside evaluator custody: the returned ExamCase objects include
    private gold. This is a transfer *split* adapter, not a heldout/retention
    bank, model attestation, or graduation decision.
    """
    from labyrinth.independent_holdout import ExamCase

    if (type(public) is not list or type(private) is not list
            or len(public) != COUNT or len(private) != COUNT):
        raise ValueError("complete evaluator-held transfer pair required")
    cases = []
    for prompt_case, key in zip(public, private):
        if type(prompt_case) is not dict or type(key) is not dict:
            raise ValueError("invalid evaluator transfer case")
        case_id = prompt_case.get("id")
        prompt = prompt_case.get("prompt")
        steps = prompt_case.get("steps")
        if (type(case_id) is not str or type(prompt) is not str
                or type(steps) is not int or steps not in (7, 8)
                or (case_id, steps, prompt_case.get("prompt_sha256"))
                != (key.get("id"), key.get("steps"), key.get("prompt_sha256"))
                or _digest(prompt.encode()) != prompt_case.get("prompt_sha256")):
            raise ValueError("mismatched evaluator transfer identities")
        observed_answer, observed_structure = independent_answer(prompt, steps)
        if (key.get("answer") != observed_answer
                or prompt_case.get("structure_sha256") != observed_structure):
            raise ValueError("evaluator transfer gold/structure differs from printed program")
        case = ExamCase(case_id, "transfer", "program_reasoning", prompt,
                        observed_answer, observed_structure)
        case.validate()
        cases.append(case)
    if (len({c.case_id for c in cases}) != COUNT
            or len({c.prompt for c in cases}) != COUNT
            or len({c.structure_key for c in cases}) != COUNT):
        raise ValueError("duplicate evaluator transfer cases")
    return tuple(cases)


def independent_answer(prompt: str, steps: int) -> tuple[str, str]:
    """At scoring time interpret printed instructions independently of keys."""
    lines = prompt.splitlines()
    if (len(lines) != steps + 3 or steps not in (7, 8)
            or lines[0] != PROGRAM_HEADER or lines[-1] != PROGRAM_FOOTER):
        raise ValueError("unrecognized held-out program header/length/footer")
    match = re.fullmatch(r"Initial registers: A=(\d+), B=(\d+)", lines[1])
    if match is None:
        raise ValueError("unrecognized held-out register initialization")
    registers = {"A": int(match[1]), "B": int(match[2])}
    initial_a, initial_b = registers["A"], registers["B"]
    if any(not 0 <= value < 97 for value in registers.values()):
        raise ValueError("registers outside supported bounds")
    structure = []
    commands: list[tuple[str, str, str, str]] = []
    cross_directions: set[tuple[str, str]] = set()
    wrapped = False
    for index, line in enumerate(lines[2:-1], 1):
        instruction = re.fullmatch(rf"{index}\. (ADD|SUB|MUL) ([AB]) ([AB]|[2-9])", line)
        if instruction is None:
            raise ValueError("unrecognized or reordered held-out instruction")
        operation, target, operand = instruction.groups()
        right = registers[operand] if operand in registers else int(operand)
        previous = registers[target]
        if operation == "ADD":
            value = previous + right
        elif operation == "SUB":
            value = previous - right
        else:
            value = previous * right
        wrapped |= not 0 <= value < 97
        registers[target] = value % 97
        kind = operand if operand in registers else "C"
        commands.append((operation, target, kind, operand))
        if kind in registers and kind != target:
            cross_directions.add((kind, target))
        structure.append(f"{operation} {target} {operand if operand in registers else 'C'}")
    final_a = registers["A"]
    if (not wrapped or cross_directions != {("A", "B"), ("B", "A")}
            or not _all_steps_reach_final_a(commands)
            or not any(_execute_commands(commands, (initial_a + delta) % 97, initial_b) != final_a
                       for delta in (1, 2, 3))
            or not any(_execute_commands(commands, initial_a, (initial_b + delta) % 97) != final_a
                       for delta in (1, 2, 3))
            or not all(any(_execute_commands(commands, initial_a, initial_b,
                                              intervene_after=position, perturb=delta) != final_a
                           for delta in (1, 2, 3))
                       for position in range(1, steps + 1))):
        raise ValueError("held-out program lacks full causal transfer structure")
    return str(registers["A"]), _digest(";".join(structure).encode())


def _disjoint_new_dirs(private_dir: Path, public_dir: Path) -> None:
    for directory in (private_dir, public_dir):
        if directory.is_symlink() or directory.exists() or not directory.parent.is_dir():
            raise FileExistsError("both evaluation directories must be new with existing parents")
    private = private_dir.resolve(strict=False)
    public = public_dir.resolve(strict=False)
    if private == public or private in public.parents or public in private.parents:
        raise ValueError("private and public evaluation directories must be disjoint")


def prepare(private_dir: Path, public_dir: Path, seed: str | None = None) -> dict[str, Any]:
    _disjoint_new_dirs(private_dir, public_dir)
    secret_seed = secrets.token_hex(16) if seed is None else seed
    public, private = make_cases(secret_seed)
    # The seed and gold answers live only in the evaluator-owned directory,
    # physically distinct from the answerless handoff to inference workers.
    prompts = _jsonl(public)
    gold = _jsonl(private)
    private_dir.mkdir(mode=0o700)
    public_dir.mkdir(mode=0o700)
    _create_new(public_dir / "prompts.jsonl", prompts)
    _create_new(private_dir / "keys.jsonl", gold)
    _create_new(private_dir / "seed.txt", secret_seed.encode("ascii") + b"\n")
    manifest = {"schema": SCHEMA, "count": COUNT,
                "slices": {"7-step": COUNT // 2, "8-step": COUNT // 2},
                "prompts_sha256": _digest(prompts),
                "keys_sha256": _digest(gold),
                "seed_commitment_sha256": _digest(secret_seed.encode()),
                "origin": "independent algorithm, 7/8-step programs; not the Labyrinth generator",
                "status": "SOURCE_PRECOMMIT_ONLY",
                "limitations": ["Separate directories do not prove independent access-control custody.",
                                "Do not supply the evaluator-only directory to training/inference workers.",
                                "Public source, deterministic template grammar and access to keys permit contamination."]}
    _create_new(private_dir / "manifest.json", _canonical(manifest) + b"\n")
    return manifest


def _arm_results(keys: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = {}
    for row in rows:
        case_id, response = row.get("id"), row.get("response")
        if not isinstance(case_id, str) or case_id in predictions or not isinstance(response, str) or len(response) > 8192:
            raise ValueError("prediction id must be unique and response must be bounded text")
        predictions[case_id] = response
    if set(predictions) != {row["id"] for row in keys}:
        raise ValueError("predictions must cover exactly every held-out case once")
    results = []
    for key in keys:
        match = ANSWER.fullmatch(predictions[key["id"]])
        formatted = match is not None
        correct = bool(match and str(int(match[1])) == key["answer"])
        results.append({"id": key["id"], "steps": key["steps"],
                        "correct": correct, "format_valid": formatted})
    by_length = {str(steps): {"correct": sum(r["correct"] for r in results if r["steps"] == steps),
                             "total": sum(r["steps"] == steps for r in results)}
                 for steps in (7, 8)}
    return {"correct": sum(r["correct"] for r in results), "total": len(results),
            "format_valid": sum(r["format_valid"] for r in results),
            "by_length": by_length, "cases": results}


def score(private_dir: Path, public_dir: Path,
          prediction_paths: dict[str, Path], output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite evaluation result")
    manifest_path = private_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > 16384:
        raise ValueError("invalid evaluator manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"),
                          object_pairs_hook=_reject_duplicate_keys)
    if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA
            or manifest.get("count") != COUNT):
        raise ValueError("wrong evaluator schema/count")
    prompts_path = public_dir / "prompts.jsonl"
    gold_path = private_dir / "keys.jsonl"
    for path in (prompts_path, gold_path):
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 2_000_000:
            raise ValueError("invalid or oversized evaluator prompt/gold file")
    if (_digest(prompts_path.read_bytes()) != manifest.get("prompts_sha256")
            or _digest(gold_path.read_bytes()) != manifest.get("keys_sha256")):
        raise ValueError("evaluator prompt/gold commitment changed")
    prompts, keys = _read_jsonl(prompts_path), _read_jsonl(gold_path)
    if (len(prompts) != COUNT or len(keys) != COUNT
            or [(r["id"], r["steps"], r["prompt_sha256"]) for r in prompts]
            != [(r["id"], r["steps"], r["prompt_sha256"]) for r in keys]
            or any(_digest(r["prompt"].encode()) != r["prompt_sha256"] for r in prompts)):
        raise ValueError("mismatched evaluator prompt and gold identities")
    for public, private in zip(prompts, keys):
        observed_answer, observed_structure = independent_answer(public["prompt"], public["steps"])
        if (private["answer"] != observed_answer
                or public["structure_sha256"] != observed_structure):
            raise ValueError("held-out gold/structure disagrees with printed program")
    if set(prediction_paths) != set(ARM_NAMES):
        raise ValueError("four specified arms required")
    arms = {arm: _arm_results(keys, _read_jsonl(path))
            for arm, path in prediction_paths.items()}
    comparisons = {}
    candidate = {c["id"]: c["correct"] for c in arms["candidate"]["cases"]}
    for arm in ("baseline", "sham", "corrupted"):
        control = {c["id"]: c["correct"] for c in arms[arm]["cases"]}
        comparisons[arm] = {
            "candidate_only": sum(candidate[key] and not control[key] for key in candidate),
            "control_only": sum(not candidate[key] and control[key] for key in candidate),
            "paired_accuracy_delta": (arms["candidate"]["correct"] - arms[arm]["correct"]) / COUNT,
        }
    result = {"schema": SCHEMA, "precommit_sha256": _digest(manifest_path.read_bytes()),
              "source": "evaluator-provided predictions; no inference provenance verification",
              "status": "SCORED_UNAUTHENTICATED_PREDICTIONS_NOT_A_GRADUATION_VERDICT",
              "arms": {arm: {k: v for k, v in values.items() if k != "cases"}
                       for arm, values in arms.items()},
              "paired_comparisons": comparisons,
              "limitations": ["No pretrained origin, checkpoint, tokenizer, decoding or runtime attestation.",
                              "One seed and 64 finite public-grammar programs cannot establish broad transfer.",
                              "A scored answer file does not prove its claimed model generated the answers."]}
    _create_new(output, _canonical(result) + b"\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("prepare")
    plan.add_argument("--private-dir", type=Path, required=True)
    plan.add_argument("--public-dir", type=Path, required=True)
    plan.add_argument("--seed", help="Optional evaluator-owned 128-bit hex seed")
    judge = sub.add_parser("score")
    judge.add_argument("--private-dir", type=Path, required=True)
    judge.add_argument("--public-dir", type=Path, required=True)
    for arm in ARM_NAMES:
        judge.add_argument(f"--{arm}", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            report = prepare(args.private_dir, args.public_dir, args.seed)
        else:
            report = score(args.private_dir, args.public_dir,
                           {name: getattr(args, name) for name in ARM_NAMES}, args.output)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"evaluator error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
