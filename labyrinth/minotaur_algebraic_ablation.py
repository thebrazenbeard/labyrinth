"""Public, read-only representation ablation for Minotaur's algebraic head.

This intentionally fits coefficients with least squares. It is not a neural
training experiment, an independent private exam, or graduation evidence.
Both arms have twelve coefficients; only the supplied fourth feature changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

from .minotaur_algebraic_rules import MOD, OPS, incomplete_training_atoms

SCHEMA = "labyrinth.minotaur-algebraic-representation-ablation.v1"
_ARMS = ("bilinear", "no_xy")
_OP_NAMES = ("ADD", "SUB", "MUL")
_MAX_STEPS = 6


def _source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("minotaur_algebraic_ablation.py", "minotaur_algebraic_rules.py")
    }


def _features(torch: Any, x: Any, y: Any, arm: str) -> Any:
    cross = x * y if arm == "bilinear" else x * x
    return torch.stack((torch.ones_like(x), x / 8.0, y / 8.0,
                        cross / 64.0), dim=-1)


def _fit_coefficients(torch: Any, examples: list[tuple[int, int, int, int]],
                      arm: str) -> tuple[Any, list[int]]:
    coefficients = []
    ranks = []
    for op in range(OPS):
        subset = [row for row in examples if row[0] == op]
        x = torch.tensor([row[1] for row in subset], dtype=torch.float64)
        y = torch.tensor([row[2] for row in subset], dtype=torch.float64)
        labels = torch.tensor([row[3] for row in subset],
                              dtype=torch.float64) / 64.0
        matrix = _features(torch, x, y, arm)
        rank = int(torch.linalg.matrix_rank(matrix).item())
        if rank != 4:
            raise ValueError("rank-deficient sparse training design")
        fitted = torch.linalg.lstsq(matrix, labels).solution
        if not bool(torch.isfinite(fitted).all()):
            raise ValueError("nonfinite fitted coefficients")
        ranks.append(rank)
        coefficients.append(fitted)
    return torch.stack(coefficients), ranks


def _predict(torch: Any, coefficients: Any, arm: str,
             op: int, x: int, y: int) -> int:
    a = torch.tensor([x], dtype=torch.float64)
    b = torch.tensor([y], dtype=torch.float64)
    raw = (_features(torch, a, b, arm)[0] * coefficients[op]).sum() * 64.0
    return int(torch.remainder(torch.round(raw).long(), MOD).item())


def _evaluate_atoms(torch: Any, coefficients: Any, arm: str,
                    examples: list[tuple[int, int, int, int]]) -> dict[str, Any]:
    residues = torch.arange(MOD, dtype=torch.float64)
    x = residues.repeat_interleave(MOD)
    y = residues.repeat(MOD)
    full, withheld, wraps, wrap_totals = [], [], [], []
    heldout_totals = []
    for op in range(OPS):
        raw = x + y if op == 0 else x - y if op == 1 else x * y
        truth = raw.long().remainder(MOD)
        predicted = torch.remainder(torch.round(
            _features(torch, x, y, arm) @ coefficients[op] * 64.0
        ).long(), MOD)
        correct = predicted == truth
        trained = {(row[1], row[2]) for row in examples if row[0] == op}
        train_mask = torch.tensor([(int(a), int(b)) in trained
                                   for a, b in zip(x, y)], dtype=torch.bool)
        wrap_mask = (raw < 0) | (raw >= MOD)
        full.append(int(correct.sum().item()))
        withheld.append(int((correct & ~train_mask).sum().item()))
        heldout_totals.append(int((~train_mask).sum().item()))
        wraps.append(int((correct & wrap_mask).sum().item()))
        wrap_totals.append(int(wrap_mask.sum().item()))
    return {
        "full_correct_by_op": full,
        "full_total_by_op": [MOD * MOD] * OPS,
        "withheld_correct_by_op": withheld,
        "withheld_total_by_op": heldout_totals,
        "wrap_correct_by_op": wraps,
        "wrap_total_by_op": wrap_totals,
    }


def _make_programs(seed: int, count: int) -> list[tuple[tuple[int, int],
                                                        list[tuple[int, int, bool, int]]]]:
    rng = random.Random(seed)
    programs = []
    for _ in range(count):
        initial = (rng.randrange(MOD), rng.randrange(MOD))
        commands = []
        for _ in range(rng.randint(1, _MAX_STEPS)):
            op = rng.randrange(OPS)
            target = rng.randrange(2)
            is_register = rng.randrange(2) == 1
            value = rng.randrange(2) if is_register else rng.randrange(MOD)
            commands.append((op, target, is_register, value))
        programs.append((initial, commands))
    return programs


def _evaluate_programs(torch: Any, coefficients: Any, arm: str,
                       programs: list[tuple[tuple[int, int],
                                            list[tuple[int, int, bool, int]]]]) -> dict[str, Any]:
    exact = correct_states = total_states = 0
    lengths = {n: [0, 0] for n in range(1, _MAX_STEPS + 1)}
    for initial, commands in programs:
        model_state = list(initial)
        gold_state = list(initial)
        whole_program_correct = True
        for op, target, is_register, value in commands:
            model_operand = model_state[value] if is_register else value
            gold_operand = gold_state[value] if is_register else value
            predicted = _predict(torch, coefficients, arm, op,
                                 model_state[target], model_operand)
            old_gold = gold_state[target]
            gold_state[target] = (old_gold + gold_operand if op == 0 else
                                  old_gold - gold_operand if op == 1 else
                                  old_gold * gold_operand) % MOD
            model_state[target] = predicted
            correct = model_state == gold_state
            correct_states += int(correct)
            total_states += 1
            whole_program_correct = whole_program_correct and correct
        exact += int(whole_program_correct)
        lengths[len(commands)][0] += int(whole_program_correct)
        lengths[len(commands)][1] += 1
    return {
        "exact": exact, "total": len(programs),
        "states_correct": correct_states, "states_total": total_states,
        "by_length": [lengths[n] for n in range(1, _MAX_STEPS + 1)],
    }


def run_ablation(torch: Any, *, seeds: tuple[int, ...] = (47, 42, 2026),
                 program_seed: int = 808, program_count: int = 128) -> dict[str, Any]:
    """Compare equal-sized feature families on the same public examples.

    Training labels are used only to fit coefficients. Whole-domain arithmetic
    and full program trajectories are computed solely by offline evaluators.
    """
    if (type(seeds) is not tuple or not 1 <= len(seeds) <= 20
            or any(type(seed) is not int or not 0 <= seed < 2**31
                   for seed in seeds)
            or len(set(seeds)) != len(seeds)
            or type(program_seed) is not int or not 0 <= program_seed < 2**31
            or type(program_count) is not int or not 1 <= program_count <= 1024):
        raise ValueError("bounded distinct seeds and program count required")
    programs = _make_programs(program_seed, program_count)
    records = []
    for seed in seeds:
        examples = incomplete_training_atoms(seed, 12)
        data_digest = hashlib.sha256(json.dumps(examples,
            separators=(",", ":")).encode("utf8")).hexdigest()
        arms = {}
        for arm in _ARMS:
            coefficients, ranks = _fit_coefficients(torch, examples, arm)
            metrics = _evaluate_atoms(torch, coefficients, arm, examples)
            metrics["rank_by_op"] = ranks
            metrics["programs"] = _evaluate_programs(torch, coefficients,
                                                      arm, programs)
            arms[arm] = metrics
        records.append({"seed": seed, "training_atoms_sha256": data_digest,
                        "training_examples": len(examples), "arms": arms})
    return {
        "schema": SCHEMA,
        "method": "float64_least_squares_representation_test",
        "basis_by_arm": {
            "bilinear": ["1", "x/8", "y/8", "xy/64"],
            "no_xy": ["1", "x/8", "y/8", "x_squared/64"],
        },
        "operation_names": list(_OP_NAMES),
        "modulus": MOD, "parameter_count_per_arm": OPS * 4,
        "training_examples_per_op": 12,
        "program_seed": program_seed, "program_count": program_count,
        "torch_version": str(torch.__version__),
        "source_manifest": _source_manifest(),
        "runs": records,
        "qualification": "NOT_ESTABLISHED", "graduation_authority": "none",
        "minotaur_graduated": False, "deployed": False,
        "limitations": [
            "Least-squares coefficient fitting is not neural-gradient training",
            "The operation selector and modulo-97 projection are supplied priors",
            "The bilinear arm is given multiplication as an input feature",
            "Public synthetic cases are not a sealed independent examination",
            "No model activation or trusted graduation authority is exercised",
        ],
    }
