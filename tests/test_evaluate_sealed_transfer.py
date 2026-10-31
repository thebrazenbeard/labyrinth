"""The external transfer script is a fixture/score tool, not a neural test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from scripts.evaluate_sealed_transfer import (
    ARM_NAMES, COUNT, as_custodied_transfer_cases, make_cases, prepare, score,
)
from labyrinth.independent_holdout import case_commitment, private_gold_seal
from labyrinth.program_maze import generate_program_challenges


SEED = "c3" * 16  # Public unit-test fixture; do not use as a final-exam secret.


def _independent_interpret(prompt: str, *, intervention_step: int | None = None,
                           intervention_delta: int = 1) -> int:
    """Derive answers only from the printed instructions, not evaluator keys."""
    lines = prompt.splitlines()
    initial = re.fullmatch(r"Initial registers: A=(\d+), B=(\d+)", lines[1])
    if initial is None:
        raise AssertionError("invalid program start")
    registers = {"A": int(initial[1]), "B": int(initial[2])}
    for number, line in enumerate(lines[2:-1], 1):
        instruction = re.fullmatch(rf"{number}\. (ADD|SUB|MUL) ([AB]) ([AB]|[2-9])", line)
        if instruction is None:
            raise AssertionError("invalid instruction numbering or grammar")
        op, target, operand = instruction.groups()
        operand_value = registers[operand] if operand in registers else int(operand)
        previous = registers[target]
        if op == "ADD":
            result = previous + operand_value
        elif op == "SUB":
            result = previous - operand_value
        else:
            result = previous * operand_value
        registers[target] = result % 97
        if number == intervention_step:
            registers[target] = (registers[target] + intervention_delta) % 97
    return registers["A"]


def _independent_dependency_slice(prompt: str) -> set[str]:
    """Trace register versions backwards from final A to every source node."""
    lines = prompt.splitlines()
    latest = {"A": "A0", "B": "B0"}
    graph: dict[str, tuple[str, ...]] = {}
    for number, line in enumerate(lines[2:-1], 1):
        instruction = re.fullmatch(rf"{number}\. (ADD|SUB|MUL) ([AB]) ([AB]|[2-9])", line)
        if instruction is None:
            raise AssertionError("invalid instruction")
        _, target, operand = instruction.groups()
        graph[str(number)] = ((latest[target], latest[operand]) if operand in latest
                              else (latest[target],))
        latest[target] = str(number)
    traced: set[str] = set()
    stack = [latest["A"]]
    while stack:
        node = stack.pop()
        if node in traced:
            continue
        traced.add(node)
        stack.extend(graph.get(node, ()))
    return traced


def _modify_initial(prompt: str, register: str, delta: int) -> str:
    lines = prompt.splitlines()
    initial = re.fullmatch(r"Initial registers: A=(\d+), B=(\d+)", lines[1])
    if initial is None:
        raise AssertionError("unrecognized initialization")
    initial_a, initial_b = int(initial[1]), int(initial[2])
    if register == "A":
        initial_a = (initial_a + delta) % 97
    elif register == "B":
        initial_b = (initial_b + delta) % 97
    else:
        raise ValueError("unsupported initial register")
    lines[1] = f"Initial registers: A={initial_a}, B={initial_b}"
    return "\n".join(lines)


class ExternalTransferTests(unittest.TestCase):
    def test_every_step_is_live_and_both_registers_causally_affect_final_a(self):
        # Counterexample before this regression: seed 000...000 case 001
        # had zero B->A reads despite two A->B reads; many other cases had
        # dead instruction padding. Numeric perturbation additionally rejects
        # nominal paths nullified by concrete modulo-97 arithmetic.
        for seed_number in range(100):
            cases, _ = make_cases(f"{seed_number:032x}")
            for case in cases:
                with self.subTest(seed=seed_number, case=case["id"]):
                    traced = _independent_dependency_slice(case["prompt"])
                    required = {str(i) for i in range(1, case["steps"] + 1)}
                    self.assertTrue(required.issubset(traced), "dead printed instruction")
                    self.assertIn("B0", traced, "initial B cannot reach final A")
                    answer = _independent_interpret(case["prompt"])
                    for register in ("A", "B"):
                        self.assertTrue(any(
                            _independent_interpret(_modify_initial(case["prompt"], register, delta)) != answer
                            for delta in (1, 2, 3)
                        ), f"initial {register} has no observed influence on final A")
                    for position in range(1, case["steps"] + 1):
                        self.assertTrue(any(
                            _independent_interpret(case["prompt"], intervention_step=position,
                                                   intervention_delta=delta) != answer
                            for delta in (1, 2, 3)
                        ), f"instruction {position} has no observed causal impact on final A")

    def test_distinct_7_8_step_structures_and_independent_oracle(self):
        public, gold = make_cases(SEED)
        self.assertEqual(len(public), COUNT)
        self.assertEqual(len(gold), COUNT)
        self.assertEqual([p["steps"] for p in public].count(7), COUNT // 2)
        self.assertEqual([p["steps"] for p in public].count(8), COUNT // 2)
        self.assertEqual(len({p["structure_sha256"] for p in public}), COUNT)
        # Production TRAIN is 3/4 instructions, PUBLIC diagnostic 5/6;
        # independent 7/8 structures cannot be those exact sequences.
        existing = (generate_program_challenges(20261030, 1000, "train")
                    + generate_program_challenges(20261030, 1000, "eval"))
        self.assertFalse({p["structure_sha256"] for p in public} &
                         {hashlib.sha256(dict(e.metadata)["structure"].encode()).hexdigest()
                          for e in existing})
        for public_case, secret in zip(public, gold):
            self.assertEqual(public_case["id"], secret["id"])
            self.assertEqual(public_case["prompt_sha256"], secret["prompt_sha256"])
            self.assertNotIn("answer", public_case)
            self.assertEqual(str(_independent_interpret(public_case["prompt"])), secret["answer"])

    def test_invalid_seed_rejected(self):
        for seed in ("", "f" * 31, "Z" * 32, "a" * 33):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                make_cases(seed)

    def test_custodied_transfer_cases_bind_to_new_independent_holdout_protocol(self):
        public, private = make_cases(SEED)
        cases = as_custodied_transfer_cases(public, private)
        self.assertEqual(len(cases), COUNT)
        self.assertEqual({case.split for case in cases}, {"transfer"})
        self.assertEqual({case.skill for case in cases}, {"program_reasoning"})
        self.assertEqual(len({case.structure_key for case in cases}), COUNT)
        self.assertEqual(len(case_commitment(cases)), 64)
        self.assertEqual(len(private_gold_seal(cases, b"test-only-evaluator-key-material-32")), 64)
        modified = [dict(key) for key in private]
        modified[0]["answer"] = "96" if private[0]["answer"] != "96" else "95"
        with self.assertRaisesRegex(ValueError, "gold/structure differs"):
            as_custodied_transfer_cases(public, modified)

    def _fixture(self, root: Path) -> tuple[Path, Path, list[dict], list[dict]]:
        private, public_dir = root / "private", root / "public"
        prepare(private, public_dir, SEED)
        public = [json.loads(line) for line in (public_dir / "prompts.jsonl").read_text().splitlines()]
        gold = [json.loads(line) for line in (private / "keys.jsonl").read_text().splitlines()]
        return private, public_dir, public, gold

    @staticmethod
    def _write_preds(root: Path, gold: list[dict], *, all_correct: bool) -> Path:
        path = root / ("correct.jsonl" if all_correct else "zero.jsonl")
        rows = [{"id": key["id"], "response": f"ANSWER: {key['answer']}" if all_correct
                 else "ANSWER: -1"} for key in gold]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_score_exact_four_arms_and_paired_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private, public, _, gold = self._fixture(root)
            right = self._write_preds(root, gold, all_correct=True)
            wrong = self._write_preds(root, gold, all_correct=False)
            paths = {arm: right if arm == "candidate" else wrong for arm in ARM_NAMES}
            result = score(private, public, paths, root / "result.json")
            self.assertEqual(result["arms"]["candidate"]["correct"], COUNT)
            self.assertEqual(result["arms"]["baseline"]["correct"], 0)
            self.assertEqual(result["paired_comparisons"]["baseline"]["candidate_only"], COUNT)
            self.assertEqual(result["paired_comparisons"]["baseline"]["control_only"], 0)
            self.assertEqual(result["paired_comparisons"]["baseline"]["paired_accuracy_delta"], 1.0)
            self.assertEqual(result["arms"]["candidate"]["by_length"]["7"]["correct"], COUNT // 2)
            self.assertIn("UNAUTHENTICATED", result["status"])
            with self.assertRaises(FileExistsError):
                score(private, public, paths, root / "result.json")

    def test_tampered_gold_commitment_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private, public, _, gold = self._fixture(root)
            predictions = self._write_preds(root, gold, all_correct=True)
            with (private / "keys.jsonl").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "commitment changed"):
                score(private, public, dict.fromkeys(ARM_NAMES, predictions), root / "report.json")

    def test_recommitted_false_gold_still_fails_independent_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private, public, _, gold = self._fixture(root)
            predictions = self._write_preds(root, gold, all_correct=True)
            gold[0]["answer"] = "99"  # Impossible modulo-97 answer.
            modified = "".join(json.dumps(row) + "\n" for row in gold).encode()
            (private / "keys.jsonl").write_bytes(modified)
            manifest = json.loads((private / "manifest.json").read_text())
            manifest["keys_sha256"] = hashlib.sha256(modified).hexdigest()
            (private / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "gold/structure disagrees"):
                score(private, public, dict.fromkeys(ARM_NAMES, predictions), root / "report.json")

    def test_recommitted_dead_instruction_rejected_during_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private, public_dir, public_cases, gold = self._fixture(root)
            predictions = self._write_preds(root, gold, all_correct=True)
            case = public_cases[0]
            lines = case["prompt"].splitlines()
            lines[-2] = "7. SUB B B"  # Last printed step cannot influence final A.
            case["prompt"] = "\n".join(lines)
            case["prompt_sha256"] = hashlib.sha256(case["prompt"].encode()).hexdigest()
            gold[0]["prompt_sha256"] = case["prompt_sha256"]
            public_blob = "".join(json.dumps(row) + "\n" for row in public_cases).encode()
            private_blob = "".join(json.dumps(row) + "\n" for row in gold).encode()
            (public_dir / "prompts.jsonl").write_bytes(public_blob)
            (private / "keys.jsonl").write_bytes(private_blob)
            manifest = json.loads((private / "manifest.json").read_text())
            manifest["prompts_sha256"] = hashlib.sha256(public_blob).hexdigest()
            manifest["keys_sha256"] = hashlib.sha256(private_blob).hexdigest()
            (private / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "causal transfer structure"):
                score(private, public_dir, dict.fromkeys(ARM_NAMES, predictions), root / "report.json")

    def test_missing_or_duplicate_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private, public, _, gold = self._fixture(root)
            predictions = self._write_preds(root, gold, all_correct=True)
            rows = predictions.read_text().splitlines()
            predictions.write_text("\n".join(rows[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "exactly every"):
                score(private, public, dict.fromkeys(ARM_NAMES, predictions), root / "report.json")
            predictions.write_text("\n".join(rows[:-1] + [rows[0]]) + "\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                score(private, public, dict.fromkeys(ARM_NAMES, predictions), root / "report.json")
            predictions.write_text("".join(
                '{"id":"' + key["id"] + '","response":"ANSWER: 0","response":"ANSWER: 1"}\n'
                for key in gold
            ))
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                score(private, public, dict.fromkeys(ARM_NAMES, predictions), root / "report.json")

    def test_public_prompts_do_not_disclose_seed_or_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private, public_dir, public, gold = self._fixture(root)
            manifest = json.loads((private / "manifest.json").read_text())
            self.assertNotIn(SEED, (public_dir / "prompts.jsonl").read_text())
            self.assertFalse((public_dir / "keys.jsonl").exists())
            self.assertFalse((public_dir / "seed.txt").exists())
            self.assertNotIn(SEED, (private / "manifest.json").read_text())
            self.assertEqual(manifest["seed_commitment_sha256"], hashlib.sha256(SEED.encode()).hexdigest())
            self.assertEqual({p["id"] for p in public}, {g["id"] for g in gold})
            self.assertEqual((private / "seed.txt").read_text().strip(), SEED)


if __name__ == "__main__":
    unittest.main()
