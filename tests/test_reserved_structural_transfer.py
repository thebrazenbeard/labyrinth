"""Frozen public structural-transfer development fixture, not a final exam.

The fixture is precommitted here to make accidental generator drift visible.
It is deliberately PUBLIC: training or tuning against these prompts destroys
their value as independent evaluation evidence. No toy learning claim follows.
"""

from __future__ import annotations

import hashlib
import json
import re
import unittest

from labyrinth.adapters import AdapterError
from labyrinth.evaluation import score_response
from labyrinth.maze import Challenge, generate_maze
from labyrinth.toy_backend import ToyLinearAdapter


SEED = 20261030
SPLIT = "devstruct_v1"
TIERS = (2, 3, 4)
COUNT_PER_TIER = 8
EXPECTED_FIXTURE_SHA256 = "1ce1da3f8a6c1e2367f634986c9ecd62e442de4f91296f11826b9f9ee5c5af13"


def _fixture() -> tuple[Challenge, ...]:
    return tuple(generate_maze(SEED, TIERS, COUNT_PER_TIER, SPLIT))


def _fixture_sha256(challenges: tuple[Challenge, ...]) -> str:
    # Fixed field order is part of this public fixture's v1 commitment.
    fields = [(c.id, c.kind, c.tier, c.split, c.prompt, c.answer, c.metadata)
              for c in challenges]
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _independent_expected_answer(challenge: Challenge) -> int:
    """Read printed instructions; never use the generator's answer as oracle."""
    prompt = challenge.prompt
    if challenge.tier == 2:
        match = re.search(
            r"Start at (\d+)\. In order: add (\d+), multiply the result by "
            r"(\d+), subtract (\d+), then take the nonnegative remainder modulo (\d+)\.",
            prompt,
        )
        if match is None:
            raise AssertionError("unrecognized operation-chain instructions")
        start, add, multiply, subtract, modulus = map(int, match.groups())
        return ((start + add) * multiply - subtract) % modulus
    if challenge.tier == 3:
        priority = re.search(r"priority is ([a-z]+(?: > [a-z]+){2})", prompt)
        if priority is None:
            raise AssertionError("missing source priority")
        values = {}
        for line in prompt.splitlines():
            match = re.match(r"- source=([a-z]+); measured_value=(\d+)", line)
            if match is not None:
                values[match[1]] = int(match[2])
        if len(values) != 3:
            raise AssertionError("source records missing or duplicated")
        return values[priority[1].split(" > ")[0]]
    if challenge.tier == 4:
        initial = re.search(r"Initial registers: A=(\d+), B=(\d+)", prompt)
        if initial is None:
            raise AssertionError("missing initial registers")
        registers = {"A": int(initial[1]), "B": int(initial[2])}
        count = 0
        for line in prompt.splitlines():
            instruction = re.fullmatch(r"\d+\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)", line)
            if instruction is None:
                continue
            count += 1
            operation, target, operand = instruction.groups()
            value = registers[operand] if operand in registers else int(operand)
            if operation == "ADD":
                registers[target] = (registers[target] + value) % 97
            elif operation == "SUB":
                registers[target] = (registers[target] - value) % 97
            else:
                registers[target] = (registers[target] * value) % 97
        if count not in (5, 6):
            raise AssertionError("unexpected program holdout length")
        return registers["A"]
    raise AssertionError("unsupported fixture tier")


class ReservedStructuralTransferTests(unittest.TestCase):
    def test_fixture_is_frozen_and_keys_match_independent_interpretation(self):
        cases = _fixture()
        self.assertEqual(len(cases), 24)
        self.assertEqual(_fixture_sha256(cases), EXPECTED_FIXTURE_SHA256)
        self.assertEqual({c.tier for c in cases}, set(TIERS))
        self.assertEqual({c.split for c in cases}, {SPLIT})
        for case in cases:
            with self.subTest(case=case.id):
                expected = _independent_expected_answer(case)
                self.assertEqual(str(expected), case.answer)
                self.assertEqual(score_response(case, f"ANSWER: {expected}").points, 1)

    def test_train_and_public_transfer_cases_do_not_overlap(self):
        heldout = _fixture()
        training = tuple(generate_maze(SEED, TIERS, COUNT_PER_TIER, "train"))
        self.assertFalse({c.id for c in heldout} & {c.id for c in training})
        self.assertFalse({c.prompt for c in heldout} & {c.prompt for c in training})
        heldout_structures = {dict(c.metadata)["structure"] for c in heldout if c.tier == 4}
        train_structures = {dict(c.metadata)["structure"] for c in training if c.tier == 4}
        self.assertTrue(heldout_structures)
        self.assertTrue(train_structures)
        self.assertFalse(heldout_structures & train_structures)

    def test_toy_arithmetic_adapter_is_not_a_structural_transfer_evaluator(self):
        learner = ToyLinearAdapter(seed=0)
        cases = _fixture()
        for case in cases:
            with self.subTest(case=case.id), self.assertRaisesRegex(
                AdapterError, "only canonical tier-1 arithmetic"
            ):
                learner.generate(case.prompt)
        # Unsupported tasks must be recorded as NOT SUPPORTED / NOT RUN,
        # never treated as evidence that a neural model generalized.


if __name__ == "__main__":
    unittest.main()
