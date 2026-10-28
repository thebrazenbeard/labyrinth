"""Independent printed-program oracle and development transfer-slice checks."""

import re
import unittest

from labyrinth.program_maze import generate_program_challenges
from labyrinth.transfer_slices import generate_transfer_slices, transfer_commitment


def interpret_printed_program(prompt):
    initial = re.search(r"Initial registers: A=(\d+), B=(\d+)", prompt)
    self_registers = {"A": int(initial[1]), "B": int(initial[2])}
    for operation, target, operand in re.findall(
        r"^\d+\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$", prompt, flags=re.MULTILINE
    ):
        value = self_registers[operand] if operand in self_registers else int(operand)
        if operation == "ADD":
            self_registers[target] = (self_registers[target] + value) % 97
        elif operation == "SUB":
            self_registers[target] = (self_registers[target] - value) % 97
        else:
            self_registers[target] = self_registers[target] * value % 97
    return self_registers["A"]


class TransferSlicesTests(unittest.TestCase):
    def test_distinct_axes_determinism_independent_oracle_and_pure_commitment(self):
        groups = generate_transfer_slices(73, count=128)
        self.assertEqual(groups, generate_transfer_slices(73, count=128))
        legacy_train = generate_program_challenges(73, 128, "train")
        legacy_eval = generate_program_challenges(73, 128, "eval")
        train_patterns = {dict(case.metadata)["structure"] for case in legacy_train}

        for name, cases in groups.items():
            self.assertEqual(len(cases), 128)
            for case in cases:
                self.assertEqual(int(case.answer), interpret_printed_program(case.prompt))
                self.assertNotEqual(case.split, "train")

        near = {dict(c.metadata)["structure"] for c in groups["fresh_numbers"]}
        structural = {dict(c.metadata)["structure"] for c in groups["new_composition"]}
        longer = {dict(c.metadata)["structure"] for c in groups["longer_program"]}
        self.assertEqual(near, train_patterns)
        self.assertTrue(train_patterns.isdisjoint(structural | longer))
        self.assertTrue(structural.isdisjoint(longer))
        self.assertEqual({s.count(";") + 1 for s in near}, {3, 4})
        self.assertEqual({s.count(";") + 1 for s in structural}, {3, 4})
        self.assertEqual({s.count(";") + 1 for s in longer}, {5, 6})
        self.assertNotEqual(groups["longer_program"], legacy_eval)

        commitment = transfer_commitment(groups)
        self.assertEqual(commitment["status"], "development_only")
        self.assertEqual(set(commitment["slices"]), set(groups))
        self.assertNotIn("ANSWER:", str(commitment))
        self.assertNotIn("prompt", str(commitment))
        self.assertNotIn("answer", str(commitment))

    def test_explicit_profile_domain_separates_same_split_and_legacy_unchanged(self):
        old = generate_program_challenges(3, 2, "eval")
        self.assertEqual(old, generate_program_challenges(3, 2, "eval"))
        matched = generate_program_challenges(3, 2, "eval", profile="matched_length_ood")
        longer = generate_program_challenges(3, 2, "eval", profile="structural_holdout")
        self.assertFalse({c.id for c in matched} & {c.id for c in longer})
        self.assertFalse({c.id for c in old} & {c.id for c in longer})

    def test_invalid_input_is_rejected(self):
        for seed, count in ((-1, 1), (True, 1), (0, 0), (0, 129), (0, True)):
            with self.subTest(seed=seed, count=count), self.assertRaises(ValueError):
                generate_transfer_slices(seed, count)
        for split, profile in (("train", "matched_length_ood"), ("eval", "none")):
            with self.subTest(split=split, profile=profile), self.assertRaises(ValueError):
                generate_program_challenges(0, 1, split, profile=profile)
        with self.assertRaises(ValueError):
            transfer_commitment({})


if __name__ == "__main__":
    unittest.main()
