import hashlib
import re
import unittest

from labyrinth.program_maze import generate_program_challenges


def oracle(prompt):
    """Interpret the printed instructions, independent of generator internals."""
    initial = re.search(r"Initial registers: A=(\d+), B=(\d+)", prompt)
    registers = {"A": int(initial[1]), "B": int(initial[2])}
    for line in prompt.splitlines():
        match = re.fullmatch(r"\d+\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)", line)
        if not match:
            continue
        op, target, operand = match.groups()
        value = registers[operand] if operand in registers else int(operand)
        if op == "ADD":
            result = registers[target] + value
        elif op == "SUB":
            result = registers[target] - value
        else:
            result = registers[target] * value
        registers[target] = result % 97
    return registers["A"]


class ProgramMazeTests(unittest.TestCase):
    def test_determinism_and_independent_oracle(self):
        for split in ("train", "eval", "holdout"):
            challenges = generate_program_challenges(27, 100, split)
            self.assertEqual(challenges, generate_program_challenges(27, 100, split))
            for challenge in challenges:
                self.assertEqual(int(challenge.answer), oracle(challenge.prompt))
                self.assertRegex(challenge.answer, r"^(0|[1-9][0-9]*)$")
                self.assertGreaterEqual(int(challenge.answer), 0)
                self.assertLess(int(challenge.answer), 97)
                self.assertEqual(challenge.tier, 4)

    def test_structure_holdout_not_just_new_literals(self):
        train = generate_program_challenges(4, 1000, "train")
        evaluated = generate_program_challenges(4, 1000, "eval")
        train_structures = {dict(c.metadata)["structure"] for c in train}
        eval_structures = {dict(c.metadata)["structure"] for c in evaluated}
        self.assertTrue(train_structures.isdisjoint(eval_structures))
        self.assertEqual({len(s.split(";")) for s in train_structures}, {3, 4})
        self.assertEqual({len(s.split(";")) for s in eval_structures}, {5, 6})
        for challenge in train + evaluated:
            meta = dict(challenge.metadata)
            self.assertEqual(meta["structure_fingerprint"], hashlib.sha256(meta["structure"].encode()).hexdigest())
            self.assertEqual(meta["profile"], "train" if challenge.split == "train" else "structural_holdout")
            instructions = re.findall(r"^\d+\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$", challenge.prompt, re.MULTILINE)
            actual_structure = ";".join(
                f"{op} {target} {operand if operand in ('A', 'B') else 'C'}"
                for op, target, operand in instructions
            )
            self.assertEqual(meta["structure"], actual_structure)

    def test_validation(self):
        for args in ((True, 1, "train"), (0, 0, "train"), (0, True, "train"),
                     (0, 1001, "train"), (0, 1, "../eval"), (0, 1, None)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                generate_program_challenges(*args)


if __name__ == "__main__":
    unittest.main()
