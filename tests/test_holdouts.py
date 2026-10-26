"""Independent prompt interpreters for the reserved holdout generation."""

import re
import unittest

from labyrinth.holdouts import generate_holdout_pairs
from labyrinth.program_maze import generate_program_challenges


def _independent_execution(initial, steps):
    state = dict(initial)
    for verb, destination, rhs in steps:
        value = state[rhs] if rhs in ("A", "B") else int(rhs)
        if verb in ("ADD", "plus"):
            state[destination] = (state[destination] + value) % 97
        elif verb in ("SUB", "minus"):
            state[destination] = (state[destination] - value) % 97
        elif verb in ("MUL", "times"):
            state[destination] = (state[destination] * value) % 97
        else:
            raise AssertionError("unrecognized operator")
    return state["A"]


def interpret_canonical(prompt):
    match = re.search(r"Initial registers: A=(\d+), B=(\d+)", prompt)
    if not match:
        raise AssertionError("missing canonical start values")
    steps = re.findall(r"^\d+\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$", prompt, re.M)
    if not steps:
        raise AssertionError("missing canonical instructions")
    return _independent_execution({"A": int(match[1]), "B": int(match[2])}, steps)


def interpret_shifted(prompt):
    match = re.search(r"Starting state \(reordered\): B=(\d+) \| A=(\d+)", prompt)
    if not match:
        raise AssertionError("missing shifted start values")
    steps = []
    for destination, verb, kind, rhs in re.findall(
        r"^step \d+ \| destination=([AB]) \| action=(plus|minus|times) "
        r"\| input=(register|literal):([AB]|\d+)$", prompt, re.M
    ):
        steps.append((verb, destination, kind, rhs))
    if not steps:
        raise AssertionError("missing shifted instructions")
    for verb, destination, kind, rhs in steps:
        if (kind == "register") != (rhs in ("A", "B")):
            raise AssertionError("invalid tagged operand")
    return _independent_execution(
        {"A": int(match[2]), "B": int(match[1])},
        [(verb, destination, rhs) for verb, destination, _, rhs in steps],
    )


class HoldoutTests(unittest.TestCase):
    def test_deterministic_semantically_paired_format_shift(self):
        pairs = generate_holdout_pairs(8105, 160)
        self.assertEqual(pairs, generate_holdout_pairs(8105, 160))
        seen = set()
        for canonical, shifted in pairs:
            self.assertNotEqual(canonical.prompt, shifted.prompt)
            self.assertNotEqual(canonical.id, shifted.id)
            self.assertEqual(canonical.answer, shifted.answer)
            self.assertEqual(canonical.split, "final")
            self.assertEqual(shifted.split, "final")
            self.assertEqual(canonical.tier, 4)
            self.assertEqual(int(canonical.answer), interpret_canonical(canonical.prompt))
            self.assertEqual(int(shifted.answer), interpret_shifted(shifted.prompt))
            self.assertEqual(dict(canonical.metadata)["pair_id"], dict(shifted.metadata)["pair_id"])
            self.assertEqual(dict(canonical.metadata)["representation"], "canonical")
            self.assertEqual(dict(shifted.metadata)["representation"], "shifted")
            self.assertNotIn(canonical.id, seen)
            seen.add(canonical.id)
            self.assertNotIn(shifted.id, seen)
            seen.add(shifted.id)

    def test_reserved_structure_disjoint_from_train_and_development(self):
        reserved = {dict(pair[0].metadata)["structure_fingerprint"]
                    for pair in generate_holdout_pairs(91, 1000)}
        for split in ("train", "dev", "eval"):
            known = {dict(case.metadata)["structure_fingerprint"]
                     for case in generate_program_challenges(91, 1000, split)}
            self.assertTrue(reserved.isdisjoint(known), split)
        self.assertEqual({len(dict(case.metadata)["structure"].split(";"))
                          for pair in generate_holdout_pairs(5, 1000) for case in pair}, {7, 8})

    def test_bad_generation_arguments(self):
        for seed, count in ((True, 1), (-1, 1), (1.5, 1), (1, True), (1, 0), (1, 1001)):
            with self.subTest(seed=seed, count=count), self.assertRaises(ValueError):
                generate_holdout_pairs(seed, count)


if __name__ == "__main__":
    unittest.main()
