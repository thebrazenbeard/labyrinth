"""Contracts for causal Minotaur V4 transition supervision."""
import re
import unittest

from labyrinth.minotaur_transition_curriculum import (
    CurriculumConfig, build_transition_splits, reconstruct_first_failure,
    replay_mod97, validate_transition_case,
)


def modulo97_oracle(question):
    """Independent replay used to protect the corpus parser from self-agreement."""
    initial = re.search(r"Initial registers: A=(\d+), B=(\d+)", question)
    values = {"A": int(initial.group(1)), "B": int(initial.group(2))}
    output = []
    for _, op, target, operand in re.findall(r"^(\d+)\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$", question, re.MULTILINE):
        value = values[operand] if operand in values else int(operand)
        values[target] = {"ADD": values[target] + value, "SUB": values[target] - value,
                          "MUL": values[target] * value}[op] % 97
        output.append((values["A"], values["B"]))
    return tuple(output)


def observed(case):
    return tuple((int(m.group(2)), int(m.group(3))) for line in case.steps
                 if (m := re.fullmatch(r"Step (\d+): A=(\d+), B=(\d+)", line)))


class TransitionCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.config = CurriculumConfig(seed=771, train_cases=32, dev_cases=24, transfer_cases=24)
        self.splits = build_transition_splits(self.config)

    def test_deterministic_for_two_seeds_and_pair_grouped(self):
        for seed in (771, 772):
            config = CurriculumConfig(seed=seed, train_cases=20, dev_cases=20, transfer_cases=20)
            self.assertEqual(build_transition_splits(config), build_transition_splits(config))
        for split in self.splits.values():
            for clean, faulty in zip(split.cases[::2], split.cases[1::2]):
                self.assertIsNone(clean.first_error_step)
                self.assertIsNotNone(faulty.first_error_step)
                self.assertEqual(clean.canonical_sequence, faulty.canonical_sequence)
                self.assertEqual(clean.question, faulty.question)

    def test_lengths_targets_and_modulo97_replay(self):
        for split in self.splits.values():
            self.assertGreater(len({case.canonical_sequence for case in split.cases}), 4)
            self.assertGreater(len({len(case.steps) for case in split.cases}), 1)
            for case in split.cases:
                self.assertEqual(replay_mod97(case.question), modulo97_oracle(case.question))
                self.assertEqual(case.transition_targets, modulo97_oracle(case.question))
                self.assertEqual(case.transition_mask, (1,) * len(case.steps))
                self.assertTrue(1 <= len(case.steps) <= 6)
                self.assertNotIn("fault", case.question.lower())
                self.assertNotIn("correct", "\n".join(case.steps).lower())
                validate_transition_case(case)

    def test_structure_disjoint_and_faults_are_causal(self):
        groups = list(self.splits.values())
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                self.assertTrue({c.canonical_sequence for c in left.cases}.isdisjoint(c.canonical_sequence for c in right.cases))
        families = set()
        for split in self.splits.values():
            for case in split.cases:
                if case.first_error_step is None:
                    continue
                expected, got = modulo97_oracle(case.question), observed(case)
                first = case.first_error_step - 1
                self.assertEqual(reconstruct_first_failure(case.question, case.steps), case.first_error_step)
                self.assertEqual(got[:first], expected[:first])
                self.assertTrue(all(a != b for a, b in zip(got[first:], expected[first:])))
                families.add(case.fault_family)
        self.assertTrue({"non_target_change", "register_swap", "subtle_final_error", "unchanged_op"} <= families)

    def test_requested_128_unique_structures_per_split(self):
        large = build_transition_splits(CurriculumConfig(seed=97, train_cases=256,
            dev_cases=256, transfer_cases=256))
        for split in large.values():
            self.assertGreaterEqual(len({c.canonical_sequence for c in split.cases}), 128)


if __name__ == "__main__":
    unittest.main()
