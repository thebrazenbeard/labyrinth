"""Tests for the independent combinatorial register-program curriculum."""
import hashlib
import re
import unittest

from labyrinth.minotaur_combinatorial_curriculum import (
    CurriculumConfig,
    build_combinatorial_splits,
    interpret_question_mod97,
    validate_curriculum_case,
)
from labyrinth.minotaur_scratch_v2 import tokenize_observed_trace_v2


def independent_oracle(question):
    """A deliberately local reimplementation of the printed instruction semantics."""
    start = re.search(r"Initial registers: A=(\d+), B=(\d+)", question)
    values = {"A": int(start.group(1)), "B": int(start.group(2))}
    states = []
    for number, op, target, operand in re.findall(
            r"^(\d+)\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$", question, re.MULTILINE):
        operand_value = values[operand] if operand in values else int(operand)
        if op == "ADD":
            values[target] = (values[target] + operand_value) % 97
        elif op == "SUB":
            values[target] = (values[target] - operand_value) % 97
        else:
            values[target] = (values[target] * operand_value) % 97
        states.append((int(number), values["A"], values["B"]))
    return tuple(states)


class CombinatorialCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.config = CurriculumConfig(seed=771, train_cases=24, dev_cases=16,
                                       transfer_cases=16)
        self.splits = build_combinatorial_splits(self.config)

    def test_reproducible_opaque_committed_balanced_splits(self):
        again = build_combinatorial_splits(self.config)
        self.assertEqual(self.splits, again)
        for name, split in self.splits.items():
            self.assertEqual(len(split.cases), self.config.count_for(name))
            self.assertEqual(split.commitment,
                             hashlib.sha256("\n".join(c.case_id for c in split.cases).encode()).hexdigest())
            self.assertEqual(sum(c.first_error_step is None for c in split.cases), len(split.cases) // 2)
            self.assertEqual(sum(c.first_error_step is not None for c in split.cases), len(split.cases) // 2)
            self.assertGreater(len({c.canonical_sequence for c in split.cases}), 4)

    def test_splits_share_neither_ids_nor_canonical_operation_sequences(self):
        groups = list(self.splits.values())
        for left_index, left in enumerate(groups):
            for right in groups[left_index + 1:]:
                self.assertTrue({c.case_id for c in left.cases}.isdisjoint(c.case_id for c in right.cases))
                self.assertTrue({c.canonical_sequence for c in left.cases}.isdisjoint(
                    c.canonical_sequence for c in right.cases))

    def test_parser_reexecution_and_tokenizer_crosscheck(self):
        seen_families = set()
        for split in self.splits.values():
            for case in split.cases:
                self.assertEqual(interpret_question_mod97(case.question), independent_oracle(case.question))
                validate_curriculum_case(case)
                self.assertEqual(len(tokenize_observed_trace_v2(case.question, case.steps)),
                                 4 + 8 * len(case.steps))
                self.assertNotIn("correct", case.question.lower())
                self.assertNotIn("fault", case.question.lower())
                self.assertNotIn("correct", "\n".join(case.steps).lower())
                seen_families.add(case.fault_family)
        self.assertTrue({"target_delta", "non_target_change", "register_swap",
                         "final_target_mutation", "no_change"}.issubset(seen_families))

    def test_faults_are_real_and_the_final_mutation_is_subtle(self):
        final_mutations = []
        for split in self.splits.values():
            for case in split.cases:
                if case.first_error_step is not None:
                    expected = interpret_question_mod97(case.question)
                    observed = tuple(tuple(map(int, re.fullmatch(
                        r"Step (\d+): A=(\d+), B=(\d+)", line).groups())) for line in case.steps)
                    self.assertNotEqual(observed[case.first_error_step - 1],
                                        expected[case.first_error_step - 1])
                    if case.fault_family == "final_target_mutation":
                        final_mutations.append((case, expected, observed))
        self.assertTrue(final_mutations)
        for case, expected, observed in final_mutations:
            self.assertEqual(case.first_error_step, len(case.steps))
            differing = sum(a != b for a, b in zip(expected[-1][1:], observed[-1][1:]))
            self.assertEqual(differing, 1)

    def test_each_split_has_varied_instruction_components_and_step_counts(self):
        for split in self.splits.values():
            programs = [re.findall(r"^\d+\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$",
                                   case.question, re.MULTILINE) for case in split.cases]
            flattened = [instruction for program in programs for instruction in program]
            self.assertEqual({item[0] for item in flattened}, {"ADD", "SUB", "MUL"})
            self.assertEqual({item[1] for item in flattened}, {"A", "B"})
            self.assertTrue(any(item[2] in {"A", "B"} for item in flattened))
            self.assertTrue(any(item[2].isdigit() for item in flattened))
            self.assertGreater(len({len(program) for program in programs}), 1)

    def test_rejects_unbounded_or_unbalanced_configuration(self):
        for kwargs in ({"train_cases": 9}, {"dev_cases": 11}, {"transfer_cases": 8},
                       {"seed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_combinatorial_splits(CurriculumConfig(**kwargs))


if __name__ == "__main__":
    unittest.main()
