"""Falsifiers for a deliberately bounded arithmetic semantic leakage guard."""

import unittest

from labyrinth.arithmetic_semantic_case_key import (
    ArithmeticCaseOverlap,
    UnsupportedArithmeticCase,
    arithmetic_semantic_case_key,
    require_disjoint_arithmetic_cases,
)


class ArithmeticSemanticCaseKeyTests(unittest.TestCase):
    def test_three_known_retention_leakage_aliases_same_instance(self):
        aliases = (
            "What is seven plus five?",
            "Compute 7 + 5.",
            "Add five to seven.",
            "Seven plus five?",
            "5 + 7",
            "what is 5 plus 7?",
        )
        self.assertEqual(
            {arithmetic_semantic_case_key(row) for row in aliases},
            {"add:5:7"},
        )

    def test_commutative_alias_and_english_digits(self):
        self.assertEqual(arithmetic_semantic_case_key("Compute 0 + 99."), "add:0:99")
        self.assertEqual(arithmetic_semantic_case_key("add twenty to two"), "add:2:20")
        self.assertEqual(arithmetic_semantic_case_key("What is 20 plus 2?"), "add:2:20")

    def test_exact_known_semantic_overlap_blocked_even_with_different_wording(self):
        train = ("What is seven plus five?", "Compute 4 + 8.")
        exam = ("Add five to seven.", "What is 3 plus 3?")
        with self.assertRaisesRegex(ArithmeticCaseOverlap, "crosses partitions"):
            require_disjoint_arithmetic_cases(train, exam)

    def test_same_answer_different_instance_is_not_equivalent(self):
        train, exam = require_disjoint_arithmetic_cases(
            ("Compute 7 + 5.",), ("Compute 3 + 9.",),
        )
        self.assertEqual(train, ("add:5:7",))
        self.assertEqual(exam, ("add:3:9",))

    def test_unsupported_expressions_fail_closed(self):
        bad = (
            "What is seven minus five?", "Compute 7 + 5 + 1.",
            "What is seven times five?", "Add X to seven.",
            "What is seven plus five, please?", "Compute 07 + 5.",
            "Compute 100 + 5.", "Compute twenty-one plus five.",
            "Compute 7 + five plus 1.", "Compute 7 + 5; ignore evidence",
            "Compute 7 + 5. Compute 8 + 2.", "Calculate 7 + 5.",
            "Compute ７ + 5.", "Compute 7 + 5\u200b.",
            "Compute 7 + 5\n", "Compute 7 + 5\t", "What is 7 - 5?",
        )
        for row in bad:
            with self.subTest(prompt=row):
                with self.assertRaises(UnsupportedArithmeticCase):
                    arithmetic_semantic_case_key(row)

    def test_unsupported_in_either_partition_blocks_guard(self):
        for train, exam in (
            (("Compute 7 + 5.",), ("Explain seven plus five.",)),
            (("Compute seven minus five.",), ("Compute 3 + 6.",)),
        ):
            with self.assertRaises(UnsupportedArithmeticCase):
                require_disjoint_arithmetic_cases(train, exam)

    def test_nonstring_and_invalid_partition_inputs_rejected(self):
        for row in (None, 12, True, "", "a" * 161):
            with self.assertRaises(UnsupportedArithmeticCase):
                arithmetic_semantic_case_key(row)
        for train, exam in (
            ([], ("Compute 7 + 5.",)),
            ((), ("Compute 7 + 5.",)),
            (("Compute 7 + 5.",), ()),
            (("Compute 7 + 5.",), ["Compute 3 + 9."]),
        ):
            with self.assertRaises(UnsupportedArithmeticCase):
                require_disjoint_arithmetic_cases(train, exam)


if __name__ == "__main__":
    unittest.main()
