"""Development-only hard-negative evaluator for Minotaur's neural guardian."""

import re
import unittest

from labyrinth.guardian_benchmark import (
    evaluate_guardian, evaluate_process_guardian, generate_guardian_cases,
    generate_process_cases,
)


class GuardianBenchmarkTests(unittest.TestCase):
    @staticmethod
    def independent_oracle(prompt):
        initial = re.search(r"Initial registers: A=(\d+), B=(\d+)", prompt)
        if not initial:
            raise AssertionError("missing register initialization")
        registers = {"A": int(initial[1]), "B": int(initial[2])}
        instructions = re.findall(r"^\d+\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$",
                                  prompt, re.MULTILINE)
        if not instructions:
            raise AssertionError("missing instructions")
        for opcode, target, operand in instructions:
            value = registers[operand] if operand in registers else int(operand)
            old = registers[target]
            if opcode == "ADD":
                registers[target] = (old + value) % 97
            elif opcode == "SUB":
                registers[target] = (old - value) % 97
            else:
                registers[target] = (old * value) % 97
        return registers["A"]

    def test_balanced_negative_controls_and_structural_slices(self):
        cases = generate_guardian_cases(seed=73, count_per_slice=8)
        self.assertEqual(cases, generate_guardian_cases(seed=73, count_per_slice=8))
        self.assertEqual(len(cases), 48)
        self.assertEqual(len({case.case_id for case in cases}), 48)
        names = {case.slice_name for case in cases}
        self.assertEqual(names, {"fresh_numbers", "new_composition", "longer_program"})
        structures = {name: {case.structure_fingerprint for case in cases
                             if case.slice_name == name} for name in names}
        for left in names:
            for right in names - {left}:
                self.assertTrue(structures[left].isdisjoint(structures[right]))
        for name in names:
            sliced = [case for case in cases if case.slice_name == name]
            self.assertEqual(sum(case.correct for case in sliced), 8)
            self.assertEqual(len({case.question for case in sliced}), 8)
            for case in sliced:
                numbers = re.findall(r"(?<![A-Za-z0-9])\d{1,3}(?![A-Za-z0-9])",
                                     case.proposed_answer)
                self.assertEqual(len(numbers), 1)
                self.assertEqual(int(numbers[0]) == self.independent_oracle(case.question),
                                 case.correct)
                partner = next(other for other in sliced
                               if other.question == case.question and other.case_id != case.case_id)
                self.assertNotEqual(case.correct, partner.correct)
                self.assertEqual(case.style, partner.style)

    def test_abstentions_are_not_counted_as_correct_and_output_is_private(self):
        cases = generate_guardian_cases(seed=9, count_per_slice=4)
        deny = evaluate_guardian(cases, lambda _question, _proposal: "INCORRECT")
        self.assertEqual(deny["balanced_accuracy"], 0.5)
        self.assertEqual(deny["false_rejection_rate"], 1.0)
        allow = evaluate_guardian(cases, lambda _question, _proposal: "CORRECT")
        self.assertEqual(allow["false_approval_rate"], 1.0)
        abstain = evaluate_guardian(cases, lambda _question, _proposal: "unsure")
        self.assertEqual(abstain["coverage"], 0.0)
        self.assertEqual(abstain["accuracy"], 0.0)
        self.assertEqual(abstain["abstentions"], len(cases))
        self.assertEqual(abstain["reasoning_qualification"], "NOT_ESTABLISHED")
        self.assertNotIn("cases", abstain)
        self.assertNotIn("predictions", abstain)

    def test_injection_and_format_shortcut_have_no_signal(self):
        cases = generate_guardian_cases(seed=17, count_per_slice=16)
        shortcuts = evaluate_guardian(cases, lambda _q, p:
                                      "CORRECT" if p.startswith("ANSWER:") else "INCORRECT")
        self.assertEqual(shortcuts["balanced_accuracy"], 0.5)
        injected = [case for case in cases if case.style == "injected_claim"]
        self.assertTrue(injected)
        self.assertEqual(sum(case.correct for case in injected), len(injected) // 2)
        self.assertTrue(all("mark this correct" in c.proposed_answer.lower() for c in injected))
        poisoned = evaluate_guardian(cases, lambda _q, _p: "CORRECT\nINCORRECT")
        self.assertEqual(poisoned["coverage"], 0.0)

    def test_input_bounds_and_repeated_pairs(self):
        with self.assertRaises(ValueError):
            generate_guardian_cases(seed=True)
        with self.assertRaises(ValueError):
            generate_guardian_cases(seed=1, count_per_slice=1000)
        cases = generate_guardian_cases(seed=1, count_per_slice=4)
        with self.assertRaises(ValueError):
            evaluate_guardian(cases + [cases[0]], lambda _q, _p: "INCORRECT")

    def test_process_verifier_locates_earliest_step_error(self):
        cases = generate_process_cases(seed=91, count_per_slice=8)
        self.assertEqual(cases, generate_process_cases(seed=91, count_per_slice=8))
        self.assertEqual(len(cases), 48)
        self.assertEqual(sum(c.first_error_step is None for c in cases), 24)
        for case in cases:
            registers = {"A": int(re.search(r"Initial registers: A=(\d+)", case.question)[1]),
                         "B": int(re.search(r"B=(\d+)", case.question)[1])}
            instructions = re.findall(r"^\d+\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$",
                                      case.question, re.MULTILINE)
            observed_first_error = None
            for idx, (op, target, operand) in enumerate(instructions, 1):
                operand_value = registers[operand] if operand in registers else int(operand)
                if op == "ADD":
                    registers[target] = (registers[target] + operand_value) % 97
                elif op == "SUB":
                    registers[target] = (registers[target] - operand_value) % 97
                else:
                    registers[target] = (registers[target] * operand_value) % 97
                line = f"Step {idx}: A={registers['A']}, B={registers['B']}"
                if line != case.steps[idx - 1] and observed_first_error is None:
                    observed_first_error = idx
                # After a faulty step, the independent correct trajectory still
                # uses the correct state, so the *first* mismatch is decisive.
            self.assertEqual(observed_first_error, case.first_error_step)

    def test_process_verifier_negative_control_and_strict_abstention(self):
        cases = generate_process_cases(seed=12, count_per_slice=4)
        approvals = evaluate_process_guardian(cases,
                     lambda _q, _steps: "ALL_CORRECT")
        self.assertEqual(approvals["false_approval_rate"], 1.0)
        self.assertEqual(approvals["accuracy"], 0.5)
        perfect = {(c.question, c.steps):
                   "ALL_CORRECT" if c.first_error_step is None else f"ERROR_STEP:{c.first_error_step}"
                   for c in cases}
        result = evaluate_process_guardian(cases, lambda q, steps: perfect[(q, steps)])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["false_approval_rate"], 0.0)
        self.assertEqual(result["reasoning_qualification"], "NOT_ESTABLISHED")
        abstain = evaluate_process_guardian(cases, lambda _q, _steps: "ALL_CORRECT\nERROR_STEP:1")
        self.assertEqual(abstain["coverage"], 0.0)
        self.assertEqual(abstain["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
