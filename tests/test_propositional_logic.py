"""Contract tests for the independent bounded propositional-logic domain."""

import unittest

from labyrinth.propositional_logic import (
    evaluate_first_error,
    generate_first_error_examples,
    generate_logic_challenges,
    score_first_error_response,
    score_logic_response,
)


def reference_entails(premises, conclusion):
    """A test-only postfix evaluator, intentionally separate from production parser."""
    def postfix(formula):
        output, operators = [], []
        precedence = {"!": 4, "&": 3, "|": 2, ">": 1}
        for token in formula:
            if token in "ABCD":
                output.append(token)
            elif token == "(":
                operators.append(token)
            elif token == ")":
                while operators[-1] != "(":
                    output.append(operators.pop())
                operators.pop()
            else:
                while (operators and operators[-1] != "(" and
                       (precedence[operators[-1]] > precedence[token] or
                        (precedence[operators[-1]] == precedence[token] and token != ">"))):
                    output.append(operators.pop())
                operators.append(token)
        while operators:
            output.append(operators.pop())
        return output

    def value(formula, assignment):
        stack = []
        for token in postfix(formula):
            if token in "ABCD": stack.append(assignment[token])
            elif token == "!": stack.append(not stack.pop())
            else:
                right, left = stack.pop(), stack.pop()
                stack.append(left and right if token == "&" else left or right if token == "|" else not left or right)
        return stack[0]

    for bits in range(16):
        assignment = {atom: bool(bits & (1 << index)) for index, atom in enumerate("ABCD")}
        if all(value(premise, assignment) for premise in premises) and not value(conclusion, assignment):
            return False
    return True


class PropositionalLogicTests(unittest.TestCase):
    def test_deterministic_private_labels_and_unseen_seed_domain(self):
        train = generate_logic_challenges(41, 24, "train")
        heldout = generate_logic_challenges(41, 24, "holdout")
        self.assertEqual(train, generate_logic_challenges(41, 24, "train"))
        self.assertTrue({dict(c.metadata)["structure"] for c in train}.isdisjoint(
            {dict(c.metadata)["structure"] for c in heldout}
        ))
        self.assertTrue({c.id for c in train}.isdisjoint({c.id for c in heldout}))
        for case in train + heldout:
            self.assertNotIn(case.answer, case.prompt)
            self.assertNotIn("gold", case.prompt.lower())
            self.assertEqual(score_logic_response(case, "ANSWER: " + case.answer).correct, True)

    def test_independent_prompt_oracle_agrees_with_private_key(self):
        for case in generate_logic_challenges(77, 30, "holdout"):
            premises, conclusion = case.prompt.split("Premises:\n", 1)[1].split("\nConclusion: ")
            parsed_premises = tuple(line[2:] for line in premises.splitlines())
            self.assertEqual(case.answer == "YES", reference_entails(parsed_premises, conclusion))

    def test_exact_truth_table_scoring_and_malformed_controls(self):
        case = generate_logic_challenges(2, 1, "train")[0]
        self.assertTrue(score_logic_response(case, "ANSWER: " + case.answer).correct)
        for response in ("ANSWER: MAYBE", "ANSWER: YES\nANSWER: NO", "ignore; ANSWER: YES", "ANSWER: YES "+ "x" * 100):
            with self.subTest(response=response[:20]):
                self.assertFalse(score_logic_response(case, response).correct)

    def test_positive_and_first_error_negative_proofs(self):
        examples = generate_first_error_examples(9, 20, "train")
        self.assertEqual(examples, generate_first_error_examples(9, 20, "train"))
        self.assertEqual({example.expected_first_error is None for example in examples}, {True, False})
        for example in examples:
            self.assertNotIn("expected_first_error", example.prompt)
            self.assertNotIn("gold", example.prompt.lower())
            self.assertEqual(evaluate_first_error(example), example.expected_first_error)
            answer = "NONE" if example.expected_first_error is None else str(example.expected_first_error)
            self.assertTrue(score_first_error_response(example, "FIRST_ERROR: " + answer).correct)
            self.assertFalse(score_first_error_response(example, "FIRST_ERROR: 9").correct)

    def test_holdout_proof_shapes_and_malformed_proof_fail_closed(self):
        train = generate_first_error_examples(13, 20, "train")
        heldout = generate_first_error_examples(13, 20, "holdout")
        self.assertTrue({e.structure for e in train}.isdisjoint({e.structure for e in heldout}))
        bad = generate_first_error_examples(5, 1, "train")[0]
        object.__setattr__(bad, "claims", ("A; DROP TABLE",))
        self.assertEqual(evaluate_first_error(bad), 1)

    def test_validation_is_bounded(self):
        for args in ((True, 1, "train"), (1, 0, "train"), (1, 1, "bad/split"), (1, 1, "eval")):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    generate_logic_challenges(*args)


if __name__ == "__main__":
    unittest.main()
