"""Adversarial tests with local fixtures, not imported generator labels."""
import unittest

from labyrinth.minotaur_independent_eval import (
    TraceEvaluationCase, evaluate_predictions, independently_label,
)


def _program():
    return ("Execute operations.\nInitial registers: A=2, B=3\n"
            "1. ADD A B\n2. MUL B A\n3. SUB A 1\n"
            "What is the final value of A? Respond exactly.")


def _clean():
    return ("Step 1: A=5, B=3", "Step 2: A=5, B=15", "Step 3: A=4, B=15")


class IndependentMinotaurEvalTests(unittest.TestCase):
    def test_replay_detects_hidden_fixture_label_and_earliest_mutation(self):
        self.assertEqual(independently_label(_program(), _clean())["first_error_step"], None)
        altered = ("Step 1: A=5, B=3", "Step 2: A=5, B=16", "Step 3: A=4, B=15")
        verdict = independently_label(_program(), altered)
        self.assertEqual(verdict["disposition"], "ERROR_STEP")
        self.assertEqual(verdict["first_error_step"], 2)

    def test_label_mismatch_or_malformed_state_refuses_entire_batch(self):
        bad_label = TraceEvaluationCase("bad", _program(), _clean(), 2)
        refused = evaluate_predictions([bad_label], {"bad": 0})
        self.assertEqual(refused["disposition"], "REFUSED")
        self.assertIn("claimed_label_disagrees", refused["audit"]["claimed_label_failures"][0]["reason"])
        bad_state = TraceEvaluationCase("state", _program(), ("Step 1: A=97, B=3",) + _clean()[1:], 1)
        self.assertEqual(evaluate_predictions([bad_state], {"state": 1})["disposition"], "REFUSED")

    def test_predictions_score_abstention_confusion_and_controls(self):
        clean = TraceEvaluationCase("clean", _program(), _clean(), None)
        faulty_steps = ("Step 1: A=6, B=3", "Step 2: A=5, B=15", "Step 3: A=4, B=15")
        faulty = TraceEvaluationCase("faulty", _program(), faulty_steps, 1)
        report = evaluate_predictions([clean, faulty], {"clean": "ERROR_STEP:1", "faulty": None})
        metrics = report["metrics"]
        self.assertEqual(report["disposition"], "SCORED")
        self.assertEqual(metrics["abstentions"], 1)
        self.assertEqual(metrics["false_rejections"], 1)
        self.assertEqual(metrics["exact_first_error_correct"], 0)
        self.assertEqual(metrics["confusion"], {"0->1": 1, "1->ABSTAIN": 1})
        self.assertEqual(report["confidence_intervals"], "not_computed_without_iid_binomial_attestation")
        self.assertIn("always_yes", report["negative_controls"])

    def test_structure_audit_and_conditional_ci(self):
        case = TraceEvaluationCase("x", _program(), _clean(), None, structure="ADD A R;MUL B R;SUB A C")
        report = evaluate_predictions([case], {"x": "ALL_CORRECT"},
                                      training_structures=("ADD A R;MUL B R;SUB A C",),
                                      iid_binomial_attested=True)
        self.assertEqual(report["audit"]["training_structure_overlap_count"], 1)
        self.assertEqual(report["confidence_intervals"]["condition"],
                         "conditional_on_caller_iid_binomial_attestation")
        forged = TraceEvaluationCase("forged", _program(), _clean(), None, structure="not-the-program")
        self.assertEqual(evaluate_predictions([forged], {"forged": 0})["disposition"], "REFUSED")


if __name__ == "__main__":
    unittest.main()
