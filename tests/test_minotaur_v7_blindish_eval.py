"""Contracts for the independent, public-but-hostile V7 evaluator."""
import unittest
from dataclasses import replace

from labyrinth.minotaur_v7_blindish_eval import (
    ABSENT_LABEL,
    audit_predictions,
    build_protocol,
    canonical_pair,
    recompute_label,
)


class BlindishEvaluationTests(unittest.TestCase):
    def test_protocol_recomputes_complete_pair_safe_public_challenges(self):
        protocol = build_protocol(seed=20261028)
        self.assertEqual(protocol.modulus, 97)
        self.assertEqual(protocol.counts, {
            "train": 8212, "unseen_group_pairs": 910,
            "missing_output_53": 94, "missing_input_53": 193,
            "mod101": 10201,
        })
        self.assertEqual(set(protocol.digests), set(protocol.counts))
        self.assertTrue(all(recompute_label(x, y, 97) == label
                            for x, y, label in protocol.train))
        self.assertTrue(all(label == ABSENT_LABEL
                            for _x, _y, label in protocol.challenges["missing_output_53"]))
        self.assertTrue(all(ABSENT_LABEL in (x, y)
                            for x, y, _label in protocol.challenges["missing_input_53"]))
        self.assertEqual(len({canonical_pair(x, y) for x, y, _ in protocol.train}),
                         len(protocol.train) - sum(x != y for x, y, _ in protocol.train) // 2)

    def test_integrity_proves_zero_canonical_overlap_and_only_claimed_label_absence(self):
        protocol = build_protocol(seed=9)
        integrity = protocol.integrity
        self.assertTrue(integrity["ok"])
        self.assertEqual(integrity["canonical_pair_overlap"], {})
        self.assertEqual(integrity["label_overlap"]["missing_output_53"], ())
        self.assertIn("missing_input_53", integrity["label_overlap"])
        self.assertTrue(integrity["mod101_training_rows"] == 0)

    def test_predictions_are_scored_per_case_with_abstention_and_pair_complete_units(self):
        protocol = build_protocol(seed=17)
        predictions = {
            name: {(x, y): label for x, y, label in rows}
            for name, rows in protocol.challenges.items()
        }
        predictions["missing_output_53"].pop(next(iter(predictions["missing_output_53"])))
        report = audit_predictions(protocol, predictions)
        missing = report["cases"]["missing_output_53"]
        self.assertEqual(missing["correct"], missing["count"] - 1)
        self.assertEqual(missing["abstained"], 1)
        self.assertFalse(missing["pair_complete"])
        self.assertIsNone(missing["iid_confidence_interval"])
        self.assertEqual(report["qualification"], "NOT_ESTABLISHED")
        self.assertIn("not a private examiner", report["examiner_status"])

    def test_audit_rejects_wrong_labels_extra_cases_and_mod101_training(self):
        protocol = build_protocol(seed=17)
        predictions = {name: {} for name in protocol.challenges}
        with self.assertRaisesRegex(ValueError, "mod101"):
            audit_predictions(protocol, predictions, training_rows=((1, 1, 1, 101),))
        predictions["mod101"][(999, 1)] = 0
        with self.assertRaisesRegex(ValueError, "outside"):
            audit_predictions(protocol, predictions)
        with self.assertRaisesRegex(ValueError, "counts or digests"):
            audit_predictions(replace(protocol, counts={}), {name: {} for name in protocol.challenges})

    def test_baselines_are_explicit_and_adversarial_is_not_a_model_claim(self):
        protocol = build_protocol(seed=5)
        report = audit_predictions(protocol, {name: {} for name in protocol.challenges})
        self.assertEqual(set(report["baselines"]), {"abstain", "zero", "train_majority", "copy_left", "anti_oracle_adversarial"})
        self.assertEqual(report["baselines"]["abstain"]["missing_output_53"]["abstained"], 94)
        self.assertEqual(report["baselines"]["anti_oracle_adversarial"]["mod101"]["correct"], 0)
        self.assertIn("not a valid predictive baseline", report["baselines"]["anti_oracle_adversarial"]["note"])


if __name__ == "__main__":
    unittest.main()
