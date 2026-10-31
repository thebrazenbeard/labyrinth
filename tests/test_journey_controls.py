"""Matched fixed-budget controls for a toy regenerative Labyrinth journey."""

from __future__ import annotations

import json
import unittest

from labyrinth.journey_controls import ARMS, SCHEMA, run_matched_toy_controls
from labyrinth.training_scope import TrainingScopeSpec


def _scope(*, skills: tuple[str, ...] = ("arithmetic",), cycles: int = 2):
    return TrainingScopeSpec(
        objective_id="toy-matched-controls", domain="arithmetic", skills=skills,
        specificity=2, scale=4, max_cycles=cycles, seed=7,
    )


class MatchedToyControlTests(unittest.TestCase):
    def test_equal_exposure_and_fresh_exam_over_two_cycles(self):
        result = run_matched_toy_controls(_scope(), epochs_by_cycle=(1, 1))
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["cycle_count"], 2)
        self.assertEqual(result["arms"], list(ARMS))
        self.assertIs(result["formal_graduation"], False)
        self.assertEqual(result["qualification"], "NOT_ESTABLISHED")
        first, second = result["cycles"]
        self.assertTrue(set(first["training_room_ids"]).isdisjoint(second["training_room_ids"]))
        self.assertTrue(set(first["training_case_ids"]).isdisjoint(second["training_case_ids"]))
        self.assertTrue(set(first["exam_case_ids"]).isdisjoint(second["exam_case_ids"]))
        all_training = {cid for cycle in result["cycles"] for cid in cycle["training_case_ids"]}
        all_exams = {cid for cycle in result["cycles"] for cid in cycle["exam_case_ids"]}
        self.assertFalse(all_training & all_exams)
        for cycle in result["cycles"]:
            visits = {arm["case_visits"] for arm in cycle["arms"].values()}
            self.assertEqual(visits, {8})
            self.assertEqual({arm["exam_total"] for arm in cycle["arms"].values()}, {16})
            first_exam = cycle["arms"]["baseline"]["observations"]
            self.assertTrue(all(
                [entry["case_id"] for entry in arm["observations"]]
                == [entry["case_id"] for entry in first_exam]
                for arm in cycle["arms"].values()
            ))
        self.assertEqual(second["arms"]["correct_labels"]["checkpoint_before_sha256"],
                         first["arms"]["correct_labels"]["checkpoint_after_sha256"])
        self.assertEqual(second["arms"]["corrupted_labels"]["checkpoint_before_sha256"],
                         first["arms"]["corrupted_labels"]["checkpoint_after_sha256"])

    def test_nonupdate_controls_unchanged_and_true_updates_observed(self):
        result = run_matched_toy_controls(_scope(), epochs_by_cycle=(1, 1))
        baseline = result["initial_checkpoint_sha256"]
        final = result["final_checkpoint_sha256_by_arm"]
        self.assertEqual(final["baseline"], baseline)
        self.assertEqual(final["zero_update"], baseline)
        self.assertNotEqual(final["correct_labels"], baseline)
        self.assertNotEqual(final["corrupted_labels"], baseline)
        for cycle in result["cycles"]:
            self.assertEqual(cycle["arms"]["baseline"]["parameter_updates"], 0)
            self.assertEqual(cycle["arms"]["zero_update"]["parameter_updates"], 0)
            self.assertGreater(cycle["arms"]["correct_labels"]["parameter_updates"], 0)
            self.assertGreater(cycle["arms"]["corrupted_labels"]["changed_training_labels"], 0)
            self.assertEqual(cycle["arms"]["baseline"]["exam_correct"],
                             cycle["arms"]["zero_update"]["exam_correct"])

    def test_deterministic_serialization(self):
        first = run_matched_toy_controls(_scope(), epochs_by_cycle=(1, 1))
        second = run_matched_toy_controls(_scope(), epochs_by_cycle=(1, 1))
        self.assertEqual(first, second)
        self.assertEqual(json.loads(json.dumps(first, allow_nan=False)), first)

    def test_bad_scope_and_budget_rejected_before_any_effect(self):
        with self.assertRaisesRegex(ValueError, "arithmetic-only"):
            run_matched_toy_controls(_scope(skills=("empathy",)))
        with self.assertRaisesRegex(ValueError, "matching every cycle"):
            run_matched_toy_controls(_scope(), epochs_by_cycle=(1,))
        with self.assertRaisesRegex(ValueError, "bounded integer"):
            run_matched_toy_controls(_scope(), epochs_by_cycle=(True, 1))
        with self.assertRaisesRegex(ValueError, "finite"):
            run_matched_toy_controls(_scope(), learning_rate=float("nan"))


if __name__ == "__main__":
    unittest.main()
