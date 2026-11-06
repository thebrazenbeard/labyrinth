"""End-to-end evidence checks for the bounded arithmetic toy journey."""

from __future__ import annotations

import json
import unittest

from labyrinth.journey_toy import SCHEMA, run_toy_journey
from labyrinth.training_scope import TrainingScopeSpec


def _scope(*, skills: tuple[str, ...] = ("arithmetic",), cycles: int = 2):
    return TrainingScopeSpec(
        objective_id="toy-arithmetic",
        domain="arithmetic",
        skills=skills,
        specificity=2,
        scale=4,
        max_cycles=cycles,
        seed=7,
    )


class ToyJourneyTests(unittest.TestCase):
    def test_failure_regenerates_new_maze_without_resetting_weights(self):
        result = run_toy_journey(_scope(), epochs_by_cycle=(1, 1))
        first, second = result["cycles"]

        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["final_phase"], "EXHAUSTED")
        self.assertIs(result["formal_graduation"], False)
        self.assertEqual(result["qualification"], "NOT_ESTABLISHED")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(first["exam_status"], second["exam_status"])
        self.assertEqual(first["exam_status"], "FAIL")
        self.assertGreater(first["parameter_updates"], 0)
        self.assertGreater(second["parameter_updates"], 0)
        self.assertEqual(first["checkpoint_before_sha256"], result["baseline_sha256"])
        self.assertNotEqual(first["checkpoint_before_sha256"], first["checkpoint_after_sha256"])
        self.assertEqual(second["checkpoint_before_sha256"], first["checkpoint_after_sha256"])
        self.assertEqual(second["checkpoint_after_sha256"], result["final_checkpoint_sha256"])
        self.assertNotEqual(second["checkpoint_before_sha256"], second["checkpoint_after_sha256"])
        self.assertTrue(set(first["training_room_ids"]).isdisjoint(second["training_room_ids"]))
        self.assertTrue(set(first["training_case_ids"]).isdisjoint(second["training_case_ids"]))
        self.assertTrue(set(first["exam_case_ids"]).isdisjoint(second["exam_case_ids"]))
        self.assertTrue(set(first["training_case_ids"]).isdisjoint(first["exam_case_ids"]))
        self.assertTrue(all(cycle["training_case_count"] == 8 for cycle in (first, second)))
        self.assertTrue(all(cycle["exam_case_count"] == 16 for cycle in (first, second)))

    def test_deterministic_json_serializable_evidence_and_no_graduation(self):
        first = run_toy_journey(_scope(), epochs_by_cycle=(1, 1))
        second = run_toy_journey(_scope(), epochs_by_cycle=(1, 1))
        self.assertEqual(first, second)
        serialized = json.dumps(first, sort_keys=True, allow_nan=False)
        self.assertEqual(json.loads(serialized), first)
        self.assertNotIn("GRADUATED", serialized)
        self.assertIn("independent", serialized)  # limitation, not independence proof

    def test_toy_backend_fails_closed_for_unimplemented_learning_domains(self):
        for skills in (
            ("cognitive_empathy",), ("reactive_empathy",),
            ("arithmetic", "cognitive_empathy"), ("source_priority",),
        ):
            with self.subTest(skills=skills), self.assertRaisesRegex(ValueError, "only an arithmetic-only"):
                run_toy_journey(_scope(skills=skills), epochs_by_cycle=(1, 1))

    def test_invalid_or_unbounded_run_parameters_rejected(self):
        for kwargs in (
            {"epochs_by_cycle": (1,)}, {"epochs_by_cycle": (0, 1)},
            {"epochs_by_cycle": (True, 1)}, {"learning_rate": float("nan")},
            {"learning_rate": 0}, {"exam_count_per_slice": 3},
            {"exam_count_per_slice": True}, {"learner_seed": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_toy_journey(_scope(), **kwargs)

    def test_smallest_scope_rejected_when_toy_training_requires_two_examples(self):
        scope = TrainingScopeSpec(
            objective_id="tiny-arithmetic", domain="arithmetic", skills=("arithmetic",),
            specificity=1, scale=1, max_cycles=1, seed=2,
        )
        with self.assertRaisesRegex(ValueError, "at least two arithmetic chambers"):
            run_toy_journey(scope)


if __name__ == "__main__":
    unittest.main()
