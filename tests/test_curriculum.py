"""Pure stdlib tests for curriculum replay, holdout isolation and calibration."""

import json
import unittest

from labyrinth.curriculum import (
    CurriculumSpec,
    CurriculumState,
    TrainingObservation,
    next_training_case,
    record_training_result,
    schedule_training_cases,
)
from labyrinth.adapters import ReplayAdapter


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.spec = CurriculumSpec(levels=("easy", "hard"), window_size=3, promotion_score=0.8, distinct_challenges_required=3)
        self.catalog = {"easy": ("logic", "math", "contradiction"), "hard": ("adversarial", "uncertainty", "causality")}

    def test_seed_reproducible_and_each_task_once_per_cycle(self):
        state = CurriculumState()
        cases = []
        for _ in range(3):
            case = next_training_case(self.spec, state, self.catalog, run_seed=17)
            cases.append(case)
            self.assertTrue(0 <= case.seed <= 2**31 - 1)
            self.assertEqual(ReplayAdapter({"prompt": "reply"}).generate("prompt", seed=case.seed), "reply")
            state, promoted = record_training_result(
                self.spec, state, TrainingObservation(case.challenge_id, case.seed, case.level, 0.0)
            )
            self.assertFalse(promoted)
        self.assertEqual({case.challenge_id for case in cases}, set(self.catalog["easy"]))
        self.assertEqual(cases[0], next_training_case(self.spec, CurriculumState(), dict(reversed(list(self.catalog.items()))), run_seed=17))

    def test_promotion_requires_mastery_and_diversity(self):
        state = CurriculumState()
        for index in range(3):
            case = next_training_case(self.spec, state, self.catalog, run_seed=12)
            state, promoted = record_training_result(self.spec, state, TrainingObservation(case.challenge_id, case.seed, case.level, 0.9))
            self.assertEqual(promoted, index == 2)
        self.assertEqual(state.level_index, 1)
        self.assertEqual(next_training_case(self.spec, state, self.catalog, run_seed=12).level, "hard")

    def test_bad_scores_do_not_promote(self):
        state = CurriculumState()
        for score in (1.0, 0.0, 1.0):
            case = next_training_case(self.spec, state, self.catalog, run_seed=5)
            state, promoted = record_training_result(self.spec, state, TrainingObservation(case.challenge_id, case.seed, case.level, score))
            self.assertFalse(promoted)
        self.assertEqual(state.level_index, 0)

    def test_holdout_and_control_cannot_advance(self):
        for split in ("validation", "test", "holdout", "placebo", "shuffled"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                record_training_result(self.spec, CurriculumState(), TrainingObservation("logic", 1, "easy", 1.0, split))

    def test_restored_holdout_contamination_rejected(self):
        contaminated = {
            "schema_version": 1,
            "level_index": 0,
            "observations": [
                {"challenge_id": "logic", "seed": 1, "level": "easy", "score": 1.0, "split": "holdout"}
            ],
        }
        with self.assertRaisesRegex(ValueError, "evaluation/control"):
            CurriculumState.from_dict(contaminated)
        forged = CurriculumState(observations=(TrainingObservation("logic", 1, "easy", 1.0, "holdout"),))
        with self.assertRaisesRegex(ValueError, "non-training"):
            record_training_result(self.spec, forged, TrainingObservation("math", 2, "easy", 1.0))

    def test_restored_progression_cannot_be_forged(self):
        with self.assertRaisesRegex(ValueError, "replayed progression"):
            next_training_case(self.spec, CurriculumState(level_index=1), self.catalog, run_seed=1)

    def test_duplicate_evidence_rejected(self):
        obs = TrainingObservation("logic", 0, "easy", 1.0)
        state, _ = record_training_result(self.spec, CurriculumState(), obs)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            record_training_result(self.spec, state, obs)

    def test_state_json_round_trip_replays_next_case(self):
        state = CurriculumState()
        case = next_training_case(self.spec, state, self.catalog, run_seed=37)
        state, _ = record_training_result(self.spec, state, TrainingObservation(case.challenge_id, case.seed, case.level, 0.5))
        restored = CurriculumState.from_dict(json.loads(json.dumps(state.to_dict())))
        self.assertEqual(restored, state)
        self.assertEqual(next_training_case(self.spec, restored, self.catalog, run_seed=37), next_training_case(self.spec, state, self.catalog, run_seed=37))

    def test_bad_input_fails_closed(self):
        for score in (float("nan"), float("inf"), -0.1, 1.1, True):
            with self.subTest(score=score), self.assertRaises(ValueError):
                TrainingObservation("logic", 1, "easy", score)
        with self.assertRaises(ValueError):
            CurriculumSpec(levels=("x", "x"))
        with self.assertRaises(ValueError):
            next_training_case(self.spec, CurriculumState(), {"easy": []}, run_seed=2)
        with self.assertRaises(ValueError):
            next_training_case(self.spec, CurriculumState(), {"easy": ["only"]}, run_seed=2)
        with self.assertRaises(ValueError):
            CurriculumState.from_dict(None)

    def test_ungraded_schedule_reproduces_each_cycle_without_grade_state(self):
        spec = CurriculumSpec(levels=("persona",), distinct_challenges_required=3)
        ids = ("contrast", "construct", "repair", "review")
        first = schedule_training_cases(spec, {"persona": ids}, run_seed=74413, steps=9)
        second = schedule_training_cases(spec, {"persona": tuple(reversed(ids))}, run_seed=74413, steps=9)
        self.assertEqual(first, second)
        self.assertEqual(first[0], next_training_case(spec, CurriculumState(), {"persona": ids}, run_seed=74413))
        self.assertEqual({item.challenge_id for item in first[:4]}, set(ids))
        self.assertEqual({item.challenge_id for item in first[4:8]}, set(ids))
        self.assertEqual(len({(item.challenge_id, item.seed) for item in first}), 9)
        self.assertTrue(all(item.level == "persona" and 0 <= item.seed < 2**31 for item in first))
        self.assertNotEqual(first, schedule_training_cases(spec, {"persona": ids}, run_seed=74414, steps=9))

    def test_ungraded_schedule_refuses_implicit_promotion_and_invalid_catalogs(self):
        single = CurriculumSpec(levels=("persona",), distinct_challenges_required=3)
        valid = {"persona": ["alpha", "beta", "gamma"]}
        for bad in ({}, {"persona": []}, {"persona": ["x", "x", "y"]},
                    {"persona": ["x", "y", 1]}, {"persona": "alpha"},
                    {"persona": ["alpha", "beta", "gamma"], "holdout": ["sealed"]}):
            with self.subTest(catalog=bad), self.assertRaises(ValueError):
                schedule_training_cases(single, bad, run_seed=3, steps=3)
        with self.assertRaisesRegex(ValueError, "single-level"):
            schedule_training_cases(self.spec, self.catalog, run_seed=3, steps=3)
        for bad_seed in (True, -1, "3"):
            with self.subTest(seed=bad_seed), self.assertRaises(ValueError):
                schedule_training_cases(single, valid, run_seed=bad_seed, steps=3)
        for bad_steps in (True, 0, -2, 129, 2.0):
            with self.subTest(steps=bad_steps), self.assertRaises(ValueError):
                schedule_training_cases(single, valid, run_seed=3, steps=bad_steps)


if __name__ == "__main__":
    unittest.main()
