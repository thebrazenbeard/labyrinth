"""Pure stdlib tests for curriculum replay, holdout isolation and calibration."""

import json
import unittest

from labyrinth.curriculum import (
    CurriculumSpec,
    CurriculumState,
    TrainingObservation,
    next_training_case,
    record_training_result,
    record_scheduled_training_result,
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

    def test_subclassed_private_split_cannot_masquerade_as_training(self):
        class ForgedSplit(str):
            def __eq__(self, other):
                return other == "train"

            def __ne__(self, other):
                return other != "train"

        private_split = ForgedSplit("private_final")
        with self.assertRaisesRegex(ValueError, "split"):
            record_training_result(
                self.spec,
                CurriculumState(),
                TrainingObservation("logic", 1, "easy", 1.0, private_split),
            )

        contaminated = {
            "schema_version": 1,
            "level_index": 0,
            "observations": [
                {"challenge_id": "logic", "seed": 1, "level": "easy", "score": 1.0,
                 "split": private_split}
            ],
        }
        with self.assertRaisesRegex(ValueError, "split"):
            CurriculumState.from_dict(contaminated)

    def test_mutated_existing_observation_cannot_admit_private_final_as_train(self):
        class ForgedSplit(str):
            def __eq__(self, other):
                return other == "train"

            def __ne__(self, other):
                return other != "train"

        spec = CurriculumSpec(levels=("easy", "hard"), window_size=1,
                              promotion_score=0.5, distinct_challenges_required=1)
        catalog = {"easy": ("one",), "hard": ("two",)}
        issued = next_training_case(spec, CurriculumState(), catalog, run_seed=42)
        observation = TrainingObservation(issued.challenge_id, issued.seed,
                                          issued.level, 1.0)
        object.__setattr__(observation, "split", ForgedSplit("private_final"))
        with self.assertRaisesRegex(ValueError, "split"):
            record_training_result(spec, CurriculumState(), observation)
        with self.assertRaisesRegex(ValueError, "split"):
            record_scheduled_training_result(spec, CurriculumState(), observation,
                                             catalog, run_seed=42)

    def test_mutated_curriculum_spec_is_revalidated_before_replay(self):
        spec = CurriculumSpec(levels=("easy", "hard"), window_size=1,
                              promotion_score=0.5, distinct_challenges_required=1)
        object.__setattr__(spec, "levels", ["easy", "hard"])
        with self.assertRaisesRegex(ValueError, "immutable tuple"):
            next_training_case(spec, CurriculumState(), self.catalog, run_seed=42)

    def test_subclassed_level_cannot_masquerade_as_active_corridor(self):
        class ForgedLevel(str):
            def __eq__(self, other):
                return other == "easy"

            def __ne__(self, other):
                return other != "easy"

        with self.assertRaisesRegex(ValueError, "level"):
            record_training_result(
                self.spec,
                CurriculumState(),
                TrainingObservation("logic", 1, ForgedLevel("private_final"), 1.0),
            )

    def test_subclassed_challenge_ids_cannot_forge_diversity(self):
        class ForgedId(str):
            def __eq__(self, other):
                return self is other

            __hash__ = str.__hash__

        state = CurriculumState()
        for seed in range(3):
            with self.assertRaisesRegex(ValueError, "challenge_id"):
                record_training_result(
                    self.spec,
                    state,
                    TrainingObservation(ForgedId("logic"), seed, "easy", 1.0),
                )

    def test_subclassed_scores_cannot_forge_promotion(self):
        class EvilFloat(float):
            def __le__(self, other):
                return True

            def __ge__(self, other):
                return True

            def __radd__(self, other):
                return 1000.0

        with self.assertRaisesRegex(ValueError, "score"):
            TrainingObservation("logic", 1, "easy", EvilFloat(0.0))
        with self.assertRaisesRegex(ValueError, "promotion_score"):
            CurriculumSpec(levels=("easy", "hard"), promotion_score=EvilFloat(1.0))

        for observation in (
            {"challenge_id": "logic", "seed": 1, "level": "easy", "score": EvilFloat(0.0)},
        ):
            with self.assertRaisesRegex(ValueError, "score"):
                CurriculumState.from_dict({"schema_version": 1, "level_index": 0,
                                           "observations": [observation]})

    def test_curriculum_level_plan_must_be_immutable_primitive_tuple(self):
        levels = ["easy", "hard"]
        with self.assertRaisesRegex(ValueError, "immutable tuple"):
            CurriculumSpec(levels=levels, window_size=1, distinct_challenges_required=1)

        class ForgedLevel(str):
            def __eq__(self, other):
                return True

            __hash__ = str.__hash__

        with self.assertRaisesRegex(ValueError, "immutable tuple"):
            CurriculumSpec(levels=("easy", ForgedLevel("private_final")))
        with self.assertRaisesRegex(ValueError, "unique"):
            CurriculumSpec(levels=("easy", "easy"))
        with self.assertRaisesRegex(ValueError, "unique"):
            next_training_case(self.spec, CurriculumState(),
                               {"easy": ("logic", "logic", "math")}, run_seed=17)
        with self.assertRaisesRegex(ValueError, "unique"):
            next_training_case(self.spec, CurriculumState(),
                               {"easy": ("logic", "math", "contradiction") + ("logic",)}, run_seed=17)
        with self.assertRaisesRegex(ValueError, "requires"):
            next_training_case(self.spec, CurriculumState(),
                               {"easy": ("logic", ForgedLevel("private_final"), "math")}, run_seed=17)

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


if __name__ == "__main__":
    unittest.main()
