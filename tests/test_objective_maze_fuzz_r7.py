"""Adversarial checks for catalog-issued curriculum observations."""

import unittest

from labyrinth.curriculum import (
    CurriculumSpec,
    CurriculumState,
    TrainingObservation,
    next_training_case,
    record_training_result,
)


class ObjectiveMazeIssuedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.spec = CurriculumSpec(levels=("easy", "hard"))
        self.catalog = {"easy": ("alpha", "beta", "gamma"),
                        "hard": ("delta", "epsilon", "zeta")}
        self.seed = 20261030

    def test_legacy_unbound_observations_can_promote_without_catalog_issuance(self):
        """This is a reproduced boundary, not approval to trust legacy calls."""
        state = CurriculumState()
        for i in range(4):
            state, promoted = record_training_result(
                self.spec, state, TrainingObservation(f"unlisted-{i}", i, "easy", 1.0)
            )
        self.assertTrue(promoted)
        self.assertEqual(state.level_index, 1)

    def test_schedule_bound_record_rejects_unissued_case_and_wrong_seed(self):
        from labyrinth.curriculum import record_scheduled_training_result

        state = CurriculumState()
        for observation in (
            TrainingObservation("unlisted", 0, "easy", 1.0),
            TrainingObservation("alpha", 0, "easy", 1.0),
        ):
            with self.subTest(observation=observation), self.assertRaisesRegex(
                ValueError, "scheduled"
            ):
                record_scheduled_training_result(
                    self.spec, state, observation, self.catalog, run_seed=self.seed
                )

    def test_schedule_bound_record_promotes_only_after_issued_sequence(self):
        from labyrinth.curriculum import record_scheduled_training_result

        state = CurriculumState()
        for i in range(4):
            issued = next_training_case(self.spec, state, self.catalog, run_seed=self.seed)
            state, promoted = record_scheduled_training_result(
                self.spec, state,
                TrainingObservation(issued.challenge_id, issued.seed, issued.level, 1.0),
                self.catalog, run_seed=self.seed,
            )
            self.assertEqual(promoted, i == 3)
        self.assertEqual(state.level_index, 1)

    def test_schedule_bound_record_replays_prior_history(self):
        from labyrinth.curriculum import record_scheduled_training_result

        state, _ = record_training_result(
            self.spec, CurriculumState(),
            TrainingObservation("not-issued", 5, "easy", 1.0),
        )
        plausible = next_training_case(self.spec, state, self.catalog, run_seed=self.seed)
        with self.assertRaisesRegex(ValueError, "history|scheduled"):
            record_scheduled_training_result(
                self.spec, state,
                TrainingObservation(plausible.challenge_id, plausible.seed, plausible.level, 1.0),
                self.catalog, run_seed=self.seed,
            )

    def test_mutated_frozen_prior_score_cannot_grant_promotion(self):
        from labyrinth.curriculum import record_scheduled_training_result

        state = CurriculumState()
        for _ in range(3):
            issued = next_training_case(self.spec, state, self.catalog, run_seed=self.seed)
            state, _ = record_scheduled_training_result(
                self.spec, state,
                TrainingObservation(issued.challenge_id, issued.seed, issued.level, 0.0),
                self.catalog, run_seed=self.seed,
            )
        # Frozen dataclasses can still be corrupted through low-level restore,
        # so replay must validate each historic observation before using score.
        issued = next_training_case(self.spec, state, self.catalog, run_seed=self.seed)
        object.__setattr__(state.observations[0], "score", 10.0)
        with self.assertRaisesRegex(ValueError, "score"):
            record_scheduled_training_result(
                self.spec, state,
                TrainingObservation(issued.challenge_id, issued.seed, issued.level, 1.0),
                self.catalog, run_seed=self.seed,
            )


if __name__ == "__main__":
    unittest.main()
