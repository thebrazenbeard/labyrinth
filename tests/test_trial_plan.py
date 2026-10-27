"""Executable controls for matched neural-training exposures.

This is a scheduling contract, not evidence of a successful neural update.
"""

import unittest

from labyrinth.maze import generate_maze
from labyrinth.trial_plan import build_trial_plan


class TrialPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = [f"train-case-{index:03d}" for index in range(128)]
        self.development = [f"dev-case-{index:03d}" for index in range(12)]
        self.independent = [f"final-case-{index:03d}" for index in range(24)]

    def make_plan(self, **overrides):
        arguments = dict(
            train_case_ids=self.train,
            development_case_ids=self.development,
            independent_case_ids=self.independent,
            steps=32,
            seed=1807,
        )
        arguments.update(overrides)
        return build_trial_plan(**arguments)

    def test_exposures_match_across_all_intervention_arms(self) -> None:
        plan = self.make_plan()
        manifest = plan.to_dict()

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["available_training_cases"], 128)
        self.assertEqual(manifest["scheduled_training_visits"], 32)
        self.assertEqual(manifest["unique_cases_scheduled"], 32)
        self.assertEqual(len(manifest["scheduled_case_ids"]), 32)
        self.assertEqual(set(manifest["scheduled_case_ids"]) - set(self.train), set())
        self.assertEqual(manifest["arms"], ["trained", "sham", "corrupted"])
        self.assertEqual(
            manifest["schedule_by_arm"],
            {arm: manifest["scheduled_case_ids"] for arm in manifest["arms"]},
        )
        self.assertTrue(set(self.development).isdisjoint(manifest["scheduled_case_ids"]))
        self.assertTrue(set(self.independent).isdisjoint(manifest["scheduled_case_ids"]))
        self.assertNotIn("completed_training_visits", manifest)
        self.assertNotIn("unique_cases_seen", manifest)

    def test_catalog_input_order_does_not_change_schedule_or_commitment(self) -> None:
        plan = self.make_plan().to_dict()
        reordered = self.make_plan(
            train_case_ids=list(reversed(self.train)),
            development_case_ids=list(reversed(self.development)),
            independent_case_ids=list(reversed(self.independent)),
        ).to_dict()
        self.assertEqual(plan, reordered)
        self.assertEqual(len(plan["manifest_sha256"]), 64)

    def test_cycle_covers_pool_before_revisiting(self) -> None:
        manifest = self.make_plan(steps=128 + 7).to_dict()
        first_cycle = manifest["scheduled_case_ids"][:128]
        second_cycle = manifest["scheduled_case_ids"][128:]
        self.assertEqual(set(first_cycle), set(self.train))
        self.assertEqual(len(set(second_cycle)), 7)
        self.assertEqual(manifest["unique_cases_scheduled"], 128)
        self.assertEqual(manifest["scheduled_training_visits"], 135)

    def test_independent_case_identifiers_never_leave_public_manifest(self) -> None:
        manifest = self.make_plan().to_dict()
        self.assertEqual(manifest["independent_case_count"], 24)
        self.assertNotIn("independent_ids_commitment_sha256", manifest)
        public_serialized = repr(manifest)
        for identifier in self.independent:
            self.assertNotIn(identifier, public_serialized)
        self.assertNotIn("independent_case_ids", manifest)

    def test_public_plan_cannot_be_used_as_dictionary_oracle_for_final_ids(self) -> None:
        original = self.make_plan(independent_case_ids=["F07"]).to_dict()
        replacement = self.make_plan(independent_case_ids=["F08"]).to_dict()
        # Final-set membership changes must not affect anything published by
        # this training-schedule module. The evaluator owns separate custody.
        self.assertEqual(original, replacement)
        self.assertNotIn("independent_ids_commitment_sha256", repr(self.make_plan()))

    def test_reject_split_collisions_and_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "across splits"):
            self.make_plan(development_case_ids=self.development + [self.train[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.make_plan(train_case_ids=self.train + [self.train[0]])

    def test_reject_invalid_case_ids_and_budget(self) -> None:
        for bad in ("", "bad\ncase", "\ud800", 17):
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                self.make_plan(train_case_ids=[bad])
        for bad in (0, True, -1, 100001):
            with self.subTest(steps=bad), self.assertRaises(ValueError):
                self.make_plan(steps=bad)
        for bad in (-1, True, 2**64):
            with self.subTest(seed=bad), self.assertRaises(ValueError):
                self.make_plan(seed=bad)
        with self.assertRaises(ValueError):
            self.make_plan(independent_case_ids=[])

    def test_seed_changes_schedule_but_not_catalog_commitments(self) -> None:
        original = self.make_plan().to_dict()
        alternate = self.make_plan(seed=1808).to_dict()
        self.assertNotEqual(original["scheduled_case_ids"], alternate["scheduled_case_ids"])
        self.assertEqual(
            original["development_ids_commitment_sha256"],
            alternate["development_ids_commitment_sha256"],
        )

    def test_real_register_program_generator_produces_disjoint_exposures(self) -> None:
        train = generate_maze(107, tiers=(4,), count=24, split="train")
        development = generate_maze(108, tiers=(4,), count=12, split="dev")
        independent = generate_maze(109, tiers=(4,), count=12, split="final")
        plan = build_trial_plan(
            train_case_ids=[case.id for case in train],
            development_case_ids=[case.id for case in development],
            independent_case_ids=[case.id for case in independent],
            steps=32,
            seed=1807,
        ).to_dict()

        self.assertEqual(plan["unique_cases_scheduled"], 24)
        self.assertTrue(set(plan["scheduled_case_ids"]) <= {case.id for case in train})
        # This generator independently supplies structural train-vs-eval separation.
        fingerprints = lambda cases: {dict(case.metadata)["structure_fingerprint"] for case in cases}
        self.assertTrue(fingerprints(train).isdisjoint(fingerprints(development)))
        self.assertTrue(fingerprints(train).isdisjoint(fingerprints(independent)))


if __name__ == "__main__":
    unittest.main()
