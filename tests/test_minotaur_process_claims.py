"""First-error claim-verification supervision; not a ProcessBench score."""
import unittest

from labyrinth.minotaur_critic_pilot import (
    CriticConfig, process_critic_examples, training_schedule, validate_dataset_separation,
    critic_datasets,
)


class ProcessClaimsTests(unittest.TestCase):
    def test_balanced_first_error_claims_are_task_paired_and_source_disjoint(self):
        train = process_critic_examples(123, "train", 2, (1, 2))
        dev = process_critic_examples(456, "dev", 2, (1, 2))
        transfer = process_critic_examples(789, "transfer", 2, (4,))
        self.assertEqual([len(v) for v in (train, dev, transfer)], [16, 16, 8])
        validate_dataset_separation(train, dev, transfer)
        pairs = training_schedule(train, steps=4, seed=19)
        self.assertEqual(len(pairs), 4)
        for a, b in pairs:
            self.assertEqual(a.case_id, b.case_id)
            self.assertEqual(a.case_prompt_sha256, b.case_prompt_sha256)
            self.assertEqual({a.is_correct, b.is_correct}, {True, False})
            self.assertIn("proposed first error", a.prompt.lower())
            self.assertTrue("ALL_CORRECT" in a.prompt or "ERROR_STEP:" in a.prompt)
        self.assertTrue(all(ex.kind.startswith("first_error.") for ex in transfer))
        self.assertEqual(len({ex.case_prompt_sha256 for ex in transfer}), 4)

    def test_reload_datasets_are_objective_bound(self):
        config = CriticConfig("unused", "unused", objective="process_first_error",
                              eval_count_per_tier=2, train_count_per_tier=2, steps=2)
        train, dev, transfer = critic_datasets(config)
        self.assertEqual([len(v) for v in (train, dev, transfer)], [16, 16, 8])
        self.assertTrue(all(ex.kind.startswith("first_error.") for ex in dev + transfer))
        answer = CriticConfig("unused", "unused", eval_count_per_tier=2,
                              train_count_per_tier=2, steps=2)
        _, ordinary, _ = critic_datasets(answer)
        self.assertTrue(all(not ex.kind.startswith("first_error.") for ex in ordinary))

    def test_process_critic_mode_requires_supported_objective(self):
        CriticConfig("unused", "unused", objective="process_first_error").validate_fields()
        with self.assertRaises(ValueError):
            CriticConfig("unused", "unused", objective="unknown").validate_fields()


if __name__ == "__main__":
    unittest.main()
