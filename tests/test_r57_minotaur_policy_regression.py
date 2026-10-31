"""R57: accepted TRAIN feedback must remain checkpoint-persistable."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import labyrinth.minotaur_policy_learning as learning
from tests.test_minotaur_policy_learning import APPROVED, SCOPE, model, outcome


class PolicyCheckpointBudgetTests(unittest.TestCase):
    def test_observation_that_exhausts_checkpoint_budget_is_atomic(self):
        policy = model()
        original = policy.model_hash
        # A deliberately smaller test cap exposes the same serialization
        # budget boundary without consuming thousands of TRAIN observations.
        with patch.object(learning, "MAX_POLICY_JSON_BYTES", 675):
            with self.assertRaisesRegex(ValueError, "checkpoint.*size cap"):
                policy.observe(outcome(0, correct=True))
            self.assertEqual(policy.model_hash, original)
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "still-valid.json"
                self.assertEqual(policy.save_policy(checkpoint), original)
                restored = learning.load_policy(
                    checkpoint, scope=SCOPE, approved_training_cases=APPROVED,
                )
                self.assertEqual(restored.model_hash, original)

    def test_observation_at_exact_checkpoint_limit_remains_persistable(self):
        policy = model()
        with patch.object(learning, "MAX_POLICY_JSON_BYTES", 697):
            self.assertTrue(policy.observe(outcome(0, correct=True))["updated"])
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "limit.json"
                digest = policy.save_policy(checkpoint)
                restored = learning.load_policy(
                    checkpoint, scope=SCOPE, approved_training_cases=APPROVED,
                )
                self.assertEqual(restored.model_hash, digest)


if __name__ == "__main__":
    unittest.main()
