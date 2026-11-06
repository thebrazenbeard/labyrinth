"""Hostile check: source-only saga receipts must not accept forged qualification labels."""

import unittest

from labyrinth.model_saga_runtime import CheckpointRef
from labyrinth.evolutionary_saga_runner_v1 import (
    FreshProcessCheckpointControl, SagaCampaignEvidence,
    SagaGenerationEvidence, SagaTrialEvidence,
)


class EvidenceLabelInvariantsTests(unittest.TestCase):
    def test_all_evidence_classes_refuse_caller_supplied_upgrade_labels(self):
        fake_hash = "0" * 64
        w0 = CheckpointRef("/tmp/fake", fake_hash, None, None, "WEIGHTS_ONLY_CONTINUITY")
        trials = dict(
            objective_id="source-only", entry_checkpoint_sha256=fake_hash,
            exit_checkpoint_sha256="1" * 64, developmental_status="PASS",
            terminal=True, train_spark_count=1, generation_cycles=2,
            ariadne_head_sha256="2" * 64, traversed_branches=("left",),
            training_tactics=("transfer",),
        )

        for forged in ("NEURAL_UPDATE_VERIFIED", "INDEPENDENT_REVIEW_PASSED", "GRADUATED"):
            with self.subTest(kind="trial", forged=forged), self.assertRaises(ValueError):
                SagaTrialEvidence(**trials, classification=forged)
            with self.subTest(kind="generation", forged=forged), self.assertRaises(ValueError):
                SagaGenerationEvidence(0, None, fake_hash, (), "winner", forged)
            with self.subTest(kind="process-control", forged=forged), self.assertRaises(ValueError):
                FreshProcessCheckpointControl(fake_hash, None, 123, forged)
            with self.subTest(kind="campaign", forged=forged), self.assertRaises(ValueError):
                SagaCampaignEvidence(w0, (), FreshProcessCheckpointControl(fake_hash, None, 123), forged)

    def test_legitimate_model_action_label_remains_explicitly_unqualified(self):
        value = SagaTrialEvidence(
            "source-only", "0" * 64, "1" * 64, "PASS", True, 1, 1, "2" * 64,
            ("left",), ("transfer",),
            "MODEL_ACTION_INTERFACE_UNQUALIFIED_NOT_NEURAL_EVIDENCE",
        )
        self.assertIn("UNQUALIFIED", value.classification)


if __name__ == "__main__":
    unittest.main()
