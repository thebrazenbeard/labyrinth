"""A physical trainer may mutate Python dataclass aliases after TRAIN starts.

These tests use mock byte writers, not neural SGD or an actual model.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.test_social_experience_training_bridge_v1 import (
    MockPhysicalTrainer, _beats, _build,
)


class TrainingCaseAliasHostileTests(unittest.TestCase):
    def test_trainer_cannot_rewrite_approved_supervised_target_after_wal(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            class MutatingPhysicalTrainer(MockPhysicalTrainer):
                def train_supervised(self, case, parent):
                    # Frozen dataclasses are mutable through object.__setattr__.
                    # A buggy adapter can update the target used by the actual
                    # byte effect while the bridge still logs its old digest.
                    object.__setattr__(case.feedback, "preferred_response",
                                       "I do not care about the promise or its consequences.")
                    object.__setattr__(case, "learner_response", "forged learner answer")
                    return super().train_supervised(case, parent)

            bridge, _, _, trainer = _build(root, trainer=MutatingPhysicalTrainer(root))
            with self.assertRaisesRegex(ValueError, "TRAIN case|approved.*train|changed"):
                bridge.run_train_beat(_beats()[0])
            self.assertTrue(bridge.indeterminate,
                            "a mutated case after physical TRAIN must remain ambiguous")
            self.assertEqual(len(trainer.cases), 1)


if __name__ == "__main__":
    unittest.main()
