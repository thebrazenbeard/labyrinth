"""Exercise real model-action adapter through the source-only Saga bridge."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from labyrinth.evolutionary_saga_runner_v1 import run_theseus_saga_trial
from tests import test_model_saga_runtime as saga_fixtures


class TheseusSagaIntegrationTests(unittest.TestCase):
    def test_policy_chosen_world_causes_real_train_route_but_no_neural_claim(self):
        with TemporaryDirectory() as tmp:
            session = saga_fixtures.SagaSessionTests().make_session(Path(tmp))[0]

            class StoryPolicy:
                def choose_action(self, context):
                    actions = context.available_actions
                    for choice in ("face", "train", "forward", "right", "left"):
                        if choice in actions:
                            return "ACTION: " + choice
                    return "ACTION: look"

            trial = run_theseus_saga_trial(session, model_policy=StoryPolicy())
            self.assertEqual(trial.developmental_status, "PASS")
            self.assertEqual(trial.train_spark_count, 4)
            self.assertTrue(trial.terminal)
            self.assertIn("NOT_NEURAL_EVIDENCE", trial.classification)
            self.assertEqual(trial.entry_checkpoint_sha256,
                             session.entrance.checkpoint.weights_sha256)
            self.assertEqual(trial.exit_checkpoint_sha256,
                             session.checkpoint.weights_sha256)
            self.assertEqual(trial.training_tactics, ("transfer",) * 4)
            # Crucial: this test learner always responds 'ANSWER: 0', and
            # the fake examiner grades by cycle. It has no generalization.
            self.assertFalse(any(spark.train_correct for spark in session.training_sparks))

    def test_policy_cannot_forge_success_from_illegal_action(self):
        with TemporaryDirectory() as tmp:
            session = saga_fixtures.SagaSessionTests().make_session(Path(tmp))[0]

            class BadPolicy:
                def choose_action(self, context):
                    return "ACTION: face"  # not a legal entrance action

            with self.assertRaisesRegex(ValueError, "did not complete"):
                run_theseus_saga_trial(session, model_policy=BadPolicy())
            self.assertEqual(len(session.training_sparks), 0)
            self.assertEqual(session.action_count, 0)


if __name__ == "__main__":
    unittest.main()
