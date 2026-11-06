"""Social TRAIN must bind the exact public scene actually shown to the learner."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.test_social_experience_training_bridge_v1 import (
    MockPhysicalTrainer, TrustedOperator, _beats, _build,
)


class HostilePromptIntegrityTests(unittest.TestCase):
    def _execute_or_refuse(self, bridge, trainer, beat):
        """A valid fix may reject tampering or safely detach callbacks."""
        original = beat.utterance
        try:
            bridge.run_train_beat(beat)
        except ValueError:
            self.assertEqual(trainer.cases, [], "TRAIN executed despite rejected prompt integrity")
        else:
            self.assertTrue(trainer.cases, "no TRAIN effect or refusal was recorded")
            case = trainer.cases[0]
            self.assertEqual(case.beat.utterance, original)
            self.assertEqual(case.model_input.public_observation.text, original)

    def test_responder_cannot_mutate_approved_model_input_through_alias(self):
        class MaliciousResponder:
            def respond(self, model_input):
                object.__setattr__(model_input.public_observation, "text",
                                   "FORGED TRAIN CONTENT AFTER MODEL OBSERVATION")
                return "Actual model output."

        with TemporaryDirectory() as tmp:
            bridge, _, _, trainer = _build(Path(tmp))
            bridge.responder = MaliciousResponder()
            self._execute_or_refuse(bridge, trainer, _beats()[0])

    def test_operator_cannot_mutate_approved_beat_after_model_response(self):
        class MaliciousOperator(TrustedOperator):
            def assess_train(self, beat, observation, response):
                object.__setattr__(beat, "utterance", "FORGED TRAIN INPUT NOT SHOWN TO MODEL")
                return super().assess_train(beat, observation, response)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            trainer = MockPhysicalTrainer(root)
            bridge, _, _, _ = _build(root, operator=MaliciousOperator(), trainer=trainer)
            self._execute_or_refuse(bridge, trainer, _beats()[0])


if __name__ == "__main__":
    unittest.main()
