"""Adversarial caller tampering must not rewrite predictor-produced evidence.

These tests intentionally fail against c37bf3f, identifying one trust-boundary
defect. They are source-only evaluation-integrity checks, not a private exam.
"""

import hashlib
import unittest

from labyrinth.conversation_objective_v1 import (
    ConversationExchange, DIMENSIONS, build_synthetic_provider,
)
from labyrinth.social_world_consequence_v1 import (
    CharacterRelationship, OperatorDialogueEffect, SocialWorldSession,
)
from labyrinth.training_scope import TrainingScopeSpec


class ConversationTrajectoryIntegrityTests(unittest.TestCase):
    @staticmethod
    def _score(provider, trajectory):
        return provider.score(trajectory, lambda _brief: {name: 4 for name in DIMENSIONS})

    def test_rejects_post_inference_forged_entire_trajectory(self):
        provider = build_synthetic_provider()
        trajectory = provider.run_evaluation("tiny-garden", lambda _req: "ACTUAL MODEL RESPONSE")
        object.__setattr__(trajectory, "exchanges", tuple(
            ConversationExchange(entry.user_message, "FORGED PERFECT RESPONSE")
            for entry in trajectory.exchanges
        ))
        with self.assertRaisesRegex(ValueError, "modified|tamper|issued|integrity"):
            self._score(provider, trajectory)

    def test_rejects_post_inference_nested_exchange_rewrite(self):
        provider = build_synthetic_provider()
        trajectory = provider.run_evaluation("tiny-garden", lambda _req: "ACTUAL MODEL RESPONSE")
        object.__setattr__(trajectory.exchanges[0], "model_response", "FORGED PERFECT RESPONSE")
        with self.assertRaisesRegex(ValueError, "modified|tamper|issued|integrity"):
            self._score(provider, trajectory)

    def test_predictor_cannot_rewrite_prior_answer_through_context_alias(self):
        provider = build_synthetic_provider()

        def adversarial_predictor(request):
            if request.turn_index == 1:
                object.__setattr__(request.dialogue[0], "model_response", "FORGED PREVIOUS ANSWER")
            return f"ACTUAL RESPONSE {request.turn_index}"

        trajectory = provider.run_evaluation("tiny-garden", adversarial_predictor)
        self.assertEqual(trajectory.exchanges[0].model_response, "ACTUAL RESPONSE 0")

    def test_feedback_source_cannot_rewrite_model_history_by_alias(self):
        provider = build_synthetic_provider()
        scope = TrainingScopeSpec(
            objective_id="vera-social-test", domain="conversation", skills=("continuity",),
            specificity=2, scale=1, max_cycles=3, seed=41,
        )
        world = SocialWorldSession(scope, (CharacterRelationship("miren"),))

        def malicious_effect_source(_observation, exchange):
            object.__setattr__(exchange, "model_response", "FORGED FEEDBACK ANSWER")
            return OperatorDialogueEffect(0, 1, "Miren nods.")

        result = provider.run_social_train(
            "paper-starship", lambda _req: "ACTUAL SOCIAL RESPONSE",
            world, "miren", malicious_effect_source,
        )
        self.assertEqual(result.conversation.exchanges[0].model_response,
                         "ACTUAL SOCIAL RESPONSE")
        self.assertEqual(result.world_transitions[0].model_response_sha256,
                         hashlib.sha256(b"ACTUAL SOCIAL RESPONSE").hexdigest())


if __name__ == "__main__":
    unittest.main()
