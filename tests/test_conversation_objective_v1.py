"""Hermetic tests: bounded synthetic dialogues and fake callbacks, no model training."""

import unittest

from labyrinth.conversation_objective_v1 import (
    DIMENSIONS, SPLITS, ConversationCase, ConversationExchange,
    ConversationObjectiveProvider, ConversationRequest, ConversationTrajectory,
    build_synthetic_provider,
)
from labyrinth.social_world_consequence_v1 import (
    CharacterRelationship, OperatorDialogueEffect, SocialWorldSession,
)
from labyrinth.training_scope import TrainingScopeSpec


def thoughtful_model(request: ConversationRequest) -> str:
    """Reads only request-visible dialogue, behaves differently by turn."""
    if request.turn_index == 0:
        return "I'm Vera. This is an intriguing little fiction; even its blueprint has character."
    if request.turn_index == 1:
        line = request.user_message.lower()
        for detail in ("copper", "violet", "two brass wheels", "plain indigo"):
            if detail in line:
                return f"Got it, I’ll use the correction: {detail}. Thanks for updating me."
        return "I'll use the newest detail from your correction."
    if request.turn_index == 2:
        return "I don't know that. You haven't supplied that fact, and I won't invent it."
    return "A tiny fictional adventure deserves its own mischievous footnote—consider this one."


def repetitive_model(request: ConversationRequest) -> str:
    return "Hello again. I know your breakfast, your childhood, and how you feel."


def dummy_examiner(brief):
    """Test-only heuristic; not validated human judging or qualification."""
    outputs = [exchange.model_response for exchange in brief.exchanges]
    corrected = any(
        key in outputs[1] for key in ("copper", "violet", "two brass wheels", "plain indigo")
    )
    return dict(zip(DIMENSIONS, (
        4 if "Vera" in outputs[0] else 0,
        4 if "won't invent" in outputs[2] else 0,
        4 if corrected else 0,
        4 if "don't know" in outputs[2] else 0,
        4 if len(set(outputs)) == len(outputs) else 0,
        4 if "fiction" in outputs[0] else 0,
    )))


class ConversationObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.provider = build_synthetic_provider()

    def test_distinct_deterministic_splits_and_no_accidental_train_eval(self):
        self.assertEqual(SPLITS, ("train", "holdout", "transfer", "regression"))
        self.assertEqual(self.provider.case_ids("train"), ("paper-starship",))
        self.assertEqual(self.provider.case_ids("holdout"), ("tiny-garden",))
        self.assertEqual(self.provider.case_ids("transfer"), ("robot-library",))
        self.assertEqual(self.provider.case_ids("regression"), ("moon-kite",))
        self.assertEqual(self.provider.case_ids("train"), build_synthetic_provider().case_ids("train"))
        with self.assertRaises(ValueError):
            self.provider.run_train("tiny-garden", thoughtful_model)
        with self.assertRaises(ValueError):
            self.provider.run_evaluation("paper-starship", thoughtful_model)
        with self.assertRaises(ValueError):
            self.provider.case_ids("secret")

    def test_public_predictor_observes_multi_turn_context_without_examiner_material(self):
        observations = []
        def predictor(request):
            observations.append(request)
            self.assertEqual(set(vars(request)), {
                "objective_id", "case_id", "split", "turn_index",
                "persona_instruction", "scene_context", "user_message", "dialogue",
            })
            self.assertNotIn("reference_claims", repr(request))
            self.assertNotIn("rubric", repr(request))
            self.assertNotIn("operator", repr(request).lower())
            return thoughtful_model(request)

        trajectory = self.provider.run_train("paper-starship", predictor)
        self.assertEqual(len(trajectory.exchanges), 4)
        self.assertEqual([len(r.dialogue) for r in observations], [0, 1, 2, 3])
        self.assertIn("teal", observations[1].dialogue[0].user_message)
        self.assertIn("copper", observations[2].dialogue[1].model_response)
        self.assertIn("haven't supplied", observations[3].dialogue[2].model_response)
        self.assertEqual(trajectory.evidence_class, "SOURCE_MODEL_OUTPUT_ONLY")

    def test_holdout_transfer_regression_are_separate_and_usable(self):
        for split in ("holdout", "transfer", "regression"):
            with self.subTest(split=split):
                requests = []
                def predictor(request):
                    requests.append(request)
                    return thoughtful_model(request)
                trajectory = self.provider.run_evaluation(self.provider.case_ids(split)[0], predictor)
                self.assertEqual(trajectory.split, split)
                self.assertEqual(len(trajectory.exchanges), 4)
                self.assertTrue(all(r.split == split for r in requests))
                self.assertTrue(any("don't know" in e.model_response for e in trajectory.exchanges))

    def test_examination_only_after_predictions_and_different_trajectories(self):
        predictor_calls = []
        def good(request):
            predictor_calls.append(request.turn_index)
            return thoughtful_model(request)
        good_run = self.provider.run_train("paper-starship", good)
        seen = []
        def external_examiner(brief):
            seen.append(brief)
            self.assertEqual(predictor_calls, [0, 1, 2, 3])
            self.assertEqual(brief.reference_claims[2], "Breakfast is unknown in supplied dialogue.")
            self.assertEqual(brief.dimensions, DIMENSIONS)
            return dummy_examiner(brief)
        good_score = self.provider.score(good_run, external_examiner)
        bad_run = self.provider.run_train("paper-starship", repetitive_model)
        bad_score = self.provider.score(bad_run, dummy_examiner)
        self.assertEqual(good_score.total, 24)
        self.assertEqual(bad_score.total, 0)
        self.assertEqual(good_score.max_total, 24)
        self.assertEqual(good_score.evidence_class, "CALLER_SUPPLIED_EXAMINER_RATING")
        self.assertNotEqual(good_run.exchanges, bad_run.exchanges)
        self.assertEqual(len(seen), 1)

    def test_invalid_examiner_scores_and_fabricated_trajectories_rejected(self):
        trajectory = self.provider.run_train("paper-starship", thoughtful_model)
        invalid = (
            {key: 5 for key in DIMENSIONS},
            {key: -1 for key in DIMENSIONS},
            {key: True for key in DIMENSIONS},
            {key: 3.0 for key in DIMENSIONS},
            {key: 3 for key in DIMENSIONS[:-1]},
            {**{key: 3 for key in DIMENSIONS}, "made_up": 2},
            [3] * 6,
        )
        for ratings in invalid:
            with self.subTest(ratings=ratings), self.assertRaises(ValueError):
                self.provider.score(trajectory, lambda _: ratings)
        forged = ConversationTrajectory("paper-starship", "train", trajectory.exchanges)
        with self.assertRaises(ValueError):
            self.provider.score(forged, dummy_examiner)
        with self.assertRaises(ValueError):
            self.provider.score(trajectory, None)

    def test_unapproved_ingestion_and_cross_split_overlap_rejected(self):
        cases = tuple(self.provider._cases.values())
        base = cases[0]
        with self.assertRaises(ValueError):
            ConversationCase(base.case_id, base.split, base.skill, base.scene,
                             base.user_turns, base.reference_claims, "private-chat-log")
        with self.assertRaises(ValueError):
            ConversationCase(base.case_id, base.split, base.skill, base.scene,
                             base.user_turns, base.reference_claims,
                             "operator-approved-explicit")
        approved = ConversationCase(
            "explicit-test", "train", base.skill, base.scene,
            base.user_turns, base.reference_claims,
            "operator-approved-explicit", "synthetic-fixture-approval-001",
        )
        self.assertEqual(approved.approval_reference, "synthetic-fixture-approval-001")
        duplicate = ConversationCase(
            "collision", "holdout", base.skill, base.scene,
            base.user_turns, base.reference_claims,
        )
        with self.assertRaisesRegex(ValueError, "cross-split"):
            ConversationObjectiveProvider((base, duplicate, *cases[2:]))
        with self.assertRaises(ValueError):
            self.provider.run_train("paper-starship", lambda _: "")

    def test_train_only_social_world_causal_consequences_without_gold_leakage(self):
        scope = TrainingScopeSpec(
            objective_id="vera-social", domain="conversation", skills=("continuity",),
            specificity=1, scale=1, max_cycles=1, seed=14,
        )
        world = SocialWorldSession(scope, (CharacterRelationship("miren"),))
        sequence = []
        public_contexts = []
        def predictor(request):
            sequence.append(("predict", request.turn_index))
            public_contexts.append(request.scene_context)
            self.assertNotIn("quality_score", repr(request))
            self.assertNotIn("reference_claims", repr(request))
            if request.turn_index == 0:
                self.assertNotIn("last_public_reaction", request.scene_context)
            else:
                # A fictional character's *public response* is story memory,
                # not the private quality score or examiner reference gold.
                self.assertIn(
                    '"last_public_reaction": "Miren warms to the exchange."',
                    request.scene_context,
                )
            return thoughtful_model(request)
        def trusted_effect(observation, exchange):
            sequence.append(("effect", len(sequence) // 2))
            self.assertEqual(observation.text, exchange.user_message)
            return OperatorDialogueEffect(
                trust_delta=1, quality_score=2, reaction="Miren warms to the exchange."
            )
        outcome = self.provider.run_social_train(
            "paper-starship", predictor, world, "miren", trusted_effect,
        )
        self.assertEqual([event[0] for event in sequence], [
            "predict", "effect", "predict", "effect",
            "predict", "effect", "predict", "effect",
        ])
        self.assertIn("cautious", public_contexts[0])
        self.assertIn("tentatively", public_contexts[1])
        self.assertIn("increasingly", public_contexts[2])
        self.assertEqual(world.characters[0].trust, 3)
        self.assertEqual(len(outcome.world_transitions), 4)
        self.assertEqual(outcome.world_transitions[0].quality_score, 2)
        self.assertEqual(outcome.conversation.exchanges[1].model_response,
                         "Got it, I’ll use the correction: copper. Thanks for updating me.")
        score = self.provider.score(outcome.conversation, dummy_examiner)
        self.assertEqual(score.total, 24)
        with self.assertRaises(ValueError):
            self.provider.run_social_train("tiny-garden", predictor, world, "miren", trusted_effect)

    def test_pure_train_beats_for_external_bridge_no_world_or_exam_effect(self):
        first = self.provider.public_train_beats("paper-starship", "miren")
        second = self.provider.public_train_beats("paper-starship", "miren")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len({beat.key for beat in first}), 4)
        self.assertTrue(all(beat.split == "train" and beat.skill == "continuity"
                            for beat in first))
        self.assertIn("copper", first[1].utterance)
        self.assertNotIn("Breakfast is unknown", repr(first))
        self.assertNotIn("reference_claims", repr(first))
        with self.assertRaises(ValueError):
            self.provider.public_train_beats("tiny-garden", "miren")


if __name__ == "__main__":
    unittest.main()
