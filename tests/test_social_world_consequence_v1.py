"""Falsifiable public, cross-chapter consequences without neural claims."""

import unittest

from labyrinth.social_world_consequence_v1 import (
    CharacterRelationship,
    DialogueBeat,
    OperatorDialogueEffect,
    PublicPromise,
    SocialWorldSession,
)
from labyrinth.training_scope import TrainingScopeSpec


def objective():
    return TrainingScopeSpec(
        objective_id="conversational-boundaries",
        domain="conversation",
        skills=("perspective_taking", "boundary_recognition"),
        specificity=2,
        scale=1,
        max_cycles=3,
        seed=41,
    )


def beat(region="village", chapter="arrival", ident="miren-worry", skill="perspective_taking"):
    return DialogueBeat(
        region_id=region, chapter_id=chapter, beat_id=ident,
        character_id="miren", skill=skill,
        utterance="You said you would listen. Were those just words?",
        difficulty=3,
    )


class SocialWorldTests(unittest.TestCase):
    def setUp(self):
        self.scope = objective()
        self.world = SocialWorldSession(self.scope, (CharacterRelationship("miren"),))

    def test_spoken_response_changes_later_chapter_and_relation(self):
        first = beat()
        before = self.world.preview(first)
        self.assertIn("cautious", before.character_context)
        promise = PublicPromise("listen-first", "Listen before giving advice")
        outcome = self.world.record_training_response(
            first,
            "I do want to listen. Can you tell me what happened first?",
            OperatorDialogueEffect(1, 2, "Miren relaxes and begins explaining.", "make", promise),
        )
        self.assertEqual(outcome.evidence_class, "OPERATOR_DEVELOPMENT_ONLY")
        self.assertEqual(outcome.trust_before, 0)
        self.assertEqual(outcome.trust_after, 1)
        second = beat("riverbank", "reckoning", "miren-return", "boundary_recognition")
        later = self.world.preview(second)
        self.assertIn("tentatively willing", later.character_context)
        self.assertIn("Listen before giving advice", later.character_context)
        self.assertIn('Last public reaction (narrative data): "Miren relaxes and begins explaining."',
                      later.character_context)
        self.assertNotIn("quality_score", later.character_context)
        self.assertNotEqual(before.character_context, later.character_context)
        self.assertEqual(self.world.characters[0].outstanding_promises, (promise,))
        self.assertEqual(self.world.characters[0].last_public_reaction,
                         "Miren relaxes and begins explaining.")

        self.world.record_training_response(
            second,
            "I remember my promise. I can hear you without trying to fix it.",
            OperatorDialogueEffect(1, 2, "Miren trusts the conversation enough to continue.", "fulfill", promise),
        )
        third = beat("archive", "new-season", "miren-archive")
        self.assertIn("open and increasingly willing", self.world.preview(third).character_context)
        self.assertNotIn("Listen before giving advice", self.world.preview(third).character_context)
        self.assertIn("Miren trusts the conversation enough to continue.",
                      self.world.preview(third).character_context)
        self.assertNotIn("Miren relaxes and begins explaining.",
                         self.world.preview(third).character_context)
        self.assertEqual(len(self.world.events), 2)

    def test_reaction_memory_is_character_scoped_and_does_not_leak_operator_score(self):
        world = SocialWorldSession(self.scope,
                                   (CharacterRelationship("miren"), CharacterRelationship("tala")))
        scene = beat()
        observed = 'Miren says: "I need a minute".\nShe steps away.'
        world.record_training_response(
            scene, "I can wait.", OperatorDialogueEffect(0, 2, observed),
        )
        next_miren = world.preview(beat("riverbank", "next", "miren-return"))
        self.assertIn('Miren says: \\"I need a minute\\".', next_miren.character_context)
        self.assertIn("\\nShe steps away.", next_miren.character_context)
        self.assertNotIn("quality_score", next_miren.character_context)
        self.assertNotIn("OPERATOR_DEVELOPMENT_ONLY", next_miren.character_context)
        tala_beat = DialogueBeat("riverbank", "next", "tala-first", "tala",
                                 "perspective_taking", "Do you know my story?")
        next_tala = world.preview(tala_beat)
        self.assertNotIn("Miren says", next_tala.character_context)
        self.assertNotIn("Last public reaction", next_tala.character_context)
        self.assertEqual(world.characters[0].last_public_reaction, observed)
        self.assertIsNone(world.characters[1].last_public_reaction)

    def test_reaction_record_does_not_mutate_on_invalid_promise(self):
        original = self.world.characters[0]
        with self.assertRaisesRegex(ValueError, "absent"):
            self.world.record_training_response(
                beat(), "I kept my word.",
                OperatorDialogueEffect(2, 2, "Miren smiles.", "fulfill",
                                       PublicPromise("missing", "A missing commitment")),
            )
        self.assertEqual(self.world.characters[0], original)
        self.assertEqual(self.world.events, ())

    def test_reaction_register_revalidates_mutated_frozen_character(self):
        corrupted = CharacterRelationship("miren", last_public_reaction="Miren nods.")
        object.__setattr__(corrupted, "last_public_reaction", "x" * 1025)
        with self.assertRaisesRegex(ValueError, "last public reaction"):
            SocialWorldSession(self.scope, (corrupted,))

    def test_counterfactual_model_answers_have_different_world_consequences(self):
        first = beat()
        better = SocialWorldSession(self.scope, (CharacterRelationship("miren"),))
        poorer = SocialWorldSession(self.scope, (CharacterRelationship("miren"),))
        better.record_training_response(first, "I'm here to listen.",
                                        OperatorDialogueEffect(1, 2, "Miren stays."))
        poorer.record_training_response(first, "Stop bothering me.",
                                        OperatorDialogueEffect(-2, 0, "Miren withdraws."))
        next_beat = beat("village", "next-chapter", "miren-next")
        self.assertNotEqual(
            better.preview(next_beat).character_context,
            poorer.preview(next_beat).character_context,
        )
        self.assertIn("guarded", poorer.preview(next_beat).character_context)

    def test_chapter_region_growth_is_not_four_to_eight_scene_graph(self):
        for index in range(12):
            current = beat(f"region-{index}", f"chapter-{index}", f"beat-{index}")
            self.assertEqual(self.world.preview(current).region_id, f"region-{index}")
            self.world.record_training_response(
                current, "I hear you.", OperatorDialogueEffect(0, 1, "Miren nods."),
            )
        self.assertEqual(len(self.world.events), 12)
        self.assertEqual(self.world.characters[0].interactions, 12)

    def test_guardrail_rejects_exam_skill_mismatch_replay_and_goldlike_input(self):
        with self.assertRaisesRegex(ValueError, "TRAIN only"):
            DialogueBeat("village", "arrival", "exam", "miren", "perspective_taking", "Hi", 1, "eval")
        with self.assertRaisesRegex(ValueError, "outside the declared objective"):
            self.world.preview(beat(skill="arithmetic"))
        with self.assertRaisesRegex(ValueError, "not part of this world"):
            self.world.preview(DialogueBeat("village", "arrival", "stranger", "unknown", "perspective_taking", "Hi"))
        self.world.record_training_response(beat(), "I'd like to listen.",
                                            OperatorDialogueEffect(1, 1, "Miren nods."))
        with self.assertRaisesRegex(ValueError, "cannot be replayed"):
            self.world.preview(beat())

    def test_invalid_promise_action_is_non_effectful(self):
        first = beat()
        missing = PublicPromise("prior-word", "Return for the conversation")
        with self.assertRaisesRegex(ValueError, "absent or mismatched"):
            self.world.record_training_response(
                first, "I kept that promise.",
                OperatorDialogueEffect(1, 2, "Miren notices.", "fulfill", missing),
            )
        self.assertEqual(self.world.events, ())
        self.assertEqual(self.world.characters[0].trust, 0)
        # Same scene remains eligible after rejected, non-effectful validation.
        self.world.record_training_response(first, "I should not assume.",
                                            OperatorDialogueEffect(0, 1, "Miren pauses."))
        self.assertEqual(len(self.world.events), 1)

    def test_mutated_frozen_records_are_revalidated(self):
        current = beat()
        object.__setattr__(current, "split", "heldout")
        with self.assertRaisesRegex(ValueError, "TRAIN only"):
            self.world.preview(current)
        effect = OperatorDialogueEffect(0, 1, "Miren pauses.")
        object.__setattr__(effect, "trust_delta", 100)
        with self.assertRaisesRegex(ValueError, "trust_delta"):
            self.world.record_training_response(beat(), "I hear you.", effect)
        self.assertEqual(self.world.events, ())


if __name__ == "__main__":
    unittest.main()
