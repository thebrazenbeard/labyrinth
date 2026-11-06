"""A source-approved model-proposed scene must alter a later social journey."""

from pathlib import Path
import unittest

from labyrinth.daedalus_social_director_v1 import DaedalusSocialPlanner, SocialScenePolicy
from labyrinth.social_world_consequence_v1 import (
    CharacterRelationship, DialogueBeat, OperatorDialogueEffect, SocialWorldSession,
)
from labyrinth.training_scope import TrainingScopeSpec


def world():
    scope = TrainingScopeSpec("social-saga", "conversation",
                              ("perspective_taking", "boundary_recognition"), 2, 1, 3, 19)
    return SocialWorldSession(scope, (CharacterRelationship("miren"),))


class ResponsiveDirector:
    def __init__(self):
        self.requests = []

    def propose_scene(self, context):
        self.requests.append(context)
        skill = context.weak_skills[0] if context.weak_skills else "perspective_taking"
        return DialogueBeat(f"region-{context.chapter_index}",
                            f"chapter-{context.chapter_index}",
                            f"encounter-{context.chapter_index}",
                            context.character_ids[0], skill,
                            "You said you would return. What should I believe?", 3)


class DaedalusScenePlannerTests(unittest.TestCase):
    def test_weakness_feedback_changes_later_authorized_train_scene(self):
        state = world()
        director = ResponsiveDirector()
        planner = DaedalusSocialPlanner(state, director, SocialScenePolicy(2, 16))
        first = planner.propose_next()
        self.assertEqual(first.skill, "perspective_taking")
        self.assertEqual(director.requests[0].prior_public_outcomes, ())
        effect = state.record_training_response(
            first, "Please tell me more.", OperatorDialogueEffect(-1, 0, "Miren turns away."),
        )
        planner.record_completed(effect)
        second = planner.propose_next()
        self.assertEqual(second.skill, "perspective_taking")
        self.assertEqual(director.requests[-1].weak_skills, ("perspective_taking",))
        self.assertEqual(director.requests[-1].prior_public_outcomes,
                         (("perspective_taking", 0),))
        self.assertNotEqual(first.key, second.key)
        self.assertEqual(second.split, "train")
        self.assertNotIn("expected", repr(director.requests[-1]).lower())

    def test_chapters_extend_beyond_fixed_four_to_eight_node_graph(self):
        state = world()
        planner = DaedalusSocialPlanner(state, ResponsiveDirector(), SocialScenePolicy(1, 16))
        for index in range(12):
            beat = planner.propose_next()
            self.assertEqual(beat.chapter_id, f"chapter-{index}")
            transition = state.record_training_response(
                beat, "I can hear you.", OperatorDialogueEffect(1 if index == 0 else 0, 1, "Miren nods."),
            )
            planner.record_completed(transition)
        self.assertEqual(planner.completed_scenes, 12)
        self.assertEqual(len(state.events), 12)

    def test_never_accept_unscoped_or_self_examined_scene(self):
        class Invalid:
            def propose_scene(self, context):
                return DialogueBeat("village", "chapter", "beat", "miren", "arithmetic", "Hi")

        with self.assertRaisesRegex(ValueError, "outside the declared objective"):
            DaedalusSocialPlanner(world(), Invalid()).propose_next()

        class NotExact:
            def propose_scene(self, context):
                return {"beat_id": "fake", "expected": "pass", "split": "heldout"}

        with self.assertRaisesRegex(ValueError, "exact TRAIN dialogue"):
            DaedalusSocialPlanner(world(), NotExact()).propose_next()

    def test_pending_scene_cannot_be_replayed_or_accepted_without_world_effect(self):
        state = world()
        planner = DaedalusSocialPlanner(state, ResponsiveDirector())
        first = planner.propose_next()
        with self.assertRaisesRegex(ValueError, "pending"):
            planner.propose_next()
        with self.assertRaisesRegex(ValueError, "world transition"):
            planner.record_completed(object())
        event = state.record_training_response(
            first, "I will listen.", OperatorDialogueEffect(0, 1, "Miren waits."),
        )
        planner.record_completed(event)
        with self.assertRaisesRegex(ValueError, "no pending"):
            planner.record_completed(event)


if __name__ == "__main__":
    unittest.main()
