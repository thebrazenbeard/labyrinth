"""Theseus policy is genuinely in the loop, but it never grades its own exam."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from labyrinth.theseus_agent_loop_v1 import drive_theseus
from tests import test_model_saga_runtime as saga_fixtures


class WalkingPolicy:
    def __init__(self, preferred: str):
        self.preferred = preferred
        self.seen = []

    def choose_action(self, context):
        self.seen.append(context)
        options = context.available_actions
        for action in ("face", "train", "forward", self.preferred):
            if action in options:
                return "ACTION: " + action
        return "ACTION: look"


class TheseusAgentLoopTests(unittest.TestCase):
    @staticmethod
    def make_session(root):
        return saga_fixtures.SagaSessionTests().make_session(root)[0]

    def test_model_actions_causally_select_branch_and_complete_two_cycles(self):
        selected_prompts = []
        for preferred in ("left", "right"):
            with TemporaryDirectory() as tmp:
                session = self.make_session(Path(tmp))
                policy = WalkingPolicy(preferred)
                receipt = drive_theseus(session, policy, max_decisions=24)
                self.assertEqual(receipt.status, "DEVELOPMENT_EPISODE_ENDED")
                self.assertEqual(receipt.qualification, "NOT_ESTABLISHED")
                self.assertEqual(receipt.maze_cycle_index, 1)
                self.assertEqual(receipt.development_status,
                                 "PASS")
                self.assertEqual(sum(step.effect_event == "TRAIN_COMPLETED"
                                     for step in receipt.decisions), 4)
                self.assertEqual(sum(step.effect_event == "MINOTAUR_FAIL"
                                     for step in receipt.decisions), 1)
                self.assertEqual(sum(step.effect_event == "MINOTAUR_PASS"
                                     for step in receipt.decisions), 1)
                self.assertEqual(len(session.training_sparks), 4)
                self.assertEqual(policy.seen[0].available_actions,
                                 ("look", "left", "right"))
                self.assertTrue(all("ANSWER:" not in ctx.observation
                                    for ctx in policy.seen))
                self.assertTrue(all("case_id" not in repr(ctx) for ctx in policy.seen))
                selected_prompts.append(session.runner.observations[0].prompt_sha256)
                self.assertEqual(receipt.decisions[0].action, preferred)
                self.assertEqual(receipt.final_checkpoint_sha256,
                                 session.checkpoint.weights_sha256)
        self.assertNotEqual(*selected_prompts)

    def test_invalid_model_output_must_not_take_an_action_or_fallback_to_train(self):
        for invalid in ("train", "ACTION: train", "ACTION: left\nACTION: train",
                        "ACTION: right and train", "ACTION: delete", "ACTION:  look"):
            with self.subTest(invalid=invalid), TemporaryDirectory() as tmp:
                session = self.make_session(Path(tmp))

                class Malformed:
                    def choose_action(self, context):
                        return invalid

                receipt = drive_theseus(session, Malformed())
                self.assertEqual(receipt.status, "HALTED_INVALID_POLICY_OUTPUT")
                self.assertEqual(receipt.decisions, ())
                self.assertEqual(session.action_count, 0)
                self.assertEqual(len(session.training_sparks), 0)

    def test_decision_budget_is_not_reported_as_success(self):
        with TemporaryDirectory() as tmp:
            session = self.make_session(Path(tmp))
            receipt = drive_theseus(session, WalkingPolicy("left"), max_decisions=1)
            self.assertEqual(receipt.status, "HALTED_DECISION_BUDGET")
            self.assertEqual(len(receipt.decisions), 1)
            self.assertEqual(len(session.training_sparks), 0)
            self.assertEqual(receipt.qualification, "NOT_ESTABLISHED")

    def test_partially_failed_train_propagates_and_never_yields_success_receipt(self):
        with TemporaryDirectory() as tmp:
            session = self.make_session(Path(tmp))

            def indeterminate_training(_case):
                raise RuntimeError("external trainer might already have changed weights")

            session.learner.train_supervised = indeterminate_training
            with self.assertRaisesRegex(RuntimeError, "might already"):
                drive_theseus(session, WalkingPolicy("left"), max_decisions=24)
            self.assertTrue(session.indeterminate)
            self.assertEqual(session.observe().available_actions, ())

    def test_invalid_policy_and_budget_types_are_rejected(self):
        with TemporaryDirectory() as tmp:
            session = self.make_session(Path(tmp))
            with self.assertRaises(ValueError):
                drive_theseus(session, object())
            with self.assertRaises(ValueError):
                drive_theseus(session, WalkingPolicy("left"), max_decisions=True)


if __name__ == "__main__":
    unittest.main()
