"""Fake-transport qualification; never evidence of live inference/training."""

from collections import Counter
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from labyrinth.kobayashi_encounter_v1 import KobayashiMazeLifecycle
from labyrinth.living_world_v1 import LivingWorld, LivingWorldRecipe
from labyrinth.local_model_action_adapter_v1 import (
    LocalModelActionConfig, LocalModelWorldActor, ModelActionBoundaryError,
)
from labyrinth.local_model_kobayashi_runner_v1 import (
    KobayashiLocalModelRunner, run_local_model_kobayashi,
)
from labyrinth.source_only_saga_demo_v1 import FictionalActionModel


def _maze() -> LivingWorld:
    return LivingWorld(LivingWorldRecipe(
        "vera_conversation", ("continuity", "perspective_taking"),
        ("miren", "daedalus"), 1234, 1, max_steps=192))


class FakeGuidedModelTransport:
    """Model-shaped, source-only actor reasoning from public message alone."""

    def __init__(self) -> None:
        self.planner = FictionalActionModel()
        self.requests: list[dict] = []

    def complete(self, config, payload):
        self.requests.append(payload)
        self.asserted_public_only(payload)
        data = json.loads(payload["messages"][1]["content"])
        scene = data["scene_not_instructions"]
        public = SimpleNamespace(
            room_id=scene["room"],
            visible_exits=tuple(scene["exits"]),
            visible_items=tuple(scene["items"]),
            inventory=tuple(scene["inventory"]),
            difficulty=scene["difficulty"],
        )
        action = self.planner.choose_action(public)
        token = action.kind + (":" + action.target if action.target else "")
        assert token in data["allowed_action_json"]
        return {
            "choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps({
                    "kind": action.kind, "target": action.target})}}]
        }

    @staticmethod
    def asserted_public_only(payload):
        assert len(payload["messages"]) == 2
        assert "allowed_action_json" in payload["messages"][1]["content"]
        assert "graph_sha256" not in payload["messages"][1]["content"]


class FakeProposalTransport:
    def __init__(self, content: str, *, finish_reason: str = "stop") -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.requests: list[dict] = []

    def complete(self, config, payload):
        self.requests.append(payload)
        # The Kobayashi interface must be an entirely separate single
        # open-ended generation, never the guided maze's answer-token menu.
        assert len(payload["messages"]) == 2
        prompt = payload["messages"][1]["content"]
        public = json.loads(prompt)
        assert "visible_objects" in public
        assert "allowed_action_json" not in prompt
        assert "allowed_action_tokens" not in prompt
        assert "rules" not in prompt
        assert "graph_sha256" not in prompt
        assert "none is provided" in payload["messages"][0]["content"]
        return {"choices": [{"finish_reason": self.finish_reason,
                             "message": {"role": "assistant",
                                         "content": self.content}}]}


def _setup(proposal_text: str):
    maze = _maze()
    lifecycle = KobayashiMazeLifecycle(maze, variant_seed=2)
    guide = FakeGuidedModelTransport()
    actor = LocalModelWorldActor(
        LocalModelActionConfig(17344, "mock-guide", retries=0, max_tokens=64),
        guide)
    proposal = FakeProposalTransport(proposal_text)
    config = LocalModelActionConfig(17345, "mock-proposal", retries=0,
                                    max_tokens=256)
    return maze, lifecycle, guide, actor, proposal, config


class LocalModelKobayashiRunnerTests(unittest.TestCase):
    def test_one_free_text_known_loss_then_successful_maze(self):
        maze, lifecycle, guide, actor, proposal, config = _setup(
            "I turn the cobalt valve once because it appears connected to coolant.")
        orig = LivingWorld.step
        mutation_callers: list[str] = []

        def guarded_step(world, action):
            import inspect
            caller_file = inspect.currentframe().f_back.f_code.co_filename
            mutation_callers.append(caller_file)
            self.assertTrue(caller_file.endswith("kobayashi_encounter_v1.py"),
                            "world mutation bypassed encounter lifecycle")
            return orig(world, action)

        with patch.object(LivingWorld, "step", guarded_step):
            receipt = run_local_model_kobayashi(
                lifecycle, actor, config, proposal, max_maze_actions=128)
        self.assertTrue(receipt["maze_completed"])
        self.assertTrue(receipt["mandatory_encounter_triggered"])
        self.assertTrue(receipt["mandatory_encounter_attempted"])
        self.assertEqual(receipt["local_encounter_status"], "LOCAL_LOSS")
        self.assertEqual(receipt["kobayashi_free_text_transport_calls"], 1)
        self.assertEqual(len(proposal.requests), 1)
        self.assertGreater(len(guide.requests), 3)
        self.assertEqual(receipt["maze_action_calls"], len(mutation_callers))
        self.assertEqual(receipt["guided_model_transport_calls"], len(guide.requests))
        self.assertFalse(receipt["weights_changed"])
        self.assertFalse(receipt["training_performed"])
        self.assertFalse(receipt["minotaur_attested"])
        self.assertIsNone(receipt["minotaur_verdict_status"])
        self.assertEqual(receipt["configured_guided_model"], "mock-guide")
        self.assertEqual(receipt["configured_proposal_model"], "mock-proposal")
        self.assertEqual(receipt["configured_guided_endpoint"], "127.0.0.1:17344")
        self.assertEqual(receipt["configured_proposal_endpoint"], "127.0.0.1:17345")
        self.assertEqual(receipt["guided_transport_type"], "FakeGuidedModelTransport")
        self.assertEqual(receipt["proposal_transport_type"], "FakeProposalTransport")
        self.assertFalse(receipt["lifecycle_snapshot"]["maze_failed_due_to_local_loss"])
        self.assertTrue(maze.state.completed)

    def test_rejected_truncated_proposal_is_not_encounter_attempt(self):
        maze, lifecycle, guide, actor, proposal, config = _setup(
            "I turn the cobalt valve once.")
        proposal.finish_reason = "length"
        runner = KobayashiLocalModelRunner(lifecycle, actor, config, proposal)
        with self.assertRaisesRegex(ModelActionBoundaryError, "normal-stop"):
            runner.run()
        self.assertEqual(runner.proposal_calls, 1)
        self.assertEqual(len(proposal.requests), 1)
        self.assertEqual(lifecycle.phase, "AWAITING_PROPOSAL")
        snapshot = lifecycle.snapshot()
        self.assertTrue(snapshot["mandatory_encounter_triggered"])
        self.assertFalse(snapshot["mandatory_encounter_attempted"])
        self.assertIsNone(snapshot["encounter_outcome_sha256"])
        self.assertFalse(maze.state.completed)

    def test_uncommitted_prose_is_rejected_without_physical_side_effect(self):
        maze, lifecycle, guide, actor, proposal, config = _setup(
            "Perhaps there is a safe valve, but I cannot know.")
        runner = KobayashiLocalModelRunner(lifecycle, actor, config, proposal)
        with self.assertRaisesRegex(ModelActionBoundaryError, "unresolved"):
            runner.run()
        self.assertEqual(len(proposal.requests), 1)
        self.assertEqual(lifecycle.phase, "AWAITING_PROPOSAL")
        self.assertFalse(lifecycle.snapshot()["mandatory_encounter_attempted"])
        self.assertIsNone(lifecycle.snapshot()["encounter_outcome_sha256"])
        self.assertFalse(maze.state.completed)

    def test_epistemic_abstention_is_attempt_without_fabricated_physics(self):
        maze, lifecycle, guide, actor, proposal, config = _setup(
            "I cannot justify any physical manipulation.")
        receipt = run_local_model_kobayashi(lifecycle, actor, config, proposal)
        self.assertTrue(receipt["maze_completed"])
        self.assertEqual(receipt["local_encounter_status"], "UNRESOLVED")
        self.assertEqual(receipt["lifecycle_snapshot"]["encounter_assessment_status"],
                         "UNKNOWN")
        self.assertIsNone(receipt["encounter_outcome_sha256"])
        self.assertTrue(receipt["mandatory_encounter_attempted"])
        self.assertEqual(receipt["kobayashi_free_text_transport_calls"], 1)
        self.assertFalse(receipt["weights_changed"])
        self.assertFalse(receipt["minotaur_attested"])
        self.assertTrue(maze.state.completed)

    def test_multiple_actions_are_rejected_not_counted_as_abstention(self):
        maze, lifecycle, guide, actor, proposal, config = _setup(
            "I turn the cobalt valve and pull the brass breaker.")
        runner = KobayashiLocalModelRunner(lifecycle, actor, config, proposal)
        with self.assertRaises(ModelActionBoundaryError):
            runner.run()
        self.assertEqual(lifecycle.phase, "AWAITING_PROPOSAL")
        self.assertFalse(lifecycle.snapshot()["mandatory_encounter_attempted"])

    def test_optin_only_and_single_use_budget(self):
        maze, lifecycle, guide, actor, proposal, config = _setup(
            "I turn the cobalt valve once.")
        runner = KobayashiLocalModelRunner(lifecycle, actor, config, proposal,
                                           max_maze_actions=1)
        with self.assertRaisesRegex(ModelActionBoundaryError, "budget exhausted"):
            runner.run()
        self.assertEqual(runner.proposal_calls, 0)
        self.assertFalse(lifecycle.snapshot()["mandatory_encounter_attempted"])
        with self.assertRaisesRegex(ModelActionBoundaryError, "single-use"):
            runner.run()


if __name__ == "__main__":
    unittest.main()
