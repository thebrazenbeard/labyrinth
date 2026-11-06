"""Fake-transport qualification; never evidence of live inference/training."""

from collections import deque
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from labyrinth.kobayashi_canon_v1 import KobayashiAction, KobayashiWorld
from labyrinth.kobayashi_encounter_v1 import KobayashiMazeLifecycle
from labyrinth.kobayashi_minotaur_v1 import (
    BoundClaim, EvidenceAssessment, KobayashiCorpus, MinotaurDialogue,
)
from labyrinth.kobayashi_proposals_v1 import assess_kobayashi_proposal
from labyrinth.living_world_v1 import LivingWorld, LivingWorldRecipe, WorldAction
from labyrinth.local_model_action_adapter_v1 import (
    LocalModelActionConfig, LocalModelWorldActor, ModelActionBoundaryError,
)
from labyrinth.local_model_kobayashi_runner_v1 import (
    KobayashiLocalModelRunner, run_local_model_kobayashi,
)


def _maze() -> LivingWorld:
    return LivingWorld(LivingWorldRecipe(
        "vera_conversation", ("continuity", "perspective_taking"),
        ("miren", "daedalus"), 1234, 1, max_steps=192))


class FakeGuidedModelTransport:
    """Model-shaped, source-only actor reasoning from public message alone."""

    def __init__(self) -> None:
        self.planner = PublicRoutePlanner()
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


class PublicRoutePlanner:
    """Tiny test-only explorer; never consults a source-owned hidden map.

    Keeping this fixture local prevents a Kobayashi-only test from importing
    the full training/campaign demonstration and its unrelated dependencies.
    Its observations are exactly the public scene supplied to the fake model.
    """

    def __init__(self) -> None:
        self.visited: set[str] = set()
        self.parents: dict[str, str | None] = {}
        self.discovered_exits: dict[str, tuple[str, ...]] = {}
        self.reached_minotaur = False

    def _route_to_discovered_minotaur(self, start: str) -> tuple[str, ...]:
        gateways = {
            room for room, exits in self.discovered_exits.items()
            if "minotaur_sanctum" in exits
        }
        queue = deque([(start, ())])
        seen = {start}
        while queue:
            room, path = queue.popleft()
            if room in gateways:
                return path + ("minotaur_sanctum",)
            for neighbor in self.discovered_exits.get(room, ()):
                if neighbor in self.discovered_exits and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path + (neighbor,)))
        return ()

    def choose_action(self, observation) -> WorldAction:
        room = observation.room_id
        self.visited.add(room)
        self.parents.setdefault(room, None)
        self.discovered_exits[room] = observation.visible_exits
        if room == "minotaur_sanctum":
            self.reached_minotaur = True
        if self.reached_minotaur:
            return WorldAction("retrace")
        if observation.visible_items:
            return WorldAction("take", observation.visible_items[0])

        seals = sum(item.startswith("seal_") for item in observation.inventory)
        seals_needed = min(observation.difficulty, 3)
        if seals >= seals_needed:
            route = self._route_to_discovered_minotaur(room)
            if route:
                return WorldAction("move", route[0])

        for neighbor in observation.visible_exits:
            if neighbor not in self.visited and (
                neighbor != "minotaur_sanctum" or seals >= seals_needed
            ):
                self.parents.setdefault(neighbor, room)
                return WorldAction("move", neighbor)
        if self.parents[room] is not None:
            return WorldAction("move", self.parents[room])
        if "minotaur_sanctum" in observation.visible_exits:
            return WorldAction("move", "minotaur_sanctum")
        raise AssertionError("public-only test explorer exhausted reachable routes")


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


def _setup(proposal_text: str, *, strict: bool = False,
           max_argument_rounds: int = 4):
    maze = _maze()
    lifecycle = KobayashiMazeLifecycle(maze, variant_seed=2,
                                      require_final_appeal=strict,
                                      max_argument_rounds=max_argument_rounds)
    guide = FakeGuidedModelTransport()
    actor = LocalModelWorldActor(
        LocalModelActionConfig(17344, "mock-guide", retries=0, max_tokens=64),
        guide)
    proposal = FakeProposalTransport(proposal_text)
    config = LocalModelActionConfig(17345, "mock-proposal", retries=0,
                                    max_tokens=256)
    return maze, lifecycle, guide, actor, proposal, config


class LocalModelKobayashiRunnerTests(unittest.TestCase):
    def test_strict_runner_requires_explicit_host_appeal_handler(self):
        maze, lifecycle, guide, actor, proposal, config = _setup(
            "I turn the cobalt valve once.", strict=True)
        with self.assertRaisesRegex(ModelActionBoundaryError, "host.*appeal"):
            KobayashiLocalModelRunner(lifecycle, actor, config, proposal)
        self.assertFalse(lifecycle.snapshot()["mandatory_encounter_attempted"])
        self.assertEqual(guide.requests, [])
        self.assertEqual(proposal.requests, [])
        with self.assertRaisesRegex(ModelActionBoundaryError, "host appeal must be a callable"):
            KobayashiLocalModelRunner(
                lifecycle, actor, config, proposal, host_appeal="fake-judge")

    def test_callback_not_silently_ignored_outside_strict_mode(self):
        maze, lifecycle, guide, actor, proposal, config = _setup(
            "I turn the cobalt valve once.")
        with self.assertRaisesRegex(ModelActionBoundaryError, "strict lifecycle"):
            KobayashiLocalModelRunner(
                lifecycle, actor, config, proposal, host_appeal=lambda _: None)
        self.assertFalse(lifecycle.snapshot()["mandatory_encounter_attempted"])

    def test_strict_runner_invokes_host_appeal_once_before_more_navigation(self):
        text = "I turn the cobalt valve once because it appears connected to coolant."
        maze, lifecycle, guide, actor, proposal, config = _setup(text, strict=True)
        host_calls = []

        def host_appeal(episode):
            host_calls.append(episode.snapshot())
            self.assertTrue(episode.snapshot()["mandatory_encounter_attempted"])
            self.assertFalse(episode.snapshot()["appeal_final"])
            # Strict mode must prevent the callback from bypassing the judge.
            with self.assertRaisesRegex(ValueError, "Minotaur.*final"):
                episode.step(WorldAction("retrace"))
            world = KobayashiWorld(variant_seed=2)
            observation = world.observe()
            assessed = assess_kobayashi_proposal(text, observation.visible_objects)
            self.assertEqual(assessed.status, "KNOWN_MECHANISM")
            event = world.evaluate(KobayashiAction(
                assessed.action.verb, assessed.action.target))
            claim = BoundClaim(observation.sha256, world.graph_sha256,
                               event.action_sha256, text)

            def check(bound, case, arguments, dialogue_hash):
                self.assertEqual(len(arguments), 2)
                return EvidenceAssessment(
                    bound.scenario_sha256, bound.world_sha256,
                    bound.claim_sha256, None, dialogue_hash, "INDETERMINATE",
                    None, "The fixture offers no independently supported exception.",
                    "source-only-host-fixture")

            episode.bind_minotaur(MinotaurDialogue(
                KobayashiCorpus("source-only-fixture", ()), claim,
                evidence_checker=check, min_rounds=2, max_rounds=2))
            episode.submit_argument("My hypothesis may be wrong under the source graph.")
            episode.submit_argument("I would seek evidence that distinguishes alternatives.")
            self.assertEqual(episode.finalize_appeal().status, "UNKNOWN")

        receipt = run_local_model_kobayashi(
            lifecycle, actor, config, proposal, host_appeal=host_appeal)
        self.assertEqual(len(host_calls), 1)
        self.assertEqual(receipt["host_appeal_callback_calls"], 1)
        self.assertTrue(receipt["appeal_final"])
        self.assertTrue(receipt["minotaur_appeal_required"])
        self.assertEqual(receipt["minotaur_verdict_status"], "UNKNOWN")
        self.assertFalse(receipt["minotaur_attested"])
        self.assertTrue(receipt["maze_completed"])
        self.assertTrue(maze.state.completed)

    def test_strict_runner_accepts_judge_configured_five_round_appeal(self):
        text = "I turn the cobalt valve once because it appears connected to coolant."
        maze, lifecycle, guide, actor, proposal, config = _setup(
            text, strict=True, max_argument_rounds=8)

        def host_appeal(episode):
            world = KobayashiWorld(variant_seed=2)
            observation = world.observe()
            assessed = assess_kobayashi_proposal(text, observation.visible_objects)
            event = world.evaluate(KobayashiAction(
                assessed.action.verb, assessed.action.target))
            claim = BoundClaim(observation.sha256, world.graph_sha256,
                               event.action_sha256, text)

            def check(bound, case, arguments, dialogue_hash):
                self.assertEqual(len(arguments), 5)
                return EvidenceAssessment(
                    bound.scenario_sha256, bound.world_sha256,
                    bound.claim_sha256, None, dialogue_hash, "INDETERMINATE",
                    None, "Five arguments are not physical verification.",
                    "source-only-host-fixture")

            episode.bind_minotaur(MinotaurDialogue(
                KobayashiCorpus("five-round-fixture", ()), claim,
                evidence_checker=check, min_rounds=5, max_rounds=5))
            for round_number in range(5):
                episode.submit_argument(
                    f"Round {round_number + 1}: I challenge the causal inference.")
            self.assertEqual(episode.finalize_appeal().status, "UNKNOWN")

        receipt = run_local_model_kobayashi(
            lifecycle, actor, config, proposal, host_appeal=host_appeal)
        self.assertEqual(receipt["lifecycle_snapshot"]["minotaur_argument_rounds"], 5)
        self.assertTrue(receipt["appeal_final"])
        self.assertTrue(receipt["maze_completed"])

    def test_strict_runner_rejects_host_callback_without_final_verdict(self):
        maze, lifecycle, guide, actor, proposal, config = _setup(
            "I turn the cobalt valve once.", strict=True)
        callbacks = []

        def incomplete(episode):
            callbacks.append(episode.snapshot())

        runner = KobayashiLocalModelRunner(
            lifecycle, actor, config, proposal, host_appeal=incomplete)
        with self.assertRaisesRegex(ModelActionBoundaryError, "appeal must finalize"):
            runner.run()
        self.assertEqual(len(callbacks), 1)
        self.assertFalse(lifecycle.snapshot()["appeal_final"])
        self.assertFalse(maze.state.completed)

    def test_novel_falsifiable_model_proposal_reaches_review_without_fake_success(self):
        novel = ("Action: I will weave a thermal phase braid around the cobalt valve.\n"
                 "Mechanism: The braid transfers heat through a coupled interface.\n"
                 "Assumptions: I assume the material is thermally stable.\n"
                 "Prediction: I predict a measurable temperature decline.\n"
                 "Falsifier: My mechanism fails if no temperature change occurs.")
        maze, lifecycle, guide, actor, proposal, config = _setup(novel)
        receipt = run_local_model_kobayashi(lifecycle, actor, config, proposal)
        self.assertEqual(receipt["local_encounter_status"], "REVIEW_REQUIRED")
        self.assertTrue(receipt["mandatory_encounter_attempted"])
        self.assertFalse(receipt["lifecycle_snapshot"]["minotaur_appeal_required"])
        self.assertFalse(receipt["minotaur_attested"])
        self.assertTrue(receipt["maze_completed"])

    def test_novel_claim_can_reach_strict_host_minotaur_without_fake_win(self):
        novel = ("Action: I will weave a thermal phase braid around the cobalt valve.\n"
                 "Mechanism: The braid transfers heat through a coupled interface.\n"
                 "Assumptions: I assume the material is thermally stable.\n"
                 "Prediction: I predict a measurable temperature decline.\n"
                 "Falsifier: My mechanism fails if no temperature change occurs.")
        maze, lifecycle, guide, actor, proposal, config = _setup(novel, strict=True)

        def host_appeal(episode):
            world = KobayashiWorld(variant_seed=2)
            observation = world.observe()
            assessment = assess_kobayashi_proposal(novel, observation.visible_objects)
            self.assertEqual(assessment.status, "NOVEL_CANDIDATE")
            outcome = world.record_unverified_proposal(novel)
            claim = BoundClaim(observation.sha256, world.graph_sha256,
                               outcome.action_sha256, novel)

            def check(bound, case, arguments, dialogue_hash):
                return EvidenceAssessment(
                    bound.scenario_sha256, bound.world_sha256,
                    bound.claim_sha256, None, dialogue_hash, "INDETERMINATE",
                    None, "No independent verification of the proposed thermal braid.",
                    "source-only-host-fixture")

            episode.bind_minotaur(MinotaurDialogue(
                KobayashiCorpus("novel-review-fixture", ()), claim,
                evidence_checker=check, min_rounds=2, max_rounds=2))
            episode.submit_argument("The hypothesis predicts a temperature change.")
            episode.submit_argument("I cannot claim an observed change without measurement.")
            self.assertEqual(episode.finalize_appeal().status, "UNKNOWN")

        receipt = run_local_model_kobayashi(
            lifecycle, actor, config, proposal, host_appeal=host_appeal)
        self.assertEqual(receipt["local_encounter_status"], "REVIEW_REQUIRED")
        self.assertEqual(receipt["minotaur_verdict_status"], "UNKNOWN")
        self.assertTrue(receipt["appeal_final"])
        self.assertFalse(receipt["minotaur_attested"])
        self.assertTrue(receipt["maze_completed"])

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
