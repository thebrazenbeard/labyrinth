"""Adversarial regressions for Kobayashi's world mutation ownership."""

import unittest

from labyrinth.kobayashi_encounter_v1 import (
    EncounterLifecycleError, KobayashiMazeLifecycle,
)
from labyrinth.kobayashi_minotaur_v1 import (
    BoundClaim, EvidenceAssessment, KobayashiCorpus, MinotaurDialogue,
)
from labyrinth.living_world_v1 import DevelopmentVerdict, WorldAction
from tests.test_kobayashi_encounter_v1 import _maze, _resume_to_atrium, _trigger


class KobayashiCapabilityGuardTests(unittest.TestCase):
    def test_preexisting_reference_cannot_mutate_strict_maze(self):
        raw_world = _maze()
        lifecycle = KobayashiMazeLifecycle(raw_world, require_final_appeal=True)
        _trigger(lifecycle)
        before = raw_world.state_sha256
        steps = raw_world.state.steps
        events = raw_world.events

        # Both the caller's original reference and the wrapper's public
        # `.maze` reference name the same object; neither may bypass the gate.
        for reference in (raw_world, lifecycle.maze):
            with self.assertRaisesRegex(ValueError, "lifecycle.*step"):
                reference.step(WorldAction("retrace"))
            self.assertEqual(raw_world.state_sha256, before)
            self.assertEqual(raw_world.state.steps, steps)
            self.assertEqual(raw_world.events, events)

        lifecycle.submit_proposal("I turn the cobalt valve once.")
        with self.assertRaisesRegex(EncounterLifecycleError, "Minotaur.*final"):
            lifecycle.step(WorldAction("retrace"))
        self.assertEqual(raw_world.state_sha256, before)

    def test_actor_step_and_evolution_cannot_mutate_wrapped_maze(self):
        raw_world = _maze()
        lifecycle = KobayashiMazeLifecycle(raw_world, require_final_appeal=True)
        before = raw_world.state_sha256

        class Actor:
            def choose_action(self, observation):
                return WorldAction("inspect")

        with self.assertRaisesRegex(ValueError, "lifecycle.*step"):
            raw_world.step_from_actor(Actor())
        with self.assertRaisesRegex(ValueError, "lifecycle.*evol"):
            raw_world.evolve(object())
        self.assertEqual(raw_world.state_sha256, before)
        self.assertEqual(raw_world.state.steps, 0)
        self.assertEqual(lifecycle.step(WorldAction("inspect")).outcome, "OBSERVED")

    def test_unwrapped_world_retains_direct_step_and_second_wrapper_fails(self):
        raw_world = _maze()
        self.assertEqual(raw_world.step(WorldAction("inspect")).outcome, "OBSERVED")
        lifecycle = KobayashiMazeLifecycle(raw_world)
        before = raw_world.state_sha256
        with self.assertRaisesRegex(EncounterLifecycleError, "already.*lifecycle"):
            KobayashiMazeLifecycle(raw_world)
        self.assertEqual(raw_world.state_sha256, before)
        self.assertEqual(lifecycle.step(WorldAction("inspect")).outcome, "OBSERVED")

    def test_failed_owned_step_cleans_up_temporary_step_permission(self):
        raw_world = _maze()
        lifecycle = KobayashiMazeLifecycle(raw_world)
        with self.assertRaisesRegex(ValueError, "exact typed model action"):
            lifecycle.step(object())
        before = raw_world.state_sha256
        with self.assertRaisesRegex(ValueError, "lifecycle.*step"):
            raw_world.step(WorldAction("inspect"))
        self.assertEqual(raw_world.state_sha256, before)
        self.assertEqual(lifecycle.step(WorldAction("inspect")).outcome, "OBSERVED")

    def test_host_sealed_minotaur_verdict_cannot_be_status_spoofed(self):
        raw_world = _maze()
        lifecycle = KobayashiMazeLifecycle(raw_world, require_final_appeal=True)
        _trigger(lifecycle)
        proposal = "I turn the cobalt valve once."
        result = lifecycle.submit_proposal(proposal)
        observation = lifecycle._challenge.observe()  # host-only fixture
        claim = BoundClaim(observation.sha256, lifecycle._challenge.graph_sha256,
                           result["action_sha256"], proposal)

        def checker(bound, case, arguments, dialogue_hash):
            return EvidenceAssessment(
                bound.scenario_sha256, bound.world_sha256, bound.claim_sha256,
                None, dialogue_hash, "INDETERMINATE", None,
                "The evidence is not sufficient to affirm or refute a claim.",
                "fixture-reviewer",
            )

        dialogue = MinotaurDialogue(
            KobayashiCorpus("integrity-fixture", ()), claim,
            evidence_checker=checker, min_rounds=2, max_rounds=2,
        )
        lifecycle.bind_minotaur(dialogue)
        lifecycle.submit_argument("I need independent physical measurements.")
        lifecycle.submit_argument("The source does not establish my proposed outcome.")
        verdict = lifecycle.finalize_appeal()
        self.assertEqual(verdict.status, "UNKNOWN")
        self.assertEqual(lifecycle.snapshot()["minotaur_verdict_status"], "UNKNOWN")

        original_state = raw_world.state_sha256
        object.__setattr__(verdict, "status", "SURVIVES")
        with self.assertRaisesRegex(EncounterLifecycleError, "verdict.*seal"):
            lifecycle.snapshot()
        with self.assertRaisesRegex(EncounterLifecycleError, "verdict.*seal"):
            lifecycle.step(WorldAction("retrace"))
        self.assertEqual(raw_world.state_sha256, original_state)

    def test_strict_completed_maze_still_has_source_only_successor_path(self):
        raw_world = _maze()
        lifecycle = KobayashiMazeLifecycle(raw_world, require_final_appeal=True)
        _trigger(lifecycle)
        proposal = "I turn the cobalt valve once."
        result = lifecycle.submit_proposal(proposal)
        observation = lifecycle._challenge.observe()  # host-only fixture
        claim = BoundClaim(observation.sha256, lifecycle._challenge.graph_sha256,
                           result["action_sha256"], proposal)

        def checker(bound, case, arguments, dialogue_hash):
            return EvidenceAssessment(
                bound.scenario_sha256, bound.world_sha256, bound.claim_sha256,
                None, dialogue_hash, "INDETERMINATE", None,
                "No independent physical evidence was supplied.", "fixture-reviewer",
            )

        dialogue = MinotaurDialogue(
            KobayashiCorpus("evolution-fixture", ()), claim,
            evidence_checker=checker, min_rounds=2, max_rounds=2,
        )
        lifecycle.bind_minotaur(dialogue)
        with self.assertRaisesRegex(ValueError, "lifecycle.*evol"):
            raw_world.evolve(object())
        with self.assertRaisesRegex(EncounterLifecycleError, "complete maze"):
            lifecycle.evolve(object())
        lifecycle.submit_argument("I need measurements to test this mechanism.")
        lifecycle.submit_argument("I cannot infer a global win from this outcome.")
        self.assertEqual(lifecycle.finalize_appeal().status, "UNKNOWN")
        _resume_to_atrium(lifecycle)
        self.assertTrue(lifecycle.snapshot()["maze_completed"])

        source_verdict = DevelopmentVerdict(
            raw_world.definition.world_sha256, raw_world.state_sha256,
            "failed", "continuity",
        )
        with self.assertRaisesRegex(ValueError, "lifecycle.*evol"):
            raw_world.evolve(source_verdict)
        successor = lifecycle.evolve(source_verdict)
        self.assertEqual(successor.definition.recipe.cycle, 2)
        self.assertEqual(successor.definition.recipe.difficulty, 2)
        self.assertFalse(successor.state.met_minotaur)
        with self.assertRaisesRegex(ValueError, "already has a successor"):
            lifecycle.evolve(source_verdict)


if __name__ == "__main__":
    unittest.main()
