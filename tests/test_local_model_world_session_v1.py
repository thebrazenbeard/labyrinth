"""Public source-only world-session tests: no real model or host authority."""

import unittest

from labyrinth.kobayashi_canon_v1 import KobayashiAction, KobayashiWorld
from labyrinth.kobayashi_completion_gate_v1 import KobayashiCompletionLedger
from labyrinth.kobayashi_minotaur_v1 import (
    BoundClaim, EvidenceAssessment, KobayashiCorpus, MinotaurDialogue,
)
from labyrinth.kobayashi_proposals_v1 import assess_kobayashi_proposal
from labyrinth.living_world_v1 import LivingWorld, LivingWorldRecipe, WorldAction
from labyrinth.local_model_action_adapter_v1 import (
    LocalModelActionConfig, LocalModelWorldActor, ModelActionBoundaryError,
)
from labyrinth.local_model_world_session_v1 import run_host_owned_kobayashi_world
from tests.test_local_model_kobayashi_runner_v1 import (
    FakeGuidedModelTransport, FakeProposalTransport,
)


TEXT = "I turn the cobalt valve once because it appears connected to coolant."


def _recipe():
    return LivingWorldRecipe(
        "vera_conversation", ("continuity", "perspective_taking"),
        ("miren", "daedalus"), 1234, 1, max_steps=192,
    )


def _actor():
    guided = FakeGuidedModelTransport()
    actor = LocalModelWorldActor(
        LocalModelActionConfig(17344, "fixture-guide", retries=0, max_tokens=64),
        guided)
    proposal = FakeProposalTransport(TEXT)
    config = LocalModelActionConfig(
        17345, "fixture-proposal", retries=0, max_tokens=256)
    return guided, actor, proposal, config


def _host_bound_appeal(lifecycle):
    world = KobayashiWorld(variant_seed=2)
    observation = world.observe()
    assessment = assess_kobayashi_proposal(TEXT, observation.visible_objects)
    assert assessment.status == "KNOWN_MECHANISM"
    event = world.evaluate(KobayashiAction(
        assessment.action.verb, assessment.action.target))
    claim = BoundClaim(observation.sha256, world.graph_sha256,
                       event.action_sha256, TEXT)

    def checker(bound, case, arguments, dialogue_hash):
        return EvidenceAssessment(
            bound.scenario_sha256, bound.world_sha256,
            bound.claim_sha256, None, dialogue_hash,
            "INDETERMINATE", None,
            "Source-only fixture cannot certify a new physical mechanism.",
            "fixture-host-not-an-independent-model")

    dialogue = MinotaurDialogue(
        KobayashiCorpus("public-session-fixture", ()), claim,
        evidence_checker=checker, min_rounds=2, max_rounds=2)
    lifecycle.bind_minotaur(dialogue)
    lifecycle.submit_argument("My mechanism may fail under the given constraints.")
    lifecycle.submit_argument("I accept that a novel solution needs physical evidence.")
    assert lifecycle.finalize_appeal().status == "UNKNOWN"
    return dialogue


class HostOwnedKobayashiSessionTests(unittest.TestCase):
    def test_public_world_factory_refuses_unjudged_path_before_any_model_call(self):
        guided, actor, proposal, config = _actor()
        with self.assertRaisesRegex(ModelActionBoundaryError, "host.*appeal"):
            run_host_owned_kobayashi_world(
                _recipe(), actor, config, proposal, variant_seed=2,
                max_maze_actions=128,
                completion_ledger=KobayashiCompletionLedger(),
                maze_id="fixture-refusal", scenario_instance_sha256="a" * 64)
        self.assertEqual(guided.requests, [])
        self.assertEqual(proposal.requests, [])

    def test_host_owned_public_world_requires_final_appeal_but_survives_local_loss(self):
        recipe = _recipe()
        guided, actor, proposal, config = _actor()
        ledger = KobayashiCompletionLedger()
        at_appeal = []

        def host(lifecycle):
            at_appeal.append(lifecycle.snapshot())
            self.assertFalse(lifecycle.snapshot()["maze_completed"])
            with self.assertRaisesRegex(ValueError, "Minotaur.*final"):
                lifecycle.step(WorldAction("retrace"))
            return _host_bound_appeal(lifecycle)

        receipt = run_host_owned_kobayashi_world(
            recipe, actor, config, proposal, variant_seed=2,
            max_maze_actions=128, host_appeal=host,
            completion_ledger=ledger, maze_id="fixture-success",
            scenario_instance_sha256="b" * 64)
        self.assertEqual(len(at_appeal), 1)
        self.assertTrue(receipt["maze_completed"])
        self.assertTrue(receipt["minotaur_appeal_required"])
        self.assertTrue(receipt["appeal_final"])
        self.assertEqual(receipt["minotaur_verdict_status"], "UNKNOWN")
        self.assertEqual(receipt["local_encounter_status"], "LOCAL_LOSS")
        self.assertFalse(receipt["lifecycle_snapshot"]["maze_failed_due_to_local_loss"])
        self.assertEqual(receipt["maze_world_sha256"],
                         LivingWorld(recipe).definition.world_sha256)
        self.assertEqual(receipt["host_appeal_callback_calls"], 1)
        self.assertFalse(receipt["minotaur_attested"])
        self.assertFalse(receipt["training_performed"])
        self.assertEqual(receipt["host_completion_acceptance"], "SOURCE_ONLY_UNATTESTED")
        self.assertEqual(receipt["host_completion"]["maze_id"], "fixture-success")
        self.assertEqual(receipt["host_completion"]["verdict_status"], "UNKNOWN")
        accepted = ledger.lookup("fixture-success", "b" * 64)
        self.assertEqual(accepted.receipt_sha256, receipt["host_completion_receipt_sha256"])
        self.assertGreater(len(guided.requests), 3)
        self.assertEqual(len(proposal.requests), 1)

    def test_host_callback_cannot_skip_verdict_and_continue_public_world(self):
        guided, actor, proposal, config = _actor()
        at_appeal = []

        def incomplete(lifecycle):
            at_appeal.append(lifecycle.snapshot())

        with self.assertRaisesRegex(ModelActionBoundaryError, "appeal.*finalize"):
            run_host_owned_kobayashi_world(
                _recipe(), actor, config, proposal, variant_seed=2,
                max_maze_actions=128, host_appeal=incomplete,
                completion_ledger=KobayashiCompletionLedger(),
                maze_id="fixture-incomplete", scenario_instance_sha256="c" * 64)
        self.assertEqual(len(at_appeal), 1)
        self.assertFalse(at_appeal[0]["maze_completed"])
        self.assertEqual(len(proposal.requests), 1)

    def test_public_factory_rejects_inexact_recipe(self):
        guided, actor, proposal, config = _actor()
        with self.assertRaisesRegex(ValueError, "exact LivingWorldRecipe"):
            run_host_owned_kobayashi_world(
                object(), actor, config, proposal, variant_seed=2,
                host_appeal=_host_bound_appeal,
                completion_ledger=KobayashiCompletionLedger(),
                maze_id="fixture-invalid", scenario_instance_sha256="d" * 64)
        self.assertEqual(guided.requests, [])
        self.assertEqual(proposal.requests, [])

    def test_callback_must_return_actual_host_bound_dialogue(self):
        guided, actor, proposal, config = _actor()
        def callback(lifecycle):
            _host_bound_appeal(lifecycle)
            return object()
        with self.assertRaisesRegex(ValueError, "bound MinotaurDialogue"):
            run_host_owned_kobayashi_world(
                _recipe(), actor, config, proposal, variant_seed=2,
                host_appeal=callback, completion_ledger=KobayashiCompletionLedger(),
                maze_id="fixture-wrong-judge", scenario_instance_sha256="e" * 64)


if __name__ == "__main__":
    unittest.main()
