"""Adversarial source-only repros: all preflight before irreversible TRAIN.

The fake physical trainer writes actual bytes to disk, but this is NOT SGD,
real Qwen learning, a private Minotaur exam, or host-side authorization.
"""

from dataclasses import replace
import unittest

from tests import test_campaign_orchestrator_v1 as campaign_fixtures
from labyrinth.evolutionary_population_v1 import Metrics


class CampaignPreflightHostileTests(unittest.TestCase):
    def _fixture(self):
        fixture = campaign_fixtures.OrchestratorTests(
            "test_no_fitness_winner_retains_exact_parent_and_no_false_generation"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def _assert_no_training_or_false_permanent_block(self, fixture, trainers):
        self.assertEqual([trainer.calls for trainer in trainers], [0, 0],
                         "preflight validation happened AFTER physical TRAIN effects")
        self.assertFalse(fixture.orchestrator._blocked,
                         "rejected no-effect preflight permanently blocked campaign")
        desc = fixture.journal.continuation_descriptor()
        self.assertEqual(desc["status"], "READY_FOR_EXTERNAL_FULL_STATE_LOAD")
        self.assertEqual(desc["candidates"], [])

    def test_missing_unchanged_control_is_rejected_before_any_training(self):
        fixture = self._fixture()
        launches, trainers, _ = fixture._launches(1)
        with self.assertRaises(ValueError):
            fixture.orchestrator.run_generation(
                launches=launches,
                control_manifest_path=fixture.root / "missing-control.json",
                evaluation_plan=fixture.plan,
            )
        self._assert_no_training_or_false_permanent_block(fixture, trainers)

    def test_invalid_second_launch_is_rejected_before_first_sibling_training(self):
        fixture = self._fixture()
        launches, trainers, _ = fixture._launches(1)
        bad = (launches[0], replace(launches[1], max_beats=0))
        with self.assertRaisesRegex(ValueError, "scene count|bounded"):
            fixture.orchestrator.run_generation(
                launches=bad,
                control_manifest_path=fixture._control(fixture.w0_root, 1),
                evaluation_plan=fixture.plan,
            )
        self._assert_no_training_or_false_permanent_block(fixture, trainers)

    def test_self_grading_plan_is_rejected_before_training_not_after(self):
        fixture = self._fixture()
        launches, trainers, _ = fixture._launches(1)
        bad_plan = replace(fixture.plan, examiner_id="fixture_trainer")
        with self.assertRaisesRegex(ValueError, "trainer cannot"):
            fixture.orchestrator.run_generation(
                launches=launches,
                control_manifest_path=fixture._control(fixture.w0_root, 1),
                evaluation_plan=bad_plan,
            )
        self._assert_no_training_or_false_permanent_block(fixture, trainers)


    def test_no_effect_preflight_failure_does_not_poison_same_launch_retry(self):
        fixture = self._fixture()
        launches, trainers, _ = fixture._launches(1)
        control = fixture._control(fixture.w0_root, 1)
        with self.assertRaisesRegex(ValueError, "scene count|bounded"):
            fixture.orchestrator.run_generation(
                launches=(launches[0], replace(launches[1], max_beats=0)),
                control_manifest_path=control,
                evaluation_plan=fixture.plan,
            )
        self._assert_no_training_or_false_permanent_block(fixture, trainers)
        pending = fixture.orchestrator.run_generation(
            launches=launches,
            control_manifest_path=control,
            evaluation_plan=fixture.plan,
        )
        self.assertEqual(len(pending.experiences), 2)
        self.assertEqual([trainer.calls for trainer in trainers], [1, 1])

    def test_trainer_cannot_rewrite_exam_preregistration_during_training(self):
        fixture = self._fixture()
        launches, trainers, _ = fixture._launches(1)
        sealed_before = fixture.plan.sealed_case_commitment
        original = trainers[0].train_supervised

        def mutating_trainer(case, parent):
            object.__setattr__(fixture.plan, "sealed_case_commitment", "f" * 64)
            return original(case, parent)

        trainers[0].train_supervised = mutating_trainer
        pending = fixture.orchestrator.run_generation(
            launches=launches,
            control_manifest_path=fixture._control(fixture.w0_root, 1),
            evaluation_plan=fixture.plan,
        )
        self.assertEqual(pending.population.plan.sealed_case_commitment,
                         sealed_before,
                         "trainer rewrote purportedly preregistered exam cohort")


    def test_reused_failed_candidate_id_rejects_recoverably_before_effect(self):
        fixture = self._fixture()
        launches, trainers, _ = fixture._launches(1)
        control = fixture._control(fixture.w0_root, 1)
        pending = fixture.orchestrator.run_generation(
            launches=launches, control_manifest_path=control,
            evaluation_plan=fixture.plan)
        exams = fixture._external_fixture_exams(pending.population, 1)
        no_winner = tuple(replace(e, metrics=Metrics(.50, .50, .99, .99, .02))
                          if e.subject_id not in ("parent", "control") else e
                          for e in exams)
        fixture.orchestrator.finalize_generation(pending, no_winner)
        self.assertEqual([t.calls for t in trainers], [1, 1])
        retry = tuple(replace(launch, bridge_journal_directory=(
            fixture.root / f"retry_{launch.candidate_id}")) for launch in launches)
        with self.assertRaises(ValueError):
            fixture.orchestrator.run_generation(
                launches=retry, control_manifest_path=control,
                evaluation_plan=fixture.plan)
        self.assertEqual([t.calls for t in trainers], [1, 1],
                         "source must never replay an already consumed candidate effect")
        self.assertFalse(fixture.orchestrator._blocked,
                         "no-effect duplicate candidate preflight must be recoverable")

    def test_pending_receipt_alias_cannot_forge_selected_social_world(self):
        fixture = self._fixture()
        launches, _, _ = fixture._launches(1)
        pending = fixture.orchestrator.run_generation(
            launches=launches,
            control_manifest_path=fixture._control(fixture.w0_root, 1),
            evaluation_plan=fixture.plan)
        source_receipt = pending.experiences[1].social_receipt
        honest_trust = source_receipt.characters[0].trust
        self.assertEqual(honest_trust, 1)
        object.__setattr__(source_receipt.characters[0], "trust", 3)
        exams = fixture._external_fixture_exams(pending.population, 1)
        try:
            selected = fixture.orchestrator.finalize_generation(pending, exams)
        except ValueError:
            return
        self.assertEqual(selected.decision.winner_id, "g1_right")
        self.assertEqual(selected.selected_characters[0].trust, honest_trust,
                         "forged callback-alias trust promoted into next world")

    def test_caller_cannot_rewrite_genesis_world_trust_before_first_train(self):
        fixture = self._fixture()
        object.__setattr__(fixture.orchestrator.selected_characters[0], "trust", 3)
        launches, trainers, responders = fixture._launches(1)
        try:
            fixture.orchestrator.run_generation(
                launches=launches,
                control_manifest_path=fixture._control(fixture.w0_root, 1),
                evaluation_plan=fixture.plan,
            )
        except ValueError:
            self.assertEqual([trainer.calls for trainer in trainers], [0, 0],
                             "genesis forgery rejected only after physical TRAIN")
            return
        self.assertIn("cautious, unsure how this conversation will go",
                      responders[0].inputs[0].public_observation.character_context,
                      "training world entrance detached from witnessed genesis")


if __name__ == "__main__":
    unittest.main()
