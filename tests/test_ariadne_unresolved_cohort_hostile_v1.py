"""A crashed unselected sibling cohort must not reopen a new model campaign."""

from dataclasses import replace
import unittest

from labyrinth.ariadne_recovery_v1 import AriadneJournal
from labyrinth.campaign_orchestrator_v1 import CampaignBoundaryError, CampaignOrchestrator
from labyrinth.social_world_consequence_v1 import CharacterRelationship
from tests import test_ariadne_recovery_v1 as ariadne_fixtures
from tests import test_campaign_orchestrator_v1 as campaign_fixtures


class AriadneUnresolvedCohortHostileTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ariadne_fixtures.AriadneRecoveryTests(
            "test_idempotency_is_exact_and_prepared_never_issues_second_permit"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_empty_genesis_has_ready_developmental_entrance(self):
        desc = self.fixture.journal.continuation_descriptor()
        self.assertEqual(desc["status"], "READY_FOR_EXTERNAL_FULL_STATE_LOAD")
        self.assertEqual(desc["development_status"], "NO_ACTIVE_COHORT")
        self.assertFalse(desc["candidates"])

    def test_single_committed_evaluated_child_without_decision_is_indeterminate(self):
        self.fixture._candidate("left", self.fixture.w0)
        desc = AriadneJournal.open(self.fixture.journal.root).continuation_descriptor()
        self.assertEqual(desc["status"], "READY_FOR_EXTERNAL_FULL_STATE_LOAD")
        self.assertEqual(desc["development_status"], "UNRESOLVED_CURRENT_COHORT")
        self.assertEqual([c["fate"] for c in desc["candidates"]], ["EVALUATED"])
        self.assertFalse(desc["selected_lineage"])

    def test_two_evaluated_siblings_without_decision_are_indeterminate(self):
        self.fixture._candidate("left", self.fixture.w0)
        self.fixture._candidate("right", self.fixture.w0)
        desc = AriadneJournal.open(self.fixture.journal.root).continuation_descriptor()
        self.assertEqual(desc["status"], "READY_FOR_EXTERNAL_FULL_STATE_LOAD")
        self.assertEqual(desc["development_status"], "UNRESOLVED_CURRENT_COHORT")
        self.assertEqual({c["fate"] for c in desc["candidates"]}, {"EVALUATED"})
        self.assertFalse(desc["selected_lineage"])

    def test_settled_selection_requires_world_transition_before_ready(self):
        self.fixture._candidate("left", self.fixture.w0)
        self.fixture._candidate("right", self.fixture.w0)
        self.fixture.journal.reject_candidate(
            candidate_id="left", reason="development fixture lower",
            idempotency_key="reject_left",
        )
        self.fixture.journal.select_candidate(
            candidate_id="right", selection_reason="development fixture higher",
            idempotency_key="select_right",
        )
        # With the existing transition contract a selected successor world
        # must be materialized before a new cohort may enter.
        self.fixture.journal.record_world(
            world_id="world_2", parent_world_id="world_1", version=2,
            curriculum={"objective": "challenging successors"},
            world_state={"rooms": ["more difficult entrance"]},
            idempotency_key="world_2",
        )
        desc = self.fixture.journal.continuation_descriptor()
        self.assertEqual(desc["status"], "READY_FOR_EXTERNAL_FULL_STATE_LOAD")
        self.assertEqual(desc["development_status"], "NO_ACTIVE_COHORT")
        self.assertEqual(desc["selected_world_id"], "world_2")
        self.assertEqual(len(desc["selected_lineage"]), 1)

    def test_new_campaign_after_crashed_exam_cannot_train_other_siblings(self):
        second_fixture = campaign_fixtures.OrchestratorTests(
            "test_no_fitness_winner_retains_exact_parent_and_no_false_generation"
        )
        second_fixture.setUp()
        self.addCleanup(second_fixture.doCleanups)
        launched, trainers, _ = second_fixture._launches(1)
        control = second_fixture._control(second_fixture.w0_root, 1)
        second_fixture.orchestrator.run_generation(
            launches=launched, control_manifest_path=control,
            evaluation_plan=second_fixture.plan,
        )
        # Simulate the Python process dying before any Minotaur decision: the
        # local pending object is gone but two physical candidates remain.
        recovered = CampaignOrchestrator(
            journal=AriadneJournal.open(second_fixture.journal.root),
            origin_manifest_path=second_fixture.w0_path,
            scope=second_fixture.scope, provider=second_fixture.provider,
            characters=(CharacterRelationship("miren"),),
        )
        self.assertEqual(
            recovered.journal.continuation_descriptor()["development_status"],
            "UNRESOLVED_CURRENT_COHORT",
        )
        new_launches = tuple(
            replace(
                item, candidate_id="next_" + item.candidate_id,
                bridge_journal_directory=second_fixture.root / ("next_" + item.candidate_id),
            ) for item in launched
        )
        with self.assertRaisesRegex(CampaignBoundaryError, "unresolved current-world cohort"):
            recovered.run_generation(
                launches=new_launches,
                control_manifest_path=control,
                evaluation_plan=second_fixture.plan,
            )
        self.assertEqual([trainer.calls for trainer in trainers], [1, 1],
                         "a second campaign retrained the finished candidates")


if __name__ == "__main__":
    unittest.main()
