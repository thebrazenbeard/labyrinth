"""Crash between selecting a physical champion and evolving its social world.

This source-only fixture tests recovery classification, not real host restart.
"""

import unittest

from labyrinth.ariadne_recovery_v1 import AriadneJournal
from tests.test_campaign_orchestrator_v1 import OrchestratorTests


class AriadneWorldCommitCrashHostileTests(unittest.TestCase):
    def test_selected_model_without_successor_world_is_not_ready_to_resume(self):
        fixture = OrchestratorTests(
            "test_two_generations_carry_selected_social_world_and_full_model_parent"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        launches, _, _ = fixture._launches(1)
        pending = fixture.orchestrator.run_generation(
            launches=launches,
            control_manifest_path=fixture._control(fixture.w0_root, 1),
            evaluation_plan=fixture.plan,
        )
        original = fixture.journal.record_world

        def crash_before_world_commit(**kwargs):
            raise RuntimeError("simulated crash after champion selected, before world WAL")

        fixture.journal.record_world = crash_before_world_commit
        try:
            with self.assertRaisesRegex(RuntimeError, "after champion selected"):
                fixture.orchestrator.finalize_generation(
                    pending, fixture._external_fixture_exams(pending.population, 1)
                )
        finally:
            fixture.journal.record_world = original
        # Real recovery reads from a FRESH journal instance, not the poisoned
        # orchestrator. It must not call this state ready to load and train.
        observed = AriadneJournal.open(fixture.journal.root).continuation_descriptor()
        self.assertEqual(observed["next_generation_index"], 1,
                         "champion physical selection was already durable")
        self.assertNotEqual(observed["status"], "READY_FOR_EXTERNAL_FULL_STATE_LOAD",
                            "half-committed model/world state falsely advertised READY")


if __name__ == "__main__":
    unittest.main()
