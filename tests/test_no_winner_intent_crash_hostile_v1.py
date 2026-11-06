"""Crash after rejecting every sibling but BEFORE the NO_WINNER WAL event."""

from dataclasses import replace
import unittest

from labyrinth.ariadne_recovery_v1 import AriadneJournal
from labyrinth.evolutionary_population_v1 import Metrics
from tests.test_campaign_orchestrator_v1 import OrchestratorTests


class NoWinnerIntentCrashHostileTests(unittest.TestCase):
    def test_all_rejected_crash_before_no_winner_intent_cannot_claim_ready(self):
        f = OrchestratorTests(
            "test_no_fitness_winner_retains_exact_parent_and_no_false_generation"
        )
        f.setUp()
        self.addCleanup(f.doCleanups)
        launches, _, _ = f._launches(1)
        pending = f.orchestrator.run_generation(
            launches=launches,
            control_manifest_path=f._control(f.w0_root, 1),
            evaluation_plan=f.plan,
        )
        exams = f._external_fixture_exams(pending.population, 1)
        all_fail = tuple(replace(e, metrics=Metrics(.5, .5, .99, .99, .02))
                         if e.subject_id not in ("parent", "control") else e
                         for e in exams)
        original = f.journal.record_no_winner

        def crash_before_intent(**kwargs):
            raise RuntimeError("simulated crash before no-winner WAL")

        f.journal.record_no_winner = crash_before_intent
        try:
            with self.assertRaisesRegex(RuntimeError, "before no-winner"):
                f.orchestrator.finalize_generation(pending, all_fail)
        finally:
            f.journal.record_no_winner = original
        desc = AriadneJournal.open(f.journal.root).continuation_descriptor()
        self.assertEqual({c["fate"] for c in desc["candidates"]}, {"REJECTED"})
        self.assertEqual(desc["next_generation_index"], 0)
        self.assertNotEqual(desc["status"], "READY_FOR_EXTERNAL_FULL_STATE_LOAD",
                            "all-rejected generation lost NO_WINNER intent and "
                            "falsely exposed easy W0 as recoverable READY")


if __name__ == "__main__":
    unittest.main()
