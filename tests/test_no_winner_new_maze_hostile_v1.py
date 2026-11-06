"""A losing learner returns with W0 into a witnessed *harder* maze.

The private Minotaur exam is NOT run: metrics here are test fixtures only.
"""

from dataclasses import replace
import unittest

from tests.test_campaign_orchestrator_v1 import OrchestratorTests
from labyrinth.evolutionary_population_v1 import Metrics


class NoWinnerNewMazeHostileTests(unittest.TestCase):
    def test_no_winner_keeps_w0_but_starts_witnessed_harder_world(self):
        fixture = OrchestratorTests(
            "test_no_fitness_winner_retains_exact_parent_and_no_false_generation"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        before = fixture.journal.continuation_descriptor()
        launches, _, _ = fixture._launches(1)
        pending = fixture.orchestrator.run_generation(
            launches=launches,
            control_manifest_path=fixture._control(fixture.w0_root, 1),
            evaluation_plan=fixture.plan,
        )
        exams = fixture._external_fixture_exams(pending.population, 1)
        no_winner = tuple(
            replace(e, metrics=Metrics(.50, .50, .99, .99, .02))
            if e.subject_id not in ("parent", "control") else e
            for e in exams
        )
        result = fixture.orchestrator.finalize_generation(pending, no_winner)
        after = fixture.journal.continuation_descriptor(
            expected_head_sha256=result.journal_head_sha256
        )
        self.assertIsNone(result.decision.winner_id)
        self.assertEqual(after["selected_parent_state_sha256"],
                         before["selected_parent_state_sha256"],
                         "failed models cannot silently advance physical weights")
        self.assertEqual(after["next_generation_index"], 0)
        self.assertNotEqual(after["selected_world_id"], before["selected_world_id"],
                            "a no-winner verdict must visibly create the harder return maze")
        next_world = next(world for world in after["worlds"]
                          if world["world_id"] == after["selected_world_id"])
        self.assertEqual(next_world["parent_world_id"], before["selected_world_id"])
        self.assertGreaterEqual(next_world["curriculum"]["minimum_difficulty"], 2)


if __name__ == "__main__":
    unittest.main()
