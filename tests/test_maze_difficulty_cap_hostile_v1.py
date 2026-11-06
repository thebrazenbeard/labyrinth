"""Repeated no-winner results must keep evolving beyond scene floor five.

All exams are operator-supplied fixtures; no real private Minotaur or SGD.
"""

from dataclasses import replace
import unittest

from labyrinth.evolutionary_population_v1 import Metrics
from tests.test_campaign_orchestrator_v1 import OrchestratorTests, FixtureDirector


class MazeDifficultyCapHostileTests(unittest.TestCase):
    def test_six_losses_keep_w0_and_evolve_depth_beyond_scene_floor_five(self):
        f = OrchestratorTests(
            "test_no_fitness_winner_retains_exact_parent_and_no_false_generation"
        )
        f.setUp()
        self.addCleanup(f.doCleanups)
        observed_prompts = []
        for attempt in range(1, 7):
            current_floor = f.orchestrator.minimum_difficulty
            current_depth = f.orchestrator.complexity_depth
            old_world = f.orchestrator.current_world_id
            launches, _, _ = f._launches(attempt)
            shaped = []
            for launch in launches:
                cases = f.provider.public_train_beats(
                    launch.case_id, launch.character_id,
                    region_id=launch.region_id, chapter_id=launch.chapter_id,
                    minimum_difficulty=current_floor,
                    complexity_depth=current_depth, world_id=old_world,
                )
                if launch.candidate_id.endswith("left"):
                    observed_prompts.append(cases[0].utterance)
                shaped.append(replace(launch, director=FixtureDirector(cases)))
            pending = f.orchestrator.run_generation(
                launches=tuple(shaped),
                control_manifest_path=f._control(f.w0_root, attempt),
                evaluation_plan=f.plan,
            )
            exams = f._external_fixture_exams(pending.population, attempt)
            all_fail = tuple(replace(e, metrics=Metrics(.5, .5, .99, .99, .02))
                             if e.subject_id not in ("parent", "control") else e
                             for e in exams)
            completed = f.orchestrator.finalize_generation(pending, all_fail)
            self.assertIsNone(completed.decision.winner_id)
            descriptor = f.journal.continuation_descriptor()
            self.assertEqual(descriptor["next_generation_index"], 0)
            self.assertEqual(descriptor["selected_parent_state_sha256"],
                             f.w0_manifest["state_sha256"])
            self.assertEqual(descriptor["status"], "READY_FOR_EXTERNAL_FULL_STATE_LOAD")
            self.assertEqual(descriptor["selected_world_id"], f"world_{attempt + 1}")
            self.assertEqual(descriptor["maze_attempt_index"], attempt + 1)
            self.assertEqual(descriptor["minimum_scene_difficulty"], min(5, attempt + 1))
            self.assertEqual(descriptor["complexity_depth"], max(0, attempt - 4))
        self.assertNotEqual(observed_prompts[4], observed_prompts[5],
                            "new higher-order maze must not replay the exact floor-five task")


if __name__ == "__main__":
    unittest.main()
