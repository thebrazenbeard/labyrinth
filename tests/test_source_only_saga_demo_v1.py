"""End-to-end fictional model actor -> dynamic dialogue -> physical fixture tests."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from labyrinth.living_world_v1 import LivingWorld, LivingWorldRecipe
from labyrinth.source_only_saga_demo_v1 import (
    EVIDENCE, FictionalActionModel, FictionalDialogueModel, _play_maze,
    run_fixture_saga,
)


class SagaDemonstrationTests(unittest.TestCase):
    def test_two_cycles_include_actions_train_sparks_selection_ancestry(self):
        with TemporaryDirectory(prefix="saga-test-") as temp:
            root = Path(temp).resolve() / "demo"
            report = run_fixture_saga(root)
            self.assertEqual(report["evidence_class"], EVIDENCE)
            self.assertFalse(report["real_model_or_sgd"])
            self.assertEqual(report["ariadne_status"],
                             "READY_FOR_EXTERNAL_FULL_STATE_LOAD")
            self.assertEqual(report["difficulty"], [1, 2])
            self.assertEqual(len(report["dialogue_training"]), 2)
            self.assertEqual({spark["fixture_physical_changes"]
                              for spark in report["dialogue_training"]}, {1})
            self.assertEqual(report["fixture_selection"]["winner_id"], "cand_ariadne")
            self.assertTrue(report["fixture_selection"]["not_independent_exam"])
            self.assertIn("Cycle 1 passed", report["public_mementos"][0])
            self.assertNotEqual(*report["world_ancestry"])
            self.assertTrue(all(cycle["completed"] for cycle in report["cycles"]))
            for cycle in report["cycles"]:
                self.assertIn("MINOTAUR_ENCOUNTER", cycle["outcomes"])
                self.assertIn("ACQUIRED", cycle["outcomes"])
                self.assertIn("RETRACED", cycle["outcomes"])
                self.assertGreater(len(cycle["actions"]), 10)
            self.assertTrue((root / "ariadne").is_dir())
            self.assertTrue((root / "cand_ariadne.json").is_file())
            self.assertTrue((root / "cand_daedalus.json").is_file())

    def test_action_actor_solves_multiple_seeded_worlds_by_observation_not_script(self):
        seen_worlds = set()
        for difficulty in (1, 2, 3):
            for seed in (3, 11, 71, 139):
                with self.subTest(difficulty=difficulty, seed=seed):
                    world = LivingWorld(LivingWorldRecipe(
                        "saga_continuity", ("continuity",), ("miren",),
                        seed, difficulty=difficulty, max_steps=192))
                    report = _play_maze(world)
                    self.assertTrue(report["completed"])
                    self.assertEqual(report["outcomes"].count("MINOTAUR_ENCOUNTER"), 1)
                    self.assertEqual(world.state.position, "ariadne_atrium")
                    self.assertIn("ariadne_thread", world.state.inventory)
                    self.assertEqual(sum(item.startswith("seal_")
                                         for item in world.state.inventory), min(difficulty, 3))
                    seen_worlds.add(report["world_sha256"])
        self.assertEqual(len(seen_worlds), 12)

    def test_model_scene_uses_current_public_room_and_no_grader(self):
        world = LivingWorld(LivingWorldRecipe(
            "saga_continuity", ("continuity",), ("miren",), 5))
        from labyrinth.living_world_v1 import PublicSceneWritingContext
        writer = FictionalDialogueModel()
        result = writer.write_scene(PublicSceneWritingContext(
            world.observe(), ("continuity",), ("miren",), (), 1))
        self.assertIn("Ariadne's Atrium", result.utterance)
        self.assertFalse(hasattr(result, "score"))
        self.assertFalse(hasattr(result, "grade"))

    def test_fixture_refuses_existing_workdir_instead_of_overwriting(self):
        with TemporaryDirectory(prefix="saga-refuse-") as temp:
            existing = Path(temp).resolve()
            with self.assertRaisesRegex(ValueError, "new absolute"):
                run_fixture_saga(existing)


if __name__ == "__main__":
    unittest.main()
