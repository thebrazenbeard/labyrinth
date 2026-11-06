"""No-winner development must lead to a new harder *public* social maze."""

from dataclasses import replace
import unittest

from labyrinth.ariadne_recovery_v1 import AriadneJournal
from labyrinth.evolutionary_population_v1 import Metrics
from tests.test_campaign_orchestrator_v1 import OrchestratorTests, FixtureDirector


class HarderMazeRetryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = OrchestratorTests(
            "test_no_fitness_winner_retains_exact_parent_and_no_false_generation")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def _fail_first_maze(self):
        f = self.fixture
        launches, _, _ = f._launches(1)
        pending = f.orchestrator.run_generation(
            launches=launches,
            control_manifest_path=f._control(f.w0_root, 1),
            evaluation_plan=f.plan)
        examinations = f._external_fixture_exams(pending.population, 1)
        all_fail = tuple(replace(e, metrics=Metrics(.50, .50, .99, .99, .02))
                         if e.subject_id not in ("parent", "control") else e
                         for e in examinations)
        return f.orchestrator.finalize_generation(pending, all_fail)

    def test_retry_same_w0_in_stronger_maze_then_selected_weight_generation_one(self):
        f = self.fixture
        first = self._fail_first_maze()
        descriptor = f.journal.continuation_descriptor(
            expected_head_sha256=first.journal_head_sha256)
        self.assertIsNone(first.decision.winner_id)
        self.assertEqual(descriptor["maze_attempt_index"], 2)
        self.assertEqual(descriptor["minimum_scene_difficulty"], 2)
        self.assertEqual(descriptor["next_generation_index"], 0)
        self.assertEqual(descriptor["selected_parent_manifest_path"], str(f.w0_path))
        self.assertEqual(descriptor["selected_world_id"], "world_2")

        # The original easy director is inadmissible *before* a GPU/trainer
        # effect or a bridge WAL appears. Reproposal must satisfy floor 2.
        easy, trainers, _ = f._launches(2)
        with self.assertRaisesRegex(ValueError, "operator scene difficulty"):
            f.orchestrator.run_generation(
                launches=easy, control_manifest_path=f._control(f.w0_root, 2),
                evaluation_plan=f.plan)
        self.assertEqual([trainer.calls for trainer in trainers], [0, 0])
        self.assertFalse(f.orchestrator._blocked)

        harder, trainers, responders = f._launches(2)
        prepared = []
        for launch in harder:
            scenes = f.provider.public_train_beats(
                launch.case_id, launch.character_id,
                region_id=launch.region_id, chapter_id=launch.chapter_id,
                minimum_difficulty=2)
            self.assertGreater(len(scenes[0].utterance),
                               len(f.provider.public_train_beats(
                                   launch.case_id, launch.character_id,
                                   region_id=launch.region_id,
                                   chapter_id=launch.chapter_id)[0].utterance))
            prepared.append(replace(launch, director=FixtureDirector(scenes)))
        retry = f.orchestrator.run_generation(
            launches=tuple(prepared),
            control_manifest_path=f.root / "control_2.json",
            evaluation_plan=f.plan)
        self.assertEqual(retry.generation_number, 1)  # no false neural generation
        self.assertEqual(retry.population.parent_state_sha256,
                         f.w0_manifest["state_sha256"])
        self.assertEqual([trainer.calls for trainer in trainers], [1, 1])
        self.assertTrue(all("most recent correction" in responder.inputs[0]
                            .public_observation.text for responder in responders))
        second = f.orchestrator.finalize_generation(
            retry, f._external_fixture_exams(retry.population, 2))
        self.assertEqual(second.decision.winner_id, "g2_left")
        recovered = AriadneJournal.open(f.journal.root).continuation_descriptor(
            expected_head_sha256=second.journal_head_sha256)
        self.assertEqual(recovered["maze_attempt_index"], 3)
        self.assertEqual(recovered["next_generation_index"], 1)
        self.assertEqual(recovered["minimum_scene_difficulty"], 2)
        self.assertEqual(recovered["selected_world_id"], "world_3")
        self.assertNotEqual(recovered["selected_parent_state_sha256"],
                            f.w0_manifest["state_sha256"])

    def test_crash_after_no_winner_intent_blocks_ready_reentry(self):
        f = self.fixture
        original = f.journal.record_world

        def crash_after_no_winner(**kwargs):
            raise RuntimeError("injected crash between no-winner and harder world")

        f.journal.record_world = crash_after_no_winner
        try:
            with self.assertRaisesRegex(RuntimeError, "injected crash"):
                self._fail_first_maze()
        finally:
            f.journal.record_world = original
        descriptor = AriadneJournal.open(f.journal.root).continuation_descriptor()
        self.assertEqual(descriptor["status"], "INDETERMINATE_WORLD_TRANSITION")
        self.assertEqual(descriptor["world_transition_kind"], "NO_WINNER")
        self.assertEqual(descriptor["selected_parent_state_sha256"],
                         f.w0_manifest["state_sha256"])
        self.assertEqual(descriptor["next_generation_index"], 0)
        self.assertEqual(descriptor["maze_attempt_index"], 1)
        with self.assertRaisesRegex(ValueError, "pending or indeterminate"):
            f.orchestrator.run_generation(
                launches=f._launches(2)[0], control_manifest_path=f.w0_path,
                evaluation_plan=f.plan)

    def test_journal_rejects_unchanged_difficulty_after_no_winner_intent(self):
        f = self.fixture
        original = f.journal.record_world

        def wrong_floor(**kwargs):
            kwargs["curriculum"]["minimum_difficulty"] = 1
            return original(**kwargs)

        f.journal.record_world = wrong_floor
        try:
            with self.assertRaisesRegex(ValueError, "bounded curriculum complexity"):
                self._fail_first_maze()
        finally:
            f.journal.record_world = original
        descriptor = AriadneJournal.open(f.journal.root).continuation_descriptor()
        self.assertEqual(descriptor["status"], "INDETERMINATE_WORLD_TRANSITION")
        self.assertEqual(descriptor["event_count"], 12)

    def test_six_failed_mazes_keep_w0_and_evolve_bounded_quest_beyond_floor_five(self):
        f = self.fixture
        observed_challenges = []
        for attempt in range(1, 7):
            difficulty = f.orchestrator.minimum_difficulty
            complexity = f.orchestrator.complexity_depth
            world_id = f.orchestrator.current_world_id
            launches, trainers, responders = f._launches(attempt)
            approved_launches = []
            for launch in launches:
                beats = f.provider.public_train_beats(
                    launch.case_id, launch.character_id,
                    region_id=launch.region_id, chapter_id=launch.chapter_id,
                    minimum_difficulty=difficulty, complexity_depth=complexity,
                    world_id=world_id)
                approved_launches.append(replace(launch, director=FixtureDirector(beats)))
            pending = f.orchestrator.run_generation(
                launches=tuple(approved_launches),
                control_manifest_path=f._control(f.w0_root, attempt),
                evaluation_plan=f.plan)
            observed_challenges.append(responders[0].inputs[0].public_observation.text)
            self.assertEqual([trainer.calls for trainer in trainers], [1, 1])
            exams = f._external_fixture_exams(pending.population, attempt)
            no_winner = tuple(replace(e, metrics=Metrics(.5, .5, .99, .99, .02))
                              if e.subject_id not in ("parent", "control") else e
                              for e in exams)
            result = f.orchestrator.finalize_generation(pending, no_winner)
            self.assertIsNone(result.decision.winner_id)
            descriptor = f.journal.continuation_descriptor(
                expected_head_sha256=result.journal_head_sha256)
            self.assertEqual(descriptor["maze_attempt_index"], attempt + 1)
            self.assertEqual(descriptor["minimum_scene_difficulty"], min(5, attempt + 1))
            self.assertEqual(descriptor["complexity_depth"], max(0, attempt - 4))
            self.assertEqual(descriptor["next_generation_index"], 0)
            self.assertEqual(descriptor["selected_parent_state_sha256"],
                             f.w0_manifest["state_sha256"])
        self.assertIn("Public multistage quest", observed_challenges[5])
        self.assertIn("Stage 2:", observed_challenges[5])
        self.assertNotEqual(observed_challenges[4], observed_challenges[5])

        stale, trainers, _ = f._launches(7)
        stale_directors = []
        for launch in stale:
            scenes = f.provider.public_train_beats(
                launch.case_id, launch.character_id,
                region_id=launch.region_id, chapter_id=launch.chapter_id,
                minimum_difficulty=5, complexity_depth=0,
                world_id=f.orchestrator.current_world_id)
            stale_directors.append(replace(launch, director=FixtureDirector(scenes)))
        with self.assertRaisesRegex(ValueError, "unapproved Vera TRAIN case"):
            f.orchestrator.run_generation(
                launches=tuple(stale_directors),
                control_manifest_path=f._control(f.w0_root, 7),
                evaluation_plan=f.plan)
        self.assertEqual([trainer.calls for trainer in trainers], [0, 0])
        self.assertFalse(f.orchestrator._blocked)

        long_quest = f.provider.public_train_beats(
            "paper-starship", "miren", minimum_difficulty=5,
            complexity_depth=10**25, world_id="world_big")[0]
        self.assertLess(len(long_quest.utterance), 2048)
        self.assertEqual(long_quest.utterance.count("Stage "), 4)


if __name__ == "__main__":
    unittest.main()
