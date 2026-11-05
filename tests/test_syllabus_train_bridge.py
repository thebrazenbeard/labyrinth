"""Optional Syllabus protocol bridge: no external package required for core tests."""

import sys
from itertools import repeat
from types import ModuleType
from unittest import TestCase
from unittest.mock import patch

from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.syllabus_train_bridge import (
    SyllabusBridgeUnavailable,
    SyllabusTrainBridge,
    make_syllabus_train_bridge,
)
from labyrinth.training_scope import TrainingScopeSpec


def scope():
    return TrainingScopeSpec(
        objective_id="syllabus-local-train",
        domain="reasoning",
        skills=("arithmetic", "source_priority"),
        specificity=1,
        scale=1,
        max_cycles=2,
        seed=13,
    )


class FakeCurriculum:
    def __init__(self):
        self.events = []
        self.samples = [0, 1]

    def update_task_progress(self, task, progress):
        self.events.append(("progress", task, progress))

    def update_on_episode(self, episode_return, length, task, progress):
        self.events.append(("episode", episode_return, length, task, progress))

    def sample(self, k=1):
        self.events.append(("sample", k))
        return self.samples[:k]


class BridgeTests(TestCase):
    def setUp(self):
        self.scope = scope()
        self.plan = generate_maze_plan(self.scope, 0)
        self.curriculum = FakeCurriculum()
        self.bridge = SyllabusTrainBridge(self.scope, self.plan, self.curriculum)

    def test_train_episode_feedback_uses_upstream_syllabus_methods(self):
        room = self.plan.rooms[0]
        self.bridge.record_completed_train_room(room, correct=False, length=3)
        task = self.scope.skills.index(room.skill)
        self.assertEqual(self.curriculum.events, [
            ("progress", task, 0.0),
            ("episode", 0.0, 3, task, 0.0),
        ])
        self.assertEqual(self.bridge.recorded_count, 1)

    def test_no_replay_or_noncanonical_room(self):
        room = self.plan.rooms[0]
        self.bridge.record_completed_train_room(room, correct=True)
        before = list(self.curriculum.events)
        with self.assertRaisesRegex(ValueError, "already recorded"):
            self.bridge.record_completed_train_room(room, correct=True)
        self.assertEqual(self.curriculum.events, before)

        other = generate_maze_plan(self.scope, 1).rooms[0]
        with self.assertRaisesRegex(ValueError, "canonical TRAIN"):
            self.bridge.record_completed_train_room(other, correct=True)
        self.assertEqual(self.curriculum.events, before)

    def test_only_strict_train_outcomes(self):
        room = self.plan.rooms[0]
        for value in (1, 0, "yes"):
            with self.assertRaises(ValueError):
                self.bridge.record_completed_train_room(room, correct=value)
        for value in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                self.bridge.record_completed_train_room(room, correct=True, length=value)
        self.assertEqual(self.curriculum.events, [])

    def test_advisory_sampling_does_not_mutate_canonical_maze(self):
        before = generate_maze_plan(self.scope, 0)
        self.curriculum.samples = [1, 1]
        self.assertEqual(self.bridge.advisory_skill_sample(2), ("source_priority",))
        self.assertEqual(generate_maze_plan(self.scope, 0), before)
        self.assertEqual(self.bridge.recorded_count, 0)

    def test_bad_syllabus_output_rejected(self):
        for samples in ([True], [3], ["arithmetic"], [-1]):
            self.curriculum.samples = samples
            with self.assertRaisesRegex(ValueError, "invalid Syllabus task"):
                self.bridge.advisory_skill_sample(1)
        with self.assertRaises(ValueError):
            self.bridge.advisory_skill_sample(0)

    def test_infinite_upstream_sample_is_consumed_only_to_bound(self):
        class InfiniteCurriculum(FakeCurriculum):
            def sample(self, k=1):
                return repeat(0)

        bridge = SyllabusTrainBridge(self.scope, self.plan, InfiniteCurriculum())
        with self.assertRaisesRegex(ValueError, "invalid Syllabus task index count"):
            bridge.advisory_skill_sample(1)

    def test_upstream_update_failure_poison_bridge_to_avoid_replay(self):
        class FailsAfterProgress(FakeCurriculum):
            def update_on_episode(self, *args):
                raise RuntimeError("upstream fail")

        curriculum = FailsAfterProgress()
        bridge = SyllabusTrainBridge(self.scope, self.plan, curriculum)
        with self.assertRaisesRegex(RuntimeError, "upstream fail"):
            bridge.record_completed_train_room(self.plan.rooms[0], correct=True)
        with self.assertRaisesRegex(RuntimeError, "indeterminate"):
            bridge.record_completed_train_room(self.plan.rooms[1], correct=True)
        with self.assertRaisesRegex(RuntimeError, "indeterminate"):
            bridge.advisory_skill_sample(1)

    def test_factory_tracks_verified_upstream_constructor_contract(self):
        # Synthetic module records the constructor contracts without importing
        # costly optional Gym/NumPy/PettingZoo dependencies into core tests.
        syllabus = ModuleType("syllabus")
        task_space = ModuleType("syllabus.task_space")
        curricula = ModuleType("syllabus.curricula")

        class DiscreteTaskSpace:
            def __init__(self, n, tasks=None):
                self.n = n
                self.tasks = tasks

        class OnlineLearningProgress(FakeCurriculum):
            def __init__(self, space, *, update_interval, normalize_success):
                super().__init__()
                self.space = space
                self.update_interval = update_interval
                self.normalize_success = normalize_success

        task_space.DiscreteTaskSpace = DiscreteTaskSpace
        curricula.OnlineLearningProgress = OnlineLearningProgress
        with patch.dict(sys.modules, {
            "syllabus": syllabus,
            "syllabus.task_space": task_space,
            "syllabus.curricula": curricula,
        }):
            bridge = make_syllabus_train_bridge(self.scope, self.plan)
        self.assertEqual(bridge.curriculum.space.tasks, list(self.scope.skills))
        self.assertEqual(bridge.curriculum.space.n, len(self.scope.skills))
        self.assertEqual(bridge.curriculum.update_interval, len(self.plan.rooms))
        self.assertFalse(bridge.curriculum.normalize_success)

    def test_missing_optional_package_is_clean_error(self):
        # This does not depend on whether Syllabus happens to be installed in
        # a developer's interpreter; simulate a genuine import rejection.
        real_import = __import__

        def blocked(name, *args, **kwargs):
            if name.startswith("syllabus"):
                raise ModuleNotFoundError("No module named 'syllabus'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked):
            with self.assertRaises(SyllabusBridgeUnavailable):
                make_syllabus_train_bridge(self.scope, self.plan)
