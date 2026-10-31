"""Signed measured neural batch execution bridge: source-only checks.

This tests the real physical execute() preflight and the exact selector used
inside its optimizer loop. The neural backend is deliberately not installed
or executed here; source checks cannot attest model updates or host custody.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import qualify_maze_conditioned_minilm_cpu as runner
from tests import test_measured_neural_batch_binding as measured_fixtures


class MeasuredMiniLMExecutionTests(unittest.TestCase):
    def setUp(self):
        fixture = measured_fixtures.BoundNeuralMeasuredBatchesTests(
            "test_actual_four_key_signed_development_produces_bound_five_arm_schedule")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.schedule = fixture.pinned(fixture.proposal())
        self.request = runner.MeasuredNeuralExecutionRequest(
            schedule=self.schedule,
            independently_pinned_schedule_sha256=self.schedule.schedule_sha256,
            run_id=self.schedule.run_id,
            candidate_artifact_sha256=self.schedule.candidate_artifact_sha256,
            predecessor_checkpoint_sha256=self.schedule.predecessor_checkpoint_sha256,
            scope_fingerprint=self.schedule.scope_fingerprint,
            fence=self.schedule.fence,
        )

    def execute(self, root: Path, **changed):
        arguments = dict(
            scope=self.fixture.signed.scope,
            cycles=self.fixture.signed.cycles,
            steps_per_cycle=16, learning_rate=.002,
            preregistered_controls=True, measured_batches=self.request,
        )
        arguments.update(changed)
        return runner.execute(root, **arguments)

    def test_same_exact_training_visits_reach_every_arm_and_labels_remain_distinct(self):
        cycles = self.fixture.signed.cycles
        self.assertNotEqual(
            runner.select_executed_training_batch(cycles[1], 0, "correct", self.schedule),
            runner.select_training_batch(cycles[1], 0))
        self.assertEqual(
            runner.select_executed_training_batch(cycles[0], 0, "correct", self.schedule),
            runner.select_training_batch(cycles[0], 0))
        observed = {
            arm: tuple(row.exposure_id for step in range(16)
                       for row in runner.select_executed_training_batch(
                           cycles[1], step, arm, self.schedule))
            for arm in runner.PREREG_ARMS
        }
        self.assertEqual(len(set(observed.values())), 1)
        self.assertEqual(len(next(iter(observed.values()))), 64)
        randomized = runner.preregistered_randomized_labels(
            cycles[1], self.fixture.signed.scope.seed)
        labels = {
            arm: tuple(tuple(runner.training_batch_labels(
                runner.select_executed_training_batch(cycles[1], step, arm, self.schedule),
                arm, randomized if arm == "randomized_labels" else None))
                for step in range(16))
            for arm in runner.PREREG_ARMS
        }
        self.assertEqual(labels["correct"], labels["zero_update"])
        self.assertEqual(labels["correct"], labels["zero_lr"])
        self.assertNotEqual(labels["correct"], labels["inverted"])
        self.assertNotEqual(labels["correct"], labels["randomized_labels"])

    def test_default_three_arm_training_batch_selection_is_unchanged(self):
        cycles = self.fixture.signed.cycles
        for cycle in cycles:
            for step in range(16):
                self.assertEqual(
                    runner.select_executed_training_batch(cycle, step, "correct"),
                    runner.select_training_batch(cycle, step))

    def test_verified_schedule_preflights_before_importing_torch_or_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "neural"
            # Missing Torch only proves all source+schedule validation passed
            # and the program reached its real neural backend import point.
            with patch.dict("sys.modules", {"torch": None}):
                with self.assertRaises(ModuleNotFoundError):
                    self.execute(destination)
            self.assertFalse(destination.exists())

    def test_foreign_signed_schedule_or_missing_five_arm_control_is_rejected(self):
        failures = (
            {"preregistered_controls": False},
            {"steps_per_cycle": 8},
            {"measured_batches": replace(
                self.request, independently_pinned_schedule_sha256="f" * 64)},
            {"measured_batches": replace(
                self.request, predecessor_checkpoint_sha256="f" * 64)},
            {"measured_batches": replace(self.request, fence=self.request.fence + 1)},
            {"measured_batches": replace(
                self.request, candidate_artifact_sha256="f" * 64)},
            {"measured_batches": replace(
                self.request, schedule=replace(self.schedule, batches=self.schedule.batches[::-1]))},
        )
        with tempfile.TemporaryDirectory() as temp:
            for index, changes in enumerate(failures):
                destination = Path(temp) / f"rejected-{index}"
                with self.subTest(index=index), self.assertRaises(ValueError):
                    self.execute(destination, **changes)
                self.assertFalse(destination.exists())

    def test_combining_optimizer_and_measured_custody_must_match_identity(self):
        foreign = runner.OptimizerContinuityRequest(
            "unrelated-run", self.schedule.predecessor_checkpoint_sha256,
            self.schedule.fence)
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "rejected"
            with self.assertRaisesRegex(ValueError, "pins diverge"):
                self.execute(destination, optimizer_continuity=foreign)
            self.assertFalse(destination.exists())

    def test_positive_combined_custody_and_optimizer_exposures_match_actual_loop(self):
        from labyrinth.minilm_optimizer_state_resume import MazeOptimizerResumePlan

        matching = runner.OptimizerContinuityRequest(
            self.schedule.run_id, self.schedule.predecessor_checkpoint_sha256,
            self.schedule.fence)
        expected_suffix = tuple(
            f"c1-s{step:02d}-o{offset}-{row.exposure_id[:12]}"
            for step in range(16) for offset, row in enumerate(
                runner.select_executed_training_batch(
                    self.fixture.signed.cycles[1], step, "correct", self.schedule)))
        exposures_by_arm = {}
        for arm in runner.PREREG_ARMS:
            plan = MazeOptimizerResumePlan(
                scope=self.fixture.signed.scope,
                cycles=self.fixture.signed.cycles,
                weak_cycle_one=(self.schedule.weak_skill,),
                arm=arm, steps_per_cycle=16, learning_rate=.002,
                logical_run_id=self.schedule.run_id,
                base_checkpoint_sha256="a" * 64,
                prior_checkpoint_sha256=self.schedule.predecessor_checkpoint_sha256,
                fence=self.schedule.fence,
                measured_cycle_one=self.schedule,
                measured_schedule_sha256=self.request.independently_pinned_schedule_sha256,
                measured_candidate_artifact_sha256=(
                    self.request.candidate_artifact_sha256),
            )
            actual = plan.exposures()[64:]
            self.assertEqual(tuple(key for key, _ in actual), expected_suffix)
            exposures_by_arm[arm] = actual
        self.assertEqual(exposures_by_arm["correct"], exposures_by_arm["zero_update"])
        self.assertEqual(exposures_by_arm["correct"], exposures_by_arm["zero_lr"])
        self.assertNotEqual(exposures_by_arm["correct"], exposures_by_arm["inverted"])
        self.assertNotEqual(exposures_by_arm["correct"], exposures_by_arm["randomized_labels"])
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "neural"
            with patch.dict("sys.modules", {"torch": None}):
                with self.assertRaises(ModuleNotFoundError):
                    self.execute(destination, optimizer_continuity=matching)
            self.assertFalse(destination.exists())

    def test_direct_python_m_uses_one_class_identity_for_signed_and_optimizer_modules(self):
        # A real -m invocation creates __main__.TrainingCycle, but the signed
        # binder/optimizer source imports the canonical qualified module.
        # Test the same entrypoint semantics in a fresh process without Torch.
        program = """
import runpy, sys
sys.argv = ['probe', '--output-parent', sys.argv[1], '--run-name', 'identity-probe']
try:
    runpy.run_module('scripts.qualify_maze_conditioned_minilm_cpu',
                     run_name='__main__', alter_sys=True)
except SystemExit as exc:
    assert exc.code == 0, exc.code
canonical = sys.modules['scripts.qualify_maze_conditioned_minilm_cpu']
from labyrinth.measured_neural_batch_binding import TrainingExample
from labyrinth.minilm_optimizer_state_resume import MazeOptimizerResumePlan
assert TrainingExample is canonical.TrainingExample
scope = canonical.default_scope()
cycles = canonical.build_training_cycles(scope, weak_cycle_one=('negative_polarity',))
p = MazeOptimizerResumePlan(
    scope=scope, cycles=cycles, weak_cycle_one=('negative_polarity',),
    arm='correct', steps_per_cycle=4, learning_rate=.002,
    logical_run_id='identity-probe', base_checkpoint_sha256='a'*64,
    prior_checkpoint_sha256='b'*64, fence=4)
p.validate()
print('SINGLE_CANONICAL_SOURCE_IDENTITY_PASS')
"""
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [sys.executable, "-c", program, temp],
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True, text=True, timeout=20,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("SINGLE_CANONICAL_SOURCE_IDENTITY_PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
