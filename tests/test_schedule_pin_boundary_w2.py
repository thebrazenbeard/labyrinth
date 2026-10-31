"""Hostile source-only snapshot and process-boundary TRAIN schedule tests."""

from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from labyrinth.maze_conditioned_minilm_cpu_experiment import (
    PREREG_ARMS, export_measured_cli_schedule_json,
)
from labyrinth.measured_neural_batch_binding import (
    propose_signed_neural_batch_schedule,
)
from labyrinth.schedule_pin_boundary_w2 import resolve_source_pinned_train_batch
from tests import test_qualify_maze_measured_tactics_cpu as signed_fixtures


class SchedulePinBoundaryW2Tests(unittest.TestCase):
    def setUp(self):
        source = signed_fixtures.SignedMeasuredTacticBatchesTests(
            "test_real_signed_two_polarity_score_changes_actual_train_visits")
        source.setUp()
        self.addCleanup(source.doCleanups)
        self.source = source
        self.schedule = propose_signed_neural_batch_schedule(
            declared_scope=source.scope, training_cycles=source.cycles,
            steps_per_cycle=16, binding=source.binding,
            training_exposures=source.training,
            expected_training_roster_sha256=source.training_sha,
            predecessor_checkpoint_sha256=(
                source.fixture.context.predecessor_lora_value_sha256),
            fence=7, **source.inputs,
        )
        self.raw = export_measured_cli_schedule_json(self.schedule)
        self.pins = dict(
            scope=source.scope, steps_per_cycle=16,
            expected_raw_sha256=hashlib.sha256(self.raw).hexdigest(),
            expected_schedule_sha256=self.schedule.schedule_sha256,
            expected_run_id=source.inputs["candidate_run_id"],
            expected_candidate_artifact_sha256=source.fixture.artifact_hash,
            expected_predecessor_checkpoint_sha256=(
                source.fixture.context.predecessor_lora_value_sha256),
            expected_scope_fingerprint=source.scope.fingerprint,
            expected_fence=7,
        )

    def read(self, raw=None, **overrides):
        pins = dict(self.pins)
        pins.update(overrides)
        return resolve_source_pinned_train_batch(
            self.raw if raw is None else raw, arm="correct", step=0, **pins)

    def test_exact_pinned_source_bytes_reconstruct_immutable_train_scalars(self):
        batch = self.read()
        self.assertEqual(len(batch), 4)
        self.assertTrue(all(type(row) is tuple and len(row) == 3
                            and type(row[0]) is str and type(row[1]) is str
                            and type(row[2]) is int and row[2] in (0, 1)
                            for row in batch))
        self.assertEqual(tuple(row[0] for row in batch),
                         tuple(row.exposure_id for row in self.schedule.batches[0]))
        with self.assertRaises(TypeError):
            batch[0] = ("forged", "forged", 1)

    def test_post_export_object_setattr_mutation_cannot_change_snapshot(self):
        expected = self.read()
        object.__setattr__(self.schedule, "batches", self.schedule.batches[::-1])
        object.__setattr__(self.schedule, "schedule_sha256", "f" * 64)
        self.assertEqual(self.read(), expected)

    def test_pre_export_spoof_with_recomputed_local_digest_fails_external_pin(self):
        fake = replace(self.schedule, batches=self.schedule.batches[::-1])
        object.__setattr__(fake, "schedule_sha256", "f" * 64)
        forged_bytes = export_measured_cli_schedule_json(fake)
        with self.assertRaisesRegex(ValueError, "pinned source bytes"):
            self.read(raw=forged_bytes)

    def test_raw_bytes_and_every_separately_supplied_pin_are_mandatory(self):
        invalid = (
            {"expected_raw_sha256": "0" * 64},
            {"expected_schedule_sha256": "0" * 64},
            {"expected_run_id": "other-run"},
            {"expected_candidate_artifact_sha256": "0" * 64},
            {"expected_predecessor_checkpoint_sha256": "0" * 64},
            {"expected_scope_fingerprint": "0" * 64},
            {"expected_fence": 8},
        )
        for altered in invalid:
            with self.subTest(**altered), self.assertRaises(ValueError):
                self.read(**altered)

    def test_untrusted_subclasses_and_noncanonical_bytes_fail_closed(self):
        class FalseArm(str):
            def __eq__(self, other):
                return True

        with self.assertRaises(ValueError):
            resolve_source_pinned_train_batch(
                self.raw, arm=FalseArm("correct"), step=0, **self.pins)
        with self.assertRaises(ValueError):
            resolve_source_pinned_train_batch(
                self.raw, arm="correct", step=True, **self.pins)
        with self.assertRaises(ValueError):
            self.read(raw=self.raw + b" ")
        with self.assertRaises(ValueError):
            self.read(raw=bytearray(self.raw))
        with self.assertRaises(ValueError):
            self.read(raw=b"{}")

    def test_complete_five_arm_inventory_same_exposures_each_step(self):
        for step in (0, 7, 15):
            common = resolve_source_pinned_train_batch(
                self.raw, arm="correct", step=step, **self.pins)
            for arm in PREREG_ARMS:
                self.assertEqual(resolve_source_pinned_train_batch(
                    self.raw, arm=arm, step=step, **self.pins), common)


if __name__ == "__main__":
    unittest.main()
