"""Exact-version and control-identity regression tests for the v2 byte window."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import unittest

from scripts.qualify_minilm_cpu import PHYSICAL_ARMS, arm_contract
from scripts.qualify_model_bytes_provenance import ModelByteWindowError, _check_report
from test_physical_report_bridge import physical_report


class V2ReportBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("/operator-pinned/base-model")
        self.runner_sha = sha256(b"pinned-v2-runner").hexdigest()
        self.report = physical_report()
        self.report.update({
            "model_id": str(self.root),
            "harness_sha256": self.runner_sha,
            "run_id": "test-run", "seed": 42,
            "steps_per_cycle": 2, "learning_rate": .002,
            "device": "cpu",
            "baseline_compute_matched_to_training": False,
            "reload_method": "NEW_MODEL_OBJECT_SAME_PROCESS",
            "zero_update_control": "SAME_FORWARD_BACKWARD_ADAMW_STEP_WITH_ZERO_LR",
            "arm_contracts": {arm: arm_contract(arm, .002) for arm in PHYSICAL_ARMS},
            "learning_qualification": "NOT_ESTABLISHED_UNTIL_INDEPENDENT_EVALUATION",
        })

    def check(self, report: dict) -> None:
        _check_report(
            report, model_root=self.root, runner_sha=self.runner_sha,
            run_name="test-run", seed=42, steps=2, rate=.002,
        )

    def test_current_v2_control_fields_bind_exactly(self) -> None:
        self.check(self.report)

    def test_historical_three_arm_report_cannot_claim_v2_harness(self) -> None:
        report = deepcopy(self.report)
        report["schema"] = "labyrinth.minilm-cpu-development.v1"
        report["arms"] = {arm: {} for arm in ("correct", "zero_update", "inverted")}
        with self.assertRaisesRegex(ModelByteWindowError, "exact runner/configuration binding"):
            self.check(report)

    def test_four_arm_downgrade_cannot_omit_baseline(self) -> None:
        report = deepcopy(self.report)
        report["arms"].pop("baseline")
        with self.assertRaisesRegex(ModelByteWindowError, "exact v2 four-arm"):
            self.check(report)

    def test_control_identity_cannot_be_rewritten(self) -> None:
        report = deepcopy(self.report)
        report["arm_contracts"]["zero_update"]["optimizer_step"] = False
        with self.assertRaisesRegex(ModelByteWindowError, "physical control contract"):
            self.check(report)

    def test_integer_does_not_masquerade_as_boolean_control(self) -> None:
        report = deepcopy(self.report)
        report["arm_contracts"]["correct_labels"]["optimizer_step"] = 1
        with self.assertRaisesRegex(ModelByteWindowError, "physical control contract"):
            self.check(report)

    def test_run_name_bound_to_output_and_runner(self) -> None:
        report = deepcopy(self.report)
        report["run_id"] = "other-run"
        with self.assertRaisesRegex(ModelByteWindowError, "exact runner/configuration binding"):
            self.check(report)


if __name__ == "__main__":
    unittest.main()
