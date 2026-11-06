"""Adversarial tests: frozen MiniLM weights cannot change public predictions.

These are source-level report-coherence checks, not proof of actual inference.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from test_physical_report_bridge import (
    digest,
    physical_report,
)
from labyrinth.physical_report_bridge import inspect_public_minilm_report


def two_cycle_report() -> dict:
    report = physical_report()
    for arm, arm_record in report["arms"].items():
        first = arm_record["cycles"][0]
        second = deepcopy(first)
        second["cycle"] = 1
        second["trainable_sha256_before"] = first["trainable_sha256_after"]
        second["lora_sha256_before"] = first["lora_sha256_after"]
        if arm in ("correct_labels", "corrupted_labels"):
            second["trainable_sha256_after"] = digest(f"{arm}-second")
            second["lora_sha256_after"] = digest(f"{arm}-lora-second")
        arm_record["cycles"].append(second)
    return report


class FrozenModelPredictionCoherenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "report.json"

    def inspect(self, report: dict):
        content = (json.dumps(report, sort_keys=True) + "\n").encode("utf-8")
        self.path.write_bytes(content)
        return inspect_public_minilm_report(
            self.path, expected_report_sha256=sha256(content).hexdigest(),
            expected_harness_sha256=digest("harness"), expected_run_id="run-001",
            expected_model_id="sentence-transformers/all-MiniLM-L6-v2",
        )

    def test_frozen_baseline_response_cannot_change_across_cycles(self) -> None:
        report = two_cycle_report()
        later = report["arms"]["baseline"]["cycles"][1]
        later["heldout_predictions"][0]["predicted_label"] = 1
        later["heldout_predictions"][0]["response"] = "1"
        later["heldout_accuracy"] = 0.5
        with self.assertRaisesRegex(ValueError, "frozen.*prediction|prediction.*frozen"):
            self.inspect(report)

    def test_sham_response_cannot_differ_from_same_weight_baseline(self) -> None:
        report = physical_report()
        sham = report["arms"]["zero_update"]["cycles"][0]
        sham["transfer_predictions"][0]["predicted_label"] = 1
        sham["transfer_predictions"][0]["response"] = "1"
        sham["transfer_accuracy"] = 0.5
        with self.assertRaisesRegex(ValueError, "frozen.*prediction|prediction.*frozen"):
            self.inspect(report)

    def test_declared_optimizer_steps_cannot_disagree_with_recorded_batches(self) -> None:
        report = physical_report()
        report["schema"] = "labyrinth.minilm-cpu-development.v2"
        report["steps_per_cycle"] = 8
        # The actual development script uses this setting to build every
        # training-arm schedule. This report claims 8 but records only 2.
        with self.assertRaisesRegex(ValueError, "steps_per_cycle|declared.*step"):
            self.inspect(report)

    def test_declared_training_prompt_digest_list_must_match_batch_rows(self) -> None:
        report = physical_report()
        report["schema"] = "labyrinth.minilm-cpu-development.v2"
        for arm, arm_record in report["arms"].items():
            if arm == "baseline":
                arm_record["cycles"][0]["training_prompt_sha256"] = []
            else:
                arm_record["cycles"][0]["training_prompt_sha256"] = [digest("alien-prompt")]
        # The physical script emits this redundant field directly from the
        # batch receipts, so a mismatch is a source-format contradiction.
        with self.assertRaisesRegex(ValueError, "training_prompt_sha256|training prompt"):
            self.inspect(report)


if __name__ == "__main__":
    unittest.main()
