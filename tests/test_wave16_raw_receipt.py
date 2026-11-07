"""Host-private raw wire retention for independently replayable batch checks."""

import hashlib
import unittest
from unittest.mock import patch

from labyrinth.minotaur_process_adapter import ProcessModelAdapter, _ABSTAIN
from scripts import switchback_model_trial as trial


class RawResponseReceiptTests(unittest.TestCase):
    def _setup(self):
        adapter = ProcessModelAdapter(
            mode="fixed", actions=(), generation_seed=0,
            timeout_seconds=1.0, max_output_bytes=1024)
        config = trial.TrialConfig(0, 0, "fixed", 1, 1.0, 3.0, 1024)
        return config, adapter

    def test_actual_fixed_child_retains_exact_bounded_raw_response(self):
        config, adapter = self._setup()
        report = trial.run_trial(config, adapter)
        turn = report["turns"][0]
        self.assertEqual(turn["raw_response"], _ABSTAIN)
        self.assertEqual(turn["response_sha256"],
                         hashlib.sha256(_ABSTAIN.encode("utf-8")).hexdigest())
        self.assertTrue(report["process_receipt_consistency_valid"])
        self.assertFalse(report["physical_child_execution_independently_attested"])
        self.assertEqual(report["report_classification"],
                         "HOST_PRIVATE_SEED_AND_MODEL_OUTPUT_DO_NOT_PUBLISH")
        self.assertFalse(report["verified_success"])

    def test_unbounded_or_invalid_utf8_output_never_enters_raw_report(self):
        for raw in ("Z" * 1025, "\ud800"):
            with self.subTest(kind=repr(raw[:10])):
                config, adapter = self._setup()
                with patch.object(ProcessModelAdapter, "generate", return_value=raw):
                    report = trial.run_trial(config, adapter)
                self.assertFalse(report["verified_success"])
                self.assertFalse(report["process_receipt_consistency_valid"])
                self.assertEqual(report["turns"][0]["raw_response"], None)
                self.assertEqual(report["terminal_reason"], "EVALUATOR_DISAGREEMENT")


if __name__ == "__main__":
    unittest.main()
