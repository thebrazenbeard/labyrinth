"""Behavioral regression for the bounded scratch-model memorization probe."""
import json
from pathlib import Path
import tempfile
import unittest


try:
    import torch  # noqa: F401
    import safetensors.torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch+safetensors are absent in CI")
class ScratchMicrofitTests(unittest.TestCase):
    def test_32_cases_receive_real_updates_and_memorize(self):
        from labyrinth.minotaur_scratch import ScratchConfig, run_microfit

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "microfit-only"
            report = run_microfit(
                output,
                ScratchConfig(seed=411, width=32, heads=4, layers=1),
            )

            self.assertEqual(report["case_count"], 32)
            self.assertEqual(report["class_counts"], {"0": 16, "nonzero": 16})
            self.assertEqual(report["lineage"]["initialization"], "random_from_seed")
            self.assertIsNone(report["lineage"]["pretrained_checkpoint"])
            self.assertGreater(report["optimizer_steps"], 0)
            self.assertNotEqual(report["initial_weight_sha256"],
                                report["trained_weight_sha256"])
            self.assertGreater(report["before"]["loss"], report["after"]["loss"])
            self.assertEqual(report["after"]["accuracy"], 1.0)
            self.assertTrue(report["success"])
            self.assertTrue(report["gradient_norms"]["all_finite"])
            self.assertTrue(report["gradient_norms"]["all_positive"])
            self.assertGreater(report["gradient_norms"]["minimum"], 0.0)
            self.assertGreaterEqual(report["gradient_norms"]["maximum"],
                                    report["gradient_norms"]["minimum"])
            self.assertTrue((output / "microfit.safetensors").is_file())
            disk_report = json.loads((output / "microfit-report.json").read_text())
            self.assertEqual(disk_report, report)
            with self.assertRaises(FileExistsError):
                run_microfit(output, ScratchConfig(width=32, heads=4, layers=1))


if __name__ == "__main__":
    unittest.main()
