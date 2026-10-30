"""Source contract for the offline three-process pretrained resume probe."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from scripts.qualify_minilm_resume_cpu import (
    SCHEMA, checkpoint_exposures, main, plan_output, training_visits,
    verify_reference_resume,
)


def _receipts():
    reference = {
        "schema": SCHEMA, "stage": "reference", "graduation_authorized": False,
        "base_digest": "a" * 64, "initial_digest": "b" * 64,
        "initial_lora_digest": "f" * 64,
        "seed": 12, "steps": 8, "split_at": 4,
        "process_pid": 101,
        "final_trainable_digest": "c" * 64,
        "final_lora_digest": "e" * 64,
        "training_exposures": [["case", "d" * 64]],
        "heldout_predictions": [1, 0], "transfer_predictions": [0, 1],
        "heldout_correct": 2, "transfer_correct": 2,
    }
    resumed = dict(reference, process_pid=103, stage="resume")
    checkpoint = {name: reference[name] for name in
                  ("base_digest", "initial_digest", "initial_lora_digest",
                   "seed", "steps", "split_at")}
    checkpoint["process_pid"] = 102
    return reference, resumed, checkpoint


class MiniLMResumeContractTests(unittest.TestCase):
    def test_readonly_default_plans_unique_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([
                    "--output-parent", temporary,
                    "--run-name", "single-safe-run",
                    "--steps", "8", "--split-at", "4",
                ]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["schema"], SCHEMA)
            self.assertFalse(report["execute"])
            self.assertFalse(report["graduation_authorized"])
            self.assertFalse(Path(report["run_root"]).exists())
            (Path(temporary) / "single-safe-run").mkdir()
            with self.assertRaises(FileExistsError):
                plan_output(temporary, "single-safe-run")
            for bad in ("../escape", "bad name", ".hidden", "bad\\name"):
                with self.subTest(bad=bad):
                    with self.assertRaises(ValueError):
                        plan_output(temporary, bad)

    def test_training_exposure_records_two_disjoint_cycles(self):
        visits = training_visits(8)
        self.assertEqual(len(visits), 32)
        self.assertEqual(len({case for case, _, _ in visits}), 32)
        self.assertTrue(all(case.startswith("cycle0-") for case, _, _ in visits[:16]))
        self.assertTrue(all(case.startswith("cycle1-") for case, _, _ in visits[16:]))
        self.assertFalse({digest for _, digest, _ in visits[:16]}
                         & {digest for _, digest, _ in visits[16:]})
        self.assertEqual(checkpoint_exposures(8, 4),
                         tuple((case, digest) for case, digest, _ in visits[:16]))
        for steps, split in ((0, 1), (7, 3), (8, 0), (8, 8), (34, 4)):
            with self.subTest(steps=steps, split=split):
                with self.assertRaises(ValueError):
                    checkpoint_exposures(steps, split)

    def test_exact_state_and_predictions_required_for_resume(self):
        reference, resumed, checkpoint = _receipts()
        self.assertIsNone(verify_reference_resume(reference, resumed, checkpoint))
        with self.assertRaisesRegex(ValueError, "stage receipt"):
            verify_reference_resume(reference, dict(resumed, stage="reference"),
                                    checkpoint)
        with self.assertRaisesRegex(ValueError, "stage receipt"):
            verify_reference_resume(reference,
                                    dict(resumed, graduation_authorized=True),
                                    checkpoint)
        for name, invalid in (
            ("final_trainable_digest", "e" * 64),
            ("final_lora_digest", "d" * 64),
            ("training_exposures", [["different-case", "d" * 64]]),
            ("heldout_predictions", [0, 0]),
            ("transfer_predictions", [1, 1]),
            ("heldout_correct", 1),
            ("transfer_correct", 1),
        ):
            with self.subTest(name=name):
                modified = dict(resumed, **{name: invalid})
                with self.assertRaisesRegex(ValueError, "interrupted continuation differs"):
                    verify_reference_resume(reference, modified, checkpoint)
        modified = dict(checkpoint, initial_digest="e" * 64)
        with self.assertRaisesRegex(ValueError, "lineage mismatch"):
            verify_reference_resume(reference, resumed, modified)
        with self.assertRaisesRegex(ValueError, "fresh-process"):
            verify_reference_resume(reference, dict(resumed, process_pid=101), checkpoint)
        missing = dict(resumed)
        missing.pop("final_trainable_digest")
        with self.assertRaisesRegex(ValueError, "missing exact resume"):
            verify_reference_resume(reference, missing, checkpoint)
        with self.assertRaisesRegex(ValueError, "did not update"):
            verify_reference_resume(
                dict(reference, initial_lora_digest="e" * 64),
                dict(resumed, initial_lora_digest="e" * 64),
                dict(checkpoint, initial_lora_digest="e" * 64),
            )


if __name__ == "__main__":
    unittest.main()
