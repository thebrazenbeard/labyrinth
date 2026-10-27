"""Scratch-initialized Minotaur is a neural process verifier, not an adapter."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from labyrinth.minotaur_scratch import (
    ScratchConfig, encoded_process_cases, tokenize_process_case,
)


class ScratchDataTests(unittest.TestCase):
    def test_classes_and_split_custody(self):
        train = encoded_process_cases(123, 3, ("fresh_numbers", "new_composition"))
        dev = encoded_process_cases(456, 3, ("fresh_numbers", "new_composition"))
        transfer = encoded_process_cases(789, 3, ("longer_program",))
        self.assertEqual([len(x) for x in (train, dev, transfer)], [12, 12, 6])
        self.assertEqual(len({c["case_id"] for s in (train, dev, transfer) for c in s}), 30)
        for dataset in (train, dev, transfer):
            self.assertEqual(sum(c["label"] == 0 for c in dataset), len(dataset) // 2)
            for c in dataset:
                self.assertTrue(0 <= c["label"] <= 6)
                self.assertGreater(len(c["tokens"]), 10)
                self.assertTrue(all(0 <= t < 256 for t in c["tokens"]))
        self.assertTrue(all(c["slice"] == "longer_program" for c in transfer))
        self.assertEqual(train, encoded_process_cases(123, 3, ("fresh_numbers", "new_composition")))

    def test_invalid_input_does_not_silently_truncate(self):
        data = encoded_process_cases(123, 1, ("fresh_numbers",))
        with self.assertRaises(ValueError):
            tokenize_process_case("not a register program", ("Step 1: A=3, B=4",))
        with self.assertRaises(ValueError):
            ScratchConfig(width=47, heads=4).validate()
        self.assertEqual(ScratchConfig().validate().max_steps, 6)
        self.assertGreater(len(data[0]["tokens"]), 10)


try:
    import torch
    import safetensors.torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch+safetensors are absent in CI")
class ScratchTrainingTests(unittest.TestCase):
    def test_full_weights_start_random_and_fresh_reload(self):
        from labyrinth.minotaur_scratch import run_scratch, reload_scratch
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "new-model"
            report = run_scratch(run, ScratchConfig(width=32, heads=4, layers=1),
                                 epochs=1, train_per_slice=2, eval_per_slice=2,
                                 batch_size=8)
            self.assertEqual(report["lineage"]["initialization"], "random_from_seed")
            self.assertIsNone(report["lineage"]["pretrained_checkpoint"])
            self.assertNotEqual(report["initial_weight_sha256"],
                                report["trained_weight_sha256"])
            self.assertEqual(report["qualification"], "NOT_ESTABLISHED")
            self.assertTrue((run / "minotaur.safetensors").is_file())
            restored = reload_scratch(run)
            self.assertEqual(restored["trained_weight_sha256"],
                             report["trained_weight_sha256"])
            self.assertEqual(restored["eval_prediction_sha256"],
                             report["eval_prediction_sha256"])
            with self.assertRaises(FileExistsError):
                run_scratch(run, ScratchConfig(width=32, heads=4, layers=1),
                            epochs=1, train_per_slice=2, eval_per_slice=2)
            (run / "minotaur.safetensors").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                reload_scratch(run)
