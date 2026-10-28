"""Source-controlled V5 compact neural checkpoint and rollout verification."""
import unittest
from pathlib import Path
from labyrinth.minotaur_algebraic_checkpoint_v5 import (
    load_pinned_neural_add, evaluate_checkpoint_programs,
)

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/"models"/"minotaur_algebraic_v5_development"/"model.safetensors"

class CheckpointV5Tests(unittest.TestCase):
    def test_real_neural_checkpoint_and_unseen_program_rollout(self):
        try:
            import torch, safetensors.torch
        except ImportError:
            self.skipTest("optional CPU neural runtime absent")
        model, op_ids=load_pinned_neural_add(CHECKPOINT)
        self.assertEqual(sum(p.numel() for p in model.parameters()),32258)
        report=evaluate_checkpoint_programs(model,op_ids,seed=818,cases=20)
        self.assertEqual(report["disposition"],"SCORED")
        self.assertEqual(report["qualification"],"NOT_ESTABLISHED")
        self.assertEqual(report["metrics"]["exact_first_error_accuracy"],1.0)

    def test_unpinned_checkpoint_refused(self):
        try:
            import torch, safetensors.torch
        except ImportError:
            self.skipTest("optional CPU neural runtime absent")
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/"tampered.safetensors"
            b=bytearray(CHECKPOINT.read_bytes())
            b[-1] ^= 1
            p.write_bytes(b)
            with self.assertRaises(ValueError):
                load_pinned_neural_add(p)
