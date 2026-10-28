"""Tests for the *experimental*, SHA-pinned Minotaur V4 inference path."""
from pathlib import Path
import unittest

from labyrinth.minotaur_transition_cli import predict_guarded_transition

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT / "models" / "minotaur_transition_v4_development" / "model.safetensors"


class TransitionCliTests(unittest.TestCase):
    def test_real_experimental_checkpoint_and_guarded_prediction(self):
        try:
            import torch, safetensors.torch
        except ImportError:
            self.skipTest("optional neural runtime absent")
        from labyrinth.minotaur_transition_curriculum import (
            CurriculumConfig, build_transition_splits,
        )
        cases=build_transition_splits(CurriculumConfig(
            seed=813, train_cases=12, dev_cases=12, transfer_cases=12))["structure_disjoint_transfer"].cases
        for case in cases:
            result=predict_guarded_transition(CHECKPOINT,case.question,case.steps)
            self.assertEqual(result["qualification"],"NOT_ESTABLISHED")
            self.assertEqual(result["graduation_authority"],"none")
            self.assertEqual(result["neural_first_error_step"],case.first_error_step)
            self.assertEqual(result["disposition"],"ALL_CORRECT" if case.first_error_step is None else "ERROR_STEP")

    def test_unsupported_program_abstains_without_model_execution(self):
        result=predict_guarded_transition(CHECKPOINT,
            "unsupported program",("Step 1: A=1, B=2",))
        self.assertEqual(result["disposition"],"ABSTAIN")
        self.assertEqual(result["graduation_authority"],"none")

if __name__=="__main__":
    unittest.main()
