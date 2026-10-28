"""Gold-state computations must be absent from inference feature encoding."""
import unittest
from unittest.mock import patch
try:
    import torch
except ImportError:
    torch=None

@unittest.skipUnless(torch is not None,"torch optional")
class OracleFreeInferenceTests(unittest.TestCase):
    def test_inference_inputs_do_not_invoke_teacher_oracle(self):
        from labyrinth.minotaur_transition_v4 import (
            TransitionConfig,encode_transition_cases,build_transition_model,
        )
        class InputCase:
            question="Execute operations.\nInitial registers: A=3, B=4\n1. ADD A B"
            steps=("Step 1: A=7, B=4",)
        with patch("labyrinth.minotaur_transition_v4._teacher_states",
                   side_effect=RuntimeError("gold oracle was run")):
            features, gold=encode_transition_cases(
                [InputCase()],TransitionConfig(architecture="algebraic_rule"),torch,
                include_supervision=False,
            )
        self.assertIsNone(gold)
        model=build_transition_model(
            TransitionConfig(architecture="algebraic_rule"))
        model.eval()
        with torch.no_grad():
            prediction=model(features)
        self.assertEqual(prediction["predicted_states"].shape,(1,6,2))
        with self.assertRaisesRegex(ValueError,"teacher.*evaluation"):
            model(features,teacher_states=torch.zeros((1,6,2),dtype=torch.long))

if __name__=="__main__":
    unittest.main()
