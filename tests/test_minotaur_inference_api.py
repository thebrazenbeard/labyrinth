"""A neural evaluation never depends on teacher states or perturbs RNG."""
import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipUnless(torch is not None, "optional torch")
class InferenceBoundaryTests(unittest.TestCase):
    def test_algebra_constructor_does_not_change_global_rng(self):
        from labyrinth.minotaur_transition_v4 import TransitionConfig, build_transition_model
        before=torch.get_rng_state().clone()
        model=build_transition_model(TransitionConfig(architecture="algebraic_rule",seed=19))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in model.parameters()),12)

    def test_inference_ignores_training_mode_and_forbids_gold(self):
        from labyrinth.minotaur_transition_v4 import (
            TransitionConfig,TransitionFeatures,build_transition_model,
            predict_transition_inference,
        )
        model=build_transition_model(TransitionConfig(architecture="algebraic_rule",seed=19))
        model.train()
        f=TransitionFeatures(
            torch.tensor([[1,2]]),
            torch.zeros((1,6),dtype=torch.long),
            torch.zeros((1,6),dtype=torch.long),
            torch.zeros((1,6),dtype=torch.bool),
            torch.zeros((1,6),dtype=torch.long),
            torch.tensor([[True,False,False,False,False,False]]),
            torch.tensor([[1,0,0,0,0,0]]),
        )
        result=predict_transition_inference(model,f)
        self.assertEqual(tuple(result["predicted_states"].shape),(1,6,2))
        self.assertTrue(model.training,"Training state must be restored after inference")
        self.assertFalse(result["predicted_states"].requires_grad)
        with self.assertRaises(TypeError):
            predict_transition_inference(model,f,
                torch.zeros((1,6,2),dtype=torch.long))


if __name__=="__main__":
    unittest.main()
