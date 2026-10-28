"""A finite-dimensional neural-addition primitive composed by explicit algorithms."""
import unittest

from labyrinth.minotaur_algebraic_composer_v5 import predict_composed_operators

class AlgebraicComposerTests(unittest.TestCase):
    def test_known_add_primitive_composes_sub_and_mul(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch tested in neural CI")
        class PerfectLearnedAddition:
            def eval(self):
                return self
            def __call__(self,op,x,y):
                return torch.nn.functional.one_hot((x+y)%97,97).float()*10
        op=torch.tensor([0,1,2,0,2,1,2],dtype=torch.long)
        x=torch.tensor([30,10,3,96,42,0,95],dtype=torch.long)
        y=torch.tensor([20,11,13,1,0,70,96],dtype=torch.long)
        values=predict_composed_operators(PerfectLearnedAddition(),op,x,y,semantic_ids=(0,1,2))
        expected=torch.tensor([50,96,39,0,0,27,(95*96)%97])
        self.assertTrue(torch.equal(values,expected))

    def test_bad_neural_addition_does_not_pass_as_symbolic_math(self):
        import torch
        class Broken:
            def eval(self):
                return self
            def __call__(self,op,x,y):
                return torch.nn.functional.one_hot(torch.zeros_like(x),97).float()
        result=predict_composed_operators(Broken(),torch.tensor([0,2]),torch.tensor([17,8]),torch.tensor([5,6]),semantic_ids=(0,1,2))
        self.assertEqual(tuple(result.tolist()),(0,0))

    def test_op_mapping_and_domain_validation(self):
        import torch
        class Dummy:
            def eval(self):return self
            def __call__(self,op,x,y):
                return torch.nn.functional.one_hot((x+y)%97,97).float()
        x=torch.tensor([9],dtype=torch.long)
        y=torch.tensor([4],dtype=torch.long)
        prediction=predict_composed_operators(Dummy(),torch.tensor([2]),x,y,semantic_ids=(2,1,0))
        self.assertEqual(int(prediction[0]),13)
        with self.assertRaises(ValueError):
            predict_composed_operators(Dummy(),torch.tensor([4]),x,y,semantic_ids=(0,1,2))
