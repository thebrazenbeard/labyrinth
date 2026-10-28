"""Independent canonical reverse-pair metrics for V6 neural predictions."""
import unittest
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None, "Optional PyTorch required for neural group audit")
class GroupAuditTests(unittest.TestCase):
    def test_pair_complete_not_correlated_directed_row_accuracy(self):
        from labyrinth.minotaur_v6_spectral import PairExamples, audit_grouped_predictions
        x=torch.tensor([1,2,2,3,4],dtype=torch.long)
        y=torch.tensor([2,1,2,4,3],dtype=torch.long)
        labels=torch.remainder(x*y,97)
        examples=PairExamples(x,y,labels)
        class OneBrokenOrientation:
            def eval(self):return self
            def __call__(self,x,y):
                answers=torch.remainder(x*y,97)
                answers=torch.where((x==3)&(y==4),torch.zeros_like(answers),answers)
                return torch.nn.functional.one_hot(answers,97).float()
        r=audit_grouped_predictions(OneBrokenOrientation(),examples)
        self.assertEqual(r["canonical_groups"],3)
        self.assertEqual(r["correct_complete_groups"],2)
        self.assertAlmostEqual(r["pair_complete_accuracy"],2/3)
        self.assertAlmostEqual(r["directed_row_accuracy"],4/5)
        self.assertAlmostEqual(r["reversal_consistency_rate"],.5)
        self.assertEqual(len(r["prediction_sha256"]),64)

    def test_perfect_symmetry_is_not_more_independent_samples(self):
        from labyrinth.minotaur_v6_spectral import PairExamples, audit_grouped_predictions
        x=torch.tensor([1,2,4,5],dtype=torch.long)
        y=torch.tensor([2,1,5,4],dtype=torch.long)
        class Perfect:
            def eval(self):return self
            def __call__(self,a,b):
                return torch.nn.functional.one_hot(torch.remainder(a*b,97),97).float()
        result=audit_grouped_predictions(Perfect(),PairExamples(x,y,torch.remainder(x*y,97)))
        self.assertEqual(result["canonical_groups"],2)
        self.assertEqual(result["directed_rows"],4)
        self.assertEqual(result["correct_complete_groups"],2)
        self.assertEqual(result["pair_complete_accuracy"],1.0)
