"""Compact Minotaur neural rules must transfer from missing atomic examples."""
from __future__ import annotations
import unittest
try:
    import torch
except ImportError:
    torch = None

@unittest.skipUnless(torch is not None, "torch optional")
class AlgebraicNeuralTests(unittest.TestCase):
    def test_learns_full_mod97_atoms_from_36_small_partial_examples(self):
        from labyrinth.minotaur_algebraic_rules import (
            build_algebraic_head, incomplete_training_atoms, train_rule_head,
            all_atom_accuracy, corrupt_labels_within_op,
        )
        torch.set_num_threads(min(2, torch.get_num_threads()))
        atoms=incomplete_training_atoms(47,12)
        self.assertEqual(len(atoms),36)
        self.assertEqual({op for op,a,b,y in atoms},{0,1,2})
        self.assertTrue(all(a<=8 and b<=8 and ((a>=b) if op==1 else True)
                            for op,a,b,y in atoms))
        head=build_algebraic_head(torch,seed=19)
        self.assertEqual(sum(p.numel() for p in head.parameters()),12)
        initial=head.coefficients.detach().clone()
        frozen=all_atom_accuracy(head,torch)
        steps=train_rule_head(head,atoms,torch,steps=500,learning_rate=0.06)
        self.assertEqual(steps,500)
        self.assertFalse(torch.equal(initial,head.coefficients.detach()))
        candidate=all_atom_accuracy(head,torch)
        self.assertEqual(candidate["correct"],3*97*97)
        self.assertLess(frozen["correct"], 10000)
        bad=corrupt_labels_within_op(atoms,seed=59)
        self.assertEqual(len(bad),len(atoms))
        self.assertTrue(all(x[:3]==y[:3] and x[3]!=y[3] for x,y in zip(atoms,bad)))
        bad_head=build_algebraic_head(torch,seed=19)
        train_rule_head(bad_head,bad,torch,steps=500,learning_rate=0.06)
        self.assertLess(all_atom_accuracy(bad_head,torch)["correct"],candidate["correct"])

    def test_zero_lr_and_rejected_invalid_examples(self):
        from labyrinth.minotaur_algebraic_rules import (
            build_algebraic_head,incomplete_training_atoms,train_rule_head,
        )
        data=incomplete_training_atoms(47,12)
        head=build_algebraic_head(torch,seed=19)
        before=head.coefficients.detach().clone()
        train_rule_head(head,data,torch,steps=40,learning_rate=0.0)
        self.assertTrue(torch.equal(before,head.coefficients.detach()))
        with self.assertRaises(ValueError):
            train_rule_head(head, data+[(0,11,11,22)],torch,steps=10,learning_rate=.06)

    def test_v4_free_running_rollout_with_modular_wrap(self):
        from labyrinth.minotaur_transition_v4 import (
            TransitionConfig,build_transition_model,TransitionFeatures,
        )
        from labyrinth.minotaur_algebraic_rules import (
            build_algebraic_head,incomplete_training_atoms,train_rule_head,
        )
        head=build_algebraic_head(torch,seed=19)
        train_rule_head(head,incomplete_training_atoms(47,12),torch,
                        steps=500,learning_rate=.06)
        net=build_transition_model(TransitionConfig(architecture="algebraic_rule",seed=19))
        net.algebraic_head.load_state_dict(head.state_dict())
        net.eval()
        # A starts 95, B starts 5. ADD A B -> 3; MUL B A -> 15;
        # SUB A B -> 85. This combines wrap and register operands unseen in train.
        data=TransitionFeatures(
            initial_states=torch.tensor([[95,5]],dtype=torch.long),
            operations=torch.tensor([[0,2,1,0,0,0]],dtype=torch.long),
            targets=torch.tensor([[0,1,0,0,0,0]],dtype=torch.long),
            operand_is_register=torch.tensor([[True,True,True,False,False,False]]),
            operand_values=torch.tensor([[1,0,1,0,0,0]],dtype=torch.long),
            mask=torch.tensor([[True,True,True,False,False,False]]),
            step_numbers=torch.tensor([[1,2,3,0,0,0]],dtype=torch.long),
        )
        with torch.no_grad():
            out=net(data)
        self.assertEqual(out["predicted_states"][0,:3].tolist(),
                         [[3,5],[3,15],[85,15]])
        # No gold target is supplied at inference.
        self.assertEqual(sum(p.numel() for p in net.parameters()),12)


if __name__ == "__main__":
    unittest.main()
