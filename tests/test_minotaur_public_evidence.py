"""Real trained Minotaur neural inference must compose with independent V5 evaluation."""
import unittest
try:
    import torch
except ImportError:
    torch=None

from labyrinth.minotaur_algebraic_eval_v5 import (
    AtomicEvaluationCase, ProgramRolloutCase,
)


@unittest.skipUnless(torch is not None,"torch optional")
class MainAlgebraicEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from labyrinth.minotaur_algebraic_rules import (
            build_algebraic_head,incomplete_training_atoms,train_rule_head,
        )
        torch.set_num_threads(min(torch.get_num_threads(),2))
        cls.train=incomplete_training_atoms(47,12)
        cls.head=build_algebraic_head(torch,seed=19)
        train_rule_head(cls.head,cls.train,torch,steps=500,learning_rate=0.06)

    def test_independent_evaluator_scores_unseen_atoms_and_mixed_programs(self):
        from labyrinth.minotaur_public_evidence import evaluate_compact_public
        cases=[
            AtomicEvaluationCase("add-wrap","ADD",95,5,3,"reserved-1"),
            AtomicEvaluationCase("sub-wrap","SUB",3,15,85,"reserved-2"),
            AtomicEvaluationCase("mul-large","MUL",23,19,(23*19)%97,"reserved-3"),
        ]
        clean=ProgramRolloutCase(
            "clean",(95,5),(
                ("ADD","A","B"),("MUL","B","A"),("SUB","A","B")
            ),((3,5),(3,15),(85,15)),None,"program-A",
        )
        faulty=ProgramRolloutCase(
            "faulty",(95,5),(
                ("ADD","A","B"),("MUL","B","A"),("SUB","A","B")
            ),((3,5),(3,15),(84,15)),3,"program-A",
        )
        report=evaluate_compact_public(
            self.head,cases,[clean,faulty],self.train,torch,
        )
        self.assertEqual(report["disposition"],"PUBLIC_DIAGNOSTIC")
        self.assertEqual(report["atomic"]["metrics"]["exact_correct"],3)
        self.assertEqual(report["program"]["metrics"]["exact_first_error_correct"],2)
        self.assertEqual(report["atomic"]["structural_overlap_audit"]["exact_training_pair_overlap"],[])
        self.assertEqual(report["graduation_authority"],"none")
        self.assertEqual(report["qualification"],"NOT_ESTABLISHED")

    def test_training_overlap_and_false_claims_are_rejected_not_scored(self):
        from labyrinth.minotaur_public_evidence import evaluate_compact_public
        overlapping=AtomicEvaluationCase("trained","ADD",0,0,0,"reserved")
        bad=AtomicEvaluationCase("false-key","ADD",95,5,4,"reserved")
        train=self.train + [(0,0,0,0)]
        report=evaluate_compact_public(self.head,[overlapping],[],train,torch)
        self.assertEqual(report["disposition"],"REFUSED")
        self.assertEqual(report["reason"],"training_atom_overlap")
        report=evaluate_compact_public(self.head,[bad],[],self.train,torch)
        self.assertEqual(report["disposition"],"REFUSED")
        self.assertEqual(report["reason"],"independent_evaluator_refused")
        self.assertEqual(report["graduation_authority"],"none")


    def test_invalid_program_refused_before_any_model_callback(self):
        from unittest.mock import patch
        from labyrinth.minotaur_public_evidence import evaluate_compact_public
        atomic=[AtomicEvaluationCase("safe","ADD",95,5,3,"heldout")]
        forged=ProgramRolloutCase(
            "forged",(95,5),(("ADD","A","B"),),
            ((3,5),),1,"bad-claim",
        )
        with patch.object(self.head,"predict",side_effect=AssertionError("model invoked")):
            result=evaluate_compact_public(self.head,atomic,[forged],self.train,torch)
        self.assertEqual(result["disposition"],"REFUSED")
        self.assertEqual(result["reason"],"independent_evaluator_refused")


    def test_inference_error_abstains_and_restores_model_state(self):
        from unittest.mock import patch
        from labyrinth.minotaur_public_evidence import evaluate_compact_public
        atomic=[AtomicEvaluationCase("safe","ADD",95,5,3,"heldout")]
        original=self.head.training
        with patch.object(self.head,"predict",side_effect=RuntimeError("broken runtime")):
            report=evaluate_compact_public(self.head,atomic,[],self.train,torch)
        self.assertEqual(report["disposition"],"REFUSED")
        self.assertEqual(report["reason"],"neural_inference_failed")
        self.assertEqual(self.head.training,original)

    def test_malformed_scalar_outputs_never_become_exact_scores(self):
        from unittest.mock import patch
        from labyrinth.minotaur_public_evidence import evaluate_compact_public
        atomic=[AtomicEvaluationCase("safe","ADD",95,5,3,"heldout")]
        for bad in (3.75,3.0,-0.1,True,float("nan"),float("inf"),-1,97):
            with self.subTest(output=bad):
                with patch.object(self.head,"predict",return_value=torch.tensor(bad)):
                    report=evaluate_compact_public(self.head,atomic,[],self.train,torch)
                self.assertEqual(report["disposition"],"REFUSED")
                self.assertEqual(report["reason"],"neural_inference_failed")
                self.assertEqual(report["qualification"],"NOT_ESTABLISHED")
        with patch.object(self.head,"predict",return_value=torch.tensor([3])):
            report=evaluate_compact_public(self.head,atomic,[],self.train,torch)
        self.assertEqual(report["disposition"],"REFUSED")


if __name__=="__main__":
    unittest.main()
