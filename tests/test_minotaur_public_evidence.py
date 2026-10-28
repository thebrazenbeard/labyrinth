"""Real trained Minotaur neural inference must compose with independent V5 evaluation."""
from contextlib import nullcontext
import unittest
try:
    import torch
except ImportError:
    torch=None

from labyrinth.minotaur_algebraic_eval_v5 import (
    AtomicEvaluationCase, ProgramRolloutCase,
)


class PublicBridgeProtocolTests(unittest.TestCase):
    class _Scalar:
        ndim = 0

        def item(self):
            return 3

    class _FakeTorch:
        long = object()

        def no_grad(self):
            return nullcontext()

        def tensor(self, value, dtype):
            return value

    def test_incomplete_neural_head_refused_before_inference(self):
        from labyrinth.minotaur_public_evidence import evaluate_compact_public

        class IncompleteHead:
            def predict(self, *_args):
                raise AssertionError("incomplete head was invoked")

            def eval(self):
                raise AssertionError("incomplete head was invoked")

        try:
            report = evaluate_compact_public(
                IncompleteHead(), [AtomicEvaluationCase("safe", "ADD", 1, 2, 3)],
                [], [(0, 3, 4, 7)], None,
            )
        except Exception as exc:
            self.fail(f"bridge leaked an exception for incomplete neural head: {exc}")
        self.assertEqual(report["disposition"], "REFUSED")
        self.assertEqual(report["reason"], "unsupported_neural_head")
        self.assertEqual(report["graduation_authority"], "none")

    def test_neural_state_restoration_failure_refuses_scored_result(self):
        from labyrinth.minotaur_public_evidence import evaluate_compact_public
        prediction = self._Scalar()

        class UnrestorableHead:
            training = True

            def eval(self):
                self.training = False

            def predict(self, *_args):
                return prediction

            def train(self, _mode):
                raise RuntimeError("state restoration failed")

        try:
            report = evaluate_compact_public(
                UnrestorableHead(), [AtomicEvaluationCase("safe", "ADD", 1, 2, 3)],
                [], [(0, 3, 4, 7)], self._FakeTorch(),
            )
        except Exception as exc:
            self.fail(f"bridge leaked an exception on state restoration: {exc}")
        self.assertEqual(report["disposition"], "REFUSED")
        self.assertEqual(report["reason"], "neural_state_restore_failed")
        self.assertEqual(report["graduation_authority"], "none")

    def test_silent_noop_state_restoration_refuses_scored_result(self):
        from labyrinth.minotaur_public_evidence import evaluate_compact_public
        prediction = self._Scalar()

        class NoopRestoreHead:
            training = True

            def eval(self):
                self.training = False

            def predict(self, *_args):
                return prediction

            def train(self, _mode):
                pass

        report = evaluate_compact_public(
            NoopRestoreHead(), [AtomicEvaluationCase("safe", "ADD", 1, 2, 3)],
            [], [(0, 3, 4, 7)], self._FakeTorch(),
        )
        self.assertEqual(report["disposition"], "REFUSED")
        self.assertEqual(report["reason"], "neural_state_restore_failed")

    def test_mixed_child_modes_are_restored_after_public_diagnostic(self):
        from labyrinth.minotaur_public_evidence import evaluate_compact_public
        prediction = self._Scalar()

        class Child:
            training = False

            def train(self, mode):
                self.training = mode

        class CompositeHead:
            training = True

            def __init__(self):
                self.child = Child()

            def modules(self):
                yield self
                yield self.child

            def eval(self):
                self.train(False)

            def train(self, mode):
                self.training = mode
                self.child.train(mode)

            def predict(self, *_args):
                return prediction

        head = CompositeHead()
        report = evaluate_compact_public(
            head, [AtomicEvaluationCase("safe", "ADD", 1, 2, 3)],
            [], [(0, 3, 4, 7)], self._FakeTorch(),
        )
        self.assertEqual(report["disposition"], "PUBLIC_DIAGNOSTIC")
        self.assertTrue(head.training)
        self.assertFalse(head.child.training)

    def test_nested_mode_restoration_noop_refuses_diagnostic(self):
        from labyrinth.minotaur_public_evidence import evaluate_compact_public
        prediction = self._Scalar()

        class Child:
            training = False

            def train(self, _mode):
                pass

        class CompositeHead:
            training = True

            def __init__(self):
                self.child = Child()

            def modules(self):
                return [self, self.child]

            def eval(self):
                self.training = False
                self.child.training = True

            def train(self, mode):
                self.training = mode

            def predict(self, *_args):
                return prediction

        head = CompositeHead()
        report = evaluate_compact_public(
            head, [AtomicEvaluationCase("safe", "ADD", 1, 2, 3)],
            [], [(0, 3, 4, 7)], self._FakeTorch(),
        )
        self.assertEqual(report["disposition"], "REFUSED")
        self.assertEqual(report["reason"], "neural_state_restore_failed")

    def test_nested_topology_drift_refuses_diagnostic(self):
        from labyrinth.minotaur_public_evidence import evaluate_compact_public
        prediction = self._Scalar()

        class Child:
            training = False

            def train(self, mode):
                self.training = mode

        class MutableHead:
            training = True

            def __init__(self):
                self.children = [Child()]

            def modules(self):
                yield self
                yield from self.children

            def eval(self):
                self.training = False
                self.children = [Child()]

            def train(self, mode):
                self.training = mode

            def predict(self, *_args):
                return prediction

        report = evaluate_compact_public(
            MutableHead(), [AtomicEvaluationCase("safe", "ADD", 1, 2, 3)],
            [], [(0, 3, 4, 7)], self._FakeTorch(),
        )
        self.assertEqual(report["disposition"], "REFUSED")
        self.assertEqual(report["reason"], "neural_state_restore_failed")


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


if __name__=="__main__":
    unittest.main()
