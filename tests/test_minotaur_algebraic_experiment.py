"""Saved rule weights must replay independently without atomic-row training."""
import tempfile
import unittest
from pathlib import Path
try:
    import torch
    import safetensors
except ImportError:
    torch = None
    safetensors = None

@unittest.skipUnless(torch is not None and safetensors is not None, "torch+safetensors optional")
class AlgebraicExperimentTests(unittest.TestCase):
    def test_heldout_programs_compose_rules_without_teacher_states(self):
        from labyrinth.minotaur_algebraic_experiment import evaluate_programs
        from labyrinth.minotaur_algebraic_rules import (
            build_algebraic_head,incomplete_training_atoms,train_rule_head,
        )
        head=build_algebraic_head(torch,seed=19)
        train_rule_head(head,incomplete_training_atoms(47,12),torch,steps=500,learning_rate=.06)
        score=evaluate_programs(head,torch,seed=808,count=128)
        self.assertEqual(score["exact_programs"],128)
        self.assertEqual(score["total_programs"],128)
        self.assertGreater(score["unseen_atomic_visits"],0)
        self.assertGreater(score["unseen_atomic_visits"],score["seen_atomic_visits"])

    def test_source_pinned_artifact_reloads_exact_predictions(self):
        from labyrinth.minotaur_algebraic_experiment import run_experiment,reload_experiment
        torch.set_num_threads(min(torch.get_num_threads(),2))
        with tempfile.TemporaryDirectory() as temp:
            report=run_experiment(Path(temp)/"output",torch,seed=47,
                                  training_per_op=12,steps=500)
            self.assertEqual(report["train_examples"],36)
            self.assertEqual(report["parameter_count"],12)
            self.assertEqual(report["candidate"]["total"],3*97*97)
            self.assertEqual(report["candidate"]["correct"],3*97*97)
            self.assertEqual(report["withheld_atomic_transfer"],
                             {"correct":3*97*97-36,"total":3*97*97-36})
            self.assertGreater(report["candidate"]["correct"],report["corrupted"]["correct"])
            replay=reload_experiment(Path(temp)/"output",torch)
            self.assertTrue(replay["matches_original"])
            self.assertTrue(replay["process_id_differs"] is False)  # same-process unit test
            with self.assertRaises(FileExistsError):
                run_experiment(Path(temp)/"output",torch,seed=47)
