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

    def test_rejects_report_tampering_and_source_drift(self):
        import json
        from unittest.mock import patch
        from labyrinth.minotaur_algebraic_experiment import (
            run_experiment, reload_experiment,
        )
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/"run"
            run_experiment(folder,torch,seed=47,steps=60)
            fp=folder/"report.json"
            original=fp.read_bytes()
            data=json.loads(original)
            data["minotaur_graduated"]=True
            fp.write_text(json.dumps(data),encoding="utf8")
            with self.assertRaisesRegex(ValueError,"report.*changed"):
                reload_experiment(folder,torch)
            fp.write_bytes(original)
            with patch("labyrinth.minotaur_algebraic_experiment._source_manifest",
                       return_value={"tampered":"source"}):
                with self.assertRaisesRegex(ValueError,"source.*changed"):
                    reload_experiment(folder,torch)

    def test_truncated_program_predictions_cannot_score_perfect(self):
        from unittest.mock import patch
        from labyrinth.minotaur_algebraic_experiment import evaluate_programs
        from labyrinth.minotaur_algebraic_rules import build_algebraic_head
        class MalformedModel:
            def __init__(self):
                self.algebraic_head=self
                self.training=True
            def load_state_dict(self, state, strict=True): pass
            def eval(self):
                self.training=False
                return self
            def train(self, mode=True):
                self.training=mode
                return self
            def __call__(self, features):
                return {"predicted_states":torch.empty((features.mask.shape[0],0,2),dtype=torch.long)}
        with patch("labyrinth.minotaur_algebraic_experiment.build_transition_model",
                   return_value=MalformedModel()):
            with self.assertRaisesRegex(ValueError,"prediction.*length"):
                evaluate_programs(build_algebraic_head(torch,seed=19),
                                  torch,seed=808,count=8)
