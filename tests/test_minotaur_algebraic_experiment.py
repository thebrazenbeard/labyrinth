"""Saved rule weights must replay independently without atomic-row training."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
try:
    import torch
    import safetensors
except ImportError:
    torch = None
    safetensors = None

@unittest.skipUnless(torch is not None and safetensors is not None, "torch+safetensors optional")
class AlgebraicExperimentTests(unittest.TestCase):
    def test_stale_imported_training_source_refuses_after_disk_edit(self):
        import shutil
        from labyrinth import minotaur_algebraic_experiment as experiment
        source = Path(experiment.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "labyrinth"
            copied.mkdir()
            for name in experiment._source_manifest():
                shutil.copy2(source / name, copied / name)
            with patch.object(experiment, "__file__", str(copied / "minotaur_algebraic_experiment.py")):
                self.assertEqual(experiment._source_manifest(), experiment._IMPORTED_SOURCE_MANIFEST)
                with (copied / "minotaur_algebraic_rules.py").open("a", encoding="utf8") as stream:
                    stream.write("\n# changed after import\n")
                with self.assertRaisesRegex(ValueError, "source.*changed.*import"):
                    experiment._verified_source_manifest()
                output = Path(temp) / "must-not-create"
                with self.assertRaisesRegex(ValueError, "source.*changed.*import"):
                    experiment.run_experiment(output, torch, seed=47, steps=1)
                self.assertFalse(output.exists())

    def test_reload_cli_treats_numeric_pid_reuse_as_diagnostic(self):
        import io
        import json
        from contextlib import redirect_stderr, redirect_stdout
        from labyrinth.minotaur_algebraic_experiment import main

        replay = {"matches_original": True, "process_id_differs": False,
                  "qualification": "NOT_ESTABLISHED"}
        with patch("labyrinth.minotaur_algebraic_experiment.reload_experiment",
                   return_value=replay):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["reload", "--output-dir", "unused"]), 0)
            self.assertIs(json.loads(output.getvalue())["process_id_differs"], False)

        for failed in (False, None, 1):
            with self.subTest(matches_original=failed):
                with patch("labyrinth.minotaur_algebraic_experiment.reload_experiment",
                           return_value={"matches_original": failed,
                                         "process_id_differs": True}):
                    with redirect_stderr(io.StringIO()) as error:
                        with self.assertRaises(SystemExit) as exit_status:
                            main(["reload", "--output-dir", "unused"])
                    self.assertEqual(exit_status.exception.code, 2)
                    self.assertIn("checkpoint replay mismatch", error.getvalue())

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

    def test_max_valid_training_seed_records_bounded_corruption_seed(self):
        from labyrinth.minotaur_algebraic_experiment import run_experiment,reload_experiment
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/"max-seed"
            report=run_experiment(folder,torch,seed=2**31-1,
                                  training_per_op=24,steps=1)
            self.assertEqual(report["seed"],2**31-1)
            self.assertEqual(report["corruption_seed"],11)
            self.assertTrue(reload_experiment(folder,torch)["matches_original"])

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
