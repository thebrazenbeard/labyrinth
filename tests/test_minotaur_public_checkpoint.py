"""Source-bound public checkpoint evaluation remains diagnostic, never a gate."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

try:
    import torch
    import safetensors
except ImportError:
    torch = None
    safetensors = None


@unittest.skipUnless(torch is not None and safetensors is not None, "torch+safetensors optional")
class PublicCheckpointTests(unittest.TestCase):
    def test_stale_imported_public_evaluator_refuses_after_disk_edit(self):
        from labyrinth import minotaur_public_checkpoint as public
        source = Path(public.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "labyrinth"
            copied.mkdir()
            for name in public._source_manifest():
                shutil.copy2(source / name, copied / name)
            with patch.object(public, "__file__", str(copied / "minotaur_public_checkpoint.py")):
                self.assertEqual(public._source_manifest(), public._IMPORTED_SOURCE_MANIFEST)
                with (copied / "minotaur_algebraic_eval_v5.py").open("a", encoding="utf8") as stream:
                    stream.write("\n# changed after import\n")
                with self.assertRaisesRegex(ValueError, "source.*changed.*import"):
                    public._verified_source_manifest()
                refused = public.evaluate_checkpoint_public(self.artifact, self.challenge)
                self.assertEqual(refused["disposition"], "REFUSED")
                self.assertEqual(refused["reason"], "invalid_artifact_or_challenge_input")

    @classmethod
    def setUpClass(cls):
        from labyrinth.minotaur_algebraic_experiment import run_experiment
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.artifact = cls.root / "trained"
        torch.set_num_threads(min(torch.get_num_threads(), 2))
        cls.training_report = run_experiment(cls.artifact, torch, seed=47, steps=500)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.temp_case = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_case.cleanup)
        self.work = Path(self.temp_case.name)
        self.challenge = self.work / "challenge.json"
        self.write_challenge()

    def write_challenge(self, *, atoms=None, programs=None):
        if atoms is None:
            atoms = [
                {"case_id": "add-wrap", "op": "ADD", "x": 95, "y": 5,
                 "claimed_output": 3, "group_id": "heldout"},
                {"case_id": "mul-large", "op": "MUL", "x": 23, "y": 19,
                 "claimed_output": (23 * 19) % 97},
            ]
        if programs is None:
            programs = [{
                "case_id": "clean", "initial_state": [95, 5],
                "instructions": [["ADD", "A", "B"], ["MUL", "B", "A"]],
                "observed_states": [[3, 5], [3, 15]],
                "claimed_first_error": None,
            }]
        data = {"schema": "labyrinth.minotaur-public-challenge.v1",
                "atomic_cases": atoms, "program_cases": programs}
        self.challenge.write_text(json.dumps(data), encoding="utf8")

    def evaluate(self, artifact=None, challenge=None, **kwargs):
        from labyrinth.minotaur_public_checkpoint import evaluate_checkpoint_public
        return evaluate_checkpoint_public(artifact or self.artifact,
                                          challenge or self.challenge, **kwargs)

    def test_checkpoint_and_public_cases_score_without_graduation(self):
        result = self.evaluate(
            expected_source_commit=self.training_report["source_commit"],
            expected_checkpoint_sha256=self.training_report["checkpoint_sha256"],
        )
        self.assertEqual(result["disposition"], "PUBLIC_DIAGNOSTIC")
        self.assertEqual(result["evaluation"]["atomic"]["metrics"]["exact_correct"], 2)
        self.assertEqual(result["evaluation"]["program"]["metrics"]["exact_first_error_correct"], 1)
        self.assertEqual(result["parameter_count"], 12)
        self.assertIn("minotaur_algebraic_eval_v5.py", result["public_evaluator_source_manifest"])
        self.assertIn("minotaur_public_evidence.py", result["public_evaluator_source_manifest"])
        self.assertFalse(result["minotaur_graduated"])
        self.assertFalse(result["deployed"])
        self.assertEqual(result["qualification"], "NOT_ESTABLISHED")
        self.assertEqual(result["graduation_authority"], "none")
        # Same interpreter test: numeric PID equality does not negate genuine
        # artifact replay, nor assert independent evaluator attestation.
        self.assertFalse(result["process_diagnostic"]["numeric_pid_differs_from_training"])
        self.assertEqual(result["process_diagnostic"]["independent_process_attestation"],
                         "NOT_ESTABLISHED")

    def test_forged_atomic_and_program_claims_refuse(self):
        atoms = [{"case_id": "forged", "op": "ADD", "x": 95, "y": 5,
                  "claimed_output": 4}]
        self.write_challenge(atoms=atoms)
        self.assertEqual(self.evaluate()["reason"], "invalid_public_atomic_claim")
        self.write_challenge()
        data = json.loads(self.challenge.read_text())
        data["program_cases"][0]["claimed_first_error"] = 1
        self.challenge.write_text(json.dumps(data))
        self.assertEqual(self.evaluate()["reason"], "invalid_public_program_claim")

    def test_training_overlap_and_duplicate_challenge_keys_refuse(self):
        from labyrinth.minotaur_algebraic_rules import incomplete_training_atoms
        op, x, y, target = incomplete_training_atoms(47, 12)[0]
        atom = {"case_id": "trained", "op": ("ADD", "SUB", "MUL")[op],
                "x": x, "y": y, "claimed_output": target}
        self.write_challenge(atoms=[atom], programs=[])
        self.assertEqual(self.evaluate()["reason"], "public_evaluator_refused")
        self.challenge.write_text(
            '{"schema":"labyrinth.minotaur-public-challenge.v1",'
            '"atomic_cases":[],"atomic_cases":[],"program_cases":[]}', encoding="utf8")
        self.assertEqual(self.evaluate()["reason"], "invalid_artifact_or_challenge_input")
        self.challenge.write_text(
            '{"schema":"labyrinth.minotaur-public-challenge.v1",'
            '"atomic_cases":[{"case_id":"bad","op":"ADD","x":NaN,"y":1,'
            '"claimed_output":1}],"program_cases":[]}', encoding="utf8")
        self.assertEqual(self.evaluate()["reason"], "invalid_artifact_or_challenge_input")

    def test_overflow_exponents_refuse_in_challenge_and_report(self):
        from labyrinth.minotaur_public_checkpoint import _strict_json
        for raw in (b'{"v":1e999}', b'{"v":-1e999}', b'{"v":[1e999]}'):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "nonfinite_json_number"):
                    _strict_json(raw)

        self.challenge.write_text(
            '{"schema":"labyrinth.minotaur-public-challenge.v1",'
            '"atomic_cases":[{"case_id":"bad","op":"ADD",'
            '"x":1e999,"y":1,"claimed_output":1}],"program_cases":[]}',
            encoding="utf8",
        )
        self.assertEqual(self.evaluate()["reason"], "invalid_artifact_or_challenge_input")
        self.write_challenge()

        copied = self.work / "overflow-report"
        shutil.copytree(self.artifact, copied)
        report_path = copied / "report.json"
        original = report_path.read_text(encoding="utf8")
        self.assertTrue(original.endswith("}\n"))
        modified = original[:-2] + ',\n  "overflow_probe": 1e999\n}\n'
        report_path.write_text(modified, encoding="utf8")
        (copied / "report.sha256").write_text(
            hashlib.sha256(report_path.read_bytes()).hexdigest() + "\n",
            encoding="utf8",
        )
        result = self.evaluate(artifact=copied)
        self.assertEqual(result["disposition"], "REFUSED")
        self.assertEqual(result["reason"], "checkpoint_or_public_evaluation_failed")
        self.assertEqual(result["qualification"], "NOT_ESTABLISHED")

    def test_tampered_report_and_checkpoint_refuse(self):
        copied = self.work / "copy"
        shutil.copytree(self.artifact, copied)
        report_path = copied / "report.json"
        report = json.loads(report_path.read_text())
        report["training_atoms_sha256"] = "0" * 64
        encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
        report_path.write_text(encoded)
        (copied / "report.sha256").write_text(hashlib.sha256(report_path.read_bytes()).hexdigest() + "\n")
        self.assertEqual(self.evaluate(artifact=copied)["reason"], "training_identity_digest_mismatch")
        shutil.rmtree(copied)
        shutil.copytree(self.artifact, copied)
        with (copied / "candidate.safetensors").open("ab") as stream:
            stream.write(b"tamper")
        self.assertEqual(self.evaluate(artifact=copied)["reason"],
                         "checkpoint_changed_during_read")

    def test_artifact_swap_between_preflight_and_replay_refuses(self):
        from labyrinth import minotaur_algebraic_experiment as algebraic
        copied = self.work / "swapped-during-replay"
        shutil.copytree(self.artifact, copied)
        report_path = copied / "report.json"

        def change_evidence_while_replay_runs(_folder, _torch):
            previous = json.loads(report_path.read_text(encoding="utf8"))
            previous["seed"] = 987654321
            report_path.write_text(json.dumps(previous, sort_keys=True) + "\\n",
                                   encoding="utf8")
            (copied / "report.sha256").write_text(
                hashlib.sha256(report_path.read_bytes()).hexdigest() + "\\n",
                encoding="utf8")
            return {"matches_original": True, "process_id_differs": True}

        with patch.object(algebraic, "reload_experiment",
                          side_effect=change_evidence_while_replay_runs):
            result = self.evaluate(artifact=copied)
        self.assertEqual(result["disposition"], "REFUSED")
        self.assertEqual(result["reason"], "artifact_changed_during_replay")

    def test_source_drift_expectations_and_symlink_refuse(self):
        result = self.evaluate(expected_source_commit="0" * 40)
        self.assertEqual(result["reason"], "expected_source_commit_mismatch")
        result = self.evaluate(expected_checkpoint_sha256="0" * 64)
        self.assertEqual(result["reason"], "expected_checkpoint_digest_mismatch")
        with patch("labyrinth.minotaur_algebraic_experiment._source_manifest",
                   return_value={"drift": "not the trained source"}):
            self.assertEqual(self.evaluate()["reason"], "checkpoint_or_public_evaluation_failed")
        linked = self.work / "link.json"
        try:
            linked.symlink_to(self.challenge)
        except OSError:
            self.skipTest("symlinks unsupported")
        self.assertEqual(self.evaluate(challenge=linked)["reason"],
                         "invalid_artifact_or_challenge_input")
        linked_directory = self.work / "artifact-alias"
        linked_directory.symlink_to(self.artifact, target_is_directory=True)
        self.assertEqual(self.evaluate(artifact=linked_directory)["reason"],
                         "invalid_artifact_or_challenge_input")

    def test_fifo_challenge_refuses_without_blocking_open(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("named pipes unsupported")
        fifo = self.work / "challenge.fifo"
        os.mkfifo(fifo)
        before = time.monotonic()
        self.assertEqual(self.evaluate(challenge=fifo)["reason"],
                         "invalid_artifact_or_challenge_input")
        self.assertLess(time.monotonic() - before, 2.0)

    def test_cli_exclusive_report_no_overwrite(self):
        from labyrinth.minotaur_public_checkpoint import main
        output = self.work / "public-report.json"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--artifact-dir", str(self.artifact),
                                   "--challenge-json", str(self.challenge),
                                   "--report", str(output)]), 0)
        result = json.loads(output.read_text())
        self.assertEqual(result["disposition"], "PUBLIC_DIAGNOSTIC")
        original = output.read_bytes()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--artifact-dir", str(self.artifact),
                                   "--challenge-json", str(self.challenge),
                                   "--report", str(output)]), 2)
        self.assertEqual(output.read_bytes(), original)

    def test_cli_executes_in_separate_child_without_claiming_attestation(self):
        output = self.work / "child-report.json"
        done = subprocess.run([
            sys.executable, "-m", "labyrinth.minotaur_public_checkpoint",
            "--artifact-dir", str(self.artifact), "--challenge-json", str(self.challenge),
            "--report", str(output),
        ], capture_output=True, text=True, check=False)
        self.assertEqual(done.returncode, 0, done.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report["disposition"], "PUBLIC_DIAGNOSTIC")
        self.assertEqual(report["process_diagnostic"]["independent_process_attestation"],
                         "NOT_ESTABLISHED")


if __name__ == "__main__":
    unittest.main()
