"""Host-private single-trial candidate output must not leak via default CLI stdout."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

from scripts import switchback_model_trial as trial


class SingleTrialCliPrivacyTests(unittest.TestCase):
    @staticmethod
    def _fixed_args() -> list[str]:
        return ["--seed", "0", "--mode", "fixed", "--max-calls", "1",
                "--timeout", "1", "--total-timeout", "3"]

    @staticmethod
    def _cli(argv: list[str]) -> int:
        with redirect_stderr(io.StringIO()):
            try:
                return trial.main(argv)
            except SystemExit as error:
                return int(error.code)

    def test_default_stdout_redacts_seed_and_arbitrary_child_model_text(self):
        secret = "CHILD_MODEL_OUTPUT_DO_NOT_PUBLISH_971A"
        source = ("import json; print(json.dumps({'kind':'NOVEL_RULE',"
                  "'description':" + repr(secret) + "}))")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = self._cli([
                "--seed", "739", "--mode", "command", "--authorize-command",
                "--max-calls", "1", "--timeout", "1", "--total-timeout", "3",
                "--command", sys.executable, "-c", source,
            ])
        self.assertEqual(status, 0)
        text = stdout.getvalue()
        public = json.loads(text)
        self.assertEqual(public["schema"],
                         "labyrinth.switchback-real-model-trial-public-summary.v1")
        self.assertFalse(public["private_report_included"])
        for field in ('"case_seed"', '"case_seed_host_only"', '"raw_response"',
                      '"turns"', '"configuration"', '"source_receipt"', secret):
            with self.subTest(field=field):
                self.assertNotIn(field, text)

    def test_exclusive_mode_0600_private_report_is_replayable_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            private_path = Path(directory) / "host-private-trial.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = self._cli([*self._fixed_args(),
                                    "--private-report-file", str(private_path)])
            self.assertEqual(status, 0)
            public = json.loads(stdout.getvalue())
            self.assertTrue(public["private_report_saved"])
            self.assertFalse(public["private_report_included"])
            self.assertEqual(stat.S_IMODE(private_path.stat().st_mode), 0o600)
            private = json.loads(private_path.read_text(encoding="utf-8"))
            self.assertEqual(private["schema"], trial.SCHEMA)
            self.assertTrue(private["source_receipt_replay_valid"])
            self.assertEqual(public["trial_receipt_sha256"],
                             private["receipt_sha256"])
            self.assertIn("raw_response", private["turns"][0])
            self.assertNotIn('"raw_response"', stdout.getvalue())

    def test_existing_file_and_symlink_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.json"
            existing.write_text("KEEP_EXISTING_SECRET", encoding="utf-8")
            link = Path(directory) / "linked.json"
            try:
                os.symlink(existing, link)
            except (OSError, NotImplementedError):
                link = existing
            for destination in (existing, link):
                stdout = io.StringIO()
                with self.subTest(target=destination.name):
                    with redirect_stdout(stdout):
                        status = self._cli([
                            *self._fixed_args(),
                            "--private-report-file", str(destination)])
                    self.assertEqual(status, 2)
                    self.assertEqual(existing.read_text(encoding="utf-8"),
                                     "KEEP_EXISTING_SECRET")
                    self.assertFalse(json.loads(stdout.getvalue())[
                        "private_report_saved"])
                    self.assertNotIn('"raw_response"', stdout.getvalue())

    def test_failed_private_file_write_never_exposes_raw_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "missing" / "receipt.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = self._cli([*self._fixed_args(),
                                    "--private-report-file", str(destination)])
            self.assertEqual(status, 2)
            public = json.loads(stdout.getvalue())
            self.assertTrue(public["private_report_save_failed"])
            self.assertFalse(public["private_report_included"])
            self.assertNotIn('"case_seed"', stdout.getvalue())

    def test_explicit_unsafe_stdout_preserves_full_replayable_legacy_json(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = self._cli([*self._fixed_args(),
                                "--emit-private-json-stdout"])
        self.assertEqual(status, 0)
        private = json.loads(stdout.getvalue())
        self.assertEqual(private["schema"], trial.SCHEMA)
        self.assertTrue(private["source_receipt_replay_valid"])
        self.assertIn("case_seed_host_only", private)
        self.assertIn("raw_response", private["turns"][0])


if __name__ == "__main__":
    unittest.main()
