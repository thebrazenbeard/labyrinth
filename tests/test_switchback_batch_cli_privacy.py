"""Source-public batch CLI must keep host-private transcripts off default stdout."""

from contextlib import redirect_stdout
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

from scripts import switchback_batch_trial as batch


class BatchCliPrivacyTests(unittest.TestCase):
    @staticmethod
    def _cli(argv: list[str]) -> int:
        # A missing CLI capability is a failing exit status, not a test error.
        with redirect_stderr(io.StringIO()):
            try:
                return batch.main(argv)
            except SystemExit as error:
                return int(error.code)

    @staticmethod
    def _fixed_args() -> list[str]:
        return ["--seeds", "0", "--mode", "fixed",
                "--max-calls-per-case", "1", "--max-total-calls", "1",
                "--max-total-seconds", "5"]

    def test_default_stdout_redacts_private_seeds_model_text_and_command_args(self):
        model_secret = "MODEL_TEXT_DO_NOT_PUBLISH_26AD"
        argv_secret = "ARGV_DO_NOT_PUBLISH_7CBD"
        fixture = ("import json; print(json.dumps({'kind':'NOVEL_RULE',"
                   "'description':" + repr(model_secret) + "})); # " + argv_secret)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = self._cli([
                "--seeds", "0", "--mode", "command", "--authorize-local",
                "--max-calls-per-case", "1", "--max-total-calls", "1",
                "--max-total-seconds", "5", "--command", sys.executable,
                "-c", fixture,
            ])
        self.assertEqual(status, 0)
        text = stdout.getvalue()
        output = json.loads(text)
        self.assertEqual(output["schema"],
                         "labyrinth.switchback-batch-public-summary.v1")
        self.assertFalse(output["private_report_included"])
        self.assertEqual(output["aggregate"]["precommitted_case_count"], 1)
        for private in ("case_seed", "generation_seed", "raw_response",
                        "command_argv", "model_artifact_path", "source_receipt",
                        model_secret, argv_secret):
            with self.subTest(private=private):
                self.assertNotIn(private, text)

    def test_private_report_file_is_exclusive_mode_0600_and_replayable(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "host-private-batch.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = self._cli([*self._fixed_args(),
                                     "--private-report-file", str(destination)])
            self.assertEqual(status, 0)
            public = json.loads(stdout.getvalue())
            self.assertFalse(public["private_report_included"])
            self.assertTrue(public["private_report_saved"])
            self.assertEqual(public["schema"],
                             "labyrinth.switchback-batch-public-summary.v1")
            self.assertTrue(destination.is_file())
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
            private = json.loads(destination.read_text(encoding="utf-8"))
            self.assertTrue(batch.verify_batch_receipt(private))
            self.assertEqual(public["batch_receipt_sha256"],
                             private["batch_receipt_sha256"])
            self.assertIn("raw_response", private["cases"][0]["trial"]["turns"][0])
            self.assertNotIn('"raw_response"', stdout.getvalue())

    def test_existing_file_or_symlink_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.json"
            existing.write_text("SENTINEL_PRIVATE_FILE", encoding="utf-8")
            symlink = Path(directory) / "linked.json"
            try:
                os.symlink(existing, symlink)
            except (OSError, NotImplementedError):
                symlink = existing
            for destination in (existing, symlink):
                stdout = io.StringIO()
                with self.subTest(destination=destination.name):
                    with redirect_stdout(stdout):
                        status = self._cli([*self._fixed_args(),
                                             "--private-report-file", str(destination)])
                    self.assertEqual(status, 2)
                    self.assertEqual(existing.read_text(encoding="utf-8"),
                                     "SENTINEL_PRIVATE_FILE")
                    public = json.loads(stdout.getvalue())
                    self.assertFalse(public["private_report_saved"])
                    self.assertNotIn('"raw_response"', stdout.getvalue())

    def test_private_output_failure_never_falls_back_to_full_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing_parent" / "report.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = self._cli([*self._fixed_args(),
                                     "--private-report-file", str(missing)])
            self.assertEqual(status, 2)
            public = json.loads(stdout.getvalue())
            self.assertFalse(public["private_report_saved"])
            self.assertFalse(public["private_report_included"])
            self.assertNotIn('"case_seed"', stdout.getvalue())

    def test_private_json_stdout_requires_explicit_opt_in(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = self._cli([*self._fixed_args(),
                                 "--emit-private-json-stdout"])
        self.assertEqual(status, 0)
        private = json.loads(stdout.getvalue())
        self.assertTrue(batch.verify_batch_receipt(private))
        self.assertEqual(private["schema"], "labyrinth.switchback-batch-process.v1")
        self.assertIn("raw_response", private["cases"][0]["trial"]["turns"][0])


if __name__ == "__main__":
    unittest.main()
