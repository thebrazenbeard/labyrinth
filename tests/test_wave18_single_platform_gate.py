"""Hostile Windows-like platform gate tests for private single-trial export.

No real model or host service is invoked; a minimal source-shaped report is
provided to the CLI so the test isolates filesystem publication behavior.
"""

from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
import io
import json
import unittest

from scripts import switchback_model_trial as trial


_REPORT = {
    "receipt_sha256": "f" * 64,
    "terminal_reason": "SUBMITTED",
    "source_verdict": "UNKNOWN",
    "source_receipt_replay_valid": True,
    "verified_success": False,
}
_WINDOWS_TARGET = r"C:\minotaur-wave18-synthetic\private.json"


class PrivateFilePlatformQualification(unittest.TestCase):
    def test_nt_with_callable_fchmod_cannot_publish_or_leak_private_cli_report(self):
        """Even on Python >=3.13 Windows, POSIX 0600 is not ACL proof."""
        with TemporaryDirectory() as directory:
            target = Path(directory) / "private.json"
            output = io.StringIO()
            with (mock.patch.object(trial, "run_trial", return_value=dict(_REPORT)),
                  mock.patch.object(trial.os, "name", "nt"),
                  mock.patch.object(trial.os, "fchmod", wraps=trial.os.fchmod) as chmod,
                  mock.patch.object(trial.tempfile, "mkstemp", wraps=trial.tempfile.mkstemp)
                  as stage,
                  redirect_stdout(output)):
                status = trial.main(["--seed", "0", "--max-calls", "1",
                                     "--timeout", "1", "--total-timeout", "3",
                                     "--private-report-file", _WINDOWS_TARGET])
            self.assertEqual(status, 2)
            summary = json.loads(output.getvalue())
            self.assertTrue(summary["private_report_save_failed"])
            self.assertFalse(summary["private_report_saved"])
            self.assertFalse(summary["private_report_included"])
            self.assertNotIn("raw_response", output.getvalue())
            self.assertFalse(target.exists())
            stage.assert_not_called()
            chmod.assert_not_called()

    def test_nt_direct_writer_refuses_before_mkstemp(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "private.json"
            with (mock.patch.object(trial.os, "name", "nt"),
                  mock.patch.object(trial.os, "fchmod", wraps=trial.os.fchmod),
                  mock.patch.object(trial.tempfile, "mkstemp", wraps=trial.tempfile.mkstemp)
                  as stage):
                with self.assertRaises(OSError):
                    trial._write_private_report_exclusive(dict(_REPORT), _WINDOWS_TARGET)
            stage.assert_not_called()
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
