"""Fail-closed source-only Windows private-export boundary for batch CLI."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from scripts import switchback_batch_trial as batch


class BatchPrivateExportPortabilityTests(unittest.TestCase):
    @staticmethod
    def _args(path: Path) -> list[str]:
        return ["--seeds", "0", "--max-calls-per-case", "1",
                "--max-total-calls", "1", "--call-timeout", "1",
                "--max-total-seconds", "3", "--private-report-file", str(path)]

    def test_missing_fchmod_is_redacted_failure_not_uncaught_exception(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "private.json"
            stdout = io.StringIO()
            with mock.patch.object(batch.os, "fchmod", side_effect=AttributeError(
                    "module 'os' has no attribute 'fchmod'")):
                with redirect_stdout(stdout):
                    try:
                        status = batch.main(self._args(destination))
                    except AttributeError:
                        status = "UNCAUGHT_ATTRIBUTE_ERROR"
            self.assertEqual(status, 2)
            public = json.loads(stdout.getvalue())
            self.assertTrue(public["private_report_save_failed"])
            self.assertFalse(public["private_report_included"])
            self.assertFalse(destination.exists())
            self.assertNotIn('"raw_response"', stdout.getvalue())

    def test_unqualified_platform_must_not_claim_owner_only_acl(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "private.json"
            stdout = io.StringIO()
            # Patch a proposed source gate, not global os.name/pathlib internals.
            with mock.patch.object(batch, "_private_file_platform_qualified",
                                   return_value=False, create=True):
                with redirect_stdout(stdout):
                    status = batch.main(self._args(destination))
            self.assertEqual(status, 2)
            public = json.loads(stdout.getvalue())
            self.assertTrue(public["private_report_save_failed"])
            self.assertFalse(destination.exists())

    def test_unimplemented_fchmod_cannot_escape_redacted_cli_error(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "private.json"
            stdout = io.StringIO()
            with mock.patch.object(batch.os, "fchmod", side_effect=NotImplementedError(
                    "owner-only file mode not implemented")):
                with redirect_stdout(stdout):
                    try:
                        status = batch.main(self._args(destination))
                    except NotImplementedError:
                        status = "UNCAUGHT_NOT_IMPLEMENTED"
            self.assertEqual(status, 2)
            public = json.loads(stdout.getvalue())
            self.assertTrue(public["private_report_save_failed"])
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
