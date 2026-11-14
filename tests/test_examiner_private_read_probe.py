"""Real read-handle probe uses the current Windows token; it never reads secrets.

The same-principal access test is negative evidence, NOT an independent
candidate/attacker authentication or a protected holdout qualification.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from labyrinth.examiner_custody_acl import probe_local_private_read_access


@unittest.skipUnless(os.name == "nt", "real Windows NTFS access token required")
class CandidateLocalPrivateReadProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.private = self.root / "examiner-private.json"
        self.private.write_text('NEVER_PRINT_OR_READ_THIS_PRIVATE_ANSWER', encoding="utf-8")
        self.receipt = self.root / "local-access-receipt.json"

    def test_real_current_process_can_open_read_handle_without_consuming_bytes(self):
        with patch("os.read", side_effect=AssertionError("must never read private bytes")):
            report = probe_local_private_read_access(self.private, self.receipt)
        self.assertEqual(report["status"], "PRIVATE_READ_HANDLE_OPEN_SUCCEEDED")
        self.assertTrue(report["read_open_succeeded"])
        self.assertFalse(report["independent_custody_verified"])
        self.assertFalse(report["candidate_principal_read_denial_verified"])
        self.assertEqual(report["private_file_bytes_read"], 0)
        self.assertEqual(report["receipt_sha256"],
                         json.loads(self.receipt.read_text())["receipt_sha256"])
        self.assertEqual(self.private.read_text(),
                         "NEVER_PRINT_OR_READ_THIS_PRIVATE_ANSWER")

    def test_same_real_identity_as_declared_examiner_is_a_failure_not_pass(self):
        # The examiner SID is supplied by the caller and cannot authenticate
        # a genuinely independent evaluator.
        from labyrinth.examiner_custody_acl import _current_windows_sid
        sid = _current_windows_sid()
        report = probe_local_private_read_access(
            self.private, self.receipt, examiner_sid=sid)
        self.assertTrue(report["same_declared_examiner_principal"])
        self.assertTrue(report["read_open_succeeded"])
        self.assertFalse(report["independent_custody_verified"])

    def test_denied_open_is_only_local_self_report_not_independent_custody(self):
        with patch("labyrinth.examiner_custody_acl._open_read_handle",
                   side_effect=PermissionError("simulated denied token")):
            report = probe_local_private_read_access(self.private, self.receipt)
        self.assertEqual(report["status"], "PRIVATE_READ_OPEN_DENIED_LOCAL_ONLY")
        self.assertFalse(report["read_open_succeeded"])
        self.assertFalse(report["candidate_principal_read_denial_verified"])
        self.assertFalse(report["independent_custody_verified"])

    def test_path_not_found_is_not_a_denied_read_or_a_receipt(self):
        self.private.unlink()
        with self.assertRaises(ValueError):
            probe_local_private_read_access(self.private, self.receipt)
        self.assertFalse(self.receipt.exists())

    def test_cannot_overwrite_existing_local_read_receipt(self):
        probe_local_private_read_access(self.private, self.receipt)
        with self.assertRaises(FileExistsError):
            probe_local_private_read_access(self.private, self.receipt)

    def test_malformed_claimed_examiner_sid_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "SID|examiner"):
            probe_local_private_read_access(
                self.private, self.receipt, examiner_sid="user-not-sid")
        self.assertFalse(self.receipt.exists())


if __name__ == "__main__":
    unittest.main()
