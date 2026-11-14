"""Real POSIX UID/GID drop in a child; not a container or independent examiner."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from labyrinth.minotaur_candidate_bridge import (
    SubprocessCandidateAdapter, audit_receipt, run_candidate,
)
from labyrinth.timegate_exam import CandidateSession, candidate_view, make_case


IS_WSL = (os.name == "posix" and "microsoft" in os.uname().release.lower())


class PosixCandidateIdentityTests(unittest.TestCase):
    def test_identity_pair_must_be_complete_and_unprivileged(self):
        for user, group in ((65534, None), (None, 65534), (0, 65534), (65534, 0)):
            with self.subTest(user=user, group=group), self.assertRaises(ValueError):
                SubprocessCandidateAdapter((sys.executable, "-c", "pass"),
                    model_id="test", unix_user_id=user, unix_group_id=group)

    @unittest.skipUnless(os.name == "posix", "POSIX identity requires POSIX")
    def test_nonroot_examiner_cannot_silently_request_identity_drop(self):
        with mock.patch("os.geteuid", return_value=1000):
            with self.assertRaises(PermissionError):
                SubprocessCandidateAdapter((sys.executable, "-c", "pass"),
                    model_id="test", unix_user_id=65534, unix_group_id=65534)

    @unittest.skipUnless(IS_WSL and os.geteuid() == 0,
                         "physical WSL guard requires root on WSL")
    def test_wsl_identity_drop_refused_before_child_execution(self):
        with self.assertRaises(PermissionError):
            SubprocessCandidateAdapter((sys.executable, "-c", "print('invalid')"),
                model_id="synthetic:wsl", unix_user_id=65534, unix_group_id=65534)

    @unittest.skipUnless(os.name == "posix" and hasattr(os, "geteuid") and
                         os.geteuid() == 0 and not IS_WSL,
                         "physical UID-drop probe needs native POSIX root (not WSL)")
    def test_child_cannot_read_examiners_known_private_path(self):
        case = make_case(7, sealed=True)
        with tempfile.TemporaryDirectory(prefix="minotaur_examiner_") as directory:
            os.chmod(directory, 0o700)
            secret = Path(directory, "heldout.txt")
            secret.write_text("EXAMINER_ONLY", encoding="utf-8")
            secret.chmod(0o600)
            script = (
                "import json,os,pathlib,sys;json.load(sys.stdin);"
                "uid_ok=os.geteuid()==65534 and os.getegid()==65534 "
                "and os.getgroups()==[];"
                "p=pathlib.Path(sys.argv[1]);"
                "readable=False;"
                "exec('try:\\n p.read_text()\\n readable=True\\nexcept PermissionError:\\n pass');"
                "print(json.dumps({'kind':'ACTION',"
                "'action':'wait' if uid_ok and not readable else 'leak'}))"
            )
            adapter = SubprocessCandidateAdapter(
                (sys.executable, "-c", script, str(secret)), model_id="synthetic:uid",
                timeout_seconds=3.0, unix_user_id=65534, unix_group_id=65534,
            )
            receipt = run_candidate(candidate_view(case), lambda: CandidateSession(case),
                adapter, authorize_local_adapter=True, require_process_separation=True,
                max_model_calls=1)
            self.assertEqual(receipt.turns[0].disposition, "ACTION", receipt.final_verdict)
            self.assertEqual(json.loads(receipt.turns[0].response_text)["action"], "wait")
            self.assertEqual(secret.read_text(encoding="utf-8"), "EXAMINER_ONLY")
            self.assertTrue(audit_receipt(receipt, candidate_view(case),
                                          lambda: CandidateSession(case)))


if __name__ == "__main__":
    unittest.main()
