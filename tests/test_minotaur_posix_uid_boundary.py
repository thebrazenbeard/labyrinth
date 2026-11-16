"""Real POSIX UID/GID drop in a child; not a container or independent examiner."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
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

    def test_nonroot_examiner_cannot_silently_request_identity_drop(self):
        if os.name != "posix":
            with self.assertRaises(ValueError):
                SubprocessCandidateAdapter((sys.executable, "-c", "pass"),
                    model_id="test", unix_user_id=65534, unix_group_id=65534)
        else:
            with mock.patch("os.geteuid", return_value=1000):
                with self.assertRaises(PermissionError):
                    SubprocessCandidateAdapter((sys.executable, "-c", "pass"),
                        model_id="test", unix_user_id=65534, unix_group_id=65534)

    def test_host_specific_uid_drop_admission(self):
        command = (sys.executable, "-c", "print('invalid')")
        if os.name != "posix":
            expected = ValueError
        elif IS_WSL or os.geteuid() != 0:
            expected = PermissionError
        else:
            expected = None
        if expected is not None:
            with self.assertRaises(expected):
                SubprocessCandidateAdapter(command, model_id="synthetic:uid",
                    unix_user_id=65534, unix_group_id=65534)
        else:
            adapter = SubprocessCandidateAdapter(command, model_id="synthetic:uid",
                unix_user_id=65534, unix_group_id=65534)
            self.assertEqual((adapter.unix_user_id, adapter.unix_group_id),
                             (65534, 65534))

    def test_uid_drop_launch_composes_with_posix_session_isolation(self):
        """A privileged POSIX launch must preserve BOTH UID/GID and a new session."""
        command = (sys.executable, "-c", "print('not executed')")
        if os.name != "posix":
            with self.assertRaises(ValueError):
                SubprocessCandidateAdapter(command, model_id="synthetic:uid",
                    unix_user_id=65534, unix_group_id=65534)
            return
        # Only intercept launch arguments: never give WSL an actual UID-drop
        # child, whose Win32 interop defeats the claimed Linux identity wall.
        with (mock.patch("labyrinth.minotaur_candidate_bridge.os.geteuid", return_value=0),
              mock.patch("labyrinth.minotaur_candidate_bridge.os.uname",
                         return_value=SimpleNamespace(release="6.8.0-native-test")),
              mock.patch("labyrinth.minotaur_candidate_bridge.subprocess.Popen",
                         side_effect=RuntimeError("launch intercepted")) as popen):
            adapter = SubprocessCandidateAdapter(command, model_id="synthetic:uid",
                unix_user_id=65534, unix_group_id=65534)
            with self.assertRaisesRegex(RuntimeError, "launch intercepted"):
                adapter.generate('{"public":true}')
        launch = popen.call_args.kwargs
        self.assertIs(launch["start_new_session"], True)
        self.assertEqual(launch["user"], 65534)
        self.assertEqual(launch["group"], 65534)
        self.assertEqual(launch["extra_groups"], ())

    def test_private_file_denied_on_native_root_else_host_rejected(self):
        if os.name != "posix" or IS_WSL or os.geteuid() != 0:
            expected = ValueError if os.name != "posix" else PermissionError
            with self.assertRaises(expected):
                SubprocessCandidateAdapter((sys.executable, "-c", "pass"),
                    model_id="synthetic:uid", unix_user_id=65534,
                    unix_group_id=65534)
            return
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
