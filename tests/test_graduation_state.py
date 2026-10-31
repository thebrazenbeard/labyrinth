"""Atomic, tamper-evident custody of Minotaur's signed qualification state."""

from pathlib import Path
from contextlib import closing
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from labyrinth.graduation_gate import GateState
from labyrinth.graduation_state import (
    CustodyError, decide_campaign, initialize_campaign,
)
from tests.test_graduation_gate import (
    KEY, POLICY, evaluated_receipt, independently_signed,
)


STATE_KEY = b"state-storage-test-key-unique-from-evaluator-key-2026"


class GraduationStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "guardian-custody.sqlite3"

    def decide(self, receipt, policy=POLICY):
        raw, signature = independently_signed(receipt)
        return decide_campaign(self.database, policy, raw, signature,
                               independent_evaluator_key=KEY, state_key=STATE_KEY)

    def test_requires_prior_policy_bound_campaign_initialization(self):
        with self.assertRaises(CustodyError):
            self.decide(evaluated_receipt())
        initialize_campaign(self.database, POLICY, state_key=STATE_KEY)
        with self.assertRaises(CustodyError):
            initialize_campaign(self.database, POLICY, state_key=STATE_KEY)

    def test_signed_graduation_survives_store_reopen_without_deployment(self):
        initialize_campaign(self.database, POLICY, state_key=STATE_KEY)
        accepted = self.decide(evaluated_receipt())
        self.assertEqual(accepted.disposition, "GRADUATED")
        self.assertEqual(accepted.state.next_attempt, 2)
        self.assertTrue(accepted.state.graduated)
        # A second database connection inherits durable qualified status.
        replay = self.decide(evaluated_receipt())
        self.assertEqual(replay.disposition, "GRADUATED")
        self.assertEqual(replay.state, accepted.state)

    def test_rejected_evidence_does_not_consume_attempt(self):
        initialize_campaign(self.database, POLICY, state_key=STATE_KEY)
        raw, _ = independently_signed(evaluated_receipt())
        denied = decide_campaign(self.database, POLICY, raw, "0" * 64,
                                 independent_evaluator_key=KEY, state_key=STATE_KEY)
        self.assertEqual(denied.disposition, "BLOCKED")
        accepted = self.decide(evaluated_receipt(candidate_correct=False))
        self.assertEqual(accepted.disposition, "STAY")
        self.assertEqual(accepted.state.next_attempt, 2)
        # Reusing an authenticated failed exam does not grant another try.
        self.assertEqual(self.decide(evaluated_receipt(candidate_correct=False)).disposition,
                         "BLOCKED")

    def test_tamper_or_wrong_state_key_fail_closed(self):
        initialize_campaign(self.database, POLICY, state_key=STATE_KEY)
        raw, signature = independently_signed(evaluated_receipt())
        with self.assertRaises(CustodyError):
            decide_campaign(self.database, POLICY, raw, signature,
                            independent_evaluator_key=KEY,
                            state_key=b"wrong-state-key-does-not-open-this-store")
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute("UPDATE campaign SET serialized_state = ? WHERE id = 1",
                                   ('{"next_attempt":99}',))
        with self.assertRaises(CustodyError):
            self.decide(evaluated_receipt())

    def test_cross_policy_and_parent_directory_boundaries(self):
        initialize_campaign(self.database, POLICY, state_key=STATE_KEY)
        from dataclasses import replace
        with self.assertRaises(CustodyError):
            self.decide(evaluated_receipt(), replace(POLICY, objective_id="different"))
        with self.assertRaises(ValueError):
            initialize_campaign(Path("relative.sqlite3"), POLICY, state_key=STATE_KEY)
        with self.assertRaises(ValueError):
            initialize_campaign(self.database.parent / "missing" / "child.sqlite3",
                                POLICY, state_key=STATE_KEY)

    def test_forged_mastery_objective_cannot_initialize_persisted_campaign(self):
        from dataclasses import replace

        class GhostKind(str):
            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return False

        # The previous constructor accepted this forged invalid kind, and
        # the persistent decision path promoted a mastery claim to GRADUATED
        # despite correct-label learning-effect controls showing no gain.
        with self.assertRaises(ValueError):
            replace(POLICY, objective_kind=GhostKind("not-an-approved-kind"))
        self.assertFalse(self.database.exists())

        initialize_campaign(self.database, POLICY, state_key=STATE_KEY)
        self.assertEqual(self.decide(evaluated_receipt(controls_correct=True)).disposition,
                         "STAY")

    def test_subclassed_policy_cannot_create_or_use_persisted_campaign(self):
        from labyrinth.graduation_gate import GraduationPolicy

        class ForgedPolicy(GraduationPolicy):
            def __getattribute__(self, name):
                if name == "objective_kind":
                    return "skill_mastery"
                return object.__getattribute__(self, name)

        fake = ForgedPolicy(**POLICY.__dict__)
        with self.assertRaises(TypeError):
            initialize_campaign(self.database, fake, state_key=STATE_KEY)
        self.assertFalse(self.database.exists())
        initialize_campaign(self.database, POLICY, state_key=STATE_KEY)
        raw, signature = independently_signed(evaluated_receipt())
        with self.assertRaises(TypeError):
            decide_campaign(self.database, fake, raw, signature,
                            independent_evaluator_key=KEY, state_key=STATE_KEY)

    def test_ancestor_symlink_cannot_redirect_custodian_database(self):
        actual_root = Path(self.temp.name) / "actual"
        actual_root.mkdir()
        custodian_dir = actual_root / "custodian"
        custodian_dir.mkdir()
        alias = Path(self.temp.name) / "redirected"
        try:
            alias.symlink_to(actual_root, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("directory symlinks unavailable on this host")

        redirected = alias / "custodian" / "campaign.sqlite3"
        self.assertFalse(redirected.parent.is_symlink())
        with self.assertRaisesRegex(ValueError, "symlink|junction"):
            initialize_campaign(redirected, POLICY, state_key=STATE_KEY)
        self.assertFalse((custodian_dir / "campaign.sqlite3").exists())

        real = custodian_dir / "campaign.sqlite3"
        initialize_campaign(real, POLICY, state_key=STATE_KEY)
        raw, signature = independently_signed(evaluated_receipt())
        with self.assertRaisesRegex(ValueError, "symlink|junction"):
            decide_campaign(redirected, POLICY, raw, signature,
                            independent_evaluator_key=KEY, state_key=STATE_KEY)

    def test_two_fresh_processes_cannot_spend_same_signed_attempt(self):
        initialize_campaign(self.database, POLICY, state_key=STATE_KEY)
        raw, signature = independently_signed(evaluated_receipt(candidate_correct=False))
        # A real separate Python interpreter re-imports the module, opens the
        # same WAL database, and tries to spend exactly the same signed exam.
        command = (
            "import sys; "
            "from pathlib import Path; "
            "from labyrinth.graduation_state import decide_campaign; "
            "from tests.test_graduation_gate import KEY,POLICY; "
            "from tests.test_graduation_state import STATE_KEY; "
            "raw,signature=sys.stdin.read().splitlines(); "
            "result=decide_campaign(Path(sys.argv[1]),POLICY,"
            "bytes.fromhex(raw),signature,"
            "independent_evaluator_key=KEY,state_key=STATE_KEY); "
            "print(result.disposition)"
        )
        argv = [sys.executable, "-c", command, str(self.database)]
        processes = []
        try:
            for _ in range(2):
                processes.append(subprocess.Popen(
                    argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True))
            # Feed both before waiting, preserving a real interprocess race.
            payload = f"{raw.hex()}\n{signature}\n"
            for process in processes:
                process.stdin.write(payload)
                process.stdin.close()
                process.stdin = None
            stdout1, stderr1 = processes[0].communicate(timeout=15)
            stdout2, stderr2 = processes[1].communicate(timeout=15)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate()
        first, second = processes
        self.assertEqual((first.returncode, second.returncode), (0, 0),
                         stderr1 + stderr2)
        self.assertEqual(sorted((stdout1.strip(), stdout2.strip())), ["BLOCKED", "STAY"])
        self.assertEqual(self.decide(evaluated_receipt(candidate_correct=False)).disposition,
                         "BLOCKED")


if __name__ == "__main__":
    unittest.main()
