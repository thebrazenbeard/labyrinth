"""Atomic, tamper-evident custody of Minotaur's signed qualification state."""

from pathlib import Path
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
        with sqlite3.connect(self.database) as connection:
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
            "result=decide_campaign(Path(sys.argv[1]),POLICY,"
            "bytes.fromhex(sys.argv[2]),sys.argv[3],"
            "independent_evaluator_key=KEY,state_key=STATE_KEY); "
            "print(result.disposition)"
        )
        argv = [sys.executable, "-c", command, str(self.database), raw.hex(), signature]
        first = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True)
        second = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True)
        stdout1, stderr1 = first.communicate(timeout=15)
        stdout2, stderr2 = second.communicate(timeout=15)
        self.assertEqual((first.returncode, second.returncode), (0, 0),
                         stderr1 + stderr2)
        self.assertEqual(sorted((stdout1.strip(), stdout2.strip())), ["BLOCKED", "STAY"])
        self.assertEqual(self.decide(evaluated_receipt(candidate_correct=False)).disposition,
                         "BLOCKED")


if __name__ == "__main__":
    unittest.main()
