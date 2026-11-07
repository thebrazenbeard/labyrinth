"""Adversarial antirollback tests for signed Minotaur graduation campaigns.

No real external monotonic witness exists in the application. The reference
fake witnesses below are deliberately *not* production custody. They test the
minimum necessary semantics of a future independent host-owned authority.
"""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from labyrinth.graduation_state import CustodyError, decide_campaign, initialize_campaign
from tests.test_graduation_gate import (
    KEY, POLICY, evaluated_receipt, independently_signed,
)


STATE_KEY = b"wave8-separate-state-secret-at-least-32-bytes"


def _snapshot(source: Path, destination: Path) -> None:
    """SQLite backup API, including committed WAL state. No user files touched."""
    with closing(sqlite3.connect(source)) as old:
        with closing(sqlite3.connect(destination)) as new:
            old.backup(new)


def _read_seal(database: Path) -> tuple[int, str]:
    with closing(sqlite3.connect(database)) as db:
        version, mac = db.execute(
            "SELECT version, state_mac FROM campaign WHERE id=1"
        ).fetchone()
    return version, mac


def _decide(database: Path, *, success: bool):
    raw, signature = independently_signed(
        evaluated_receipt(candidate_correct=success)
    )
    return decide_campaign(
        database, POLICY, raw, signature,
        independent_evaluator_key=KEY, state_key=STATE_KEY,
    )


class ReferenceWitness:
    """An IN-MEMORY mock: neither durable, protected nor host-qualified."""

    def __init__(self, version: int, mac: str | None = None):
        self.version = version
        self.mac = mac

    def assert_current(self, database: Path, *, bind_state: bool):
        version, mac = _read_seal(database)
        if version != self.version or (bind_state and mac != self.mac):
            raise CustodyError("database diverges from independent monotonic witness")

    def advance(self, database: Path):
        self.version, self.mac = _read_seal(database)


def _qualification_reference(database: Path, receipt_success: bool,
                             witness: ReferenceWitness | None,
                             *, bind_state: bool = True):
    """Deliberately external reference wrapper; NOT an installed API."""
    if witness is None:
        raise CustodyError("host-owned durable monotonic witness missing")
    witness.assert_current(database, bind_state=bind_state)
    verdict = _decide(database, success=receipt_success)
    witness.advance(database)
    return verdict


class MonotonicityHostileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "campaign.sqlite3"
        self.base = Path(self.temp.name) / "baseline.sqlite3"
        initialize_campaign(self.db, POLICY, state_key=STATE_KEY)
        _snapshot(self.db, self.base)

    def test_authentic_sqlite_snapshot_restores_accepted_attempt_budget(self):
        self.assertEqual(_decide(self.db, success=False).disposition, "STAY")
        self.assertEqual(_decide(self.db, success=True).disposition, "BLOCKED")
        _snapshot(self.base, self.db)
        self.assertEqual(_decide(self.db, success=True).disposition, "GRADUATED")

    def test_independent_witness_in_memory_rejects_database_only_rollback(self):
        version, mac = _read_seal(self.db)
        witness = ReferenceWitness(version, mac)
        self.assertEqual(_qualification_reference(
            self.db, False, witness).disposition, "STAY")
        _snapshot(self.base, self.db)
        with self.assertRaises(CustodyError):
            _qualification_reference(self.db, True, witness)

    def test_ephemeral_witness_restart_makes_rollback_invisible(self):
        version, mac = _read_seal(self.db)
        witness = ReferenceWitness(version, mac)
        self.assertEqual(_qualification_reference(
            self.db, False, witness).disposition, "STAY")
        _snapshot(self.base, self.db)
        # Simulated loss of volatile witness across restart: both state
        # objects still contain valid MACs, but the historical high-water
        # mark has been lost. No claim of real independent custody.
        lost_witness = ReferenceWitness(*_read_seal(self.db))
        self.assertEqual(_qualification_reference(
            self.db, True, lost_witness).disposition, "GRADUATED")

    def test_version_only_witness_fails_on_same_version_divergent_state(self):
        """Two genuinely sealed branches with equal version, different MACs."""
        self.assertEqual(_decide(self.db, success=False).disposition, "STAY")
        branch_a = _read_seal(self.db)
        witness = ReferenceWitness(*branch_a)
        _snapshot(self.base, self.db)
        self.assertEqual(_decide(self.db, success=True).disposition, "GRADUATED")
        branch_b = _read_seal(self.db)
        self.assertEqual(branch_a[0], branch_b[0])
        self.assertNotEqual(branch_a[1], branch_b[1])
        witness.assert_current(self.db, bind_state=False)  # VERSION-ONLY FALSE ACCEPT
        with self.assertRaises(CustodyError):
            witness.assert_current(self.db, bind_state=True)

    def test_missing_monotonic_witness_must_fail_closed_for_qualification(self):
        with self.assertRaises(CustodyError):
            _qualification_reference(self.db, True, None)
        self.assertEqual(_read_seal(self.db)[0], 0)


if __name__ == "__main__":
    unittest.main()
