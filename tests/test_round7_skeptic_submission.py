"""Independent RED: unsupported one-shot submissions may be retried for free.

Target baseline: 3aaff23c23e2694a2398775becfde385d19ba762
This is a host API trust-boundary/attempt-accounting test, not neural evidence.
"""

import unittest

from labyrinth.cipher_wheel_world import CandidateSession, make_world
from labyrinth.cipher_wheel_oracle import oracle_table, verify_receipt_replay


class UnsupportedSubmissionAccounting(unittest.TestCase):
    def test_unsupported_python_submission_does_not_grant_free_retry(self):
        world = make_world(19, heldout=True)
        session = CandidateSession(world)
        # This entrypoint is a terminal answer channel; an unsupported but
        # user-supplied answer should be recorded as failed or quarantined.
        try:
            session.submit_table({"A": {1: "amber"}, "B": {}})
        except ValueError:
            pass
        receipt = session.submit_table(oracle_table(world))
        self.assertFalse(receipt["verified_success"],
                         "malformed terminal submission silently got a second chance")

    def test_many_malformed_submissions_cannot_precede_replayable_win(self):
        world = make_world(5)
        session = CandidateSession(world)
        for bad in (object(), {"A": {"x": (1, 2)}, "B": {}},
                    {"A": float("nan"), "B": {}}):
            try:
                session.submit_table(bad)
            except ValueError:
                pass
        receipt = session.submit_table(oracle_table(world))
        self.assertFalse(receipt["verified_success"]
                         and verify_receipt_replay(receipt),
                         "unrecorded malformed submissions can precede a valid win")


if __name__ == "__main__":
    unittest.main()
