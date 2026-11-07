"""Independent red-team falsifiers: exact-world trust and charged-input custody.

These are adversarial properties, not duplicates of implementation fixtures.
They intentionally fail on Cipher Wheel commit 1a5c88d.
"""

import unittest

from labyrinth.cipher_wheel_world import CandidateSession, World, make_world
from labyrinth.cipher_wheel_oracle import oracle_table, replay_trace, verify_receipt_replay


class CipherWheelSkepticProvenance(unittest.TestCase):
    @staticmethod
    def _charged_receipt(action):
        session = CandidateSession(make_world(0))
        session.act(action)
        return session.receipt()

    def test_distinct_long_invalid_inputs_must_not_share_replayable_receipt(self):
        first = self._charged_receipt("X" * 65)
        second = self._charged_receipt("Y" * 65)
        # Either exact provenance distinguishes them or at least one attempt
        # is quarantined as nonreplayable. A shared *verified* hash is false.
        self.assertFalse(first["attempts"] == second["attempts"] and
                         first["receipt_sha256"] == second["receipt_sha256"] and
                         verify_receipt_replay(first) and verify_receipt_replay(second))

    def test_distinct_json_scalar_types_must_not_share_replayable_receipt(self):
        for first_action, second_action in ((True, None), (1, "1"), (False, 0)):
            with self.subTest(first=repr(first_action), second=repr(second_action)):
                first = self._charged_receipt(first_action)
                second = self._charged_receipt(second_action)
                self.assertFalse(first["attempts"] == second["attempts"] and
                                 first["receipt_sha256"] == second["receipt_sha256"] and
                                 verify_receipt_replay(first) and verify_receipt_replay(second))

    def test_forged_world_subclass_equality_cannot_certify_table(self):
        class LyingStr(str):
            def __eq__(self, other):
                return True

            __hash__ = str.__hash__

        actual = make_world(0)
        forged = World(actual.seed, actual.heldout, actual.variant,
                       tuple(LyingStr("NOT-A-COLOR") for _ in range(4)),
                       LyingStr("NOT-A-REAL-WORLD-ID"))
        self.assertIsNone(replay_trace(forged, ()),
                          "forged noncanonical world passed independent replay")
        with self.assertRaises(ValueError):
            oracle_table(forged)
        try:
            candidate_session = CandidateSession(forged)
        except (ValueError, TypeError):
            return  # Early rejection is strongest.
        result = candidate_session.submit_table(oracle_table(actual))
        self.assertFalse(result.get("verified_success", False),
                         "forged world obtained positive certification")


if __name__ == "__main__":
    unittest.main()
