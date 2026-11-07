"""Second hostile wave: JSON coercion may erase exact charged action identity.

Independent regression written against Cipher Wheel fix commit 1dab344.
"""

import unittest

from labyrinth.cipher_wheel_world import CandidateSession, make_world
from labyrinth.cipher_wheel_oracle import oracle_table, verify_receipt_replay


class NestedActionCustody(unittest.TestCase):
    @staticmethod
    def _receipt(input_value):
        session = CandidateSession(make_world(0))
        session.act(input_value)
        return session.receipt()

    def test_nested_json_coercions_cannot_share_verified_receipts(self):
        collision_pairs = (
            ({1: "x"}, {"1": "x"}),
            ({None: 7}, {"null": 7}),
            ({True: "x"}, {"true": "x"}),
            ([(1, 2)], [[1, 2]]),
        )
        for left, right in collision_pairs:
            with self.subTest(left=repr(left), right=repr(right)):
                a = self._receipt(left)
                b = self._receipt(right)
                self.assertFalse(
                    a["attempts"] == b["attempts"] and
                    a["receipt_sha256"] == b["receipt_sha256"] and
                    verify_receipt_replay(a) and verify_receipt_replay(b),
                    "distinct Python action values share one verified trace")

    def test_unrepresentable_nested_action_must_not_precede_verified_win(self):
        world = make_world(0)
        session = CandidateSession(world)
        session.act({1: "hidden-as-string-key"})
        receipt = session.submit_table(oracle_table(world))
        self.assertFalse(receipt.get("verified_success", False),
                         "candidate graduated after unauditable charged action")
        self.assertFalse(verify_receipt_replay(receipt),
                         "unrepresentable input was laundered as valid replay")


if __name__ == "__main__":
    unittest.main()
