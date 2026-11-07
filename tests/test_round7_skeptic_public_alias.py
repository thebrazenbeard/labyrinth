"""Independent RED: public mutable views must not rewrite session evidence.

Original failing source: 3aaff23c23e2694a2398775becfde385d19ba762.
The oracle's replay can validate a forged history if two actions share a state.
"""

import json
import unittest

from labyrinth.cipher_wheel_world import CandidateSession, make_world
from labyrinth.cipher_wheel_oracle import oracle_table, verify_receipt_replay


class PublicMutableAliasEvidence(unittest.TestCase):
    def test_view_nested_history_cannot_rewrite_actual_action(self):
        world = make_world(0)
        session = CandidateSession(world)
        public_view = session.act("B")
        self.assertEqual(public_view["history"][0]["action"]["json"], '"B"')
        # B and RESET both leave state zero unchanged under train mechanics.
        public_view["history"][0]["action"]["json"] = json.dumps("RESET")
        receipt = session.submit_table(oracle_table(world))
        self.assertTrue(verify_receipt_replay(receipt))
        self.assertEqual(receipt["attempts"][0]["action"]["json"], '"B"',
                         "public view altered private attempted-action record")

    def test_prior_receipt_nested_history_cannot_rewrite_actual_action(self):
        world = make_world(0)
        session = CandidateSession(world)
        session.act("B")
        preliminary = session.receipt()
        self.assertTrue(verify_receipt_replay(preliminary))
        preliminary["attempts"][0]["action"]["json"] = json.dumps("RESET")
        final = session.submit_table(oracle_table(world))
        self.assertTrue(verify_receipt_replay(final))
        self.assertEqual(final["attempts"][0]["action"]["json"], '"B"',
                         "older emitted receipt changed subsequent factual transcript")

    def test_prior_receipt_submitted_raw_cannot_poison_final_session(self):
        world = make_world(0)
        session = CandidateSession(world)
        first = session.submit_table(oracle_table(world))
        self.assertTrue(first["verified_success"])
        self.assertTrue(verify_receipt_replay(first))
        # Caller owns the returned dict, never private accepted submission.
        label = next(iter(first["submitted_raw"]["A"]))
        first["submitted_raw"]["A"][label] = "FAKE"
        later = session.receipt()
        self.assertEqual(later["verdict"], "VERIFIED_WIN",
                         "mutating an emitted receipt changed final session verdict")
        self.assertTrue(verify_receipt_replay(later))


if __name__ == "__main__":
    unittest.main()
