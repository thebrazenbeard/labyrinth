"""Hostile exact-witness checks independent from the solver's answer key."""

import unittest
from unittest.mock import patch

import labyrinth.exact_cover_tribunal as cover


class UnexpectedSolutionWitnessTests(unittest.TestCase):
    def test_forged_verifier_win_cannot_credit_invalid_specific_allocation(self):
        scenario = cover.families()["cyclic_deficiency"]
        invalid = (0, 1)  # Covers site a twice while omitting another site.
        self.assertEqual(cover.verify_allocation(scenario, invalid)["verdict"],
                         "VERIFIED_FAILURE")
        forged = {"verdict": "VERIFIED_WIN",
                  "rules_fingerprint": cover.fingerprint(scenario)}
        with patch.object(cover, "verify_allocation", return_value=forged):
            result = cover.adjudicate_unexpected(
                scenario, invalid, "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["overturned_prior"])

    def test_forged_verifier_failure_cannot_dismiss_valid_surprise(self):
        scenario = cover.families()["repairable_by_declared_team"]
        valid = (0, 4)  # A direct nonoverlapping six-site cover.
        self.assertEqual(cover.verify_allocation(scenario, valid)["verdict"],
                         "VERIFIED_WIN")
        forged = {"verdict": "VERIFIED_FAILURE",
                  "rules_fingerprint": cover.fingerprint(scenario)}
        with patch.object(cover, "verify_allocation", return_value=forged):
            result = cover.adjudicate_unexpected(
                scenario, valid, "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["overturned_prior"])

    def test_deceptive_replay_verdict_never_overturns_prior(self):
        class Liar(str):
            def __eq__(self, other):
                return other == "VERIFIED_WIN"

            def __ne__(self, other):
                return other != "VERIFIED_WIN"

        scenario = cover.families()["cyclic_deficiency"]
        forged = {"verdict": Liar("VERIFIED_FAILURE"),
                  "rules_fingerprint": cover.fingerprint(scenario)}
        with patch.object(cover, "verify_allocation", return_value=forged):
            result = cover.adjudicate_unexpected(
                scenario, (0, 1), "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["overturned_prior"])

    def test_actual_alternative_win_still_overturns_prior(self):
        scenario = cover.families()["alternative_matching"]
        result = cover.adjudicate_unexpected(
            scenario, (5, 6), "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(result["verdict"], "VERIFIED_WIN")
        self.assertTrue(result["overturned_prior"])


if __name__ == "__main__":
    unittest.main()
