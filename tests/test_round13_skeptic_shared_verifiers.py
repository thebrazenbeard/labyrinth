"""Hostile shared-verifier faults: one patched check cannot certify itself.

Test-only independent attack against Experimenter's clean Switchback source
3d64a2c88223112e0ded48f8382204a3aeef75a0.  Production code untouched.
"""

from copy import deepcopy
from hashlib import sha256
import json
import unittest
from unittest.mock import patch

from labyrinth import switchback_oracle as oracle
from labyrinth.switchback_circuit import make_case


class IndependentVerifiersMustNotSelfCertify(unittest.TestCase):
    @staticmethod
    def _resign(payload):
        unsigned = {key: value for key, value in payload.items() if key != "sha256"}
        payload["sha256"] = sha256(json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode()).hexdigest()

    def test_single_corrupted_plan_scorer_cannot_certify_false_win(self):
        case = make_case(0)
        fake_plan = ["COMMIT"]  # No post-switch input; physically impossible win.
        forged_score = {
            "verdict": "VERIFIED_WIN", "reason": "FAKE_WIN",
            "counterexample": None, "steps_checked": 1,
        }
        with patch.object(oracle, "verify_plan", return_value=forged_score):
            receipt = oracle.adjudicate(
                case, {"kind": "PLAN", "actions": fake_plan})
            accepted = receipt.get("verified_success") is True and oracle.verify_receipt_replay(receipt)
        self.assertFalse(accepted, "one patched plan scorer self-certified a false WIN")

    def test_single_corrupted_probe_trace_checker_cannot_launder_forgery(self):
        case = make_case(0)
        valid_win = oracle.search(case)
        self.assertEqual(valid_win["verdict"], "VERIFIED_WIN")
        # Contradictory impossible transcript, not from the bounded ProbeSession.
        forged_trace = [{
            "action": "WAIT", "valid": False, "applied": True,
            "before": "111", "after": "000",
        }]
        self.assertFalse(oracle.verify_probe_trace(case, forged_trace))
        with patch.object(oracle, "verify_probe_trace", return_value=True):
            try:
                receipt = oracle.adjudicate_session(
                    case, forged_trace,
                    {"kind": "PLAN", "actions": valid_win["witness"]})
                accepted = receipt.get("verified_success") is True and oracle.verify_receipt_replay(receipt)
            except (ValueError, TypeError):
                accepted = False
        self.assertFalse(accepted, "one patched trace checker replay-certified forged probe history")

    def test_probe_forgery_crossing_reset_must_remain_rejected(self):
        case = make_case(0)
        valid_win = oracle.search(case)
        malformed = [
            {"action": "RESET", "valid": True, "applied": True,
             "before": "000", "after": "111"},
            {"action": "0", "valid": True, "applied": True,
             "before": "111", "after": "111"},
        ]
        with patch.object(oracle, "verify_probe_trace", return_value=True):
            try:
                receipt = oracle.adjudicate_session(case, malformed, {
                    "kind": "PLAN", "actions": valid_win["witness"]})
                accepted = receipt.get("verified_success") is True and oracle.verify_receipt_replay(receipt)
            except (ValueError, TypeError):
                accepted = False
        self.assertFalse(accepted, "forged reset/continuation bypassed physical replay")

    def test_rehashed_mutations_cannot_promote_custody_or_semantics(self):
        case = make_case(0)
        win_path = oracle.search(case)["witness"]
        original = oracle.adjudicate_session(case, [], {
            "kind": "PLAN", "actions": win_path})
        self.assertTrue(oracle.verify_receipt_replay(original))

        def mutate_core_reason(receipt):
            receipt["adjudication"]["review"]["reason"] = "FORGED_INDEPENDENCE"

        def mutate_core_provenance(receipt):
            receipt["adjudication"]["private_custody"] = True

        def mutate_outer_graduation(receipt):
            receipt["graduation_eligible"] = True

        def mutate_outer_count_bool(receipt):
            receipt["probe_count"] = False  # Python False == 0, but wrong type.

        def mutate_extra_field(receipt):
            receipt["adjudication"]["invented_witness"] = {"verified": True}

        for mutation in (mutate_core_reason, mutate_core_provenance,
                         mutate_outer_graduation, mutate_outer_count_bool,
                         mutate_extra_field):
            with self.subTest(mutation=mutation.__name__):
                forged = deepcopy(original)
                mutation(forged)
                self._resign(forged["adjudication"])
                self._resign(forged)
                self.assertFalse(oracle.verify_receipt_replay(forged),
                                 "rehashed semantic/provenance mutation passed replay")


if __name__ == "__main__":
    unittest.main()
