"""Independent receipt-language integrity regressions for Wave18 OOD.

Rehashing public SHA-256 digests is deliberately in scope: these digests are
not signatures.  The test asks only whether the semantic verifier checks its
own declared reason/counterexample fields against the finite verdict.
"""

from copy import deepcopy
from hashlib import sha256
import unittest

from labyrinth import novel_router_ood_family as ood


class EvilStr(str):
    """Serialize as a false claim while comparing equal to expected literals."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


def _rehash(receipt):
    core = receipt["adjudication"]
    core["sha256"] = sha256(ood._canonical({
        key: value for key, value in core.items() if key != "sha256"
    }).encode()).hexdigest()
    receipt["sha256"] = sha256(ood._canonical({
        key: value for key, value in receipt.items() if key != "sha256"
    }).encode()).hexdigest()
    return receipt


class OodReceiptMathTests(unittest.TestCase):
    def test_untampered_plan_receipt_replays(self):
        case = ood.make_case(0)
        honest = ood.ProbeSession(case).submit({
            "kind": "PLAN", "actions": ["COMMIT"]})
        self.assertEqual(honest["verdict"], "VERIFIED_FAILURE")
        self.assertTrue(ood.verify_session_receipt(honest))

    def test_rehashed_incorrect_plan_reason_does_not_replay(self):
        case = ood.make_case(0)
        honest = ood.ProbeSession(case).submit({
            "kind": "PLAN", "actions": ["COMMIT"]})
        forged = deepcopy(honest)
        forged["adjudication"]["reason"] = "INDEPENDENT_VERIFIED_ALL_POSSIBLE_RULES"
        self.assertFalse(ood.verify_session_receipt(_rehash(forged)))

    def test_rehashed_novel_rule_counterexample_does_not_replay(self):
        case = ood.make_case(0)
        honest = ood.ProbeSession(case).submit({
            "kind": "NOVEL_RULE", "description": "Could we revise the seal law?"})
        self.assertEqual(honest["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertTrue(ood.verify_session_receipt(honest))
        forged = deepcopy(honest)
        forged["adjudication"]["counterexample"] = ["COMMIT"]
        self.assertFalse(ood.verify_session_receipt(_rehash(forged)))

    def test_rehashed_unsat_reason_does_not_replay(self):
        case = ood.make_case(0)
        honest = ood.ProbeSession(case).submit({"kind": "CLAIM_UNSAT"})
        self.assertTrue(ood.verify_session_receipt(honest))
        forged = deepcopy(honest)
        forged["adjudication"]["reason"] = "EXACT_VERIFIED_WIN"
        self.assertFalse(ood.verify_session_receipt(_rehash(forged)))

    def test_evilstr_false_verified_win_does_not_replay(self):
        case = ood.make_case(0)
        honest = ood.ProbeSession(case).submit({
            "kind": "PLAN", "actions": ["COMMIT"]})
        self.assertEqual(honest["verdict"], "VERIFIED_FAILURE")
        forged = deepcopy(honest)
        forged["adjudication"]["verdict"] = EvilStr("VERIFIED_WIN")
        forged["adjudication"]["verified_success"] = True
        forged["verdict"] = EvilStr("VERIFIED_WIN")
        forged["verified_success"] = True
        self.assertFalse(ood.verify_session_receipt(_rehash(forged)))

    def test_evilstr_forged_probe_trace_does_not_replay(self):
        case = ood.make_case(0)
        session = ood.ProbeSession(case)
        session.probe("PULSE")
        honest = session.submit({"kind": "ABSTAIN"})
        self.assertTrue(ood.verify_session_receipt(honest))
        forged = deepcopy(honest)
        forged["probe_trace"][0]["before"] = EvilStr("00000")
        forged["probe_trace"][0]["after"] = EvilStr("11111")
        forged["probe_sha256"] = sha256(ood._canonical({
            "trace": forged["probe_trace"]
        }).encode()).hexdigest()
        self.assertFalse(ood.verify_session_receipt(_rehash(forged)))


if __name__ == "__main__":
    unittest.main()
