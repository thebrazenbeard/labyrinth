"""Independent receipt-language integrity regressions for Wave18 OOD.

Rehashing public SHA-256 digests is deliberately in scope: these digests are
not signatures.  The test asks only whether the semantic verifier checks its
own declared reason/counterexample fields against the finite verdict.
"""

from copy import deepcopy
from hashlib import sha256
import unittest

from labyrinth import novel_router_ood_family as ood


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


if __name__ == "__main__":
    unittest.main()
