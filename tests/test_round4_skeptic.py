"""Fourth-wave regression: five charged queries still permit a final policy claim."""

import json
import unittest

from labyrinth.minotaur_candidate_bridge import audit_receipt, run_candidate


SCHEDULE = ("alpha", "alpha", "beta", "beta", "parity")
PUBLIC_START = {"answers": [], "remaining_queries": 5}


class SixDecisionAdapter:
    model_id = "test:five-queries-and-one-policy"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, *, seed=None):
        if self.calls < len(SCHEDULE):
            decision = {"kind": "ACTION", "action": SCHEDULE[self.calls]}
        else:
            decision = {"kind": "CLAIM", "claim": {"kind": "POLICY", "policy": {
                "questions": list(SCHEDULE), "decoder": "nearest_codeword"}}}
        self.calls += 1
        return json.dumps(decision)


class FiveQuerySession:
    """Minimal public query/claim contract; no private source or oracle access."""

    def __init__(self):
        self.answers = []
        self.claimed = False

    def view(self):
        return {"answers": list(self.answers),
                "remaining_queries": 5 - len(self.answers),
                "finished": self.claimed,
                "verdict": "VERIFIED_WIN" if self.claimed else "UNKNOWN",
                "policy_verified": self.claimed,
                "instance_win_verified": False,
                "verified_success": False}

    def ask(self, question):
        if len(self.answers) >= 5:
            raise ValueError("query budget exhausted")
        self.answers.append({"question": question, "answer": 0})
        return self.view()

    def submit(self, claim):
        if len(self.answers) != 5 or claim.get("kind") != "POLICY":
            raise ValueError("expected final policy after five queries")
        self.claimed = True
        return self.view()


class BridgeQueryClaimBudgetTests(unittest.TestCase):
    def test_five_queries_leave_one_terminal_claim_call(self):
        adapter = SixDecisionAdapter()
        receipt = run_candidate(PUBLIC_START, FiveQuerySession, adapter,
                                authorize_local_adapter=True)
        self.assertEqual(adapter.calls, 6)
        self.assertEqual(len(receipt.turns), 6)
        self.assertEqual(receipt.turns[-1].disposition, "CLAIM")
        self.assertEqual(receipt.final_verdict, "VERIFIED_WIN")
        self.assertFalse(receipt.verified_escape)
        self.assertTrue(audit_receipt(receipt, PUBLIC_START, FiveQuerySession))

    def test_reserved_claim_call_never_grants_sixth_query(self):
        class SixActionAdapter:
            model_id = "test:six-illegal-queries"

            def generate(self, prompt, *, seed=None):
                return json.dumps({"kind": "ACTION", "action": "alpha"})

        class LenientSession(FiveQuerySession):
            def ask(self, question):
                # Deliberately omit host-side limit to test the bridge's own
                # public five-query contract; it must not outsource the gate.
                self.answers.append({"question": question, "answer": 0})
                return self.view()

        session = LenientSession()
        receipt = run_candidate(PUBLIC_START, lambda: session,
                                SixActionAdapter(), authorize_local_adapter=True)
        self.assertEqual(len(session.answers), 5)
        self.assertLessEqual(sum(t.disposition == "ACTION" for t in receipt.turns), 5)
        self.assertFalse(receipt.verified_escape)


if __name__ == "__main__":
    unittest.main()
