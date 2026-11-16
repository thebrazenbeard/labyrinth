"""Regression: proof of a policy must not become an instance escape receipt."""

import json
import unittest

from labyrinth.minotaur_candidate_bridge import (FixedActionAdapter, audit_receipt,
                                                 run_candidate, run_matched_pair)
from labyrinth.timegate_exam import (CandidateSession as TimegateSession,
                                     candidate_view as timegate_public_view,
                                     make_case as make_timegate_case,
                                     search as timegate_search)


PUBLIC = {"observation": "A public evidence chamber.", "remaining_actions": 5}


class PolicyOnlyAdapter:
    model_id = "test:deterministic-policy-claim"

    def generate(self, prompt, *, seed=None):
        return json.dumps({"kind": "CLAIM", "claim": {"kind": "POLICY", "policy": {}}})


class PolicyOnlySession:
    """Host-shaped semantics: a valid policy is not a played instance win."""

    def __init__(self):
        self._outcome = None

    def view(self):
        return self._outcome or {
            "observation": PUBLIC["observation"], "finished": False,
            "verdict": "UNKNOWN", "instance_win_verified": False,
        }

    def submit(self, claim):
        if claim.get("kind") != "POLICY":
            raise ValueError("unsupported claim")
        self._outcome = {
            "observation": PUBLIC["observation"], "finished": True,
            "verdict": "VERIFIED_WIN", "policy_verified": True,
            "instance_win_verified": False, "verified_success": False,
        }
        return self.view()


class BridgeNonEscapeTests(unittest.TestCase):
    def test_verified_policy_does_not_count_as_instance_escape(self):
        receipt = run_candidate(PUBLIC, PolicyOnlySession, PolicyOnlyAdapter(),
                                authorize_local_adapter=True)
        self.assertEqual(receipt.final_verdict, "VERIFIED_WIN")
        self.assertFalse(receipt.verified_escape)
        self.assertTrue(audit_receipt(receipt, PUBLIC, PolicyOnlySession))

    def test_overridden_control_subclass_requires_model_authorization(self):
        class UntrustedSubclass(FixedActionAdapter):
            def __init__(self):
                super().__init__((), model_id="untrusted:overridden-generate")

            def generate(self, prompt, *, seed=None):
                return json.dumps({"kind": "CLAIM", "claim": {
                    "kind": "POLICY", "policy": {}}})

        # A Python subclass can call arbitrary code, so mere isinstance()
        # must not confer deterministic-control provenance or authorization.
        with self.assertRaises(PermissionError):
            run_candidate(PUBLIC, PolicyOnlySession, UntrustedSubclass())

    def test_matched_pair_must_reject_shared_underlying_examiner(self):
        case = make_timegate_case(0)
        public = timegate_public_view(case)
        witness = timegate_search(case).witness
        shared = TimegateSession(case)

        class DistinctProxy:
            def __init__(self, backend):
                self.backend = backend

            def view(self):
                return self.backend.view()

            def act(self, action):
                return self.backend.act(action)

            def submit(self, claim):
                return self.backend.submit(claim)

        # Two newly created wrappers have distinct Python IDs but share state.
        # Candidate ABSTAIN terminalizes the backend; a contaminated control
        # then gets zero turns despite having a winning fixed script.
        try:
            pair = run_matched_pair(
                public, lambda: DistinctProxy(shared),
                candidate=FixedActionAdapter(()), control=FixedActionAdapter(witness))
        except (ValueError, RuntimeError):
            return  # Fail closed on evidence of a shared underlying examiner.
        self.assertTrue(pair.control_win)
        self.assertEqual(len(pair.control.turns), len(witness))
        self.assertTrue(audit_receipt(pair.control, public,
                                      lambda: TimegateSession(case)))


if __name__ == "__main__":
    unittest.main()
