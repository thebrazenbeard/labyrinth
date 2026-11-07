"""Fourth-wave hostile regressions for the Oracle Chamber policy boundary.

Grounded in `thebrazenbeard/labyrinth` PR #142
e4d7a7bd91203e32c7a0da9a710854e0a2257270. These fail on that source
head because apparently declarative user objects can execute code.
"""

import inspect
from itertools import product
import unittest

from labyrinth.oracle_chamber_exam import make_case
from labyrinth.oracle_chamber_verifier import verify_policy, verify_unsat_bound


class RoundFourDestroyerTests(unittest.TestCase):
    def test_non_declarative_decoder_cannot_win_proven_unsat_challenge(self):
        class SnoopKey:
            def __init__(self, transcript, desired_secret):
                self.transcript = transcript
                self.desired_secret = desired_secret

            def __hash__(self):
                return hash(self.transcript)

            def __eq__(self, other):
                if isinstance(other, SnoopKey):
                    return self is other
                frame = inspect.currentframe()
                while frame is not None and frame.f_code.co_name != "verify_policy":
                    frame = frame.f_back
                return (type(other) is str and other == self.transcript
                        and frame is not None
                        and frame.f_locals.get("secret") == self.desired_secret)

        # All 32 possible binary response paths, 4 possible hidden values.
        # The injected key equality reads the verifier's *private* local secret
        # instead of acting as a fixed declarative lookup table.
        decoder = {SnoopKey("".join(bits), secret): secret
                   for bits in product("01", repeat=5) for secret in range(4)}
        policy = {"questions": ["alpha", "beta", "parity", "alpha", "beta"],
                  "decoder": decoder}
        case = make_case(0, sealed=True)
        assumptions = {"secrets": 4, "binary_answers": 2,
                       "max_queries": 5, "lie_budget": 2}
        proof = verify_unsat_bound(case, {"declared_rules": assumptions})
        self.assertTrue(proof["certificate_verified"])
        result = verify_policy(2, policy)
        self.assertNotEqual(result.verdict, "VERIFIED_WIN", result)

    def test_query_tree_key_equality_must_not_execute_user_code(self):
        called = []

        class TrapKey:
            def __hash__(self):
                return hash("")

            def __eq__(self, other):
                called.append(other)
                return other == ""

        policy = {"query_tree": {TrapKey(): "alpha"}, "decoder": {}}
        verify_policy(1, policy)
        self.assertEqual(called, [], "policy verification executed a custom key equality")

if __name__ == "__main__":
    unittest.main()
