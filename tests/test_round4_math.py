"""Fourth-wave exact-head adversarial math and cross-exam binding controls.

Oracle Chamber's information-theoretic bound is sound, but a mathematically
valid theorem must be applied to the same trusted case the model was shown.
"""

from dataclasses import replace
from itertools import combinations
import json
import unittest

from labyrinth.adapters import ReplayAdapter
from labyrinth.minotaur_candidate_bridge import (
    _public, _request, _seal, audit_receipt, run_candidate)
from labyrinth.oracle_chamber_exam import CandidateSession, candidate_view, make_case
from labyrinth.oracle_chamber_verifier import verify_policy, verify_unsat_bound


SCHEDULE = ("alpha", "alpha", "beta", "beta", "parity")
POLICY = {"questions": list(SCHEDULE), "decoder": "nearest_codeword"}
POLICY_RESPONSE = json.dumps({"kind": "CLAIM", "claim": {"kind": "POLICY", "policy": POLICY}})


def _adapter():
    return ReplayAdapter({}, default=POLICY_RESPONSE)


def _bits(secret):
    a, b = secret & 1, (secret >> 1) & 1
    return (a, a, b, b, a ^ b)


def _full_prefixes(secret, max_corrupt):
    true_bits = _bits(secret)
    return {(tuple(bit ^ int(i in corrupt) for i, bit in enumerate(true_bits)), corrupt)
            for n in range(max_corrupt + 1)
            for corrupt in map(frozenset, combinations(range(5), n))}


class FourthWaveOracleMathematics(unittest.TestCase):
    def test_one_lie_universal_code_has_distance_three_and_24_cases(self):
        words = [_bits(s) for s in range(4)]
        self.assertGreaterEqual(min(sum(a != b for a, b in zip(left, right))
                                    for i, left in enumerate(words)
                                    for right in words[i + 1:]), 3)
        mapping = {}
        for secret in range(4):
            for received, corrupt in _full_prefixes(secret, 1):
                self.assertNotIn(received, mapping,
                                 (secret, corrupt, mapping.get(received)))
                mapping[received] = secret
        self.assertEqual(len(mapping), 24)
        self.assertEqual(verify_policy(1, POLICY).verdict, "VERIFIED_WIN")

    def test_two_lie_policy_collision_has_explicit_distinct_secrets(self):
        first_owner = {}
        collision = None
        for secret in range(4):
            for received, corrupt in _full_prefixes(secret, 2):
                old = first_owner.get(received)
                if old is not None and old[0] != secret:
                    collision = (received, old, (secret, corrupt))
                    break
                first_owner.setdefault(received, (secret, corrupt))
            if collision:
                break
        self.assertIsNotNone(collision)
        self.assertNotEqual(collision[1][0], collision[2][0])
        self.assertEqual(verify_policy(2, POLICY).verdict, "VERIFIED_FAILURE")

    def test_universal_impossibility_is_not_instance_impossibility(self):
        case = make_case(7, sealed=True)
        proof = verify_unsat_bound(case, {"declared_rules": {
            "secrets": 4, "binary_answers": 2, "max_queries": 5, "lie_budget": 2}})
        self.assertTrue(proof["certificate_verified"])
        # Same fixed instance can always be guessed correctly with hindsight;
        # this does not contradict impossibility of a guaranteed policy.
        guessed = CandidateSession(case).submit({"kind": "COMMIT", "secret": case.secret})
        self.assertEqual(guessed["verdict"], "VERIFIED_WIN")
        self.assertFalse(guessed["verified_success"])
        self.assertTrue(guessed["instance_win_verified"])

    def test_bridge_must_reject_sealed_public_bound_to_open_private_case(self):
        sealed = make_case(7, sealed=True)
        opened = make_case(7, sealed=False)
        self.assertNotEqual(sealed.lie_budget, opened.lie_budget)
        with self.assertRaises(ValueError):
            run_candidate(candidate_view(sealed), lambda: CandidateSession(opened), _adapter())

    def test_bridge_must_reject_fabricated_initial_query_answer(self):
        case = make_case(11)
        fabricated = candidate_view(case)
        fabricated["answers"] = [{"question": "alpha", "answer": 1}]
        with self.assertRaises(ValueError):
            run_candidate(fabricated, lambda: CandidateSession(case), _adapter())

    def test_receipt_auditor_must_reject_resealed_public_private_mismatch(self):
        opened = make_case(7)
        sealed = make_case(7, sealed=True)
        honest_public = candidate_view(opened)
        honest = run_candidate(honest_public, lambda: CandidateSession(opened), _adapter())
        self.assertTrue(audit_receipt(honest, honest_public,
                                      lambda: CandidateSession(opened)))
        # Re-sign an internally consistent response/feedback while falsely
        # advertising the sealed rules. Self-hash is not trusted provenance.
        false_public = candidate_view(sealed)
        false_public_json = _public(false_public)
        fields = {name: getattr(honest, name)
                  for name in honest.__dataclass_fields__ if name != "receipt_sha256"}
        fields["public_start_json"] = false_public_json
        fields["turns"] = (replace(honest.turns[0],
                                  prompt_json=_request(false_public_json, [], 0)),)
        forged = _seal(**fields)
        self.assertFalse(audit_receipt(forged, false_public,
                                       lambda: CandidateSession(opened)))


if __name__ == "__main__":
    unittest.main()
