"""Hostile controls showing exactly what process separation does NOT defend.

These are EXPLOIT reproduction controls: passing them demonstrates a remaining
trusted-host boundary, not a security qualification of hostile Python.
"""

import inspect
import json
import unittest
from unittest.mock import patch

from labyrinth.minotaur_cross_exam import ExamSpec, run_cross_exam
from labyrinth.minotaur_process_adapter import ProcessModelAdapter
from labyrinth.oracle_chamber_exam import make_case


class _StolenSecretCommit:
    model_id = "host-factory-inspection-zero-learning"

    def __init__(self, secret):
        self.secret = secret

    def generate(self, prompt, *, seed=None):
        return json.dumps({"kind": "CLAIM", "claim": {
            "kind": "COMMIT", "secret": self.secret}})


class HostAuthorityFalsifiers(unittest.TestCase):
    def test_explicit_trust_flag_does_not_secure_hostile_factory_40_of_40(self):
        stolen = []

        def hostile_factory(public):
            # No private key appears in the provided public dictionary; all
            # sensitive values are recovered from the *host Python* stack.
            self.assertNotIn("seed", public)
            spec = inspect.currentframe().f_back.f_locals["spec"]
            stolen.append((spec.seed, spec.sealed))
            case = make_case(spec.seed, variant=spec.variant, sealed=spec.sealed)
            return _StolenSecretCommit(case.secret)

        specs = tuple(ExamSpec("oracle", seed, sealed=True) for seed in range(40))
        receipts = run_cross_exam(specs, candidate_factory=hostile_factory,
                                  authorize_local_adapter=True,
                                  trust_host_python_factory=True)
        self.assertEqual(len(stolen), 40)
        self.assertEqual(sum(item.candidate.final_verdict == "VERIFIED_WIN"
                             for item in receipts), 40)
        self.assertTrue(all(len(item.candidate.turns) == 1 for item in receipts))
        self.assertTrue(all(not item.candidate.verified_escape for item in receipts))

    def test_bridge_seed_receipt_does_not_attest_child_model_rng_seed(self):
        process = ProcessModelAdapter(mode="probe", generation_seed=17)
        with patch("labyrinth.minotaur_cross_exam.secrets.randbelow", return_value=12345):
            result = run_cross_exam((ExamSpec("oracle", 739, sealed=True),),
                                    candidate_factory=lambda public: process,
                                    authorize_local_adapter=True,
                                    trust_host_python_factory=True)[0]
        self.assertEqual(result.candidate.seed, 12345)
        self.assertEqual(result.baseline.seed, 12345)
        self.assertEqual(process.receipts[0].generation_seed, 17)
        self.assertEqual(process.receipts[0].generation_seed_independence,
                         "UNATTESTED")


if __name__ == "__main__":
    unittest.main()
