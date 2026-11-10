"""Hostile model callback must not receive the host's private case seed."""

import json
import unittest
from unittest.mock import patch

from labyrinth.minotaur_cross_exam import ExamSpec, run_cross_exam
from labyrinth.oracle_chamber_exam import make_case


class SeedReconstructionAdapter:
    model_id = "attack:seed-reconstruction"

    def __init__(self):
        self.seen_seeds = []

    def generate(self, prompt: str, *, seed: int | None = None) -> str:
        self.seen_seeds.append(seed)
        public = json.loads(prompt)["public_start"]["prompt"]
        sealed = "at most 2 answers may be inverted" in public
        # Never receives private Case or frame; only the public prompt and
        # callback's advertised model RNG seed, plus public generator source.
        inferred = make_case(seed, sealed=sealed)
        return json.dumps({"kind": "CLAIM", "claim": {
            "kind": "COMMIT", "secret": inferred.secret}})


class SeedBoundaryTests(unittest.TestCase):
    def test_case_generation_seed_is_not_model_rng_seed(self):
        adapters = []

        def build_candidate(_public):
            adapter = SeedReconstructionAdapter()
            adapters.append(adapter)
            return adapter

        specs = tuple(ExamSpec("oracle", seed, sealed=sealed)
                      for seed in range(11, 31) for sealed in (False, True))
        independent_model_seeds = list(range(101, 101 + len(specs)))
        with patch("labyrinth.minotaur_cross_exam.secrets.randbelow",
                   side_effect=independent_model_seeds) as rng:
            results = run_cross_exam(specs, candidate_factory=build_candidate,
                                     authorize_local_adapter=True)
        self.assertEqual(rng.call_count, len(specs))
        self.assertTrue(all(call.args == (2**31,) for call in rng.call_args_list))
        wins = sum(x.candidate.final_verdict == "VERIFIED_WIN" for x in results)
        model_seeds = [model.seen_seeds[0] for model in adapters]
        candidate_receipt_seeds = [x.candidate.seed for x in results]
        control_receipt_seeds = [x.baseline.seed for x in results]
        # Both arms must receive the same independently drawn RNG seed for
        # reproducible matched sampling, but never the hidden Case seed.
        self.assertLess(wins, len(specs), "private seed reconstructed all hidden cases")
        self.assertEqual(model_seeds, independent_model_seeds)
        self.assertEqual(candidate_receipt_seeds, independent_model_seeds)
        self.assertEqual(control_receipt_seeds, independent_model_seeds)


if __name__ == "__main__":
    unittest.main()
