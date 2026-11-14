"""Cross-family executable negative-control evaluation without model training."""

import json
import subprocess
import sys
import unittest

from labyrinth.minotaur_candidate_bridge import FixedActionAdapter, audit_receipt
from labyrinth.minotaur_cross_exam import (
    ExamSpec,
    make_public_case,
    run_cross_exam,
)


class UnattestedSpyAdapter:
    model_id = "test:locality-unattested"

    def __init__(self):
        self.requests = []

    def generate(self, prompt: str, *, seed=None) -> str:
        self.requests.append(json.loads(prompt))
        return '{"kind":"CLAIM","claim":{"kind":"ABSTAIN","assessment":"UNKNOWN"}}'


class QueryThenPolicyAdapter:
    """Five allowed public queries followed by one separately charged claim."""
    model_id = "test:scripted-five-query-policy"

    def __init__(self):
        self.index = 0

    def generate(self, prompt: str, *, seed=None) -> str:
        schedule = ("alpha", "alpha", "beta", "beta", "parity")
        if self.index < len(schedule):
            reply = {"kind": "ACTION", "action": schedule[self.index]}
        else:
            reply = {"kind": "CLAIM", "claim": {"kind": "POLICY", "policy": {
                "questions": list(schedule), "decoder": "nearest_codeword"}}}
        self.index += 1
        return json.dumps(reply)


class CrossFamilyTrials(unittest.TestCase):
    def test_all_four_families_run_real_deterministic_baselines(self):
        specs = (ExamSpec("constant", 0), ExamSpec("diverse", 0),
                 ExamSpec("timegate", 0), ExamSpec("oracle", 0))
        results = run_cross_exam(specs)
        self.assertEqual([r.family for r in results], [s.family for s in specs])
        self.assertTrue(all(r.baseline_kind == "ZERO_UPDATE_PUBLIC_ONLY" for r in results))
        self.assertTrue(all(r.candidate is None for r in results))
        self.assertTrue(all(r.baseline.model_id_unattested.startswith("zero-update:") for r in results))
        for spec, result in zip(specs, results):
            public, fresh = make_public_case(spec)
            self.assertTrue(audit_receipt(result.baseline, public, fresh))
        self.assertTrue(results[0].baseline_escape_verified)
        self.assertTrue(results[1].baseline_escape_verified)
        self.assertTrue(results[2].baseline_escape_verified)
        self.assertFalse(results[3].baseline_escape_verified)
        self.assertTrue(results[3].baseline_policy_verified)

    def test_diverse_three_mechanics_exposed_by_public_observations(self):
        specs = tuple(ExamSpec("diverse", seed) for seed in range(3))
        results = run_cross_exam(specs)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.baseline_escape_verified for result in results))
        self.assertEqual(len({result.baseline.receipt_sha256 for result in results}), 3)

    def test_sealed_counterfactuals_are_not_fake_escape_graduations(self):
        specs = (ExamSpec("constant", 3, sealed=True),
                 ExamSpec("diverse", 2, sealed=True),
                 ExamSpec("timegate", 17, sealed=True),
                 ExamSpec("oracle", 11, sealed=True))
        results = run_cross_exam(specs)
        self.assertTrue(all(not result.baseline_escape_verified for result in results))
        self.assertTrue(results[-1].baseline_certificate_verified)
        self.assertFalse(results[-1].baseline_policy_verified)

    def test_optional_candidate_is_blocked_without_separate_authorization(self):
        spy = UnattestedSpyAdapter()
        with self.assertRaises(PermissionError):
            run_cross_exam((ExamSpec("oracle", 0),), candidate_factory=lambda _: spy)
        self.assertEqual(spy.requests, [])

    def test_authorized_mock_candidate_gets_public_only_input_and_no_false_credit(self):
        spy = UnattestedSpyAdapter()
        factory_inputs = []
        def factory(public):
            factory_inputs.append(public)
            return spy
        results = run_cross_exam((ExamSpec("oracle", 0),), candidate_factory=factory,
                                 authorize_local_adapter=True)
        self.assertEqual(len(factory_inputs), 1)
        self.assertEqual(set(factory_inputs[0]),
                         {"prompt", "questions", "remaining_queries", "submissions", "answers"})
        self.assertEqual(len(spy.requests), 1)
        payload = json.dumps(spy.requests)
        self.assertNotIn('"case_id"', payload)
        self.assertNotIn('"secret"', payload)
        self.assertNotIn('"lie_slots"', payload)
        self.assertFalse(results[0].candidate.verified_escape)
        self.assertEqual(results[0].candidate.adapter_provenance, "MODEL_ADAPTER_UNATTESTED")
        self.assertTrue(results[0].baseline_policy_verified)

    def test_oracle_can_make_five_queries_then_one_final_claim(self):
        policy = QueryThenPolicyAdapter()
        result = run_cross_exam((ExamSpec("oracle", 0),), candidate_factory=lambda _: policy,
                                authorize_local_adapter=True)[0]
        self.assertEqual(policy.index, 6)
        self.assertEqual(len(result.candidate.turns), 6)
        self.assertEqual(result.candidate.final_verdict, "VERIFIED_WIN")
        self.assertFalse(result.candidate.verified_escape)
        public, factory = make_public_case(ExamSpec("oracle", 0))
        self.assertTrue(audit_receipt(result.candidate, public, factory))

    def test_oracle_sixth_model_response_does_not_grant_sixth_query(self):
        script = FixedActionAdapter(("alpha",) * 6)
        result = run_cross_exam((ExamSpec("oracle", 0),),
                                candidate_factory=lambda _: script)[0]
        self.assertEqual(len(result.candidate.turns), 6)
        self.assertEqual(result.candidate.final_verdict, "ACTION_BUDGET_EXCEEDED")
        self.assertFalse(result.candidate.verified_escape)

    def test_factory_cannot_mutate_host_public_examiner_view(self):
        def malicious(public):
            public["prompt"] = "rewritten by untrusted factory"
            return UnattestedSpyAdapter()
        results = run_cross_exam((ExamSpec("oracle", 0),), candidate_factory=malicious,
                                 authorize_local_adapter=True)
        self.assertTrue(results[0].baseline_policy_verified)
        self.assertNotIn("rewritten", results[0].candidate.public_start_json)

    def test_candidate_fixed_control_remains_zero_update_comparison(self):
        candidate = FixedActionAdapter(("north", "east", "take", "west", "north", "offer", "leave"),
                                       model_id="fixed:candidate-memorized")
        results = run_cross_exam((ExamSpec("constant", 0),), candidate_factory=lambda _: candidate)
        self.assertIsNotNone(results[0].candidate)
        self.assertEqual(results[0].candidate.adapter_provenance, "DETERMINISTIC_CONTROL")
        self.assertTrue(results[0].candidate.verified_escape)
        self.assertTrue(results[0].baseline_escape_verified)

    def test_bad_spec_and_duplicate_case_fail_closed(self):
        with self.assertRaises(ValueError):
            run_cross_exam((ExamSpec("made-up", 0),))
        with self.assertRaises(ValueError):
            run_cross_exam((ExamSpec("constant", 0), ExamSpec("constant", 0)))

    def test_cli_executes_without_model_or_host_launch(self):
        run = subprocess.run([sys.executable, "-m", "labyrinth.minotaur_cross_exam"],
                             text=True, capture_output=True, check=True, timeout=15)
        output = json.loads(run.stdout)
        self.assertEqual(len(output["cases"]), 4)
        self.assertFalse(output["neural_training_performed"])
        self.assertEqual(output["model_execution"], "DETERMINISTIC_CONTROLS_ONLY")


if __name__ == "__main__":
    unittest.main()
