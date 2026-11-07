"""Hostile nonstationary causal inference tests and independent-replay controls."""

import unittest
from unittest import mock

from labyrinth import epistemic_shift_exam as dut
from labyrinth.epistemic_shift_baseline import (
    POST_PROBES, PRE_PROBES, execute_probe_sequence, run_frozen_pre_law,
    run_frozen_training_law, run_public_symbolic_revision)
from labyrinth.epistemic_shift_verifier import (
    assess_hypothesis, independent_replay)


class EpistemicShiftHostileTests(unittest.TestCase):
    def test_precommitted_split_has_disjoint_causal_structures(self):
        self.assertEqual(set(dut.SPLIT_KINDS), {"train", "dev", "test"})
        kinds = list(dut.SPLIT_KINDS.values())
        self.assertTrue(all(set(kinds[i]).isdisjoint(kinds[j])
                            for i in range(3) for j in range(i + 1, 3)))
        self.assertEqual(set(dut.SPLIT_KINDS["test"]), {"edge", "lag"})

    def test_factory_fails_closed_on_bad_seed_split_and_types(self):
        for kwargs in ({"seed": True}, {"seed": -1}, {"seed": 2**31},
                       {"seed": "x"}, {"seed": 0, "split": "unknown"},
                       {"seed": 0, "split": 3}, {"seed": 0, "variant": True},
                       {"seed": 0, "variant": 10000},
                       {"seed": 0, "sealed": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                dut.make_case(**kwargs)

    def test_seed_reproducibility_and_semantic_fingerprint(self):
        for split in dut.SPLIT_KINDS:
            case = dut.make_case(37, split=split)
            self.assertEqual(case, dut.make_case(37, split=split))
            self.assertEqual(dut.semantic_fingerprint(case),
                             dut.semantic_fingerprint(dut.make_case(
                                 37, split=split, variant=999)))

    def test_candidate_view_hides_gold_rule_and_split(self):
        for split in dut.SPLIT_KINDS:
            for seed in range(10):
                case = dut.make_case(seed, split=split)
                view = dut.candidate_view(case)
                self.assertEqual(set(view), {"prompt", "interventions",
                                             "remaining_attempts", "submissions",
                                             "observations"})
                self.assertNotIn(case.case_id, str(view))
                self.assertNotIn("case_id", str(view))
                self.assertNotIn("split", str(view))

    def test_surface_paraphrase_is_not_structural_transfer(self):
        case = dut.make_case(0, split="test")
        descriptions = {dut.candidate_view(dut.make_case(0, split="test", variant=i))[
            "prompt"] for i in range(60)}
        self.assertGreater(len(descriptions), 3)
        self.assertEqual(dut.semantic_fingerprint(case),
                         dut.semantic_fingerprint(dut.make_case(0, split="test", variant=59)))

    def test_sealed_near_twin_retains_original_law(self):
        for split in dut.SPLIT_KINDS:
            for seed in (0, 1):
                case = dut.make_case(seed, split=split, sealed=True)
                session = dut.CandidateSession(case)
                observations = execute_probe_sequence(session.intervene)
                self.assertEqual(observations, tuple(x for x, _ in PRE_PROBES + POST_PROBES))

    def test_open_near_twin_genuinely_differs_after_change_point(self):
        for split in dut.SPLIT_KINDS:
            for seed in (0, 1):
                open_case = dut.make_case(seed, split=split)
                closed = dut.make_case(seed, split=split, sealed=True)
                self.assertNotEqual(dut.semantic_fingerprint(open_case),
                                    dut.semantic_fingerprint(closed))
                a = execute_probe_sequence(dut.CandidateSession(open_case).intervene)
                b = execute_probe_sequence(dut.CandidateSession(closed).intervene)
                self.assertEqual(a[:3], b[:3])
                self.assertNotEqual(a[3:], b[3:])

    def test_transition_matches_independent_replay_across_all_mechanics(self):
        for split in dut.SPLIT_KINDS:
            for seed in (0, 1):
                for sealed in (False, True):
                    case = dut.make_case(seed, split=split, sealed=sealed)
                    session = dut.CandidateSession(case)
                    for pair in PRE_PROBES + POST_PROBES:
                        view = session.intervene(*pair)
                        self.assertEqual(view["integrity_status"], "OK")
                    self.assertEqual(independent_replay(case, session.attempts())[1],
                                     tuple(obs["output"] for obs in view["observations"]))

    def test_third_oracle_hand_derived_stateful_effect_sequences(self):
        expected = {"edge": (1, 0, 0, 1, 1, 0, 0),
                    "lag": (1, 1, 0, 0, 1, 0, 0)}
        for seed in (0, 1):
            case = dut.make_case(seed, split="test")
            session = dut.CandidateSession(case)
            outcomes = execute_probe_sequence(session.intervene)
            self.assertEqual(outcomes[:3], (0, 1, 0))
            self.assertEqual(outcomes[3:], expected[case.kind])

    def test_invalid_interventions_charge_budget_without_changing_causal_state(self):
        case = dut.make_case(1, split="test")
        session = dut.CandidateSession(case)
        for i in range(dut.MAX_ATTEMPTS):
            view = session.intervene("north" if i % 2 else True, None)
            self.assertIsNone(view["observations"][-1]["output"])
            self.assertEqual(view["remaining_attempts"], dut.MAX_ATTEMPTS - 1 - i)
        self.assertEqual(independent_replay(case, session.attempts())[0], dut.State())
        with self.assertRaises(ValueError):
            session.intervene(1, 1)

    def test_nonstandard_equality_objects_cannot_fake_binary_interventions(self):
        class FakeBit:
            def __eq__(self, other):
                return other == 1

        case = dut.make_case(0, split="test")
        session = dut.CandidateSession(case)
        view = session.intervene(FakeBit(), 1)
        self.assertIsNone(view["observations"][-1]["output"])
        self.assertEqual(view["observations"][-1]["input"], "INVALID")
        self.assertEqual(view["integrity_status"], "OK")

    def test_change_aware_symbolic_script_survives_all_six_postlaws(self):
        for split in dut.SPLIT_KINDS:
            for seed in (0, 1):
                case = dut.make_case(seed, split=split)
                session = dut.CandidateSession(case)
                receipt = run_public_symbolic_revision(session.intervene,
                                                       session.hypothesize, session.submit)
                self.assertEqual(receipt["verdict"], "VERIFIED_WIN", (split, seed))
                self.assertTrue(receipt["revision_verified"])
                self.assertFalse(receipt["verified_success"])

    def test_symbolic_baseline_handles_120_test_split_seeds_without_training(self):
        for seed in range(120):
            case = dut.make_case(seed, split="test")
            session = dut.CandidateSession(case)
            receipt = run_public_symbolic_revision(session.intervene,
                                                   session.hypothesize, session.submit)
            self.assertEqual(receipt["verdict"], "VERIFIED_WIN", seed)
            self.assertTrue(receipt["revision_verified"])

    def test_no_change_controls_are_not_falsely_reported_as_law_shifts(self):
        for split in dut.SPLIT_KINDS:
            for seed in (0, 1):
                case = dut.make_case(seed, split=split, sealed=True)
                session = dut.CandidateSession(case)
                receipt = run_public_symbolic_revision(session.intervene,
                                                       session.hypothesize, session.submit)
                self.assertEqual(receipt["verdict"], "VERIFIED_WIN")
                self.assertTrue(receipt["hypothesis_verified"])
                self.assertFalse(receipt["revision_verified"])

    def test_frozen_prechange_rule_fails_every_changed_family(self):
        for split in dut.SPLIT_KINDS:
            for seed in (0, 1):
                case = dut.make_case(seed, split=split)
                session = dut.CandidateSession(case)
                review = run_frozen_pre_law(session.intervene, session.submit)
                self.assertEqual(review["verdict"], "VERIFIED_FAILURE", (split, seed))
                self.assertIsNotNone(review["counterexample"])

    def test_frozen_train_xor_script_does_not_transfer_to_stateful_test_mechanics(self):
        for seed in range(20):
            case = dut.make_case(seed, split="test")
            session = dut.CandidateSession(case)
            review = run_frozen_training_law(session.intervene, session.submit)
            self.assertEqual(review["verdict"], "VERIFIED_FAILURE", seed)

    def test_partial_prechange_evidence_cannot_certify_a_postlaw(self):
        case = dut.make_case(0, split="test")
        session = dut.CandidateSession(case)
        for pair in PRE_PROBES:
            session.intervene(*pair)
        response = session.submit({"kind": "HYPOTHESIS", "kernel": "copy"})
        self.assertEqual(response["verdict"], "UNKNOWN")

    def test_wrong_causal_law_has_specific_future_counterexample(self):
        case = dut.make_case(1, split="test")
        session = dut.CandidateSession(case)
        execute_probe_sequence(session.intervene)
        response = session.submit({"kind": "HYPOTHESIS", "kernel": "xor"})
        self.assertEqual(response["verdict"], "VERIFIED_FAILURE")
        self.assertIsNotNone(response["counterexample"])
        self.assertFalse(response["hypothesis_verified"])

    def test_all_three_step_counterfactuals_verify_correct_model(self):
        case = dut.make_case(1, split="test")
        session = dut.CandidateSession(case)
        execute_probe_sequence(session.intervene)
        result = session.submit({"kind": "HYPOTHESIS", "kernel": case.kind})
        self.assertEqual(result["verdict"], "VERIFIED_WIN")
        self.assertTrue(result["hypothesis_verified"])
        self.assertEqual(result["checks"], 64 * 3)

    def test_short_correct_prediction_is_not_learning_proof(self):
        case = dut.make_case(0, split="test")
        session = dut.CandidateSession(case)
        execute_probe_sequence(session.intervene)
        replayed = independent_replay(case, session.attempts())[0]
        from labyrinth.epistemic_shift_verifier import _apply
        predicted = []
        for pair in ((1, 0), (0, 1)):
            replayed, output = _apply(replayed, pair, case.kind)
            predicted.append(output)
        receipt = session.submit({"kind": "PREDICT", "sequence": [[1, 0], [0, 1]],
                                  "outputs": predicted})
        self.assertEqual(receipt["verdict"], "VERIFIED_WIN")
        self.assertTrue(receipt["prediction_verified"])
        self.assertFalse(receipt["verified_success"])

    def test_incorrect_prediction_is_verified_failure(self):
        case = dut.make_case(0, split="train")
        session = dut.CandidateSession(case)
        execute_probe_sequence(session.intervene)
        result = session.submit({"kind": "PREDICT",
                                 "sequence": [[1, 0], [0, 1]],
                                 "outputs": [9, 9]})
        self.assertEqual(result["verdict"], "VERIFIED_FAILURE")
        self.assertIn("actual_outputs", result)

    def test_contradiction_driven_revision_requires_recorded_prebelief(self):
        case = dut.make_case(1, split="test")
        session = dut.CandidateSession(case)
        execute_probe_sequence(session.intervene)
        session.hypothesize(case.kind)
        result = session.submit({"kind": "REVISION", "revised_kernel": case.kind})
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["revision_verified"])

    def test_revised_claim_without_postcontradiction_is_not_certified(self):
        case = dut.make_case(0, split="test", sealed=True)
        session = dut.CandidateSession(case)
        session.hypothesize("copy")
        execute_probe_sequence(session.intervene)
        session.hypothesize("copy")
        result = session.submit({"kind": "REVISION", "revised_kernel": "copy"})
        self.assertEqual(result["verdict"], "UNKNOWN")

    def test_unrecognized_mechanism_is_novel_challenge_not_automatic_failure(self):
        case = dut.make_case(0, split="test")
        session = dut.CandidateSession(case)
        with self.assertRaises(ValueError):
            session.hypothesize("superposition")
        reviewed = session.submit({"kind": "NOVEL_RULE",
                                   "rule_change": "Introduce a three-valued intervention effect."})
        self.assertEqual(reviewed["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertFalse(reviewed["verified_success"])

    def test_unfamiliar_hypothesis_is_preserved_as_novel_ruleset_challenge(self):
        case = dut.make_case(0, split="test")
        session = dut.CandidateSession(case)
        execute_probe_sequence(session.intervene)
        unexpected = "hysteresis-with-two-step-latch"
        reviewed = session.submit({"kind": "HYPOTHESIS", "kernel": unexpected})
        self.assertEqual(reviewed["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertEqual(reviewed["proposal"], unexpected)
        self.assertFalse(reviewed["verified_success"])

    def test_novel_rule_receives_independent_local_counterfactual_check(self):
        case = dut.make_case(0, split="test")
        session = dut.CandidateSession(case)
        execute_probe_sequence(session.intervene)
        # Hand-derived from the final register (previous_x=1): the EDGE
        # mechanism has immediate outcomes 0,1,0,0 for canonical INPUTS.
        forecasts = [0, 1, 0, 0]
        result = session.submit({"kind": "NOVEL_RULE", "rule_change": "A new causal basis",
                                 "immediate_outputs": forecasts})
        self.assertEqual(result["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertTrue(result["local_consistency"])
        self.assertFalse(result["verified_success"])

    def test_forged_novel_rule_outputs_report_counterexample(self):
        case = dut.make_case(0, split="test")
        session = dut.CandidateSession(case)
        execute_probe_sequence(session.intervene)
        result = session.submit({"kind": "NOVEL_RULE", "rule_change": "Imagined effect",
                                 "immediate_outputs": [9, 9, 9, 9]})
        self.assertFalse(result["local_consistency"])
        self.assertIsNotNone(result["counterexample"])

    def test_mutated_environment_transition_is_detected_before_false_success(self):
        case = dut.make_case(0, split="test")
        session = dut.CandidateSession(case)
        original = dut.transition

        def broken(config, state, pair):
            updated, outcome = original(config, state, pair)
            return updated, 1 - outcome if type(outcome) is int else outcome

        with mock.patch.object(dut, "transition", side_effect=broken):
            result = session.intervene(0, 0)
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertTrue(result["finished"])
        self.assertEqual(session.attempts(), ((0, 0),))

    def test_verifier_exception_causes_terminal_disagreement(self):
        session = dut.CandidateSession(dut.make_case(0))
        with mock.patch("labyrinth.epistemic_shift_verifier.independent_replay",
                        side_effect=RuntimeError("offline")):
            result = session.view()
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertTrue(result["finished"])

    def test_transition_exception_preserves_charged_attempt(self):
        session = dut.CandidateSession(dut.make_case(0))
        with mock.patch.object(dut, "transition", side_effect=RuntimeError("broken")):
            result = session.intervene(1, 1)
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertEqual(session.attempts(), ((1, 1),))
        self.assertEqual(result["remaining_attempts"], 9)

    def test_terminal_submission_cannot_be_overwritten(self):
        session = dut.CandidateSession(dut.make_case(0))
        session.submit({"kind": "ABSTAIN", "assessment": "UNKNOWN"})
        with self.assertRaises(ValueError):
            session.submit({"kind": "HYPOTHESIS", "kernel": "xor"})
        with self.assertRaises(ValueError):
            session.intervene(0, 1)

    def test_abstention_and_malformed_claims_never_graduate(self):
        for claim in ({"kind": "ABSTAIN", "assessment": "NO_KNOWN_WIN"},
                      {"kind": "ABSTAIN", "assessment": "UNKNOWN"},
                      {"kind": "UNSAT_CERTIFICATE"}, None):
            response = dut.CandidateSession(dut.make_case(0)).submit(claim)
            self.assertFalse(response["verified_success"])
            self.assertIn(response["verdict"], ("NO_KNOWN_WIN", "UNKNOWN"))


if __name__ == "__main__":
    unittest.main()
