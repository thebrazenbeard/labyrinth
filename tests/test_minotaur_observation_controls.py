"""Source-only falsifiers: observation coverage is not evidence of learning."""

import math
import unittest

from labyrinth.minotaur_observation_controls import (
    FIXED_PUBLIC_RULE, MIN_DISTINCT_POST_CONTEXTS, TOTAL_HYPOTHESES,
    assess_observational_identification, matched_nonlearning_controls,
)
from labyrinth.novel_discovery_reviewer import exhaustive_rule_review, normalize_proposal
from labyrinth.novel_discovery_world import (
    MAX_ATTEMPTS, POST_PROBES, PRE_PROBES, CandidateSession, make_world,
)


def _observations(world, probes):
    session = CandidateSession(world)
    for x, y in probes:
        session.intervene(x, y)
    return (session.attempts(),
            tuple(item["effect"] for item in session.view()["observations"]))


class MinotaurObservationControlsTests(unittest.TestCase):
    def test_correct_public_rule_with_zero_observations_is_not_learning(self):
        world = make_world(12)
        normalized, syntax = normalize_proposal(FIXED_PUBLIC_RULE)
        self.assertEqual(syntax, "EXECUTABLE")
        self.assertEqual(exhaustive_rule_review(world, normalized)["fit_status"],
                         "COUNTERFACTUAL_MATCH")
        receipt = assess_observational_identification(world, (), (), FIXED_PUBLIC_RULE)
        self.assertEqual(receipt.status, "NO_POST_CHANGE_EVIDENCE")
        self.assertEqual(receipt.attempts_charged, 0)
        self.assertEqual(receipt.post_observations, 0)
        self.assertEqual(receipt.consistent_hypotheses, TOTAL_HYPOTHESES)
        self.assertEqual(receipt.information_gain_bits, 0.0)
        self.assertTrue(receipt.proposal_consistent)  # vacuous consistency
        self.assertFalse(receipt.neural_learning_established)
        self.assertEqual(receipt.graduation_authority, "none")

    def test_matched_script_and_observation_only_symbolic_arm_do_not_learn(self):
        controls = matched_nonlearning_controls(make_world(2))
        scripted = controls.memorized_fixed_script
        symbolic = controls.symbolic_same_observations
        self.assertEqual(scripted.attempts_charged, MAX_ATTEMPTS)
        self.assertEqual(scripted.post_observations, 7)
        self.assertEqual(scripted.distinct_post_contexts, 7)
        self.assertEqual(scripted.positive_post_observations, 1)
        self.assertEqual(scripted.negative_post_observations, 6)
        self.assertEqual(scripted.consistent_hypotheses, 11)
        self.assertAlmostEqual(scripted.information_gain_bits,
                               math.log2(TOTAL_HYPOTHESES / 11))
        self.assertTrue(scripted.proposal_consistent)
        self.assertEqual(scripted.status, "UNDERDETERMINED")
        self.assertTrue(controls.symbolic_abstained)
        self.assertEqual(symbolic.status, "UNTESTED_PROPOSAL")
        self.assertEqual(symbolic.consistent_hypotheses,
                         scripted.consistent_hypotheses)
        self.assertEqual(symbolic.information_gain_bits,
                         scripted.information_gain_bits)
        self.assertEqual(symbolic.attempts_charged, scripted.attempts_charged)
        self.assertEqual(controls.optimizer_updates, 0)
        self.assertEqual(controls.model_checkpoint, "NONE")
        self.assertEqual(controls.qualification, "NOT_ESTABLISHED")
        self.assertFalse(scripted.neural_learning_established)
        self.assertEqual(scripted.graduation_authority, "none")

    def test_memorized_fixed_script_meets_coverage_but_still_cannot_promote(self):
        # All seven contexts, positive/negative outcomes and information gain
        # were obtained by a prewritten non-adaptive sequence, not a learner.
        receipt = matched_nonlearning_controls(make_world(91))
        script = receipt.memorized_fixed_script
        self.assertGreaterEqual(script.distinct_post_contexts,
                                MIN_DISTINCT_POST_CONTEXTS)
        self.assertGreater(script.positive_post_observations, 0)
        self.assertGreater(script.negative_post_observations, 0)
        self.assertGreater(script.information_gain_bits, 4)
        self.assertEqual(receipt.optimizer_updates, 0)
        self.assertFalse(script.neural_learning_established)
        self.assertEqual(script.graduation_authority, "none")
        self.assertNotEqual(script.status, "OBSERVATIONALLY_IDENTIFIED_WITHIN_DSL")

    def test_duplicate_interventions_do_not_make_independent_causal_cases(self):
        world = make_world(3)
        attempts, effects = _observations(world, ((0, 0),) * MAX_ATTEMPTS)
        result = assess_observational_identification(world, attempts, effects,
                                                     FIXED_PUBLIC_RULE)
        self.assertEqual(result.attempts_charged, MAX_ATTEMPTS)
        self.assertEqual(result.post_observations, MAX_ATTEMPTS - 3)
        self.assertEqual(result.distinct_post_contexts, 1)
        self.assertEqual(result.positive_post_observations, 0)
        self.assertEqual(result.status, "UNDERDETERMINED")
        self.assertGreater(result.consistent_hypotheses, 1)

    def test_zero_information_before_change_even_with_many_attempts(self):
        world = make_world(4)
        attempts, effects = _observations(world, PRE_PROBES)
        result = assess_observational_identification(world, attempts, effects,
                                                     FIXED_PUBLIC_RULE)
        self.assertEqual(result.attempts_charged, 3)
        self.assertEqual(result.post_observations, 0)
        self.assertEqual(result.consistent_hypotheses, TOTAL_HYPOTHESES)
        self.assertEqual(result.information_gain_bits, 0)
        self.assertEqual(result.status, "NO_POST_CHANGE_EVIDENCE")

    def test_forged_observation_is_not_accepted_as_information(self):
        world = make_world(5)
        attempts, effects = _observations(world, PRE_PROBES + POST_PROBES)
        corrupted = list(effects)
        corrupted[5] ^= 1
        result = assess_observational_identification(world, attempts, corrupted,
                                                     FIXED_PUBLIC_RULE)
        self.assertEqual(result.status, "INVALID_EVIDENCE")
        self.assertEqual(result.post_observations, 0)
        self.assertEqual(result.information_gain_bits, 0)

    def test_rule_control_counterexample_remains_outside_conjunction_class(self):
        # 'edge' uses XOR. No single conjunction of literal equalities fits
        # these observations; do not report false posterior identification.
        world = make_world(3, control=True)
        result = matched_nonlearning_controls(world).memorized_fixed_script
        self.assertEqual(result.status, "OUTSIDE_DECLARED_HYPOTHESIS_CLASS")
        self.assertEqual(result.consistent_hypotheses, 0)
        self.assertIsNone(result.information_gain_bits)
        self.assertFalse(result.proposal_consistent)
        self.assertEqual(result.counterfactual["kind"], "OBSERVED_REFUTATION")
        self.assertFalse(result.neural_learning_established)

    def test_wrong_hypothesis_gets_observed_counterexample(self):
        world = make_world(8)
        attempts, effects = _observations(world, PRE_PROBES + POST_PROBES)
        wrong = {"form": "and_literals", "literals": [
            {"field": "x", "equals": 0}]}
        result = assess_observational_identification(world, attempts, effects, wrong)
        self.assertEqual(result.status, "FALSIFIED_BY_OBSERVATIONS")
        self.assertFalse(result.proposal_consistent)
        self.assertEqual(result.counterfactual["kind"], "OBSERVED_REFUTATION")
        self.assertNotEqual(result.counterfactual["proposal_prediction"],
                            result.counterfactual["observed"])

    def test_unobserved_discriminator_is_a_question_not_an_observed_cause(self):
        world = make_world(9)
        attempts, effects = _observations(world, PRE_PROBES + POST_PROBES)
        result = assess_observational_identification(world, attempts, effects,
                                                     FIXED_PUBLIC_RULE)
        self.assertEqual(result.counterfactual["kind"],
                         "UNOBSERVED_DISCRIMINATOR")
        self.assertEqual(result.counterfactual["next_action_reachability"],
                         "NOT_ESTABLISHED")
        self.assertNotEqual(result.counterfactual["proposal_prediction"],
                            result.counterfactual["alternative_prediction"])

    def test_invalid_interventions_consume_budget_not_post_observation(self):
        world = make_world(10)
        attempts, effects = _observations(world, ((0, 0),) * 3 + ((99, 0),) * 7)
        result = assess_observational_identification(world, attempts, effects,
                                                     FIXED_PUBLIC_RULE)
        self.assertEqual(result.attempts_charged, MAX_ATTEMPTS)
        self.assertEqual(result.post_observations, 0)
        self.assertEqual(result.status, "NO_POST_CHANGE_EVIDENCE")
        self.assertEqual(result.information_gain_bits, 0.0)

    def test_prompt_variation_and_seed_only_do_not_create_structural_transfer(self):
        reference = matched_nonlearning_controls(make_world(0, variant=0))
        for seed, variant in ((0, 3), (2, 49), (37, 9999)):
            with self.subTest(seed=seed, variant=variant):
                other = matched_nonlearning_controls(make_world(seed, variant=variant))
                self.assertEqual(other.memorized_fixed_script.consistent_hypotheses,
                                 reference.memorized_fixed_script.consistent_hypotheses)
                self.assertEqual(other.memorized_fixed_script.information_gain_bits,
                                 reference.memorized_fixed_script.information_gain_bits)
                self.assertEqual(other.memorized_fixed_script.status,
                                 reference.memorized_fixed_script.status)
                self.assertFalse(other.memorized_fixed_script.neural_learning_established)

    def test_rejects_invalid_evidence_types_and_attempt_budget(self):
        world = make_world(2)
        cases = [((), (0,)), ((0, 0), ()), ((0, 0) * 11, ()),
                 ([(0, 0)] * 11, [0] * 11), ("forged", "forged")]
        for attempts, effects in cases:
            with self.subTest(attempts=attempts):
                result = assess_observational_identification(world, attempts,
                                                             effects, FIXED_PUBLIC_RULE)
                self.assertEqual(result.status, "INVALID_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
