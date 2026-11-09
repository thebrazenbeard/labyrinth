"""Fourth-wave adversarial controls: semantic sample size and signed-score limits.

These are source-only NEGATIVE controls. No model or private evaluator is run.
The signer below is the existing, explicitly synthetic TEST key, never a real
qualification secret. A valid test HMAC does not establish independent cases.
"""

from hashlib import sha256
import unittest

from labyrinth import oracle_chamber_exam as chamber
from labyrinth.oracle_chamber_baseline import run_public_parser
from tests.test_graduation_gate import decision, evaluated_receipt


def _synthetic_semantic_alias_receipt(*, controls_correct=False):
    """Thirty-two distinct ID/prompt hashes, but ONE public structural task.

    A cosmetic label makes each candidate-facing prompt byte-distinct while
    preserving the exact same four-secret, five-query, one-lie mechanism.
    This models an evaluator that incorrectly considers aliasing a holdout.
    """
    receipt = evaluated_receipt(controls_correct=controls_correct)
    observed_families = set()
    for slice_number, name in enumerate(("heldout", "transfer")):
        for index, trial in enumerate(receipt["slices"][name]):
            seed = slice_number * 16 + index
            case = chamber.make_case(seed, variant=seed, sealed=False)
            observed_families.add(chamber.world_fingerprint(case))
            prompt = chamber.candidate_view(case)["prompt"]
            # The marker changes source bytes, not task mechanics or labels.
            prompt = f"{prompt} Story reference: {name}-{index}."
            trial["case_id"] = f"oracle-{name}-{index:03d}"
            trial["prompt_commitment"] = sha256(prompt.encode()).hexdigest()
    return receipt, observed_families


class FourthWaveLearningControls(unittest.TestCase):
    def test_all_disjoint_seed_holdouts_repeat_exact_training_prompt_strings(self):
        # No optimizer needed. The complete public prompt has just nine
        # possible open-world wordings regardless of seed, while the hidden
        # secret/lie sample varies. This is an *instance* challenge at best.
        training_texts = {chamber.candidate_view(chamber.make_case(
            seed, variant=seed))["prompt"] for seed in range(120)}
        self.assertEqual(len(training_texts), 9)
        heldout_texts = [chamber.candidate_view(chamber.make_case(
            seed, variant=seed))["prompt"] for seed in range(120, 1120)]
        self.assertEqual(len(heldout_texts), 1000)
        self.assertEqual(sum(prompt in training_texts for prompt in heldout_texts), 1000)

    def test_oracle_chamber_nominal_cases_have_one_open_mechanism(self):
        for sealed, expected in ((False, 1), (True, 1)):
            worlds = {chamber.world_fingerprint(chamber.make_case(
                seed, variant=seed % 10000, sealed=sealed))
                for seed in range(1000)}
            self.assertEqual(len(worlds), expected)
        self.assertNotEqual(chamber.world_fingerprint(chamber.make_case(0)),
                            chamber.world_fingerprint(chamber.make_case(0, sealed=True)))

    def test_public_zero_update_policy_succeeds_without_neural_updates(self):
        outcomes = {"open": [], "sealed": []}
        for seed in range(100):
            for kind, sealed in (("open", False), ("sealed", True)):
                case = chamber.make_case(seed, variant=seed % 10, sealed=sealed)
                session = chamber.CandidateSession(case)
                review = run_public_parser(chamber.candidate_view(case),
                                           session.ask, session.submit)
                outcomes[kind].append(review)
        self.assertEqual(sum(r.get("policy_verified") is True for r in outcomes["open"]), 100)
        self.assertEqual(sum(r.get("certificate_verified") is True
                             for r in outcomes["sealed"]), 100)
        # Public symbolic policy/certification is never neural-model success.
        self.assertTrue(all(r.get("verified_success") is False
                            for arm in outcomes.values() for r in arm))

    def test_meaning_preserving_rule_paraphrase_breaks_frozen_parser(self):
        case = chamber.make_case(17)
        initial = chamber.candidate_view(case)
        ordinary = chamber.CandidateSession(case)
        baseline = run_public_parser(initial, ordinary.ask, ordinary.submit)
        self.assertTrue(baseline.get("policy_verified"))
        reworded = dict(initial)
        reworded["prompt"] = initial["prompt"].replace(
            "at most 1 answer may be inverted adversarially",
            "no more than one of the five responses can be a deliberate inversion",
        )
        self.assertNotEqual(reworded["prompt"], initial["prompt"])
        changed = chamber.CandidateSession(case)
        result = run_public_parser(reworded, changed.ask, changed.submit)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result.get("verified_success", False))
        self.assertEqual(changed.attempts(), ())

    def test_signed_semantic_aliases_can_pass_gate_despite_effective_n_one(self):
        receipt, families = _synthetic_semantic_alias_receipt()
        self.assertEqual(len(families), 1)
        all_trials = [trial for group in receipt["slices"].values() for trial in group]
        self.assertEqual(len(all_trials), 32)
        self.assertEqual(len({t["case_id"] for t in all_trials}), 32)
        self.assertEqual(len({t["prompt_commitment"] for t in all_trials}), 32)
        # The gate sees 16 independent successes PER slice and a strong paired
        # p-value; actual semantic independence is NOT represented in schema.
        result = decision(receipt)
        self.assertEqual(result.disposition, "GRADUATED")
        self.assertTrue(result.state.graduated)

    def test_truthful_equally_capable_zero_update_control_prevents_training_grant(self):
        same_mechanics, families = _synthetic_semantic_alias_receipt(controls_correct=True)
        self.assertEqual(len(families), 1)
        result = decision(same_mechanics)
        self.assertEqual(result.disposition, "STAY")
        self.assertFalse(result.state.graduated)


if __name__ == "__main__":
    unittest.main()
