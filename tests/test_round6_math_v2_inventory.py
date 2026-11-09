"""Independent v2 receipt-roster adversarial checks.

These tests intentionally exercise a differently owned source revision (the
trusted semantic-family registry introduced after protocol v1). They do not
represent genuine checkpoint or neural-model experimental evidence.
"""

from dataclasses import replace
import hashlib
import math
import random
import unittest

from labyrinth.graduation_gate import _one_sided_exact_paired_p
from tests.test_graduation_gate import POLICY, decision, evaluated_receipt


def _tamper_registry_with_hidden_case(policy, slice_name):
    """Freeze an additional case which the otherwise valid signer omits."""
    template = next(
        entry for entry in policy.independent_semantic_registry
        if entry.attempt == 1 and entry.run_id == "fresh-run-001"
        and entry.slice_name == slice_name
    )
    hidden = replace(
        template,
        case_id=f"precommitted-but-hidden-{slice_name}",
        prompt_commitment=hashlib.sha256(
            f"unfavorably-scored-unreported-case:{slice_name}".encode()
        ).hexdigest(),
        semantic_family_id=hashlib.sha256(
            f"independent-unreported-mechanism:{slice_name}".encode()
        ).hexdigest(),
    )
    return replace(
        policy,
        independent_semantic_registry=policy.independent_semantic_registry + (hidden,),
    )


def _two_aliases_per_family():
    """Eight externally declared families, two prompt variants per family."""
    receipt = evaluated_receipt()
    family_by_case = {}
    for slice_name, trials in receipt["slices"].items():
        for index, trial in enumerate(trials):
            family = hashlib.sha256(
                f"independent-family:{slice_name}:{index // 2}".encode()
            ).hexdigest()
            trial["semantic_family_id"] = family
            family_by_case[trial["case_id"]] = family
    registry = tuple(
        replace(entry, semantic_family_id=family_by_case[entry.case_id])
        if entry.attempt == 1 and entry.run_id == receipt["run_id"]
        and entry.slice_name != "train"
        else entry
        for entry in POLICY.independent_semantic_registry
    )
    return replace(POLICY, independent_semantic_registry=registry), receipt


def _collapse_all_case_aliases_by_slice():
    """Supply one frozen semantic mechanism per slice with 16 aliases."""
    policy, receipt = _two_aliases_per_family()
    family_by_case = {}
    for slice_name, trials in receipt["slices"].items():
        family = hashlib.sha256(f"one-mechanism:{slice_name}".encode()).hexdigest()
        for trial in trials:
            trial["semantic_family_id"] = family
            family_by_case[trial["case_id"]] = family
    registry = tuple(
        replace(entry, semantic_family_id=family_by_case[entry.case_id])
        if entry.attempt == 1 and entry.run_id == receipt["run_id"]
        and entry.slice_name != "train"
        else entry
        for entry in policy.independent_semantic_registry
    )
    return replace(policy, independent_semantic_registry=registry), receipt


class FrozenRosterSelectionBiasFalsifiers(unittest.TestCase):
    def test_signer_cannot_suppress_unfavorable_predeclared_heldout_case(self):
        policy = _tamper_registry_with_hidden_case(POLICY, "heldout")
        result = decision(evaluated_receipt(), policy=policy)
        self.assertEqual(result.disposition, "BLOCKED", result.reasons)

    def test_signer_cannot_suppress_unfavorable_predeclared_transfer_case(self):
        policy = _tamper_registry_with_hidden_case(POLICY, "transfer")
        result = decision(evaluated_receipt(), policy=policy)
        self.assertEqual(result.disposition, "BLOCKED", result.reasons)

    def test_signer_cannot_hide_predeclared_training_exposure(self):
        policy = _tamper_registry_with_hidden_case(POLICY, "train")
        result = decision(evaluated_receipt(), policy=policy)
        self.assertEqual(result.disposition, "BLOCKED", result.reasons)

    def test_signed_32_aliases_predeclared_as_two_mechanisms_do_not_graduate(self):
        policy, receipt = _collapse_all_case_aliases_by_slice()
        result = decision(receipt, policy=policy)
        self.assertEqual(result.disposition, "BLOCKED", result.reasons)

    def test_exact_cluster_effective_n_eight_may_graduate_with_two_attempts(self):
        policy, receipt = _two_aliases_per_family()
        # Independent oracle: 8 family-level paired wins, 0 losses.
        # p=1/256, multiplicity=3 controls*2 slices*2 attempts=12.
        self.assertLess(12 / 256, policy.significance_alpha)
        result = decision(receipt, policy=policy)
        self.assertEqual(result.disposition, "GRADUATED", result.reasons)

    def test_one_missed_alias_downgrades_entire_family_not_just_one_row(self):
        policy, receipt = _two_aliases_per_family()
        trial = receipt["slices"]["heldout"][0]
        trial["candidate_correct"] = False
        trial["candidate_semantic_correct"] = False
        # Now heldout has seven family-level paired wins. A bad row must not
        # be diluted as 15/16 mostly correct cosmetic aliases.
        self.assertGreaterEqual(12 / 128, policy.significance_alpha)
        result = decision(receipt, policy=policy)
        self.assertEqual(result.disposition, "STAY", result.reasons)

    def test_independent_binomial_enumerator_checks_paired_tail_arithmetic(self):
        rng = random.Random(20261107)
        for _ in range(300):
            wins = rng.randrange(0, 33)
            losses = rng.randrange(0, 33 - wins)
            n = wins + losses
            exact_oracle = sum(math.comb(n, k) for k in range(wins, n + 1)) / (1 << n)
            with self.subTest(wins=wins, losses=losses):
                self.assertAlmostEqual(
                    _one_sided_exact_paired_p(wins, losses), exact_oracle,
                    delta=1e-13,
                )


if __name__ == "__main__":
    unittest.main()
