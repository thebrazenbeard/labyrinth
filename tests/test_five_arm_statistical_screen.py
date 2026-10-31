"""Source-only statistical falsifiers; no real model or private bank is run."""

from dataclasses import replace
from hashlib import sha256
import unittest

from labyrinth.five_arm_experiment_contract import (
    PriorCaseObservation, build_frozen_plan,
)
from labyrinth.five_arm_statistical_screen import (
    MINIMUM_SEEDS, PairedCase, SeedOutcome, StatisticalScreenError,
    _case_digest, screen_five_arm_seed_outcomes,
)
from scripts.qualify_maze_conditioned_minilm_cpu import PREREG_ARMS, default_scope


def digest(text: str) -> str:
    return sha256(text.encode()).hexdigest()


def sample_outcomes(*, seed_count=MINIMUM_SEEDS, correct_superiority=True):
    answer = []
    # Select exact generator-valid fixture seeds; 20261036 collides with a
    # public development prompt and is correctly rejected upstream.
    seeds = (20261030, 20261031, 20261032, 20261033,
             20261034, 20261035, 20261037, 20261038)
    if seed_count > len(seeds):
        raise ValueError("fixture declares only eight valid independent seeds")
    for i in range(seed_count):
        scope = replace(default_scope(), seed=seeds[i])
        ids = tuple(f"prior{n:02d}" for n in range(20))
        plan = build_frozen_plan(
            scope, run_id=f"screen-{i:03d}", source_head="a" * 40,
            source_tree="b" * 40, base_model_sha256="c" * 64,
            initial_trainable_sha256="d" * 64,
            initial_optimizer_state_sha256="0" * 64,
            private_transfer_roster_commitment=digest(f"transfer-salt-{i}"),
            private_final_bank_commitment=digest(f"final-salt-{i}"),
            retention_case_ids=ids, maximum_prior_correct_regressions=0,
        )

        def cases(split):
            return tuple(PairedCase(
                digest(f"case:{i}:{split}:{n}"),
                tuple((arm, arm == "correct" if correct_superiority else True)
                      for arm in PREREG_ARMS),
            ) for n in range(20))

        development, transfer = cases("development"), cases("transfer")
        answer.append(SeedOutcome(
            plan, _case_digest(development), _case_digest(transfer),
            development, transfer,
            tuple(PriorCaseObservation(name, True, True) for name in ids),
        ))
    return tuple(answer)


class FiveArmSourceStatisticalScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid = sample_outcomes()

    def test_perfect_eight_seed_screen_passes_but_authorizes_nothing(self):
        screen = screen_five_arm_seed_outcomes(self.valid)
        self.assertTrue(screen["source_screen_pass"])
        self.assertEqual(len(screen["comparisons"]), 8)
        self.assertTrue(all(row["seed_wins"] == 8 for row in screen["comparisons"]))
        self.assertTrue(all(row["unadjusted_p"] == 1 / 256 for row in screen["comparisons"]))
        self.assertFalse(screen["actual_model_updates_verified"])
        self.assertFalse(screen["independent_case_custody_verified"])
        self.assertFalse(screen["semantic_transfer_verified"])
        self.assertFalse(screen["physical_retention_verified"])
        self.assertFalse(screen["can_authorize_training"])
        self.assertFalse(screen["can_authorize_graduation"])

    def test_three_reused_pilot_seeds_cannot_be_called_independent_screen(self):
        with self.assertRaisesRegex(StatisticalScreenError, "eight"):
            screen_five_arm_seed_outcomes(self.valid[:3])

    def test_reused_case_pseudonyms_across_seed_or_split_rejected(self):
        swapped = replace(self.valid[1], development=self.valid[0].development,
                          development_case_digest=self.valid[0].development_case_digest)
        with self.assertRaisesRegex(StatisticalScreenError, "duplicate"):
            screen_five_arm_seed_outcomes((self.valid[0], swapped) + self.valid[2:])
        swapped_split = replace(self.valid[0], transfer=self.valid[0].development,
                                transfer_case_digest=self.valid[0].development_case_digest)
        with self.assertRaisesRegex(StatisticalScreenError, "duplicate"):
            screen_five_arm_seed_outcomes((swapped_split,) + self.valid[1:])

    def test_pseudonym_commitment_change_rejected(self):
        case = replace(self.valid[0].development[0], case_sha256=digest("new-case"))
        rows = (case,) + self.valid[0].development[1:]
        tampered = replace(self.valid[0], development=rows)
        with self.assertRaisesRegex(StatisticalScreenError, "changed"):
            screen_five_arm_seed_outcomes((tampered,) + self.valid[1:])

    def test_missing_arm_order_or_hostile_types_rejected(self):
        good = self.valid[0].transfer[0]
        ghost = type("GhostStr", (str,), {"__eq__": lambda self, other: True,
                                        "__hash__": str.__hash__})
        mutations = (
            replace(good, arm_correctness=good.arm_correctness[:-1]),
            replace(good, arm_correctness=(("inverted", True),) + good.arm_correctness[1:]),
            replace(good, arm_correctness=(("correct", 1),) + good.arm_correctness[1:]),
            replace(good, case_sha256=ghost(good.case_sha256)),
        )
        for case in mutations:
            with self.subTest(case=case), self.assertRaises(StatisticalScreenError):
                rows = (case,) + self.valid[0].transfer[1:]
                tampered = replace(self.valid[0], transfer=rows,
                                   transfer_case_digest=_case_digest(rows))
                screen_five_arm_seed_outcomes((tampered,) + self.valid[1:])

    def test_invalid_initial_optimizer_or_case_roster_reuse_rejected(self):
        changed_plan = replace(self.valid[1].plan,
                               initial_optimizer_state_sha256="1" * 64)
        # The modified plan fails its own frozen digest first.
        with self.assertRaises(StatisticalScreenError):
            screen_five_arm_seed_outcomes(
                (self.valid[0], replace(self.valid[1], plan=changed_plan)) + self.valid[2:])
        with self.assertRaisesRegex(StatisticalScreenError, "retention"):
            invalid = replace(self.valid[0], prior_retention=tuple(
                replace(x, after_correct=False) if n == 0 else x
                for n, x in enumerate(self.valid[0].prior_retention)))
            screen_five_arm_seed_outcomes((invalid,) + self.valid[1:])

    def test_no_control_benefit_can_never_pass_statistical_screen(self):
        outcome = screen_five_arm_seed_outcomes(sample_outcomes(correct_superiority=False))
        self.assertFalse(outcome["source_screen_pass"])
        self.assertTrue(all(row["seed_ties"] == MINIMUM_SEEDS for row in outcome["comparisons"]))
        self.assertTrue(all(row["unadjusted_p"] == 1.0 for row in outcome["comparisons"]))

    def test_one_seed_failure_prevents_eight_seed_holm_pass(self):
        # Seven wins and one loss yield p=9/256, which misses Holm's
        # transfer thresholds despite deceptively strong aggregate means.
        bad = self.valid[0]
        rows = tuple(replace(case, arm_correctness=tuple(
            (arm, arm != "correct") for arm in PREREG_ARMS)) for case in bad.transfer)
        changed = replace(bad, transfer=rows)
        outcome = screen_five_arm_seed_outcomes((changed,) + self.valid[1:])
        self.assertFalse(outcome["source_screen_pass"])
        self.assertTrue(any(row["split"] == "transfer" and not row["holm_pass"]
                            for row in outcome["comparisons"]))


if __name__ == "__main__":
    unittest.main()
