"""Source-only falsifiers for the five-arm MiniLM experimental evidence seam."""

from dataclasses import asdict, replace
import unittest

from labyrinth.five_arm_experiment_contract import (
    ArmClaim, ExperimentContractError, PriorCaseObservation, TransferCase,
    audit_distinct_eval_seeds, audit_five_arm_claims,
    audit_prior_skill_retention, audit_transfer_roster,
    build_frozen_plan, transfer_roster_commitment,
)
from labyrinth.five_arm_experiment_contract import _digest as recompute_untrusted_digest
from scripts.qualify_maze_conditioned_minilm_cpu import (
    PREREG_ARMS, default_scope,
)


class FiveArmEvidenceContractTests(unittest.TestCase):
    def setUp(self):
        self.source_head = "a" * 40
        self.source_tree = "b" * 40
        self.initial = "d" * 64
        self.optimizer_initial = "0" * 64
        self.nonce = "f" * 64
        self.transfer = (
            TransferCase("new1", "contextual.negation.v1",
                         "It looked promising, but the promise did not hold.", 0),
            TransferCase("new2", "contextual.reversal.v1",
                         "The attempt failed; that does not mean every attempt will.", 1),
        )
        self.transfer_digest = transfer_roster_commitment(self.transfer, self.nonce)
        self.plan = self.make_plan()

    def make_plan(self, *, seed=20261030, run_id="five-arm-001",
                  transfer_digest=None, final_digest=None,
                  retention_ids=("prior00", "prior01")):
        return build_frozen_plan(
            replace(default_scope(), seed=seed),
            run_id=run_id,
            source_head=self.source_head, source_tree=self.source_tree,
            base_model_sha256="c" * 64, initial_trainable_sha256=self.initial,
            initial_optimizer_state_sha256=self.optimizer_initial,
            private_transfer_roster_commitment=transfer_digest or self.transfer_digest,
            private_final_bank_commitment=final_digest or "e" * 64,
            retention_case_ids=retention_ids,
            maximum_prior_correct_regressions=0,
        )

    def claims(self):
        labels = dict(self.plan.expected_labels)
        return tuple(ArmClaim(
            arm=arm, plan_sha256=self.plan.source_contract_sha256,
            initial_trainable_sha256=self.initial,
            initial_optimizer_state_sha256=self.optimizer_initial,
            executed_visits=self.plan.train_visits,
            executed_labels=labels[arm],
            effective_learning_rate=(0.0 if arm == "zero_lr" else .001),
            forward_calls=16, backward_calls=16,
            optimizer_step_calls=0 if arm == "zero_update" else 16,
            elapsed_wall_milliseconds=2000,
            cycle_zero_trainable_sha256=self.initial
            if arm in ("zero_update", "zero_lr") else ("7" * 64),
            cycle_one_start_trainable_sha256=self.initial
            if arm in ("zero_update", "zero_lr") else ("7" * 64),
            cycle_zero_optimizer_state_sha256=self.optimizer_initial
            if arm == "zero_update" else "9" * 64,
            cycle_one_start_optimizer_state_sha256=self.optimizer_initial
            if arm == "zero_update" else "9" * 64,
            final_trainable_sha256=self.initial
            if arm in ("zero_update", "zero_lr") else ("1" * 64),
            final_optimizer_state_sha256=self.optimizer_initial
            if arm == "zero_update" else "2" * 64,
        ) for arm in PREREG_ARMS)

    def replace_arm(self, claims, arm, **kwargs):
        return tuple(replace(c, **kwargs) if c.arm == arm else c for c in claims)

    def self_resign_plan(self, plan):
        """Attacker controls the manifest and can recompute its public digest."""
        raw = asdict(plan)
        raw.pop("source_contract_sha256")
        return replace(plan, source_contract_sha256=recompute_untrusted_digest(raw))

    def test_frozen_manifest_is_deterministic_and_references_real_selector(self):
        other = self.make_plan()
        self.assertEqual(self.plan, other)
        self.assertEqual(len(self.plan.train_visits), 64)
        self.assertEqual(tuple(x[0] for x in self.plan.expected_labels), PREREG_ARMS)
        labels = dict(self.plan.expected_labels)
        self.assertNotEqual(labels["randomized_labels"], labels["correct"])
        self.assertNotEqual(labels["randomized_labels"], labels["inverted"])
        self.assertEqual(labels["zero_lr"], labels["correct"])
        self.assertEqual(labels["zero_update"], labels["correct"])
        self.assertEqual(len(self.plan.train_prompt_hashes), 24)
        self.assertFalse(any("It looked promising" in x for x in
                             self.plan.train_prompt_hashes))

    def test_valid_self_report_is_still_never_physical_qualification(self):
        report = audit_five_arm_claims(self.plan, self.claims())
        self.assertEqual(report["status"], "SOURCE_ONLY_UNATTESTED")
        self.assertTrue(report["exposures_and_budgets_source_consistent"])
        self.assertFalse(report["physical_optimizer_execution_verified"])
        self.assertFalse(report["independent_evaluation_verified"])
        self.assertFalse(report["can_authorize_graduation"])

    def test_wrong_zero_learning_rate_rejected(self):
        bad = self.replace_arm(self.claims(), "zero_lr", effective_learning_rate=.001)
        with self.assertRaisesRegex(ExperimentContractError, "learning rate"):
            audit_five_arm_claims(self.plan, bad)

    def test_zero_lr_must_still_call_optimizer_step(self):
        bad = self.replace_arm(self.claims(), "zero_lr", optimizer_step_calls=0)
        with self.assertRaisesRegex(ExperimentContractError, "optimizer"):
            audit_five_arm_claims(self.plan, bad)

    def test_zero_update_must_not_call_optimizer_step(self):
        bad = self.replace_arm(self.claims(), "zero_update", optimizer_step_calls=16)
        with self.assertRaisesRegex(ExperimentContractError, "optimizer"):
            audit_five_arm_claims(self.plan, bad)

    def test_zero_update_and_zero_lr_cannot_report_changed_weights(self):
        for arm in ("zero_update", "zero_lr"):
            with self.subTest(arm=arm):
                bad = self.replace_arm(self.claims(), arm,
                                       final_trainable_sha256="3" * 64)
                with self.assertRaisesRegex(ExperimentContractError, "updated parameters"):
                    audit_five_arm_claims(self.plan, bad)

    def test_real_training_arm_requires_changed_digest(self):
        bad = self.replace_arm(self.claims(), "correct",
                               final_trainable_sha256=self.initial)
        with self.assertRaisesRegex(ExperimentContractError, "no parameter changes"):
            audit_five_arm_claims(self.plan, bad)

    def test_maze_reset_must_not_reset_weights_or_optimizer_moments(self):
        for change in (
            {"cycle_one_start_trainable_sha256": self.initial},
            {"cycle_one_start_optimizer_state_sha256": "8" * 64},
        ):
            with self.subTest(change=change):
                bad = self.replace_arm(self.claims(), "correct", **change)
                with self.assertRaisesRegex(ExperimentContractError, "reset with Maze"):
                    audit_five_arm_claims(self.plan, bad)

    def test_first_cycle_must_change_trainable_parameters_for_learning_arm(self):
        bad = self.replace_arm(self.claims(), "correct",
                               cycle_zero_trainable_sha256=self.initial,
                               cycle_one_start_trainable_sha256=self.initial)
        with self.assertRaisesRegex(ExperimentContractError, "first training cycle"):
            audit_five_arm_claims(self.plan, bad)

    def test_all_arms_must_share_initial_optimizer_state(self):
        bad = self.replace_arm(self.claims(), "correct",
                               initial_optimizer_state_sha256="3" * 64)
        with self.assertRaisesRegex(ExperimentContractError, "cases/labels"):
            audit_five_arm_claims(self.plan, bad)

    def test_zero_update_optimizer_moments_must_remain_initial(self):
        bad = self.replace_arm(self.claims(), "zero_update",
                               cycle_zero_optimizer_state_sha256="9" * 64,
                               cycle_one_start_optimizer_state_sha256="9" * 64,
                               final_optimizer_state_sha256="9" * 64)
        with self.assertRaisesRegex(ExperimentContractError, "zero-update control"):
            audit_five_arm_claims(self.plan, bad)

    def test_zero_lr_optimizer_moments_must_evolve_despite_static_weights(self):
        bad = self.replace_arm(self.claims(), "zero_lr",
                               cycle_zero_optimizer_state_sha256=self.optimizer_initial,
                               cycle_one_start_optimizer_state_sha256=self.optimizer_initial)
        with self.assertRaisesRegex(ExperimentContractError, "state evolution"):
            audit_five_arm_claims(self.plan, bad)

    def test_randomized_labels_must_match_exact_pinned_permutation(self):
        bad = self.replace_arm(self.claims(), "randomized_labels",
                               executed_labels=dict(self.plan.expected_labels)["correct"])
        with self.assertRaisesRegex(ExperimentContractError, "cases/labels"):
            audit_five_arm_claims(self.plan, bad)

    def test_randomized_label_inversion_cheat_rejected(self):
        bad = self.replace_arm(self.claims(), "randomized_labels",
                               executed_labels=dict(self.plan.expected_labels)["inverted"])
        with self.assertRaisesRegex(ExperimentContractError, "cases/labels"):
            audit_five_arm_claims(self.plan, bad)

    def test_case_visit_difference_between_arms_rejected(self):
        altered = tuple(reversed(self.plan.train_visits))
        bad = self.replace_arm(self.claims(), "inverted", executed_visits=altered)
        with self.assertRaisesRegex(ExperimentContractError, "cases/labels"):
            audit_five_arm_claims(self.plan, bad)

    def test_initial_weights_or_contract_commitment_drift_rejected(self):
        for change in ({"initial_trainable_sha256": "4" * 64},
                       {"plan_sha256": "5" * 64}):
            with self.subTest(change=change):
                bad = self.replace_arm(self.claims(), "correct", **change)
                with self.assertRaises(ExperimentContractError):
                    audit_five_arm_claims(self.plan, bad)

    def test_step_and_wall_budget_mismatch_rejected(self):
        for change in ({"backward_calls": 15},
                       {"forward_calls": 15},
                       {"elapsed_wall_milliseconds": self.plan.maximum_wall_milliseconds_per_arm + 1}):
            with self.subTest(change=change):
                bad = self.replace_arm(self.claims(), "correct", **change)
                with self.assertRaisesRegex(ExperimentContractError, "budget"):
                    audit_five_arm_claims(self.plan, bad)

    def test_precommit_change_detected(self):
        tampered = replace(self.plan, source_tree="f" * 40)
        with self.assertRaisesRegex(ExperimentContractError, "changed after"):
            tampered.validate()

    def test_self_resigned_all_zero_arm_labels_cannot_spoof_canonical_runner(self):
        forged_labels = tuple(
            (arm, (0,) * len(self.plan.train_visits)) for arm in PREREG_ARMS
        )
        forged = self.self_resign_plan(
            replace(self.plan, expected_labels=forged_labels)
        )
        bad_claims = tuple(
            replace(c, plan_sha256=forged.source_contract_sha256,
                    executed_labels=(0,) * len(self.plan.train_visits))
            for c in self.claims()
        )
        with self.assertRaisesRegex(ExperimentContractError, "canonical runner"):
            audit_five_arm_claims(forged, bad_claims)

    def test_self_resigned_train_visit_reorder_cannot_spoof_source(self):
        forged_visits = tuple(reversed(self.plan.train_visits))
        forged = self.self_resign_plan(
            replace(self.plan, train_visits=forged_visits)
        )
        claims = tuple(
            replace(c, plan_sha256=forged.source_contract_sha256,
                    executed_visits=forged_visits)
            for c in self.claims()
        )
        with self.assertRaisesRegex(ExperimentContractError, "canonical runner"):
            audit_five_arm_claims(forged, claims)

    def test_self_resigned_forged_weakness_changes_generated_maze(self):
        forged = self.self_resign_plan(
            replace(self.plan, weak_cycle_one=("negative_polarity",))
        )
        with self.assertRaisesRegex(ExperimentContractError, "canonical runner"):
            forged.validate()

    def test_unhashable_wrong_type_members_rejected_as_contract_errors(self):
        hostile = (
            {"weak_cycle_one": ([],)},
            {"train_prompt_hashes": ([],)},
            {"retention_case_ids": ([],)},
        )
        for kwargs in hostile:
            with self.subTest(kwargs=kwargs):
                forged = replace(self.plan, **kwargs)
                with self.assertRaises(ExperimentContractError):
                    forged.validate()

    def test_builder_rejects_custom_float_side_effect_objects(self):
        class EvilFloat:
            def __float__(self):
                raise AssertionError("ATTACKER_FLOAT_EXECUTED")
        with self.assertRaises(ExperimentContractError):
            build_frozen_plan(
                default_scope(), run_id="five-arm-001",
                source_head=self.source_head, source_tree=self.source_tree,
                base_model_sha256="c" * 64,
                initial_trainable_sha256=self.initial,
                initial_optimizer_state_sha256=self.optimizer_initial,
                private_transfer_roster_commitment=self.transfer_digest,
                private_final_bank_commitment="e" * 64,
                retention_case_ids=("prior00", "prior01"),
                maximum_prior_correct_regressions=0,
                nominal_learning_rate=EvilFloat(),
            )

    def test_builder_rejects_huge_int_rate_before_float_overflow(self):
        with self.assertRaises(ExperimentContractError):
            build_frozen_plan(
                default_scope(), run_id="five-arm-001",
                source_head=self.source_head, source_tree=self.source_tree,
                base_model_sha256="c" * 64,
                initial_trainable_sha256=self.initial,
                initial_optimizer_state_sha256=self.optimizer_initial,
                private_transfer_roster_commitment=self.transfer_digest,
                private_final_bank_commitment="e" * 64,
                retention_case_ids=("prior00", "prior01"),
                maximum_prior_correct_regressions=0,
                nominal_learning_rate=10**1000,
            )

    def test_hostile_equality_never_executes_before_nested_primitive_checks(self):
        class ExplodingStr(str):
            def __eq__(self, other):
                raise AssertionError("ATTACKER_EQ_EXECUTED")
            def __ne__(self, other):
                raise AssertionError("ATTACKER_NE_EXECUTED")
            __hash__ = str.__hash__

        class ExplodingInt(int):
            def __eq__(self, other):
                raise AssertionError("ATTACKER_INT_EQ_EXECUTED")
            def __ne__(self, other):
                raise AssertionError("ATTACKER_INT_NE_EXECUTED")
            __hash__ = int.__hash__

        all_labels = self.plan.expected_labels
        hostile_plans = (
            replace(self.plan, generated_family_ids=(
                ExplodingStr("sentiment.generated.template.0"),) +
                self.plan.generated_family_ids[1:]),
            replace(self.plan, expected_labels=(
                (ExplodingStr("correct"), all_labels[0][1]),) +
                all_labels[1:]),
        )
        for plan in hostile_plans:
            with self.subTest(plan_field=plan):
                with self.assertRaises(ExperimentContractError):
                    plan.validate()

        valid = self.claims()
        hostile_claims = (
            tuple(replace(c, arm=ExplodingStr("correct"))
                  if c.arm == "correct" else c for c in valid),
            self.replace_arm(valid, "correct", executed_visits=(
                ExplodingStr(self.plan.train_visits[0]),) +
                self.plan.train_visits[1:]),
            self.replace_arm(valid, "correct", executed_labels=(
                ExplodingInt(0),) + valid[0].executed_labels[1:]),
        )
        for claims in hostile_claims:
            with self.subTest(claims=claims):
                with self.assertRaises(ExperimentContractError):
                    audit_five_arm_claims(self.plan, claims)

    def test_forged_equality_subclass_source_head_rejected(self):
        class GhostStr(str):
            def __eq__(self, other):
                return True
            def __ne__(self, other):
                return False
            __hash__ = str.__hash__
        forged = replace(self.plan, source_head=GhostStr("f" * 40))
        with self.assertRaisesRegex(ExperimentContractError, "exact Git"):
            forged.validate()

    def test_missing_or_reordered_control_arm_rejected(self):
        self.assertRaises(ExperimentContractError, audit_five_arm_claims,
                          self.plan, self.claims()[:-1])
        bad = tuple(reversed(self.claims()))
        with self.assertRaisesRegex(ExperimentContractError, "omitted or reordered"):
            audit_five_arm_claims(self.plan, bad)

    def test_transfer_new_lexical_and_family_roster_is_only_source_veto(self):
        result = audit_transfer_roster(self.plan, self.transfer, self.nonce)
        self.assertEqual(result["status"], "SOURCE_OVERLAP_VETO_ONLY")
        self.assertFalse(result["independent_semantic_transfer_verified"])
        self.assertFalse(result["private_bank_custody_verified"])

    def test_transfer_exact_train_prompt_rejected(self):
        from scripts.qualify_maze_conditioned_minilm_cpu import build_training_cycles
        first = build_training_cycles(default_scope())[0].examples[0].prompt
        cases = (TransferCase("old1", "new.family", first, 0),)
        p = self.make_plan(transfer_digest=transfer_roster_commitment(cases, self.nonce))
        with self.assertRaisesRegex(ExperimentContractError, "overlaps TRAIN"):
            audit_transfer_roster(p, cases, self.nonce)

    def test_transfer_generator_dictionary_cue_rejected_even_if_not_sampled(self):
        cases = (TransferCase("clearcue", "new.family",
                              "It was clear that patience had run out.", 0),)
        p = self.make_plan(transfer_digest=transfer_roster_commitment(cases, self.nonce))
        with self.assertRaisesRegex(ExperimentContractError, "overlaps TRAIN"):
            audit_transfer_roster(p, cases, self.nonce)

    def test_transfer_source_family_alias_rejected(self):
        cases = (TransferCase("shape", "sentiment.generated.template.0",
                              "It failed despite good appearances.", 0),)
        p = self.make_plan(transfer_digest=transfer_roster_commitment(cases, self.nonce))
        with self.assertRaisesRegex(ExperimentContractError, "overlaps TRAIN"):
            audit_transfer_roster(p, cases, self.nonce)

    def test_transfer_uncommitted_roster_salt_and_repeated_case_rejected(self):
        self.assertRaises(ExperimentContractError, audit_transfer_roster,
                          self.plan, self.transfer, "1" * 64)
        duplicate = (self.transfer[0], self.transfer[0])
        p = self.make_plan(transfer_digest=transfer_roster_commitment(duplicate, self.nonce))
        with self.assertRaisesRegex(ExperimentContractError, "reused private transfer case"):
            audit_transfer_roster(p, duplicate, self.nonce)

    def test_prior_correct_regression_not_masked_by_new_gains(self):
        ids = tuple(f"prior{i:02d}" for i in range(20))
        p = self.make_plan(retention_ids=ids)
        prior = tuple(
            PriorCaseObservation(id_, i < 16, i < 12 or i >= 16)
            for i, id_ in enumerate(ids)
        )
        # Identical aggregate 80% accuracy, but four mastered cases lost.
        self.assertEqual(sum(x.before_correct for x in prior),
                         sum(x.after_correct for x in prior))
        with self.assertRaisesRegex(ExperimentContractError, "regressed"):
            audit_prior_skill_retention(p, prior)

    def test_prior_skill_no_established_baseline_rejected(self):
        observations = tuple(PriorCaseObservation(i, False, True)
                             for i in self.plan.retention_case_ids)
        with self.assertRaisesRegex(ExperimentContractError, "no established"):
            audit_prior_skill_retention(self.plan, observations)

    def test_prior_skill_low_baseline_cannot_mask_retention(self):
        observations = (
            PriorCaseObservation("prior00", True, True),
            PriorCaseObservation("prior01", False, True),
        )
        with self.assertRaisesRegex(ExperimentContractError, "below 75%"):
            audit_prior_skill_retention(self.plan, observations)

    def test_prior_skill_source_accounting_does_not_qualify_learner(self):
        observations = tuple(PriorCaseObservation(i, True, True)
                             for i in self.plan.retention_case_ids)
        result = audit_prior_skill_retention(self.plan, observations)
        self.assertEqual(result["previously_correct_regressed"], 0)
        self.assertFalse(result["independent_observation_verified"])
        self.assertFalse(result["can_authorize_graduation"])

    def test_seed_reuse_of_same_private_evaluator_roster_rejected(self):
        other = self.make_plan(seed=20261031, run_id="five-arm-002",
                               final_digest="6" * 64)
        with self.assertRaisesRegex(ExperimentContractError, "reuse private_transfer"):
            audit_distinct_eval_seeds((self.plan, other))

    def test_private_final_must_not_reuse_transfer_bank_commitment(self):
        with self.assertRaisesRegex(ExperimentContractError, "must differ"):
            self.make_plan(final_digest=self.transfer_digest)

    def test_distinct_source_seed_rosters_still_cannot_claim_exam_custody(self):
        other = self.make_plan(seed=20261031, run_id="five-arm-002",
                               transfer_digest="8" * 64, final_digest="6" * 64)
        self.assertIsNone(audit_distinct_eval_seeds((self.plan, other)))
        self.assertNotEqual(other.source_contract_sha256,
                            self.plan.source_contract_sha256)


if __name__ == "__main__":
    unittest.main()
