"""Synthetic source falsifiers; NO real private cases or hosted evaluator."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import unittest

from labyrinth.retention_family_registry import canonical_prompt_alias_sha256
from labyrinth.sealed_semantic_exam_protocol import (
    ARMS, LocalOneShotLedger, PredictionSubmission, SealedExamError,
    SealedSituation, TrainingOverlapInventory, build_sealed_plan,
    score_sealed_exam, sign_prediction_submission, training_inventory_sha256,
)


EVALUATOR_KEY = b"evaluator-proposal-20261030-independent-key"
BANK_KEY = b"bank-proposal-20261030-separate-key-material"
TRAINER_KEY = b"trainer-proposal-20261030-separate-key"


def synthetic_bank() -> tuple[SealedSituation, ...]:
    # Synthetic, fully disclosed test data. Never elevate to private custody.
    return tuple(SealedSituation(
        case_id=f"case-{family:02d}-{case:02d}",
        family_id=(f"transfer-family-{family:02d}" if family < 10
                   else f"retention-family-{family:02d}"),
        split="transfer" if family < 10 else "retention",
        after_prompt=f"synthetic family {family} situation {case} after",
        before_prompt=(None if family < 10 else
                       f"synthetic family {family} situation {case} before"),
        gold_label=case % 2,
        semantic_key=f"situation-{family:02d}-{case:02d}",
    ) for family in range(20) for case in range(20))


class SealedSemanticProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = synthetic_bank()
        cls.inventory = TrainingOverlapInventory(
            family_ids=("training-family",),
            prompt_alias_sha256s=(canonical_prompt_alias_sha256(
                "Synthetic training prompt"),),
            semantic_keys=("training-situation",),
        )
        cls.settings = dict(
            cases=cls.cases, training_inventory=cls.inventory,
            externally_pinned_training_inventory_sha256=training_inventory_sha256(
                cls.inventory),
            exam_id="candidate-semantic-final-1", event_id="event-semantic-1",
            objective_id="sentiment.binary_polarity", scope_sha256="a" * 64,
            pretrained_model_sha256="b" * 64,
            tokenizer_sha256="c" * 64, optimizer_config_sha256="d" * 64,
            source_sha256="e" * 64, prior_skill_checkpoint_sha256="f" * 64,
            seeds=(201, 202, 203, 204, 205), step_budgets=(8,),
            evaluator_key=EVALUATOR_KEY, bank_key=BANK_KEY,
            trainer_key=TRAINER_KEY,
        )
        cls.plan = build_sealed_plan(**cls.settings)

    def submissions(self, *, seeds=(201, 202, 203, 204, 205),
                    steps=(8,)):
        predictions = tuple((case.case_id, case.gold_label) for case in self.cases)
        before = tuple((case.case_id, case.gold_label)
                       for case in self.cases if case.split == "retention")
        return tuple(sign_prediction_submission(PredictionSubmission(
            arm=arm, seed=seed, steps_per_cycle=budget,
            run_id=f"run-{seed}-{budget}-{arm}",
            initial_trainable_sha256=sha256(
                f"init-{seed}-{budget}".encode()).hexdigest(),
            training_visits_sha256=sha256(
                f"visits-{seed}-{budget}".encode()).hexdigest(),
            candidate_artifact_sha256=sha256(
                f"artifact-{seed}-{budget}-{arm}".encode()).hexdigest(),
            predictions=predictions, retention_before_predictions=before,
            prior_skill_checkpoint_sha256="f" * 64,
        ), trainer_key=TRAINER_KEY) for seed in seeds for budget in steps for arm in ARMS)

    def score(self, *, cases=None, submissions=None, ledger=None, plan=None,
              inventory=None, evaluator_key=EVALUATOR_KEY,
              trainer_key=TRAINER_KEY):
        return score_sealed_exam(
            plan=self.plan if plan is None else plan,
            cases=self.cases if cases is None else cases,
            submissions=self.submissions() if submissions is None else submissions,
            training_inventory=self.inventory if inventory is None else inventory,
            evaluator_key=evaluator_key, bank_key=BANK_KEY,
            trainer_key=trainer_key,
            local_ledger=LocalOneShotLedger() if ledger is None else ledger,
        )

    def test_400_situation_twenty_family_public_plan_has_no_private_data(self):
        plan = self.plan
        body = json.loads(plan.public_bytes)
        self.assertEqual(body["family_count"], 20)
        self.assertEqual(body["case_count"], 400)
        self.assertEqual((body["transfer_situations"],
                          body["retention_situations"]), (200, 200))
        self.assertEqual(body["arms"], list(ARMS))
        self.assertEqual(body["seeds"], [201, 202, 203, 204, 205])
        self.assertEqual(body["decision_policy_status"], "PROPOSED_NOT_APPROVED")
        self.assertFalse(body["decision_policy_qualified"])
        for sensitive in ("synthetic family", "case-00-00", "situation-00-00",
                          "gold_label", "before_prompt"):
            self.assertNotIn(sensitive, plan.public_bytes.decode())
        self.assertFalse(plan.graduation_authorized)
        self.assertFalse(plan.independent_evaluator_host_verified)
        self.assertFalse(body["external_timestamp_verified"])

    def test_complete_matched_five_arm_source_score_is_aggregate_only(self):
        scored = self.score()
        body = json.loads(scored.receipt_bytes)
        self.assertEqual(len(body["public_aggregate"]), 25)
        for row in body["public_aggregate"]:
            self.assertEqual(row["correct_transfer"], 200)
            self.assertEqual(row["correct_retention_before"], 200)
            self.assertEqual(row["correct_retention_after"], 200)
            self.assertEqual(row["retention_total"], 200)
        for private in ("synthetic family", "case-01-01", "situation-01-01",
                        "gold_label", "before_prompt", "training prompt"):
            self.assertNotIn(private, scored.receipt_bytes.decode())
        self.assertFalse(scored.graduation_authorized)
        self.assertFalse(scored.physical_execution_authenticated)
        self.assertFalse(body["private_exam_authorizes_graduation"])
        self.assertEqual(len(body["public_aggregate"][0][
            "blinded_retention_families"]), 10)
        self.assertNotIn("retention-family-10", scored.receipt_bytes.decode())

    def test_replays_refused_locally_but_new_untrusted_ledger_can_replay(self):
        ledger = LocalOneShotLedger()
        self.score(ledger=ledger)
        with self.assertRaisesRegex(SealedExamError, "already consumed"):
            self.score(ledger=ledger)
        self.assertIsNotNone(self.score(ledger=LocalOneShotLedger()))

    def test_private_mutation_and_plan_signature_forgery_rejected(self):
        changed = (replace(self.cases[0],
                           after_prompt="synthetic family zero changed after"),) + self.cases[1:]
        with self.assertRaisesRegex(SealedExamError, "private bank changed"):
            self.score(cases=changed)
        forged = replace(self.plan, signature_hex="0" * 64)
        with self.assertRaisesRegex(SealedExamError, "signature"):
            self.score(plan=forged)
        with self.assertRaisesRegex(SealedExamError, "signature"):
            self.score(evaluator_key=b"different-evaluator-key-material-000000000000")

    def test_family_alias_and_train_semantic_contamination_fail(self):
        for edited in (
            replace(self.cases[0], family_id="TRAINING Family"),
            replace(self.cases[0], semantic_key="training-situation"),
            replace(self.cases[0], after_prompt="Synthetic training prompt"),
        ):
            with self.subTest(edited=edited), self.assertRaises(SealedExamError):
                build_sealed_plan(**{**self.settings,
                                     "cases": (edited,) + self.cases[1:]})

    def test_known_arithmetic_alias_cannot_fake_distinct_private_families(self):
        a = replace(self.cases[0], after_prompt="What is seven plus five?")
        b = replace(self.cases[20], after_prompt="Compute 7 + 5.")
        changed = list(self.cases)
        changed[0], changed[20] = a, b
        with self.assertRaisesRegex(SealedExamError, "arithmetic semantic alias"):
            build_sealed_plan(**{**self.settings, "cases": tuple(changed)})

    def test_frozen_train_arithmetic_roster_rejects_paraphrased_exam_case(self):
        overlapping = replace(self.inventory,
                              arithmetic_semantic_keys=("add:5:7",))
        changed = list(self.cases)
        changed[0] = replace(changed[0], after_prompt="Compute seven plus five.")
        with self.assertRaisesRegex(SealedExamError,
                                    "arithmetic semantics overlap frozen TRAIN"):
            build_sealed_plan(**{
                **self.settings, "cases": tuple(changed),
                "training_inventory": overlapping,
                "externally_pinned_training_inventory_sha256":
                    training_inventory_sha256(overlapping),
            })

    def test_parallel_retention_forms_may_share_one_arithmetic_situation(self):
        retention_index = 10 * 20
        changed_case = replace(
            self.cases[retention_index],
            before_prompt="What is seven plus five?",
            after_prompt="Compute 7 + 5.",
        )
        changed = list(self.cases)
        changed[retention_index] = changed_case
        proposed = build_sealed_plan(**{**self.settings,
                                        "cases": tuple(changed)})
        self.assertFalse(proposed.graduation_authorized)

    def test_parallel_retention_forms_must_not_change_arithmetic_semantics(self):
        retention_index = 10 * 20
        for before_prompt in ("Compute 6 + 8.",
                              "An unparsed linguistic paraphrase of seven plus five"):
            changed = list(self.cases)
            changed[retention_index] = replace(
                changed[retention_index],
                after_prompt="What is seven plus five?",
                before_prompt=before_prompt,
            )
            with self.subTest(before=before_prompt), self.assertRaisesRegex(
                    SealedExamError, "before/after arithmetic semantics"):
                build_sealed_plan(**{**self.settings, "cases": tuple(changed)})

    def test_cross_split_or_unbalanced_private_family_rejected(self):
        for changed in (
            replace(self.cases[0], split="retention",
                    before_prompt="separate synthetic prior prompt"),
            replace(self.cases[0], gold_label=1),
            replace(self.cases[0], family_id="retention-family-10"),
        ):
            with self.subTest(changed=changed), self.assertRaises(SealedExamError):
                build_sealed_plan(**{**self.settings,
                                     "cases": (changed,) + self.cases[1:]})

    def test_subclassed_training_split_cannot_impersonate_private_transfer(self):
        class ForgedSplit(str):
            def __eq__(self, other):
                return other == "transfer"

            def __ne__(self, other):
                return not self.__eq__(other)

            __hash__ = str.__hash__

        contaminated = replace(self.cases[0], split=ForgedSplit("training"))
        self.assertEqual(str(contaminated.split), "training")
        with self.assertRaisesRegex(SealedExamError, "split"):
            build_sealed_plan(**{
                **self.settings,
                "cases": (contaminated,) + self.cases[1:],
            })

    def test_missing_arm_duplicate_job_and_posthoc_budget_rejected(self):
        complete = self.submissions()
        for rows in (
            complete[:-1],
            complete[:-1] + (complete[0],),
            (replace(complete[0], steps_per_cycle=16),) + complete[1:],
        ):
            with self.subTest(rows=len(rows)), self.assertRaises(SealedExamError):
                self.score(submissions=rows)
        with self.assertRaisesRegex(SealedExamError, "seeds"):
            build_sealed_plan(**{**self.settings, "seeds": (201, 202, 203)})

    def test_controls_must_share_weights_visits_and_distinct_run_ids(self):
        rows = self.submissions()
        for broken, rejection in (
            (replace(rows[1], initial_trainable_sha256="e" * 64), "controls used"),
            (replace(rows[1], training_visits_sha256="e" * 64), "controls used"),
            (replace(rows[1], run_id=rows[0].run_id), "repeated neural"),
        ):
            authentic = sign_prediction_submission(
                replace(broken, signature_hex=""), trainer_key=TRAINER_KEY)
            with self.subTest(field=broken), self.assertRaisesRegex(
                    SealedExamError, rejection):
                self.score(submissions=(rows[0], authentic) + rows[2:])

    def test_matched_arms_reject_different_prior_skill_before_answers(self):
        rows = self.submissions()
        prior = list(rows[1].retention_before_predictions)
        case_id, prediction = prior[0]
        prior[0] = (case_id, 1 - prediction)
        altered = sign_prediction_submission(replace(
            rows[1], retention_before_predictions=tuple(prior),
            signature_hex=""), trainer_key=TRAINER_KEY)
        with self.assertRaisesRegex(SealedExamError,
                                    "disagree on prior-skill before"):
            self.score(submissions=(rows[0], altered) + rows[2:])

    def test_trainer_submission_signatures_and_pinned_key_are_mandatory(self):
        rows = self.submissions()
        with self.assertRaisesRegex(SealedExamError, "trainer verifier identity"):
            self.score(trainer_key=b"substituted-uncertified-trainer-key-000000")
        with self.assertRaisesRegex(SealedExamError, "trainer submission signature"):
            self.score(submissions=(replace(rows[0], signature_hex="0" * 64),)
                       + rows[1:])
        tampered = replace(rows[0], predictions=tuple(
            (case_id, (1 - prediction if index == 0 else prediction))
            for index, (case_id, prediction) in enumerate(rows[0].predictions)))
        with self.assertRaisesRegex(SealedExamError, "trainer submission signature"):
            self.score(submissions=(tampered,) + rows[1:])

    def test_family_concentrated_forgetting_not_hidden_by_equal_aggregate(self):
        rows = self.submissions()

        def signed_wrong(wrong_ids: set[str]):
            altered = tuple((case_id, 1 - value if case_id in wrong_ids else value)
                            for case_id, value in rows[0].predictions)
            return sign_prediction_submission(replace(
                rows[0], predictions=altered, signature_hex=""),
                trainer_key=TRAINER_KEY)

        concentrated = {f"case-10-{index:02d}" for index in range(10)}
        spread = {f"case-{family:02d}-00" for family in range(10, 20)}
        one = self.score(submissions=(signed_wrong(concentrated),) + rows[1:])
        two = self.score(submissions=(signed_wrong(spread),) + rows[1:])
        def examined(score):
            return next(item for item in json.loads(score.receipt_bytes)[
                "public_aggregate"] if (item["seed"], item["arm"]) ==
                (201, "correct"))
        first, second = examined(one), examined(two)
        self.assertEqual(first["correct_retention_after"], 190)
        self.assertEqual(second["correct_retention_after"], 190)
        self.assertEqual(first["worst_family_retention_loss_count"], 10)
        self.assertEqual(second["worst_family_retention_loss_count"], 1)
        self.assertEqual(first["worst_family_retention_loss_pp"], 50)
        self.assertEqual(second["worst_family_retention_loss_pp"], 5)
        self.assertNotEqual(one.receipt_bytes, two.receipt_bytes)
        self.assertNotEqual(one.signature_hex, two.signature_hex)
        for sensitive in ("retention-family-10", "case-10-00", "situation-10-00"):
            self.assertNotIn(sensitive, one.receipt_bytes.decode())

    def test_pinned_prior_skill_and_complete_parallel_before_required(self):
        rows = self.submissions()
        for broken in (
            replace(rows[0], prior_skill_checkpoint_sha256="a" * 64),
            replace(rows[0], retention_before_predictions=()),
            replace(rows[0], retention_before_predictions=(
                rows[0].retention_before_predictions[0],) * 200),
        ):
            with self.subTest(field=broken), self.assertRaises(SealedExamError):
                self.score(submissions=(broken,) + rows[1:])

    def test_unsigned_replacement_training_inventory_rejected(self):
        mutated = replace(self.inventory,
                          semantic_keys=("different-situation",))
        with self.assertRaisesRegex(SealedExamError, "TRAIN inventory"):
            self.score(inventory=mutated)
        with self.assertRaisesRegex(SealedExamError, "TRAIN overlap inventory"):
            build_sealed_plan(**{**self.settings,
                                 "externally_pinned_training_inventory_sha256":
                                 "0" * 64})


if __name__ == "__main__":
    unittest.main()
