"""Source-only preregistration checks; no external examiner is installed."""

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
import unittest

from labyrinth.preregistered_control_matrix import (
    ARMS, BASELINES, ArmRef, CaseRef, PrecommitError,
    build_precommitted_matrix, signed_final_placeholder,
    verify_final_placeholder,
)


def hx(value: str) -> str:
    return sha256(value.encode("ascii")).hexdigest()


class PreregisteredControlsTests(unittest.TestCase):
    def setUp(self):
        self.evaluator_key = b"E" * 32
        self.trainer_key = b"T" * 32
        self.train = tuple(CaseRef(
            f"train-{index}", "train-family", hx(f"train prompt {index}"),
            index % 2, "qualitative", index,
        ) for index in range(8))
        self.development = tuple(CaseRef(
            f"dev-{index}", "dev-family", hx(f"dev prompt {index}"),
            index % 2, "qualitative", index + 100,
        ) for index in range(8))
        visits = tuple(case.case_id for case in self.train)
        self.arms = (
            ArmRef("correct", "AdamW", 0.002, hx("weights"), 42,
                   visits, "identity"),
            ArmRef("zero_lr", "AdamW", 0.0, hx("weights"), 42,
                   visits, "identity"),
            ArmRef("permuted", "AdamW", 0.002, hx("weights"), 42,
                   visits, "fixed_permutation", permutation=(1, 0)),
            ArmRef("randomized", "AdamW", 0.002, hx("weights"), 42,
                   visits, "random_balanced_shuffling", corruption_seed=987),
        )
        self.final = signed_final_placeholder(
            roster_sha256=hx("secret final roster"),
            signer_id="eval-A", evaluator_host_id="EXAM-HOST",
            trainer_host_id="TRAIN-HOST", nonce_sha256=hx("precommit nonce"),
            evaluator_key=self.evaluator_key,
        )

    def build(self, **changes):
        args = {
            "objective_id": "taskmaster-tactic-classification",
            "scope_sha256": hx("scope"),
            "pretrained_model_sha256": hx("frozen-base"),
            "nonce_sha256": hx("precommit nonce"),
            "training_cases": self.train,
            "public_development_cases": self.development,
            "arms": self.arms,
            "label_count": 2,
            "representation_policy": "same_representation",
            "final_placeholder": self.final,
            "independent_evaluator_key": self.evaluator_key,
            "trainer_key": self.trainer_key,
            "expected_final_signer_id": "eval-A",
            "expected_evaluator_host_id": "EXAM-HOST",
            "expected_trainer_host_id": "TRAIN-HOST",
        }
        return build_precommitted_matrix(**(args | changes))

    def test_immutable_stable_source_commit_and_no_exam_authority(self):
        first = self.build()
        self.assertEqual(first, self.build())
        self.assertEqual(len(first.sha256), 64)
        self.assertFalse(first.graduation_authorized)
        self.assertFalse(first.independent_exam_authorized)
        manifest = json.loads(first.canonical_bytes)
        self.assertEqual(tuple(manifest["baseline_roster"]), BASELINES)
        self.assertEqual(manifest["final_exam_release"],
                         "BLOCKED_PENDING_INDEPENDENT_CUSTODY")
        self.assertEqual(manifest["final_exam_source_authentication"],
                         "SOURCE_SIGNATURE_VERIFIED_FINAL_EXAM_STILL_BLOCKED")
        self.assertEqual(manifest["final_exam_key_custody"],
                         "CALLER_ASSERTED_NOT_EXTERNALLY_ATTESTED")
        self.assertEqual(manifest["semantic_family_independence"],
                         "UNVERIFIED_EXTERNAL_REVIEW_REQUIRED")
        self.assertEqual(manifest["precommit_timestamp_authority"],
                         "NONE_SOURCE_DIGEST_ONLY")
        self.assertFalse(manifest["graduation_authorized"])
        self.assertEqual(set(manifest["effective_training_labels_by_arm"]), set(ARMS))
        with self.assertRaises(FrozenInstanceError):
            first.sha256 = hx("another")

    def test_control_arms_are_mapped_to_exact_training_visits(self):
        manifest = json.loads(self.build().canonical_bytes)
        generated = manifest["effective_training_labels_by_arm"]
        gold = [case.label for case in self.train]
        self.assertEqual(generated["correct"], gold)
        self.assertEqual(generated["zero_lr"], gold)
        self.assertEqual(generated["permuted"], [1 - item for item in gold])
        self.assertEqual(sorted(generated["randomized"]), sorted(gold))
        self.assertNotEqual(generated["randomized"], gold)
        self.assertTrue(all(row["optimizer_step_required"] for row in manifest["arms"]))
        self.assertEqual(next(row for row in manifest["arms"]
                              if row["name"] == "zero_lr")["learning_rate"], 0)

    def test_train_exam_same_semantic_family_is_blocked(self):
        poisoned = tuple(replace(row, family_id="train-family")
                         for row in self.development)
        with self.assertRaisesRegex(PrecommitError, "families overlap"):
            self.build(public_development_cases=poisoned)

    def test_duplicate_prompt_alias_or_seed_cross_split_is_blocked(self):
        with self.assertRaisesRegex(PrecommitError, "reused"):
            self.build(public_development_cases=(
                replace(self.development[0], prompt_sha256=self.train[0].prompt_sha256),
                *self.development[1:],
            ))
        with self.assertRaisesRegex(PrecommitError, "reused"):
            self.build(public_development_cases=(
                replace(self.development[0], seed=self.train[0].seed),
                *self.development[1:],
            ))

    def test_noncanonical_family_alias_and_unbalanced_gold_blocked(self):
        with self.assertRaisesRegex(PrecommitError, "canonical"):
            self.build(training_cases=(
                replace(self.train[0], family_id="Train_Family"),
                *self.train[1:],
            ))
        with self.assertRaisesRegex(PrecommitError, "balanced"):
            self.build(training_cases=(
                replace(self.train[0], label=1),
                *self.train[1:],
            ))

    def test_unregistered_representation_mix_blocked(self):
        mixed = tuple(replace(row, representation="quantitative")
                      for row in self.development)
        with self.assertRaisesRegex(PrecommitError, "representation"):
            self.build(public_development_cases=mixed)
        cross = self.build(public_development_cases=mixed,
                           representation_policy="cross_representation")
        self.assertEqual(json.loads(cross.canonical_bytes)["representation_policy"],
                         "cross_representation")
        with self.assertRaisesRegex(PrecommitError, "representation"):
            self.build(representation_policy="cross_representation")

    def test_global_balance_cannot_hide_representation_label_confounding(self):
        # Balanced 4/4 overall, but qualitative always means class 0.
        # The development roster has the inverse shortcut. Neither is valid.
        train = tuple(replace(
            row, representation=("qualitative" if row.label == 0
                                 else "quantitative"))
            for row in self.train)
        dev = tuple(replace(
            row, representation=("quantitative" if row.label == 0
                                 else "qualitative"))
            for row in self.development)
        with self.assertRaisesRegex(PrecommitError, "strata are confounded"):
            self.build(training_cases=train, public_development_cases=dev)
        with self.assertRaisesRegex(PrecommitError, "strata are confounded"):
            self.build(public_development_cases=dev)

    def test_balanced_label_strata_in_both_representations_allowed(self):
        train = tuple(replace(
            row, representation=("qualitative" if index < 4
                                 else "quantitative"))
            for index, row in enumerate(self.train))
        dev = tuple(replace(
            row, representation=("quantitative" if index < 4
                                 else "qualitative"))
            for index, row in enumerate(self.development))
        accepted = self.build(training_cases=train,
                              public_development_cases=dev)
        self.assertEqual(json.loads(accepted.canonical_bytes)["representation_policy"],
                         "same_representation")

    def test_arms_must_share_original_seed_visits_and_weights(self):
        wrong = replace(self.arms[1], training_case_visits=
                        tuple(reversed(self.arms[1].training_case_visits)))
        with self.assertRaisesRegex(PrecommitError, "mismatched"):
            self.build(arms=(self.arms[0], wrong, *self.arms[2:]))
        wrong = replace(self.arms[2], initial_weights_sha256=hx("changed"))
        with self.assertRaisesRegex(PrecommitError, "mismatched"):
            self.build(arms=(*self.arms[:2], wrong, self.arms[3]))
        wrong = replace(self.arms[3], model_seed=87)
        with self.assertRaisesRegex(PrecommitError, "mismatched"):
            self.build(arms=(*self.arms[:3], wrong))

    def test_optimizer_step_and_zero_lr_are_mandatory(self):
        wrong = replace(self.arms[1], optimizer_step_required=False)
        with self.assertRaisesRegex(PrecommitError, "mismatched"):
            self.build(arms=(self.arms[0], wrong, *self.arms[2:]))
        wrong = replace(self.arms[1], learning_rate=0.002)
        with self.assertRaisesRegex(PrecommitError, "zero-learning-rate"):
            self.build(arms=(self.arms[0], wrong, *self.arms[2:]))
        wrong = replace(self.arms[0], learning_rate=0.0)
        with self.assertRaisesRegex(PrecommitError, "zero-learning-rate"):
            self.build(arms=(wrong, *self.arms[1:]))

    def test_corruption_contract_is_exact_and_nonidentity(self):
        invalid = replace(self.arms[2], permutation=(0, 1))
        with self.assertRaisesRegex(PrecommitError, "nonidentity"):
            self.build(arms=(*self.arms[:2], invalid, self.arms[3]))
        invalid = replace(self.arms[3], corruption_seed=42)
        with self.assertRaisesRegex(PrecommitError, "separate randomized"):
            self.build(arms=(*self.arms[:3], invalid))
        invalid = replace(self.arms[3], permutation=(1, 0))
        with self.assertRaisesRegex(PrecommitError, "separate randomized"):
            self.build(arms=(*self.arms[:3], invalid))

    def test_unregistered_baselines_are_blocked(self):
        with self.assertRaisesRegex(PrecommitError, "baseline roster"):
            self.build(baseline_roster=("majority_train_only",))
        manifest = json.loads(self.build().canonical_bytes)
        self.assertEqual(manifest["numeric_rule_baseline_role"],
                         "TASK_FEASIBILITY_ONLY_NOT_NEURAL_COMPETITOR")

    def test_final_custody_metadata_cannot_authorize_graduation(self):
        value = verify_final_placeholder(
            self.final, evaluator_key=self.evaluator_key,
            trainer_key=self.trainer_key, expected_signer_id="eval-A",
            expected_evaluator_host_id="EXAM-HOST",
            expected_trainer_host_id="TRAIN-HOST",
        )
        self.assertEqual(value, "SOURCE_SIGNATURE_VERIFIED_FINAL_EXAM_STILL_BLOCKED")
        with self.assertRaisesRegex(PrecommitError, "custody"):
            verify_final_placeholder(
                self.final, evaluator_key=self.evaluator_key,
                trainer_key=self.evaluator_key, expected_signer_id="eval-A",
                expected_evaluator_host_id="EXAM-HOST",
                expected_trainer_host_id="TRAIN-HOST",
            )
        with self.assertRaisesRegex(PrecommitError, "signature"):
            verify_final_placeholder(
                replace(self.final, signature_hex=hx("fake")),
                evaluator_key=self.evaluator_key, trainer_key=self.trainer_key,
                expected_signer_id="eval-A",
                expected_evaluator_host_id="EXAM-HOST",
                expected_trainer_host_id="TRAIN-HOST",
            )

    def test_final_host_isolation_required_for_source_metadata(self):
        with self.assertRaisesRegex(PrecommitError, "metadata"):
            signed_final_placeholder(
                roster_sha256=hx("exam"), signer_id="same",
                evaluator_host_id="same", trainer_host_id="same",
                nonce_sha256=hx("nonce"), evaluator_key=self.evaluator_key,
            )
        with self.assertRaisesRegex(PrecommitError, "must differ"):
            self.build(final_placeholder=replace(
                self.final, trainer_host_id="EXAM-HOST"))

    def test_final_nonce_must_match_pinned_experiment_request(self):
        with self.assertRaisesRegex(PrecommitError, "nonce differs"):
            self.build(final_placeholder=replace(
                self.final, nonce_sha256=hx("attacker changed nonce")))

    def test_hex_shaped_forgery_not_accepted_as_valid_signature(self):
        with self.assertRaisesRegex(PrecommitError, "signature invalid"):
            self.build(final_placeholder=replace(
                self.final, signature_hex=hx("attacker forged a signature")))
        with self.assertRaisesRegex(PrecommitError, "signature invalid"):
            self.build(independent_evaluator_key=b"F" * 32)

    def test_pinned_host_signer_and_distinct_key_are_mandatory(self):
        with self.assertRaisesRegex(PrecommitError, "custody"):
            self.build(expected_final_signer_id="unexpected")
        with self.assertRaisesRegex(PrecommitError, "custody"):
            self.build(expected_evaluator_host_id="another-host")
        with self.assertRaisesRegex(PrecommitError, "custody"):
            self.build(trainer_key=self.evaluator_key)

    def test_malformed_control_arm_rejected_with_bounded_error(self):
        with self.assertRaisesRegex(PrecommitError, "control arms"):
            self.build(arms=(*self.arms[:3], object()))


if __name__ == "__main__":
    unittest.main()
