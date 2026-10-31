"""R72: registry admission is a separately traceable non-learning transition."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from labyrinth.minotaur_policy_learning import load_policy
from tests.test_minotaur_policy_learning import SCOPE, APPROVED, approval, model, outcome


class MinotaurRegistryTransitionTests(unittest.TestCase):
    def test_registry_admission_emits_atomic_nonlearning_lineage_and_reloads(self):
        policy = model()
        first_hash = policy.model_hash
        added = approval(100)
        self.assertTrue(hasattr(policy, "extend_training_registry_with_receipt"),
                        "atomic non-learning registry transition receipt is missing")

        receipt = policy.extend_training_registry_with_receipt((added,))

        self.assertEqual(receipt.before_policy_sha256, first_hash)
        self.assertEqual(receipt.after_policy_sha256, policy.model_hash)
        self.assertNotEqual(receipt.before_policy_sha256, receipt.after_policy_sha256)
        self.assertNotEqual(receipt.before_registry_sha256, receipt.after_registry_sha256)
        self.assertEqual(receipt.added_case_ids, (added.case_id,))
        self.assertEqual(receipt.transition_kind, "train_registry_admission_nonlearning")
        self.assertIs(receipt.train_observation, False)
        self.assertIs(receipt.authorizes_model_update, False)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "policy.json"
            self.assertEqual(policy.save_policy(destination), receipt.after_policy_sha256)
            restored = load_policy(destination, scope=SCOPE,
                                   approved_training_cases=APPROVED + (added,))
            self.assertEqual(restored.model_hash, receipt.after_policy_sha256)

    def test_concurrent_registry_admissions_form_one_policy_hash_chain(self):
        policy = model()
        initial = policy.model_hash
        cases = (approval(100), approval(101))
        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(pool.map(
                lambda admitted: policy.extend_training_registry_with_receipt((admitted,)),
                cases,
            ))
        first = next(item for item in receipts if item.before_policy_sha256 == initial)
        second = next(item for item in receipts if item is not first)
        self.assertEqual(first.after_policy_sha256, second.before_policy_sha256)
        self.assertEqual(second.after_policy_sha256, policy.model_hash)
        self.assertEqual(
            {item.added_case_ids for item in receipts}, {(cases[0].case_id,), (cases[1].case_id,)},
        )

    def test_next_scored_train_receipt_follows_nonlearning_registry_transition(self):
        policy = model()
        registry_receipt = policy.extend_training_registry_with_receipt((approval(100),))
        learning_receipt = policy.observe(outcome(100, correct=True))
        self.assertIs(registry_receipt.train_observation, False)
        self.assertIs(learning_receipt["updated"], True)
        self.assertEqual(learning_receipt["before_hash"],
                         registry_receipt.after_policy_sha256)

    def test_rejected_duplicate_registry_admission_leaves_state_unchanged(self):
        policy = model()
        before = policy.model_hash
        with self.assertRaisesRegex(ValueError, "duplicate case or prompt"):
            policy.extend_training_registry_with_receipt((APPROVED[0],))
        self.assertEqual(policy.model_hash, before)


if __name__ == "__main__":
    unittest.main()
