"""R68: simultaneous TRAIN events must keep one serial, checkpointable ledger."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, BrokenBarrierError
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.minotaur_policy_learning import MinotaurTrainingPolicy, load_policy
from tests.test_minotaur_policy_learning import APPROVED, SCOPE, model, outcome


class MinotaurConcurrentObservationTests(unittest.TestCase):
    def test_concurrent_distinct_train_observations_have_linear_receipt_hashes(self):
        policy = model()
        initial = policy.model_hash
        original_getter = MinotaurTrainingPolicy.model_hash.fget
        rendezvous = Barrier(2)

        def collide_at_predecessor_hash(instance: MinotaurTrainingPolicy) -> str:
            predecessor = original_getter(instance)
            try:
                # On a broken implementation both threads can read the same
                # predecessor before either commits. A serialized policy
                # instead times out the first gate while holding its lock.
                rendezvous.wait(timeout=0.5)
            except BrokenBarrierError:
                pass
            return predecessor

        with patch.object(MinotaurTrainingPolicy, "model_hash",
                          property(collide_at_predecessor_hash)):
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = [pool.submit(policy.observe, outcome(index, correct=True))
                        for index in (0, 1)]
                receipts = [job.result(timeout=5) for job in jobs]

        self.assertTrue(all(receipt["updated"] for receipt in receipts))
        first = next(receipt for receipt in receipts
                     if receipt["before_hash"] == initial)
        second = next((receipt for receipt in receipts
                       if receipt["before_hash"] == first["after_hash"]), None)
        self.assertIsNotNone(second, "TRAIN receipts share an uncommitted predecessor")
        self.assertEqual(second["after_hash"], policy.model_hash)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "linear-train-ledger.json"
            self.assertEqual(policy.save_policy(checkpoint), second["after_hash"])
            restored = load_policy(checkpoint, scope=SCOPE,
                                   approved_training_cases=APPROVED)
            self.assertEqual(restored.model_hash, second["after_hash"])


if __name__ == "__main__":
    unittest.main()
