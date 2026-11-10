"""Wave20 attacks on non-JSON values entering the batch receipt verifier.

The source-owned receipt can contain tuples before JSON serialization, but
arbitrary Python subclasses must not get to redefine verdict comparisons.
"""

from copy import deepcopy
import tracemalloc
import unittest
from unittest import mock

from scripts import novel_router_batch_trial as batch
from scripts import novel_router_model_trial as single


class LyingSchema(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class ExplodingSchema(str):
    def __ne__(self, other):
        raise RuntimeError("hostile Python equality callback")


class Wave20PlainReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = batch.run_batch(
            batch.BatchConfig((0,), (41,), 1, 1, 0.03, 2., 256),
            single.AdapterSpec())
        assert batch.verify_batch_receipt(cls.baseline)

    def test_lying_str_subclass_cannot_self_certify_wrong_schema(self):
        tampered = deepcopy(self.baseline)
        tampered["schema"] = LyingSchema("WRONG_SCHEMA")
        tampered["batch_receipt_sha256"] = batch._sha({
            key: value for key, value in tampered.items()
            if key != "batch_receipt_sha256"})
        self.assertFalse(batch.verify_batch_receipt(tampered))

    def test_hostile_schema_comparison_fails_closed_without_crashing(self):
        tampered = {"schema": ExplodingSchema(batch.SCHEMA)}
        self.assertFalse(batch.verify_batch_receipt(tampered))

    def test_lying_nested_schema_is_rejected_not_compared(self):
        tampered = deepcopy(self.baseline)
        tampered["manifest"]["schema"] = LyingSchema("WRONG_SCHEMA")
        tampered["manifest_sha256"] = batch._sha(tampered["manifest"])
        tampered["batch_receipt_sha256"] = batch._sha({
            key: value for key, value in tampered.items()
            if key != "batch_receipt_sha256"})
        self.assertFalse(batch.verify_batch_receipt(tampered))

    def test_cyclic_malformed_receipt_is_rejected_boundedly(self):
        tampered = deepcopy(self.baseline)
        tampered["manifest"]["new_cycle"] = tampered
        self.assertFalse(batch.verify_batch_receipt(tampered))

    def test_overwide_container_is_rejected_before_queue_amplification(self):
        # Skeptic counterexample: the first guard queued every element of a
        # 350k-wide list, allocating ~24 MiB despite its 250k-node budget.
        # Prepare the hostile object before measuring *verifier* allocations.
        attack = {"cases": [None] * 350_000}
        tracemalloc.start()
        try:
            self.assertFalse(batch.verify_batch_receipt(attack))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 5 * 1024 * 1024)

    def test_exact_node_cap_does_not_charge_exit_sentinels(self):
        # A graph with five ordinary nodes must fit a five-node budget;
        # internal traversal bookkeeping must not alter the contract.
        with mock.patch.object(batch, "MAX_RECEIPT_NODES", 5):
            self.assertTrue(batch._plain_receipt_tree({
                "cases": [None, None, None]}))
            self.assertFalse(batch._plain_receipt_tree({
                "cases": [None, None, None, None]}))


if __name__ == "__main__":
    unittest.main()
