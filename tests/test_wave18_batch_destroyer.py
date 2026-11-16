"""Independent Wave18 batch-manifest admission challenges.

Frozen source: 1e214df4fc94d42aaa868b288602589ba593b101.
All tests are source-only; no external model or untrusted command runs.
"""

from copy import deepcopy
import unittest

from scripts import novel_router_batch_trial as batch
from scripts import novel_router_model_trial as single


class Wave18BatchDestroyer(unittest.TestCase):
    @staticmethod
    def _baseline() -> dict:
        config = batch.BatchConfig(
            case_seeds=(0, 1), generation_seeds=(41, 42),
            max_calls_per_case=1, max_total_calls=2,
            call_timeout_seconds=2., max_total_seconds=7.,
            max_output_bytes=1024)
        report = batch.run_batch(config, single.AdapterSpec())
        assert batch.verify_batch_receipt(report)
        assert [case["admission"] for case in report["cases"]] == [
            "COMPLETED", "COMPLETED"]
        return report

    @staticmethod
    def _rehash(report: dict) -> None:
        report["batch_receipt_sha256"] = batch._sha({
            name: value for name, value in report.items()
            if name != "batch_receipt_sha256"})

    def test_wall_timeout_skip_cannot_be_followed_by_later_admission(self):
        original = self._baseline()
        tampered = deepcopy(original)
        first = tampered["cases"][0]
        first["admission"] = "SKIPPED_GLOBAL_WALL_TIMEOUT"
        first["trial"] = None
        first["trial_receipt_valid"] = False
        first["verified_win"] = False
        first["reserved_calls"] = 0
        first["reserved_seconds"] = 0.0
        first.pop("review_reason")
        # All aggregate counters are consistent and the wall clock itself is
        # forged to look exhausted. The impossible suffix must still fail.
        summary = tampered["aggregate"]
        summary["receipts_valid"] = 1
        summary["skipped_or_invalid"] = 1
        summary["calls_reserved"] = 1
        summary["calls_executed"] = 1
        summary["seconds_reserved"] = 3.0
        summary["wall_elapsed_ms"] = 7001
        self._rehash(tampered)
        self.assertFalse(batch.verify_batch_receipt(tampered),
                         "monotonic wall-budget skip cannot be followed by "
                         "a later committed and scored candidate trial")

    def test_rehashed_nested_wrong_generation_does_not_change_valid_case(self):
        original = self._baseline()
        tampered = deepcopy(original)
        tampered["cases"][0]["generation_seed"] = 42
        self._rehash(tampered)
        self.assertFalse(batch.verify_batch_receipt(tampered),
                         "case's reserved public generation seed must remain "
                         "bound to its manifest index")


if __name__ == "__main__":
    unittest.main()
