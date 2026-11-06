"""Source-only custody contract tests; all predictions and receipts are synthetic."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

from labyrinth.independent_holdout import (
    InferenceClaim, case_commitment, evaluate_one_shot_holdout,
    predictions_commitment, private_gold_seal,
)
from test_independent_holdout import EXAMINER_KEY, _fixture


def _claims(arms):
    return tuple(InferenceClaim(
        arm_name=arm.name,
        process_id=f"fresh-process-{i}",
        adapter_sha256=arm.final_adapter_sha256,
        predictions_sha256=predictions_commitment(arm.responses),
        inference_report_sha256=hashlib.sha256(f"report-{i}".encode()).hexdigest(),
    ) for i, arm in enumerate(arms))


class ExamCustodyR6Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "exam-custodian.sqlite3"
        self.cases, self.exposure, self.arms, self.before = _fixture()

    def evaluate(self, *, arms=None, claims=None, attempt="attempt-one"):
        actual_arms = self.arms if arms is None else arms
        actual_claims = _claims(actual_arms) if claims is None else claims
        return evaluate_one_shot_holdout(
            self.cases, case_commitment(self.cases), self.exposure,
            actual_arms, self.before,
            examiner_gold_seal=private_gold_seal(self.cases, EXAMINER_KEY),
            examiner_key=EXAMINER_KEY, attempt_id=attempt,
            inference_claims=actual_claims, custodian_db_path=self.db,
        )

    def test_first_attempt_is_provisional_and_second_is_rejected_even_if_renamed(self):
        first = self.evaluate()
        self.assertEqual(first.status, "PROVISIONAL")
        self.assertFalse(first.graduation_authorized)
        self.assertFalse(first.evidence_authenticated)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.evaluate(attempt="attempt-renamed")
        with sqlite3.connect(self.db) as connection:
            rows = connection.execute(
                "SELECT exam_commitment, attempt_id, submission_sha256 FROM exam_attempts"
            ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], case_commitment(self.cases))
        self.assertEqual(rows[0][1], "attempt-one")
        self.assertEqual(len(rows[0][2]), 64)

    def test_changed_response_cannot_reuse_inference_claim(self):
        claims = _claims(self.arms)
        changed = tuple(replace(arm, responses=arm.responses[:-1] +
                                ((arm.responses[-1][0], "ANSWER: 999"),))
                        if arm.name == "correct_labels" else arm for arm in self.arms)
        with self.assertRaisesRegex(ValueError, "prediction claim"):
            self.evaluate(arms=changed, claims=claims)
        self.assertFalse(self.db.exists())

    def test_claimed_adapter_must_match_control_arm(self):
        claims = _claims(self.arms)
        tampered = (replace(claims[0], adapter_sha256="f" * 64),) + claims[1:]
        with self.assertRaisesRegex(ValueError, "adapter claim"):
            self.evaluate(claims=tampered)

    def test_distinct_fresh_process_claims_required(self):
        claims = _claims(self.arms)
        tampered = claims[:-1] + (replace(claims[-1], process_id=claims[0].process_id),)
        with self.assertRaisesRegex(ValueError, "fresh process"):
            self.evaluate(claims=tampered)

    def test_failed_scoring_consumes_reserved_exam_and_cannot_probe_again(self):
        missing = tuple(replace(arm, responses=arm.responses[:-1])
                        if arm.name == "correct_labels" else arm for arm in self.arms)
        with self.assertRaisesRegex(ValueError, "predictions"):
            self.evaluate(arms=missing)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.evaluate(attempt="attempt-two")

    def test_parallel_attempts_cannot_both_reserve_same_exam(self):
        def attempt(index):
            try:
                self.evaluate(attempt=f"attempt-{index}")
            except ValueError as exc:
                return str(exc)
            return "accepted"

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(attempt, range(8)))
        self.assertEqual(results.count("accepted"), 1)
        self.assertEqual(sum("already consumed" in result for result in results), 7)

    def test_unavailable_custodian_store_fails_closed(self):
        self.db = self.db.parent / "no-such-directory" / "exam.sqlite3"
        with self.assertRaises(sqlite3.OperationalError):
            self.evaluate()
        self.assertFalse(self.db.exists())

    def test_fabricated_process_ids_cannot_be_elevated_to_attestation(self):
        # These self-reported process IDs are syntactically valid but were
        # fabricated in this test. Passing claims cannot authenticate them.
        verdict = self.evaluate(claims=_claims(self.arms))
        self.assertEqual(verdict.status, "PROVISIONAL")
        self.assertFalse(verdict.evidence_authenticated)
        self.assertFalse(verdict.graduation_authorized)

    def test_in_memory_uri_and_relative_db_paths_cannot_reset_one_shot_policy(self):
        unsafe_paths = (
            "", ":memory:", "file::memory:?cache=shared",
            "file:/tmp/exam?mode=memory", "exam-relative.sqlite3", Path("."),
        )
        for unsafe in unsafe_paths:
            with self.subTest(path=str(unsafe)):
                self.db = unsafe
                with self.assertRaisesRegex(ValueError, "stable file-backed"):
                    self.evaluate()

    def test_symlink_db_cannot_redirect_custody_to_different_store(self):
        real = Path(self.temp.name) / "real-exam.sqlite3"
        real.touch()
        link = Path(self.temp.name) / "linked-exam.sqlite3"
        link.symlink_to(real)
        self.db = link
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.evaluate()

    def test_symlink_parent_and_swapped_alias_cannot_reopen_exam(self):
        actual_parent = Path(self.temp.name) / "custody"
        actual_parent.mkdir()
        alias_parent = Path(self.temp.name) / "alias"
        alias_parent.symlink_to(actual_parent, target_is_directory=True)
        self.db = alias_parent / "exam.sqlite3"
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.evaluate()

        first = Path(self.temp.name) / "first.sqlite3"
        second = Path(self.temp.name) / "second.sqlite3"
        alias_file = Path(self.temp.name) / "swappable.sqlite3"
        first.touch()
        second.touch()
        alias_file.symlink_to(first)
        self.db = alias_file
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.evaluate()
        alias_file.unlink()
        alias_file.symlink_to(second)
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.evaluate()


if __name__ == "__main__":
    unittest.main()
