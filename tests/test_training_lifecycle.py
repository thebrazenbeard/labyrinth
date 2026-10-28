"""Lifecycle routing is not itself a release authority."""

from pathlib import Path
import tempfile
import unittest

from labyrinth.graduation_gate import GateDecision, GateState
from labyrinth.graduation_state import initialize_campaign
from labyrinth.training_lifecycle import (
    LifecycleState,
    ObjectiveFeedback,
    complete_targeted_retraining,
    record_public_bounded_check,
    submit_external_reexam,
)
from tests.test_graduation_gate import KEY, POLICY, evaluated_receipt, independently_signed


STATE_KEY = b"lifecycle-state-key-separate-from-examiner-key-2026"


class TrainingLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "custody.sqlite3"
        initialize_campaign(self.database, POLICY, state_key=STATE_KEY)

    def feedback(self, challenge="challenge-1", score=0.2, targets=("format",)):
        return ObjectiveFeedback(challenge, score, targets)

    def ready_for_exam(self, state=LifecycleState()):
        state = record_public_bounded_check(state, self.feedback())
        return complete_targeted_retraining(state, "format")

    def submit(self, state, receipt, signature=None):
        raw, signed = independently_signed(receipt)
        return submit_external_reexam(
            state, self.database, POLICY, raw, signed if signature is None else signature,
            independent_evaluator_key=KEY, state_key=STATE_KEY,
        )

    def test_public_bounded_checker_routes_training_but_never_releases(self):
        state = record_public_bounded_check(
            LifecycleState(), ObjectiveFeedback("public-check", 1.0, ())
        )
        self.assertEqual(state.phase, "EXTERNAL_REEXAM")
        self.assertNotEqual(state.phase, "CUSTODY_AUTHORIZED")
        self.assertEqual(state.public_feedback[0].score, 1.0)

    def test_feedback_replay_and_targeted_retraining_are_deterministic(self):
        state = record_public_bounded_check(LifecycleState(), self.feedback())
        self.assertEqual(state.phase, "TARGETED_RETRAINING")
        with self.assertRaisesRegex(ValueError, "replayed"):
            record_public_bounded_check(state, self.feedback())
        with self.assertRaisesRegex(ValueError, "unknown"):
            complete_targeted_retraining(state, "reasoning")
        self.assertEqual(self.ready_for_exam().phase, "EXTERNAL_REEXAM")

    def test_absent_and_malformed_exam_evidence_leave_exit_blocked(self):
        ready = self.ready_for_exam()
        absent = submit_external_reexam(
            ready, self.database, POLICY, None, None,
            independent_evaluator_key=KEY, state_key=STATE_KEY,
        )
        self.assertEqual(absent.decision.disposition, "BLOCKED")
        self.assertEqual(absent.state, ready)
        raw, _ = independently_signed(evaluated_receipt())
        malformed = submit_external_reexam(
            ready, self.database, POLICY, raw, "not-a-signature",
            independent_evaluator_key=KEY, state_key=STATE_KEY,
        )
        self.assertEqual(malformed.decision.disposition, "BLOCKED")
        self.assertEqual(malformed.state, ready)

    def test_failed_exam_requires_retraining_and_receipt_replay_is_blocked(self):
        ready = self.ready_for_exam()
        failed = self.submit(ready, evaluated_receipt(candidate_correct=False))
        self.assertEqual(failed.decision.disposition, "STAY")
        self.assertEqual(failed.state.phase, "TARGETED_RETRAINING")
        self.assertEqual(failed.state.pending_targets, ("external-reexam-failed",))
        retry = complete_targeted_retraining(failed.state, "external-reexam-failed")
        replay = self.submit(retry, evaluated_receipt(candidate_correct=False))
        self.assertEqual(replay.decision.disposition, "BLOCKED")
        self.assertEqual(replay.state, retry)

    def test_only_custody_graduation_routes_to_authorized_status(self):
        ready = self.ready_for_exam()
        # The API accepts receipt bytes and a detached signature, never a
        # caller-supplied boolean or GateDecision object as release proof.
        forged = submit_external_reexam(
            ready, self.database, POLICY, GateDecision("GRADUATED", (), GateState()), None,
            independent_evaluator_key=KEY, state_key=STATE_KEY,
        )
        self.assertEqual(forged.decision.disposition, "BLOCKED")
        self.assertEqual(forged.state, ready)
        qualified = self.submit(ready, evaluated_receipt())
        self.assertEqual(qualified.decision.disposition, "GRADUATED")
        self.assertEqual(qualified.state.phase, "CUSTODY_AUTHORIZED")


if __name__ == "__main__":
    unittest.main()
