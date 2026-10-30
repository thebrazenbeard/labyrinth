"""Graduation request is not a graduation; signed external custody remains sole gate."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from labyrinth.graduation_gate import GateState
from labyrinth.journey import (
    AWAITING_EXAM,
    DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION,
    TrainingCompletionReceipt,
    begin_journey,
    complete_maze,
    face_minotaur,
)
from labyrinth.journey_qualification import (
    PublicQualificationRequest,
    decide_in_evaluator_custody,
    prepare_public_qualification_request,
)
from labyrinth.minotaur_exam import ExamCaseResult, MinotaurExamReceipt
from labyrinth.training_scope import TrainingScopeSpec

# This pre-existing test fixture is *synthetically* signed with a test-only
# HMAC key. It exercises the exact trusted gate contract, not live custody.
from tests.test_graduation_gate import (
    KEY, MODEL, POLICY, evaluated_receipt, independently_signed,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _journey_phases():
    scope = TrainingScopeSpec(
        objective_id=POLICY.objective_id, domain="reasoning",
        skills=("arithmetic",), specificity=2, scale=1, max_cycles=2, seed=233,
    )
    initial = begin_journey(scope, checkpoint_sha256=_sha("initial-adapter"))
    receipt = TrainingCompletionReceipt(
        objective_id=scope.objective_id, cycle_index=0,
        checkpoint_before_sha256=initial.checkpoint_sha256,
        checkpoint_after_sha256=MODEL,
        room_case_ids=tuple(
            (room.room_id, f"train-{i}", _sha(f"train-prompt-{i}"))
            for i, room in enumerate(initial.maze.rooms)
        ),
    )
    trained = complete_maze(initial, receipt)
    cases = tuple(
        ExamCaseResult(
            case_id=f"development-{section}-{i}", skill="arithmetic", split=section,
            score=0.95, baseline_score=0.20,
            prompt_sha256=_sha(f"development-prompt-{section}-{i}"),
        )
        for section in ("heldout", "transfer") for i in range(4)
    )
    exam = MinotaurExamReceipt(
        exam_id="development-exam-0", objective_id=scope.objective_id, cycle_index=0,
        checkpoint_sha256=MODEL, baseline_sha256=initial.baseline_checkpoint_sha256,
        evaluator_id="internal-development-only", cases=cases,
    )
    passed = face_minotaur(trained, exam)
    return initial, trained, passed


class QualificationBoundaryTests(unittest.TestCase):
    def test_trainer_cannot_request_graduation_before_dev_pass(self):
        initial, awaiting, passed = _journey_phases()
        self.assertEqual(awaiting.phase, AWAITING_EXAM)
        for state in (initial, awaiting):
            with self.subTest(phase=state.phase), self.assertRaisesRegex(ValueError, "developmental pass"):
                prepare_public_qualification_request(state)
        self.assertEqual(passed.phase, DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION)

    def test_public_request_reveals_no_private_exam_or_signing_material(self):
        _, _, passed = _journey_phases()
        request = prepare_public_qualification_request(passed)
        public = request.to_public_dict()
        self.assertEqual(public["candidate_checkpoint_sha256"], MODEL)
        self.assertEqual(public["scope_fingerprint"], passed.scope.fingerprint)
        self.assertEqual(public["qualification"], "NOT_ESTABLISHED")
        self.assertFalse({"cases", "scores", "weak_skills", "signature", "key", "private_exam"} & set(public))
        self.assertNotIn(passed.attempts[0].exam_prompt_sha256[0], str(public))

    def test_missing_or_invalid_signature_cannot_consume_attempt(self):
        request = prepare_public_qualification_request(_journey_phases()[2])
        state = GateState()
        blocked = decide_in_evaluator_custody(request, POLICY, state, None, None,
                                               expected_scope_fingerprint=request.scope_fingerprint,
                                               expected_development_exam_id=request.development_exam_id,
                                               independent_evaluator_key=None)
        self.assertEqual(blocked.disposition, "BLOCKED")
        self.assertEqual(blocked.state, state)
        self.assertNotIn("GRADUATED", blocked.reasons)

        raw, sig = independently_signed(evaluated_receipt())
        blocked = decide_in_evaluator_custody(request, POLICY, state, raw, "0" * len(sig),
                                               expected_scope_fingerprint=request.scope_fingerprint,
                                               expected_development_exam_id=request.development_exam_id,
                                               independent_evaluator_key=KEY)
        self.assertEqual(blocked.disposition, "BLOCKED")
        self.assertEqual(blocked.state, state)

    def test_wrong_subject_or_model_blocks_even_if_signed(self):
        request = prepare_public_qualification_request(_journey_phases()[2])
        raw, sig = independently_signed(evaluated_receipt())
        state = GateState()
        changed = replace(request, candidate_checkpoint_sha256="b" * 64)
        blocked = decide_in_evaluator_custody(changed, POLICY, state, raw, sig,
                                               expected_scope_fingerprint=request.scope_fingerprint,
                                               expected_development_exam_id=request.development_exam_id,
                                               independent_evaluator_key=KEY)
        self.assertEqual(blocked.disposition, "BLOCKED")
        self.assertEqual(blocked.state, state)

        changed = replace(request, objective_id="different-objective")
        blocked = decide_in_evaluator_custody(changed, POLICY, state, raw, sig,
                                               expected_scope_fingerprint=request.scope_fingerprint,
                                               expected_development_exam_id=request.development_exam_id,
                                               independent_evaluator_key=KEY)
        self.assertEqual(blocked.disposition, "BLOCKED")

    def test_exact_signed_gate_can_graduate_only_in_custody_without_mutating_journey(self):
        _, _, journey = _journey_phases()
        request = prepare_public_qualification_request(journey)
        raw, sig = independently_signed(evaluated_receipt())
        decision = decide_in_evaluator_custody(
            request, POLICY, GateState(), raw, sig,
            expected_scope_fingerprint=request.scope_fingerprint,
            expected_development_exam_id=request.development_exam_id,
            independent_evaluator_key=KEY,
        )
        self.assertEqual(decision.disposition, "GRADUATED")
        self.assertTrue(decision.state.graduated)
        self.assertEqual(decision.state.next_attempt, 2)
        self.assertEqual(journey.phase, DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION)
        self.assertFalse(hasattr(journey, "graduated"))
        self.assertNotIn("graduated", request.to_public_dict())

    def test_externally_pinned_scope_and_dev_exam_must_match_before_signature(self):
        request = prepare_public_qualification_request(_journey_phases()[2])
        raw, sig = independently_signed(evaluated_receipt())
        trusted = GateState()
        for scope_digest, exam_id in (
            ("0" * 64, request.development_exam_id),
            (request.scope_fingerprint, "substituted-development-exam"),
        ):
            with self.subTest(scope_digest=scope_digest, exam_id=exam_id):
                decision = decide_in_evaluator_custody(
                    request, POLICY, trusted, raw, sig,
                    expected_scope_fingerprint=scope_digest,
                    expected_development_exam_id=exam_id,
                    independent_evaluator_key=KEY,
                )
                self.assertEqual(decision.disposition, "BLOCKED")
                self.assertEqual(decision.state, trusted)

    def test_mutated_request_revalidated_before_trusted_gate(self):
        request = prepare_public_qualification_request(_journey_phases()[2])
        object.__setattr__(request, "candidate_checkpoint_sha256", "not-a-hash")
        with self.assertRaises(ValueError):
            decide_in_evaluator_custody(request, POLICY, GateState(), None, None,
                                         expected_scope_fingerprint="0" * 64,
                                         expected_development_exam_id="development-exam-0",
                                         independent_evaluator_key=KEY)
        with self.assertRaises(TypeError):
            decide_in_evaluator_custody({"objective_id": "forged"}, POLICY, GateState(),
                                         None, None, expected_scope_fingerprint="0" * 64,
                                         expected_development_exam_id="development-exam-0",
                                         independent_evaluator_key=KEY)


if __name__ == "__main__":
    unittest.main()
