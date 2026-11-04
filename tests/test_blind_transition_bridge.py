"""A scored private development exam cannot silently authorize an HF successor."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import hmac
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from labyrinth.blind_evaluation_custody import (
    PrivateExamCase, initialize_assessment, score_assessment,
)
from labyrinth.blind_transition_bridge import (
    CANDIDATE_LINEAGE_DOMAIN, CANDIDATE_LINEAGE_SCHEMA,
    PublicAssessmentCase, authorize_signed_regeneration,
)
from labyrinth.hf_development_transition import (
    DevelopmentCaseRef, development_case_commitment, run_journey_bound_successor,
    verify_development_transition,
)
from labyrinth.hf_pilot import PilotConfig
from labyrinth.hf_resume_bridge import ResumeContext
from labyrinth.journey import JourneyAttempt, JourneyState, TRAINING
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.training_scope import TrainingScopeSpec


EVALUATOR_KEY = b"e" * 32
STATE_KEY = b"s" * 32
LINEAGE_KEY = b"l" * 32
TRANSITION_KEY = b"t" * 32


def _cases():
    return (
        PrivateExamCase("held-1", "sentiment", "heldout", "A pleasant welcome",
                        "direct", "positive"),
        PrivateExamCase("held-2", "sentiment", "heldout", "A broken machine",
                        "direct", "negative"),
        PrivateExamCase("trans-1", "sentiment", "transfer", "Her client smiled",
                        "narrative", "positive"),
        PrivateExamCase("trans-2", "sentiment", "transfer", "Her client frowned",
                        "narrative", "negative"),
    )


def _public():
    return tuple(PublicAssessmentCase(
        c.case_id, c.skill, c.split, sha256(c.prompt.encode()).hexdigest(),
        c.structure_key,
    ) for c in _cases())


class BlindTransitionBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.artifact = self.path / "candidate.safetensors"
        self.artifact.write_bytes(b"safetensors fixture only")
        self.artifact_hash = sha256(self.artifact.read_bytes()).hexdigest()
        self.scope = TrainingScopeSpec(
            "learn-sentiment", "language", ("sentiment",), 1, 1, 3, 17,
        )
        self.context = ResumeContext(
            self.scope.fingerprint, 1, "a" * 64, "b" * 64, "operator-cycle-one",
        )
        self.db = self.path / "assessment.sqlite"
        self.public_precommit = initialize_assessment(
            self.db, _cases(), run_id="candidate-run-0",
            objective_id=self.scope.objective_id,
            checkpoint_sha256=self.artifact_hash,
            evaluator_key=EVALUATOR_KEY, state_key=STATE_KEY,
        )
        responses = {c.case_id: c.gold_answer for c in _cases()}
        responses["trans-2"] = "positive"
        self.scored = score_assessment(
            self.db, _cases(), responses, artifact_root=self.path,
            artifact_name="candidate.safetensors",
            evaluator_key=EVALUATOR_KEY, state_key=STATE_KEY,
        )
        self.attestation = {
            "schema": CANDIDATE_LINEAGE_SCHEMA,
            "run_id": "candidate-run-0",
            "exam_id": "dev-exam-0",
            "objective_id": self.scope.objective_id,
            "scope_fingerprint": self.scope.fingerprint,
            "cycle_index": 0,
            "checkpoint_artifact_sha256": self.artifact_hash,
            "predecessor_report_sha256": self.context.predecessor_report_sha256,
            "predecessor_lora_value_sha256": self.context.predecessor_lora_value_sha256,
            "public_exam_sha256": self.public_precommit,
        }

    def lineage(self, updates=None):
        values = {**self.attestation, **(updates or {})}
        raw = (json.dumps(values, sort_keys=True, separators=(",", ":")) + "\n").encode()
        return raw, hmac.new(LINEAGE_KEY, CANDIDATE_LINEAGE_DOMAIN + raw, sha256).hexdigest()

    def bridge(self, **changes):
        raw, signature = self.lineage()
        public = _public()
        dev_public = tuple(DevelopmentCaseRef(
            c.case_id, c.skill, c.prompt_sha256, c.structure_key,
        ) for c in public)
        options = {
            "scope": self.scope, "context": self.context,
            "scored_receipt_bytes": self.scored.receipt_bytes,
            "scored_signature_hex": self.scored.signature_hex,
            "candidate_run_id": "candidate-run-0",
            "candidate_artifact_sha256": self.artifact_hash,
            "public_cases": public,
            "evaluator_precommitment_sha256": self.public_precommit,
            "development_roster_precommitment_sha256": development_case_commitment(
                dev_public),
            "candidate_lineage_bytes": raw,
            "candidate_lineage_signature_hex": signature,
            "exam_id": "dev-exam-0", "event_id": "dev-event-1",
            "failure_threshold_percent": 75,
            "independent_evaluator_key": EVALUATOR_KEY,
            "independent_lineage_key": LINEAGE_KEY,
            "independent_transition_key": TRANSITION_KEY,
        }
        options.update(changes)
        return authorize_signed_regeneration(**options)

    def test_authenticated_score_plus_lineage_produces_verified_development_fail(self):
        signed = self.bridge()
        expected_roster = development_case_commitment(signed.development_public_roster)
        verified = verify_development_transition(
            self.scope, self.context, signed.receipt_bytes,
            signed.signature_hex, independent_development_key=TRANSITION_KEY,
        )
        self.assertEqual(verified.exam_id, "dev-exam-0")
        self.assertEqual(verified.event_id, "dev-event-1")
        self.assertEqual(verified.weak_skills, ("sentiment",))
        self.assertEqual(verified.exam_case_commitment_sha256, expected_roster)
        self.assertFalse(signed.graduation_authorized)
        self.assertFalse(signed.model_execution_authenticated)
        self.assertNotIn(b"A pleasant welcome", signed.receipt_bytes)
        self.assertNotIn(b"positive", signed.receipt_bytes)

    def test_real_successor_contract_accepts_bridge_when_journey_matches(self):
        signed = self.bridge()
        ids = tuple(c.case_id for c in _public())
        prompts = tuple(c.prompt_sha256 for c in _public())
        attempt = JourneyAttempt(
            cycle_index=0, exam_id="dev-exam-0",
            checkpoint_before_sha256="d" * 64,
            checkpoint_after_sha256=self.context.predecessor_lora_value_sha256,
            training_case_ids=("training-0",),
            training_prompt_sha256=("e" * 64,),
            exam_case_ids=ids, exam_prompt_sha256=prompts,
            exam_status="FAIL", weak_skills=("sentiment",),
        )
        journey = JourneyState(
            scope=self.scope, baseline_checkpoint_sha256="d" * 64,
            checkpoint_sha256=self.context.predecessor_lora_value_sha256,
            maze=generate_maze_plan(self.scope, 1, weak_skills=("sentiment",)),
            phase=TRAINING, attempts=(attempt,),
            used_training_case_ids=("training-0",),
            used_training_prompt_sha256=("e" * 64,),
            sealed_exam_ids=("dev-exam-0",),
            exam_event_statuses=("FAIL",),
            sealed_exam_case_ids=ids,
            sealed_exam_prompt_sha256=prompts,
        )
        consumed = []
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume"),
              mock.patch("labyrinth.hf_development_transition.run_pilot",
                         return_value={"trained": True}) as pilot):
            result = run_journey_bound_successor(
                PilotConfig(model_path="unused", output_dir="unused"),
                self.scope, "unused-run", self.context,
                signed.receipt_bytes, signed.signature_hex,
                journey=journey,
                evaluator_precommitment_sha256=development_case_commitment(
                    signed.development_public_roster),
                evaluator_public_roster=signed.development_public_roster,
                independent_development_key=TRANSITION_KEY,
                reserve_development_event=lambda event_id: consumed.append(event_id) or True,
            )
        self.assertEqual(result, {"trained": True})
        self.assertEqual(consumed, ["dev-event-1"])
        pilot.assert_called_once()

    def test_perfect_development_score_cannot_authorize_regeneration(self):
        perfect = json.loads(self.scored.receipt_bytes)
        for group in perfect["groups"]:
            group["correct"] = group["total"]
        raw = (json.dumps(perfect, sort_keys=True, separators=(",", ":")) + "\n").encode()
        sig = hmac.new(EVALUATOR_KEY, b"labyrinth.blind-development-result.v1\0" + raw,
                       sha256).hexdigest()
        with self.assertRaisesRegex(ValueError, "no authenticated weakness"):
            self.bridge(scored_receipt_bytes=raw, scored_signature_hex=sig)

    def test_tampering_key_substitution_and_wrong_base_rejected(self):
        raw, signature = self.lineage()
        for changes in (
            {"scored_signature_hex": "0" * 64},
            {"candidate_lineage_signature_hex": "0" * 64},
            {"candidate_lineage_bytes": raw + b" "},
            {"independent_lineage_key": EVALUATOR_KEY},
            {"independent_transition_key": EVALUATOR_KEY},
            {"context": replace(self.context, predecessor_lora_value_sha256="c" * 64)},
            {"candidate_artifact_sha256": "f" * 64},
            {"candidate_run_id": "other-run"},
            {"evaluator_precommitment_sha256": "f" * 64},
            {"development_roster_precommitment_sha256": "f" * 64},
            {"failure_threshold_percent": 0},
            {"failure_threshold_percent": True},
            {"public_cases": _public()[:-1]},
        ):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.bridge(**changes)

    def test_externally_signed_lineage_mismatch_does_not_bridge(self):
        for updates in (
            {"cycle_index": 1}, {"exam_id": "other-exam"},
            {"predecessor_report_sha256": "c" * 64},
            {"public_exam_sha256": "e" * 64},
            {"checkpoint_artifact_sha256": "f" * 64},
            {"scope_fingerprint": "e" * 64},
        ):
            with self.subTest(changes=updates):
                raw, sig = self.lineage(updates)
                with self.assertRaises(ValueError):
                    self.bridge(candidate_lineage_bytes=raw,
                                candidate_lineage_signature_hex=sig)

    def test_duplicate_prompt_and_structure_split_contamination_rejected(self):
        public = _public()
        with self.assertRaises(ValueError):
            self.bridge(public_cases=public[:2] + (
                replace(public[2], prompt_sha256=public[0].prompt_sha256), public[3]))
        with self.assertRaises(ValueError):
            self.bridge(public_cases=public[:2] + (
                replace(public[2], structure_key="direct"), public[3]))

    def test_partial_scope_exam_cannot_sign_full_objective_regeneration(self):
        # A declaration with two skills cannot treat a one-skill exam as
        # comprehensive development evidence for the declared objective.
        self.scope = replace(self.scope, skills=("sentiment", "negation"))
        self.context = replace(self.context,
                               scope_fingerprint=self.scope.fingerprint)
        self.attestation["scope_fingerprint"] = self.scope.fingerprint
        with self.assertRaisesRegex(ValueError, "declared scope skills"):
            self.bridge()

    def test_complete_two_skill_exam_can_regenerate_only_failed_skill(self):
        self.scope = replace(self.scope, skills=("sentiment", "negation"))
        self.context = replace(self.context,
                               scope_fingerprint=self.scope.fingerprint)
        cases = _cases() + (
            PrivateExamCase("held-n1", "negation", "heldout",
                            "Today is not a pleasant day", "direct-negation", "negative"),
            PrivateExamCase("held-n2", "negation", "heldout",
                            "The deal is not a disaster", "direct-negation", "positive"),
            PrivateExamCase("trans-n1", "negation", "transfer",
                            "Unexpectedly the service did not fail", "context-negation", "positive"),
            PrivateExamCase("trans-n2", "negation", "transfer",
                            "Despite promises the system did not work", "context-negation", "negative"),
        )
        self.public_precommit = initialize_assessment(
            self.path / "two-skill-assessment.sqlite", cases,
            run_id="candidate-run-0", objective_id=self.scope.objective_id,
            checkpoint_sha256=self.artifact_hash,
            evaluator_key=EVALUATOR_KEY, state_key=STATE_KEY,
        )
        responses = {case.case_id: case.gold_answer for case in cases}
        responses["trans-2"] = "positive"  # Only sentiment is weak.
        self.scored = score_assessment(
            self.path / "two-skill-assessment.sqlite", cases, responses,
            artifact_root=self.path, artifact_name="candidate.safetensors",
            evaluator_key=EVALUATOR_KEY, state_key=STATE_KEY,
        )
        self.attestation["scope_fingerprint"] = self.scope.fingerprint
        self.attestation["public_exam_sha256"] = self.public_precommit
        public = tuple(PublicAssessmentCase(
            c.case_id, c.skill, c.split, sha256(c.prompt.encode()).hexdigest(),
            c.structure_key,
        ) for c in cases)
        dev_roster = tuple(DevelopmentCaseRef(
            c.case_id, c.skill, c.prompt_sha256, c.structure_key,
        ) for c in public)
        signed = self.bridge(
            public_cases=public,
            development_roster_precommitment_sha256=development_case_commitment(
                dev_roster),
        )
        verified = verify_development_transition(
            self.scope, self.context, signed.receipt_bytes,
            signed.signature_hex, independent_development_key=TRANSITION_KEY,
        )
        self.assertEqual(verified.weak_skills, ("sentiment",))


if __name__ == "__main__":
    unittest.main()
