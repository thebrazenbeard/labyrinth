"""Public, source-only evidence transport; never an authority fixture."""

from dataclasses import replace
from hashlib import sha256
import json
import unittest

from test_trainingday_graduation_bridge import INTENT, CURRICULUM, TRAINING, POLICY, assess
from labyrinth.graduation_gate import _policy_fingerprint

from labyrinth.training_evidence_handoff_r26 import (
    HandoffSourceProvenance, ExpectedConsumerBinding,
    export_handoff, inspect_handoff, SCHEMA,
)


PROVENANCE = HandoffSourceProvenance(
    labyrinth_commit="1" * 40,
    training_report_sha256="b" * 64,
    optimizer_receipt_sha256="c" * 64,
    examiner_receipt_sha256="d" * 64,
    examiner_public_key_sha256="e" * 64,
    minotaur_policy_sha256=_policy_fingerprint(POLICY),
    training_run_id="trial-01",
)
EXPECTED = ExpectedConsumerBinding(
    objective_id=INTENT.objective_id,
    objective_kind=INTENT.objective_kind,
    intent_sha256=INTENT.intent_sha256,
    curriculum_manifest_sha256=CURRICULUM.manifest_sha256,
    admitted_corpus_sha256=TRAINING.admitted_corpus_sha256,
    training_source_sha256=TRAINING.source_sha256,
    tokenizer_sha256=TRAINING.tokenizer_sha256,
    base_model_sha256=TRAINING.base_sha256,
    initial_adapter_sha256=TRAINING.initial_adapter_sha256,
    trained_adapter_sha256=TRAINING.trained_adapter_sha256,
    completed_steps=TRAINING.completed_steps,
    evaluator_source_sha256=POLICY.expected_evaluator_sha256,
    evaluator_conditions_sha256=POLICY.expected_conditions_sha256,
    minotaur_policy_sha256=_policy_fingerprint(POLICY),
    examiner_public_key_sha256=PROVENANCE.examiner_public_key_sha256,
    examiner_receipt_sha256=PROVENANCE.examiner_receipt_sha256,
    optimizer_receipt_sha256=PROVENANCE.optimizer_receipt_sha256,
    training_report_sha256=PROVENANCE.training_report_sha256,
    training_run_id=PROVENANCE.training_run_id,
    labyrinth_commit=PROVENANCE.labyrinth_commit,
    reported_gate_status="EXAM_REQUIRED",
)


def packet(*, intent=INTENT, curriculum=CURRICULUM, training=TRAINING,
           provenance=PROVENANCE, decision=None):
    return export_handoff(
        intent, curriculum, training, POLICY,
        assess() if decision is None else decision, provenance,
    )


def reseal(value):
    """An attacker can recompute a self-digest; it cannot gain authority."""
    canonical_record = json.dumps(value["record"], sort_keys=True,
                                  separators=(",", ":"), ensure_ascii=True).encode()
    value["sha256"] = sha256(b"labyrinth.training-evidence-handoff.v1\x00"
                             + canonical_record).hexdigest()
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode()


class TrainingEvidenceHandoffR26Tests(unittest.TestCase):
    def test_public_handoff_binds_real_existing_bridge_contract_but_grants_nothing(self):
        raw = packet()
        result = inspect_handoff(raw, EXPECTED)
        self.assertEqual(result.schema, SCHEMA)
        self.assertEqual(result.objective_id, INTENT.objective_id)
        self.assertEqual(result.trained_adapter_sha256, TRAINING.trained_adapter_sha256)
        self.assertEqual(result.reported_gate_status, "EXAM_REQUIRED")
        self.assertEqual(result.disposition, "SOURCE_ONLY_NONAUTHORIZING")
        self.assertFalse(result.evaluator_independence_verified)
        self.assertFalse(result.host_authority_verified)
        self.assertFalse(result.physical_learning_verified)
        self.assertFalse(result.graduation_authorized)
        self.assertFalse(result.plastic_write_authorized)
        self.assertFalse(result.deployment_authorized)
        self.assertFalse(result.identity_consent_transferred)
        self.assertEqual(raw, packet())

    def test_expected_external_binding_rejects_cross_subject_cross_model_and_policy(self):
        raw = packet()
        for changed in (
            replace(EXPECTED, objective_id="other"),
            replace(EXPECTED, objective_kind="skill_mastery"),
            replace(EXPECTED, intent_sha256="a" * 64),
            replace(EXPECTED, curriculum_manifest_sha256="a" * 64),
            replace(EXPECTED, admitted_corpus_sha256="0" * 64),
            replace(EXPECTED, training_source_sha256="0" * 64),
            replace(EXPECTED, tokenizer_sha256="0" * 64),
            replace(EXPECTED, base_model_sha256="0" * 64),
            replace(EXPECTED, initial_adapter_sha256="0" * 64),
            replace(EXPECTED, trained_adapter_sha256="0" * 64),
            replace(EXPECTED, completed_steps=99999),
            replace(EXPECTED, completed_steps=True),
            replace(EXPECTED, completed_steps=-1),
            replace(EXPECTED, evaluator_source_sha256="a" * 64),
            replace(EXPECTED, evaluator_conditions_sha256="0" * 64),
            replace(EXPECTED, minotaur_policy_sha256="a" * 64),
            replace(EXPECTED, examiner_public_key_sha256="a" * 64),
            replace(EXPECTED, examiner_receipt_sha256="0" * 64),
            replace(EXPECTED, optimizer_receipt_sha256="0" * 64),
            replace(EXPECTED, training_report_sha256="0" * 64),
            replace(EXPECTED, training_run_id="other-run"),
            replace(EXPECTED, labyrinth_commit="0" * 40),
            replace(EXPECTED, reported_gate_status="GATE_BLOCKED"),
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    inspect_handoff(raw, changed)

    def test_export_rejects_mismatched_existing_training_lineage(self):
        for kw in (
            {"intent": replace(INTENT, objective_id="other")},
            {"curriculum": replace(CURRICULUM, manifest_sha256="a" * 64)},
            {"training": replace(TRAINING, trained_adapter_sha256="b" * 64)},
            {"training": replace(TRAINING, tokenizer_sha256="a" * 64)},
            {"training": replace(TRAINING, source_sha256="a" * 64)},
        ):
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError):
                    packet(**kw)

    def test_source_provenance_is_mandatory_with_complete_strict_digests(self):
        for item in (
            None,
            replace(PROVENANCE, labyrinth_commit="not-a-commit"),
            replace(PROVENANCE, training_report_sha256=""),
            replace(PROVENANCE, optimizer_receipt_sha256=""),
            replace(PROVENANCE, examiner_receipt_sha256=""),
            replace(PROVENANCE, examiner_public_key_sha256=""),
            replace(PROVENANCE, minotaur_policy_sha256=""),
            replace(PROVENANCE, training_run_id=""),
        ):
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    packet(provenance=item)

    def test_unverifiable_evaluator_claim_cannot_be_upgraded_to_independence(self):
        raw = packet()
        for path, value in (
            (("evaluator", "independence"), "INDEPENDENT_VERIFIED"),
            (("evaluator", "receipt_signature_verified"), True),
            (("authority", "host_authority_verified"), True),
            (("authority", "graduation_authorized"), True),
            (("authority", "deployment_authorized"), True),
            (("authority", "plastic_write_authorized"), True),
            (("authority", "identity_consent_transferred"), True),
            (("authority", "identity_consent_transferred"), 0),
            (("authority", "host_authority_verified"), 0),
        ):
            with self.subTest(path=path):
                obj = json.loads(raw)
                obj["record"][path[0]][path[1]] = value
                # Even if a malicious producer recomputes the self-digest,
                # unsupported authority declarations must be rejected.
                with self.assertRaises(ValueError):
                    inspect_handoff(reseal(obj), EXPECTED)

    def test_tamper_duplicate_keys_unknown_fields_and_noncannonical_rejected(self):
        raw = packet()
        with self.assertRaises(ValueError):
            inspect_handoff(raw.replace(b'"schema":', b'"schema":"evil","schema":', 1), EXPECTED)
        with self.assertRaises(ValueError):
            inspect_handoff(b" " + raw, EXPECTED)
        obj = json.loads(raw)
        obj["record"]["surprise"] = "hidden-consent"
        with self.assertRaises(ValueError):
            inspect_handoff(reseal(obj), EXPECTED)
        obj = json.loads(raw)
        obj["record"]["training"]["trained_adapter_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            inspect_handoff(reseal(obj), EXPECTED)

    def test_resigned_packet_with_relabelled_policy_or_bogus_optimization_is_rejected(self):
        for path, value in (
            (("evaluator", "minotaur_policy_sha256"), "a" * 64),
            (("evaluator", "examiner_public_key_sha256"), "a" * 64),
            (("training", "completed_steps"), True),
            (("training", "time_integrity_status"), "TIME_BUDGET_OVERRUN_INVALID"),
            (("minotaur", "effect_status"), "PROMOTION_ALLOWED"),
        ):
            with self.subTest(path=path):
                obj = json.loads(packet())
                obj["record"][path[0]][path[1]] = value
                with self.assertRaises(ValueError):
                    inspect_handoff(reseal(obj), EXPECTED)

    def test_resigned_claims_must_match_consumer_owned_evidence_bindings(self):
        # The hostile review correctly found these all passed at 15cde15.
        # Every field is essential to source evidence identity, even though
        # this packet never authorizes effects.
        for path, value in (
            (("subject", "objective_kind"), "skill_mastery"),
            (("evaluator", "conditions_sha256"), "0" * 64),
            (("evaluator", "examiner_receipt_sha256"), "0" * 64),
            (("provenance", "optimizer_receipt_sha256"), "0" * 64),
            (("provenance", "training_run_id"), "trial-02"),
            (("provenance", "labyrinth_commit"), "0" * 40),
            (("provenance", "training_report_sha256"), "0" * 64),
            (("training", "source_sha256"), "0" * 64),
            (("training", "tokenizer_sha256"), "0" * 64),
            (("training", "completed_steps"), 99999),
        ):
            with self.subTest(path=path):
                obj = json.loads(packet())
                obj["record"][path[0]][path[1]] = value
                with self.assertRaises(ValueError):
                    inspect_handoff(reseal(obj), EXPECTED)

    def test_gate_status_disposition_has_exact_public_semantics(self):
        # Neither the export nor readback should accept a syntactically valid
        # but self-contradictory gate result.
        invalid = replace(assess(), status="GATE_BLOCKED",
                          minotaur_gate_disposition="GRADUATED")
        with self.assertRaises(ValueError):
            packet(decision=invalid)
        obj = json.loads(packet())
        obj["record"]["minotaur"]["reported_gate_status"] = "GATE_BLOCKED"
        obj["record"]["minotaur"]["reported_gate_disposition"] = "GRADUATED"
        with self.assertRaises(ValueError):
            inspect_handoff(reseal(obj), EXPECTED)
        obj = json.loads(packet())
        obj["record"]["minotaur"]["reported_gate_status"] = "GATE_STAY"
        obj["record"]["minotaur"]["reported_gate_disposition"] = "STAY"
        with self.assertRaises(ValueError):
            inspect_handoff(reseal(obj), EXPECTED)

    def test_reported_signed_graduation_does_not_authorize_effects(self):
        reported = replace(assess(), status="SIGNED_GATE_REPORTED_GRADUATED",
                           minotaur_gate_disposition="GRADUATED")
        result = inspect_handoff(packet(decision=reported), replace(
            EXPECTED, reported_gate_status="SIGNED_GATE_REPORTED_GRADUATED"))
        self.assertEqual(result.reported_gate_status, "SIGNED_GATE_REPORTED_GRADUATED")
        self.assertFalse(result.graduation_authorized)
        self.assertFalse(result.deployment_authorized)

    def test_source_claim_of_host_verification_or_promotion_rejected(self):
        for changed in (
            replace(assess(), independently_verified_host_receipts=True),
            replace(assess(), qualified_neural_improvement=True),
            replace(assess(), promotion_allowed=True),
            replace(assess(), status="GRADUATED"),
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    packet(decision=changed)


if __name__ == "__main__":
    unittest.main()
