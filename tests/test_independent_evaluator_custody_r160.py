"""Synthetic-only Q5–Q9 examiner custody admission boundary regressions."""

from __future__ import annotations

import hashlib
import json
import unittest

from labyrinth.independent_evaluator_custody_r160 import (
    IndependentEvaluatorIssuerUnavailable,
    InvalidEvaluatorCustodyClaim,
    UNVERIFIED_REJECTED,
    inspect_evaluator_custody_claim,
    require_independent_evaluator_admission,
)


def _claim() -> dict:
    return {
        "schema": "labyrinth.independent-evaluator-custody-claim.v1",
        "examiner_id": "examiner-source-1",
        "bank_custodian_id": "custodian-source-2",
        "rights_steward_id": "steward-source-3",
        "q6_witness_owner_id": "witness-source-4",
        "transport_principal_id": "transport-source-5",
        "protected_store_id": "opaque-store-claim-6",
        "issuer_key_generation": 1,
        "issuer_public_key_hex": "a1" * 32,
        "receipt_signature_hex": "b2" * 64,
        "private_bank_commitment_sha256": "01" * 32,
        "train_exclusion_sha256": "02" * 32,
        "pretraining_precommit_sha256": "03" * 32,
        "model_freeze_sha256": "04" * 32,
        "q6_witness_head_sha256": "05" * 32,
        "rights_receipt_sha256": "06" * 32,
        "exam_policy_sha256": "07" * 32,
    }


def _wire(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


class IndependentEvaluatorCustodySourceTests(unittest.TestCase):
    def test_even_plausible_signed_claim_stays_unverified_and_non_authorizing(self):
        raw = _wire(_claim())
        report = inspect_evaluator_custody_claim(raw)
        self.assertEqual(report["status"], UNVERIFIED_REJECTED)
        self.assertEqual(report["claim_sha256"], hashlib.sha256(raw).hexdigest())
        for name in (
            "issuer_registry_verified", "signing_key_custody_verified",
            "transport_identity_verified", "protected_store_verified",
            "receipt_signature_verified", "revocation_verified",
            "rights_admitted", "train_exclusions_verified",
            "pretraining_chronology_verified", "q6_monotonic_witness_verified",
            "model_freeze_verified", "authorizes_private_exam",
            "authorizes_training", "authorizes_graduation",
        ):
            with self.subTest(name=name):
                self.assertIs(report[name], False)
        serialized = json.dumps(report)
        for private_or_sensitive in (
            "examiner-source-1", "opaque-store-claim-6",
            "b2" * 64, "a1" * 32, "private_prompt", "private_gold",
        ):
            self.assertNotIn(private_or_sensitive, serialized)

    def test_source_only_claim_cannot_be_promoted_by_an_api_option(self):
        raw = _wire(_claim())
        with self.assertRaisesRegex(IndependentEvaluatorIssuerUnavailable,
                                    "HARD HOLD"):
            require_independent_evaluator_admission(raw)

    def test_forged_signature_and_issuer_key_remain_rejected(self):
        fake = _claim()
        fake["issuer_public_key_hex"] = "c3" * 32
        fake["receipt_signature_hex"] = "d4" * 64
        report = inspect_evaluator_custody_claim(_wire(fake))
        self.assertEqual(report["status"], UNVERIFIED_REJECTED)
        self.assertIs(report["receipt_signature_verified"], False)
        self.assertIs(report["issuer_registry_verified"], False)

    def test_rehashed_self_owned_bank_and_precommit_cannot_admit(self):
        fake = _claim()
        fake["private_bank_commitment_sha256"] = hashlib.sha256(
            b"public synthetic bank chosen by attacker").hexdigest()
        fake["pretraining_precommit_sha256"] = hashlib.sha256(
            b"retrospective precommit chosen by attacker").hexdigest()
        report = inspect_evaluator_custody_claim(_wire(fake))
        self.assertEqual(report["status"], UNVERIFIED_REJECTED)
        self.assertIs(report["pretraining_chronology_verified"], False)
        self.assertIs(report["protected_store_verified"], False)

    def test_role_name_collision_is_refused_even_as_source_claim(self):
        fake = _claim()
        fake["bank_custodian_id"] = fake["examiner_id"]
        with self.assertRaisesRegex(InvalidEvaluatorCustodyClaim, "role"):
            inspect_evaluator_custody_claim(_wire(fake))

    def test_caller_cannot_smuggle_gold_prompt_or_authority_fields(self):
        for field, value in (
            ("private_prompt", "synthetic placeholder"),
            ("private_gold", 1),
            ("admitted", True),
            ("authorizes_private_exam", True),
            ("source_verified", True),
        ):
            with self.subTest(field=field):
                fake = _claim()
                fake[field] = value
                with self.assertRaisesRegex(InvalidEvaluatorCustodyClaim,
                                            "exact metadata fields"):
                    inspect_evaluator_custody_claim(_wire(fake))

    def test_missing_required_commitment_is_refused(self):
        fake = _claim()
        fake.pop("q6_witness_head_sha256")
        with self.assertRaises(InvalidEvaluatorCustodyClaim):
            inspect_evaluator_custody_claim(_wire(fake))

    def test_untrusted_role_and_path_strings_are_bounded_opaque_ids(self):
        for field, value in (
            ("examiner_id", ""),
            ("examiner_id", "../../trainer"),
            ("transport_principal_id", "https://localhost:443/claim"),
            ("protected_store_id", "/secret/private-bank"),
            ("rights_steward_id", "admin\ntrainer"),
        ):
            with self.subTest(field=field, value=value):
                fake = _claim()
                fake[field] = value
                with self.assertRaises(InvalidEvaluatorCustodyClaim):
                    inspect_evaluator_custody_claim(_wire(fake))

    def test_invalid_key_receipt_and_digest_formats_are_refused(self):
        for field, value in (
            ("issuer_public_key_hex", "f" * 63),
            ("receipt_signature_hex", "f" * 126),
            ("private_bank_commitment_sha256", "G" * 64),
            ("rights_receipt_sha256", True),
            ("issuer_key_generation", True),
            ("issuer_key_generation", 0),
        ):
            with self.subTest(field=field):
                fake = _claim()
                fake[field] = value
                with self.assertRaises(InvalidEvaluatorCustodyClaim):
                    inspect_evaluator_custody_claim(_wire(fake))

    def test_duplicate_json_members_and_noncanonical_wire_are_refused(self):
        raw = _wire(_claim())
        repeated = raw.replace(b'"schema":', b'"schema":"forged","schema":', 1)
        with self.assertRaises(InvalidEvaluatorCustodyClaim):
            inspect_evaluator_custody_claim(repeated)
        with self.assertRaises(InvalidEvaluatorCustodyClaim):
            inspect_evaluator_custody_claim(raw.replace(b'"schema":', b'"schema": ', 1))
        with self.assertRaises(InvalidEvaluatorCustodyClaim):
            inspect_evaluator_custody_claim(raw + b"\n")

    def test_invalid_utf8_json_shape_and_oversize_are_refused(self):
        for raw in (b"", b"not JSON", b"[]", b"{\xff}", b"{" + b" " * 8193 + b"}"):
            with self.subTest(raw_length=len(raw)):
                with self.assertRaises(InvalidEvaluatorCustodyClaim):
                    inspect_evaluator_custody_claim(raw)

    def test_caller_supplied_metadata_cannot_prove_transport_or_external_cas(self):
        fake = _claim()
        fake["transport_principal_id"] = "root-owner-claimed"
        fake["q6_witness_head_sha256"] = "f" * 64
        report = inspect_evaluator_custody_claim(_wire(fake))
        self.assertIs(report["transport_identity_verified"], False)
        self.assertIs(report["q6_monotonic_witness_verified"], False)


if __name__ == "__main__":
    unittest.main()
