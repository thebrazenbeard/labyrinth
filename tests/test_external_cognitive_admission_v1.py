"""Synthetic signature fixtures; none represents real TRAIN rights or evaluation custody."""

import hashlib
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from unittest import TestCase

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from labyrinth.cognitive_rights_candidate_v1 import prepare_private_cognitive_steward_manifest
from labyrinth.cognitive_train_dataset_export import export_cognitive_train_jsonl
from labyrinth.external_cognitive_admission_v1 import (
    ExternalAdmissionHeld,
    signing_bytes_for_external_claim,
    verify_external_cognitive_admission,
)
from labyrinth.journey import begin_journey
from tests.test_cognitive_objective_training_run import cognitive_contract


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode()


class ExternalAdmissionConsumerTests(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        # Generated only in a synthetic test. This is NOT a rights issuer or grant.
        self.steward_key = Ed25519PrivateKey.generate()
        self.evaluator_key = Ed25519PrivateKey.generate()
        self.anchors_path = self.work / "installed-outside-source-trust-anchors.json"
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        self.anchors_path.write_bytes(canonical({
            "schema": "labyrinth.external-cognitive-trust-anchors.v1",
            "steward": {"id": "synthetic-independent-steward",
                        "ed25519_public_key_hex": self.steward_key.public_key().public_bytes(
                            Encoding.Raw, PublicFormat.Raw).hex()},
            "evaluator": {"id": "synthetic-protected-evaluator",
                          "ed25519_public_key_hex": self.evaluator_key.public_key().public_bytes(
                              Encoding.Raw, PublicFormat.Raw).hex()},
        }))
        self.anchors_path.chmod(0o600)
        self.nonce_path = self.work / "nonces.sqlite3"
        contract = cognitive_contract(seed=1201)
        self.dataset = export_cognitive_train_jsonl(
            contract, begin_journey(contract.scope, checkpoint_sha256="1" * 64)
        )
        self.exclusions = ("f" * 64, "e" * 64)
        self.roster_digest = hashlib.sha256(canonical({
            "schema": "labyrinth.protected-semantic-exclusions.v1",
            "complete": True,
            "semantic_sha256": sorted(self.exclusions),
        })).hexdigest()
        self.source_commit = "a" * 40
        self.source_digest = "b" * 64
        self.private = prepare_private_cognitive_steward_manifest(
            self.dataset,
            source_repo="thebrazenbeard/labyrinth",
            source_commit=self.source_commit,
            source_file_sha256=self.source_digest,
            exclusion_semantic_sha256=self.exclusions,
            exclusion_roster_sha256=self.roster_digest,
        )
        self.now = int(time.time())
        subject = json.loads(self.private.private_bytes())["source"]
        self.rights_evidence = {
            "source_authorship": b"SYNTHETIC source-review exhibit, NOT permission",
            "license": b"SYNTHETIC license-review exhibit, NOT permission",
            "consent": b"SYNTHETIC consent-review exhibit, NOT permission",
            "privacy_review": b"SYNTHETIC privacy exhibit, NOT permission",
        }
        self.steward_claim = {
            "schema": "labyrinth.external-rights-attestation.v1",
            "issuer_id": "synthetic-independent-steward",
            "nonce": "steward-unique-0001",
            "issued_at": self.now - 10,
            "expires_at": self.now + 3600,
            "source": {k: subject[k] for k in (
                "repository", "producer_commit", "producer_file_sha256",
                "dataset_sha256", "objective_id", "contract_sha256", "cycle_index")},
            "private_manifest_sha256": hashlib.sha256(self.private.private_bytes()).hexdigest(),
            "protected_roster_sha256": self.roster_digest,
            "rights": {"permission": "GRANTED_FOR_CANDIDATE",
                       "source_authorship_evidence_ref": "private://source-review/001",
                       "license_identifier": "PRIVATE-LICENSE-REVIEW-001",
                       "license_evidence_ref": "private://license/001",
                       "consent_evidence_ref": "private://consent/001",
                       "privacy_review_evidence_ref": "private://privacy/001",
                       "evidence_sha256": {name: hashlib.sha256(value).hexdigest()
                                           for name, value in self.rights_evidence.items()}},
        }
        self.evaluator_claim = {
            "schema": "labyrinth.external-protected-exclusion-attestation.v1",
            "issuer_id": "synthetic-protected-evaluator",
            "nonce": "evaluator-unique-0001",
            "issued_at": self.now - 10,
            "expires_at": self.now + 3600,
            "dataset_sha256": self.dataset.sha256,
            "protected_roster_sha256": self.roster_digest,
            "protected_roster_count": len(self.exclusions),
            "roster_status": "SEALED_COMPLETE",
            "cross_system_exposure_ledger": "REVIEWED_NO_TRAIN_EXPOSURE",
        }

    def _call(self, **overrides):
        steward_bytes = canonical(overrides.pop("steward_claim", self.steward_claim))
        evaluator_bytes = canonical(overrides.pop("evaluator_claim", self.evaluator_claim))
        steward_sig = self.steward_key.sign(signing_bytes_for_external_claim(
            "steward", steward_bytes)).hex()
        evaluator_sig = self.evaluator_key.sign(signing_bytes_for_external_claim(
            "evaluator", evaluator_bytes)).hex()
        params = dict(
            dataset=self.dataset, private_steward_manifest=self.private,
            private_rights_evidence=self.rights_evidence,
            exclusion_semantic_sha256=self.exclusions,
            trust_anchor_path=self.anchors_path, nonce_registry_path=self.nonce_path,
            steward_claim_bytes=steward_bytes, steward_signature_hex=steward_sig,
            evaluator_claim_bytes=evaluator_bytes, evaluator_signature_hex=evaluator_sig,
            expected_source_commit=self.source_commit,
            expected_source_file_sha256=self.source_digest, now=self.now,
        )
        params.update(overrides)
        return verify_external_cognitive_admission(**params)

    def test_valid_synthetic_signers_only_verify_private_evidence_not_training(self):
        result = self._call()
        self.assertTrue(result.private_evidence_verified())
        self.assertFalse(result.authorizes_training)
        self.assertFalse(result.authorizes_host_execution)
        self.assertNotIn(self.dataset.sha256, repr(result))
        self.assertNotIn(self.roster_digest, repr(result))
        with self.assertRaises(TypeError):
            asdict(result)

    def test_missing_trust_anchor_and_missing_signature_hold(self):
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(trust_anchor_path=self.work / "missing.json")
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(steward_signature_hex="00" * 64)
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(evaluator_signature_hex="00" * 64)

    def test_tampering_signed_bytes_or_claimed_signer_or_wrong_issuer_hold(self):
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(steward_claim_bytes=b"{}")
        forged = dict(self.steward_claim, issuer_id="untrusted-self-issuer")
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(steward_claim=forged)
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(steward_signature_hex=self.evaluator_key.sign(
                signing_bytes_for_external_claim("steward", canonical(self.steward_claim))).hex())

    def test_signed_wrong_corpus_revision_manifest_or_roster_hold(self):
        for mutation in (
            dict(self.steward_claim, source={**self.steward_claim["source"],
                                            "producer_commit": "c" * 40}),
            dict(self.steward_claim, source={**self.steward_claim["source"],
                                            "dataset_sha256": "d" * 64}),
            dict(self.steward_claim, private_manifest_sha256="e" * 64),
            dict(self.steward_claim, protected_roster_sha256="0" * 64),
        ):
            with self.assertRaises(ExternalAdmissionHeld):
                self._call(steward_claim=mutation)
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(evaluator_claim=dict(self.evaluator_claim, protected_roster_count=3))

    def test_expiry_future_issued_and_replay_hold(self):
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(steward_claim=dict(self.steward_claim, expires_at=self.now - 1))
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(evaluator_claim=dict(self.evaluator_claim, issued_at=self.now + 300))
        self._call()
        with self.assertRaises(ExternalAdmissionHeld):
            self._call()

    def test_protected_overlap_and_absent_consent_hold(self):
        row = json.loads(self.dataset.jsonl_bytes.splitlines()[0])
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(exclusion_semantic_sha256=(row["semantic_sha256"],))
        bad_rights = {**self.steward_claim["rights"], "consent_evidence_ref": ""}
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(steward_claim=dict(self.steward_claim, rights=bad_rights))
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(private_rights_evidence={**self.rights_evidence,
                                                "consent": b"tampered evidence"})

    def test_evaluator_missing_ledger_completeness_or_wrong_dataset_hold(self):
        for mutated in (
            dict(self.evaluator_claim, cross_system_exposure_ledger="NOT_REVIEWED"),
            dict(self.evaluator_claim, roster_status="INCOMPLETE"),
            dict(self.evaluator_claim, dataset_sha256="0" * 64),
            dict(self.evaluator_claim, protected_roster_sha256="1" * 64),
        ):
            with self.assertRaises(ExternalAdmissionHeld):
                self._call(evaluator_claim=mutated)

    def test_duplicate_json_keys_and_role_signature_substitution_hold(self):
        raw = canonical(self.steward_claim)
        duplicate = raw[:-1] + b',"nonce":"spoofed-nonce-value"}'
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(steward_claim_bytes=duplicate)
        role_wrong_signature = self.steward_key.sign(
            signing_bytes_for_external_claim("evaluator", raw)).hex()
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(steward_signature_hex=role_wrong_signature)

    def test_same_signer_for_two_roles_and_repo_managed_pins_hold(self):
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        same_key = self.steward_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw).hex()
        bad_anchor = {"schema": "labyrinth.external-cognitive-trust-anchors.v1",
                      "steward": {"id": "synthetic-independent-steward",
                                  "ed25519_public_key_hex": same_key},
                      "evaluator": {"id": "synthetic-protected-evaluator",
                                    "ed25519_public_key_hex": same_key}}
        self.anchors_path.write_bytes(canonical(bad_anchor))
        with self.assertRaises(ExternalAdmissionHeld):
            self._call()
        self.anchors_path.chmod(0o666)
        with self.assertRaises(ExternalAdmissionHeld):
            self._call()

    def test_expiry_cannot_be_bypassed_by_reusing_only_one_nonce(self):
        self._call()
        second_steward = dict(self.steward_claim, nonce="steward-unique-0002")
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(steward_claim=second_steward)
        second_evaluator = dict(self.evaluator_claim, nonce="evaluator-unique-0002")
        # Failed two-nonce insert must roll back the new steward nonce.
        receipt = self._call(steward_claim=second_steward, evaluator_claim=second_evaluator)
        self.assertTrue(receipt.private_evidence_verified())

    def test_private_manifest_mismatch_holds_even_with_valid_signed_claims(self):
        other = prepare_private_cognitive_steward_manifest(
            self.dataset,
            source_repo="thebrazenbeard/labyrinth",
            source_commit="c" * 40,
            source_file_sha256=self.source_digest,
            exclusion_semantic_sha256=self.exclusions,
            exclusion_roster_sha256=self.roster_digest,
        )
        with self.assertRaises(ExternalAdmissionHeld):
            self._call(private_steward_manifest=other)
