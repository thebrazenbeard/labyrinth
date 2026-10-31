"""Synthetic source-only Q7 phase-chain attacks; NO real host custody."""

from dataclasses import replace
from hashlib import sha256
import json
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from labyrinth.final_exam_precommit import SCHEMA as PRECOMMIT_SCHEMA
from labyrinth.final_exam_precommit import SignedFinalExamPrecommit
from labyrinth.q7_temporal_custody_source import (
    Q7SourceOrderError, SignedQ7SourcePhase, create_q7_source_phase,
    receipt_sha256, verify_q7_source_phase_order,
)


def _public_key(private: Ed25519PrivateKey) -> bytes:
    return private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def _json_bytes(body: dict) -> bytes:
    return (json.dumps(body, sort_keys=True, separators=(",", ":"),
                       allow_nan=False, ensure_ascii=True) + "\n").encode("ascii")


class Q7TemporalSourceTests(unittest.TestCase):
    def setUp(self):
        self.precommit_signer = Ed25519PrivateKey.generate()
        self.anchor_signer = Ed25519PrivateKey.generate()
        self.training_signer = Ed25519PrivateKey.generate()
        self.public_precommit = _public_key(self.precommit_signer)
        self.public_anchor = _public_key(self.anchor_signer)
        self.public_training = _public_key(self.training_signer)
        body = dict(
            schema=PRECOMMIT_SCHEMA,
            exam_id="synthetic-exam", event_id="synthetic-event",
            objective_id="objective.synthetic", scope_sha256="a" * 64,
            sealed_plan_sha256="b" * 64,
            source_sha256="c" * 64,
            training_inventory_sha256="d" * 64,
            signer_public_key_sha256=sha256(self.public_precommit).hexdigest(),
            external_anchor_verified=False, host_custody_verified=False,
            physical_model_execution_verified=False,
            graduation_authorized=False,
        )
        raw = _json_bytes(body)
        self.precommit = SignedFinalExamPrecommit(
            raw, self.precommit_signer.sign(
                b"labyrinth.final-exam-pretraining-precommit.v1\0" + raw).hex())
        self.precommit_hash = sha256(raw).hexdigest()
        self.anchor = create_q7_source_phase(
            event_id="synthetic-event", precommit_sha256=self.precommit_hash,
            phase="anchor", prior_receipt_sha256="0" * 64,
            signing_key=self.anchor_signer)
        self.training_start = create_q7_source_phase(
            event_id="synthetic-event", precommit_sha256=self.precommit_hash,
            phase="training_start", prior_receipt_sha256=receipt_sha256(self.anchor),
            signing_key=self.training_signer)

    def verify(self, **overrides):
        kwargs = dict(
            pinned_precommit_signer_key=self.public_precommit,
            expected_exam_id="synthetic-exam",
            expected_event_id="synthetic-event",
            expected_objective_id="objective.synthetic",
            expected_scope_sha256="a" * 64,
            expected_sealed_plan_sha256="b" * 64,
            expected_source_sha256="c" * 64,
            expected_training_inventory_sha256="d" * 64,
            consumed_event_sha256s=frozenset(),
            anchor=self.anchor,
            pinned_anchor_key=self.public_anchor,
            expected_anchor_receipt_sha256=receipt_sha256(self.anchor),
            training_start=self.training_start,
            pinned_training_owner_key=self.public_training,
            expected_training_start_receipt_sha256=receipt_sha256(self.training_start),
        )
        kwargs.update(overrides)
        return verify_q7_source_phase_order(self.precommit, **kwargs)

    def test_public_two_phase_source_order_does_not_graduate(self):
        result = self.verify()
        self.assertFalse(result.temporal_custody_verified)
        self.assertFalse(result.graduation_authorized)
        self.assertEqual(result.anchor_receipt_sha256, receipt_sha256(self.anchor))
        self.assertEqual(result.training_start_receipt_sha256,
                         receipt_sha256(self.training_start))
        for sensitive in ("secret-case", "private-prompt", "gold_label", "answer"):
            self.assertNotIn(sensitive, self.anchor.public_bytes.decode())
            self.assertNotIn(sensitive, self.training_start.public_bytes.decode())

    def test_tampered_or_replayed_phase_fails_closed(self):
        tampered = replace(self.training_start,
                          public_bytes=self.training_start.public_bytes.replace(
                              b"training_start", b"anchor"))
        with self.assertRaises(Q7SourceOrderError):
            self.verify(training_start=tampered,
                        expected_training_start_receipt_sha256=receipt_sha256(tampered))
        with self.assertRaises(Q7SourceOrderError):
            self.verify(expected_event_id="synthetic-other")
        with self.assertRaises(Q7SourceOrderError):
            self.verify(training_start=self.anchor,
                        expected_training_start_receipt_sha256=receipt_sha256(self.anchor))

    def test_external_digest_and_signer_are_not_taken_from_payload(self):
        with self.assertRaisesRegex(Q7SourceOrderError, "external expected"):
            self.verify(expected_anchor_receipt_sha256="f" * 64)
        with self.assertRaisesRegex(Q7SourceOrderError, "external expected"):
            self.verify(expected_training_start_receipt_sha256="f" * 64)
        with self.assertRaises(Q7SourceOrderError):
            self.verify(pinned_anchor_key=self.public_training)

    def test_self_authorization_and_signer_reuse_rejected(self):
        with self.assertRaises(Q7SourceOrderError):
            self.verify(pinned_training_owner_key=self.public_anchor)
        with self.assertRaises(Q7SourceOrderError):
            self.verify(anchor=replace(self.anchor, graduation_authorized=True))
        with self.assertRaises(Q7SourceOrderError):
            self.verify(training_start=replace(self.training_start, graduation_authorized=True))

    def test_fake_ordering_and_wrong_predecessor_are_detected(self):
        bad = create_q7_source_phase(
            event_id="synthetic-event", precommit_sha256=self.precommit_hash,
            phase="training_start", prior_receipt_sha256="f" * 64,
            signing_key=self.training_signer)
        with self.assertRaisesRegex(Q7SourceOrderError, "phase chain"):
            self.verify(training_start=bad,
                        expected_training_start_receipt_sha256=receipt_sha256(bad))
        with self.assertRaises(Q7SourceOrderError):
            create_q7_source_phase(
                event_id="synthetic-event", precommit_sha256=self.precommit_hash,
                phase="training_start", prior_receipt_sha256="0" * 64,
                signing_key=self.training_signer)

    def test_duplicate_fields_and_noncanonical_signed_receipts_rejected(self):
        raw = self.anchor.public_bytes.replace(
            b'"event_id":"synthetic-event",',
            b'"event_id":"synthetic-event","event_id":"synthetic-event",')
        forged = SignedQ7SourcePhase(
            raw, self.anchor_signer.sign(
                b"labyrinth.q7-public-phase-source.v1\0" + raw).hex())
        with self.assertRaises(Q7SourceOrderError):
            self.verify(anchor=forged,
                        expected_anchor_receipt_sha256=receipt_sha256(forged))

    def test_resigned_overflowed_json_sequence_fails_with_q7_error(self):
        # The signer and caller can collude; malformed signed source must not
        # leak a raw ValueError after JSON parses 1e309 as floating infinity.
        for bad_token in (b"1e309", b"NaN"):
            with self.subTest(token=bad_token):
                raw = self.anchor.public_bytes.replace(
                    b'"sequence":1}', b'"sequence":' + bad_token + b'}', 1)
                self.assertNotEqual(raw, self.anchor.public_bytes)
                forged = SignedQ7SourcePhase(
                    raw, self.anchor_signer.sign(
                        b"labyrinth.q7-public-phase-source.v1\0" + raw).hex())
                with self.assertRaises(Q7SourceOrderError):
                    self.verify(anchor=forged,
                                expected_anchor_receipt_sha256=receipt_sha256(forged))

    def test_valid_cryptography_does_not_prove_real_time(self):
        # Both records can be signed after training by an actor controlling
        # both keys. This source verifier MUST not claim independent time.
        replay = self.verify()
        self.assertFalse(replay.temporal_custody_verified)
        self.assertFalse(replay.graduation_authorized)


if __name__ == "__main__":
    unittest.main()
