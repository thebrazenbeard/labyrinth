"""Adversarial source-only Q7 witness intake; no host or sealed exam bank."""

import hashlib
import json
import unittest
from dataclasses import replace

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from labyrinth.final_exam_precommit import SCHEMA as PRECOMMIT_SCHEMA
from labyrinth.final_exam_precommit import SignedFinalExamPrecommit
from labyrinth.q7_temporal_custody_source import create_q7_source_phase, receipt_sha256
from labyrinth.q7_witnessed_cas_intake_r93 import (
    Q7WitnessError, PinnedQ7Witness, inspect_q7_witnessed_source,
    q7_witness_next_head,
)


def _wire(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _public(private):
    return private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


class WitnessedQ7SourceTests(unittest.TestCase):
    def setUp(self):
        self.pre_signer = Ed25519PrivateKey.generate()
        self.anchor_signer = Ed25519PrivateKey.generate()
        self.training_signer = Ed25519PrivateKey.generate()
        self.witness_signer = Ed25519PrivateKey.generate()
        self.pre_public = _public(self.pre_signer)
        body = {
            "schema": PRECOMMIT_SCHEMA, "exam_id": "synthetic-exam-r93",
            "event_id": "synthetic-event-r93", "objective_id": "synthetic.objective",
            "scope_sha256": "a" * 64, "sealed_plan_sha256": "b" * 64,
            "source_sha256": "c" * 64, "training_inventory_sha256": "d" * 64,
            "signer_public_key_sha256": hashlib.sha256(self.pre_public).hexdigest(),
            "external_anchor_verified": False, "host_custody_verified": False,
            "physical_model_execution_verified": False, "graduation_authorized": False,
        }
        pre_raw = _wire(body)
        self.precommit = SignedFinalExamPrecommit(
            pre_raw, self.pre_signer.sign(
                b"labyrinth.final-exam-pretraining-precommit.v1\0" + pre_raw).hex())
        psha = hashlib.sha256(pre_raw).hexdigest()
        self.anchor = create_q7_source_phase(
            event_id=body["event_id"], precommit_sha256=psha,
            phase="anchor", prior_receipt_sha256="0" * 64,
            signing_key=self.anchor_signer)
        self.start = create_q7_source_phase(
            event_id=body["event_id"], precommit_sha256=psha,
            phase="training_start", prior_receipt_sha256=receipt_sha256(self.anchor),
            signing_key=self.training_signer)
        self.anchor_sha = receipt_sha256(self.anchor)
        self.start_sha = receipt_sha256(self.start)
        self.prior_index = 41
        self.prior_head = "f" * 64
        self.anchor_head = q7_witness_next_head(self.prior_head, 42, self.anchor_sha)
        self.next_head = q7_witness_next_head(self.anchor_head, 43, self.start_sha)
        self.pins = PinnedQ7Witness(
            host_id="host.example", exam_id=body["exam_id"],
            event_id=body["event_id"], objective_id=body["objective_id"],
            scope_sha256=body["scope_sha256"],
            sealed_plan_sha256=body["sealed_plan_sha256"],
            source_sha256=body["source_sha256"],
            training_inventory_sha256=body["training_inventory_sha256"],
            precommit_signer_key=self.pre_public,
            anchor_signer_key=_public(self.anchor_signer),
            training_signer_key=_public(self.training_signer),
            witness_signer_key=_public(self.witness_signer),
            anchor_receipt_sha256=self.anchor_sha,
            training_start_receipt_sha256=self.start_sha,
            expected_before_index=self.prior_index,
            expected_before_head_sha256=self.prior_head,
            expected_after_head_sha256=self.next_head,
            consumed_event_sha256s=frozenset(),
        )
        self.claims = {
            "schema": "labyrinth.q7.witnessed-source-chain.v1",
            "host_id": self.pins.host_id,
            "exam_id": self.pins.exam_id,
            "event_id": self.pins.event_id,
            "precommit_sha256": psha,
            "anchor_receipt_sha256": self.anchor_sha,
            "training_start_receipt_sha256": self.start_sha,
            "before_index": self.prior_index,
            "anchor_index": self.prior_index + 1,
            "after_index": self.prior_index + 2,
            "before_head_sha256": self.prior_head,
            "anchor_head_sha256": self.anchor_head,
            "after_head_sha256": self.next_head,
            "custody_verified": False,
            "graduation_authorized": False,
        }

    def _signed(self, claims=None, signer=None):
        body = self.claims if claims is None else claims
        signed = dict(body)
        signed["witness_signature_hex"] = (signer or self.witness_signer).sign(
            b"labyrinth.q7.witnessed-source-chain.v1\0" + _wire(body)).hex()
        return _wire(signed)

    def _inspect(self, *, raw=None, pins=None, anchor=None, start=None,
                 precommit=None):
        return inspect_q7_witnessed_source(
            self._signed() if raw is None else raw,
            pins=self.pins if pins is None else pins,
            precommit=self.precommit if precommit is None else precommit,
            anchor=self.anchor if anchor is None else anchor,
            training_start=self.start if start is None else start)

    def test_signed_source_only_chain_accepts_and_stays_nonauthorizing(self):
        result = self._inspect()
        self.assertEqual(result.classification, "SIGNED_Q7_CHAIN_SOURCE_ONLY_NO_TEMPORAL_CUSTODY")
        self.assertEqual(result.before_index, 41)
        self.assertEqual(result.after_index, 43)
        self.assertEqual(result.witness_sha256, hashlib.sha256(self._signed()).hexdigest())
        self.assertFalse(result.temporal_custody_verified)
        self.assertFalse(result.graduation_authorized)
        for field in ("temporal_custody_verified", "graduation_authorized"):
            with self.assertRaises(AttributeError):
                object.__setattr__(result, field, True)
        self.assertNotIn(b"private_bank", self._signed())

    def test_head_algorithm_has_independently_calculated_exact_vector(self):
        expected_anchor = hashlib.sha256(
            b"labyrinth.q7.witnessed-chain-head.v1\0"
            + bytes.fromhex("f" * 64)
            + (42).to_bytes(8, "big")
            + bytes.fromhex(self.anchor_sha)
        ).hexdigest()
        self.assertEqual(self.anchor_head, expected_anchor)
        expected_after = hashlib.sha256(
            b"labyrinth.q7.witnessed-chain-head.v1\0"
            + bytes.fromhex(expected_anchor)
            + (43).to_bytes(8, "big")
            + bytes.fromhex(self.start_sha)
        ).hexdigest()
        self.assertEqual(self.next_head, expected_after)

    def test_replay_detected_when_external_consumed_roster_is_present(self):
        event_digest = hashlib.sha256((
            PRECOMMIT_SCHEMA + "\0" + self.pins.event_id).encode("ascii")).hexdigest()
        with self.assertRaisesRegex(Q7WitnessError, "precommit|consumed"):
            self._inspect(pins=replace(self.pins,
                                       consumed_event_sha256s=frozenset({event_digest})))

    def test_cloned_external_head_cannot_be_proven_here(self):
        # A copied empty external state allows identical claims to pass. The
        # verifier must never call such a source-only replay temporal custody.
        first, replay = self._inspect(), self._inspect()
        self.assertEqual(first, replay)
        self.assertFalse(first.temporal_custody_verified)

    def test_distinct_witness_key_cannot_be_collapsed_after_construction(self):
        pins = replace(self.pins)
        object.__setattr__(pins, "witness_signer_key", self.pins.anchor_signer_key)
        with self.assertRaisesRegex(Q7WitnessError, "distinct"):
            self._inspect(pins=pins)

    def test_owner_scope_and_signer_substitution_fail_closed(self):
        for changed in ("host_id", "exam_id", "event_id", "objective_id"):
            with self.subTest(field=changed):
                with self.assertRaises(Q7WitnessError):
                    self._inspect(pins=replace(self.pins, **{changed: "foreign-id"}))
        with self.assertRaises(Q7WitnessError):
            self._inspect(pins=replace(self.pins,
                                       witness_signer_key=_public(Ed25519PrivateKey.generate())))

    def test_mismatched_host_head_or_nonmonotonic_sequence_rejected(self):
        with self.assertRaisesRegex(Q7WitnessError, "head|index"):
            self._inspect(pins=replace(self.pins, expected_before_head_sha256="0" * 64))
        for changed in ({"anchor_index": 41}, {"after_index": 44},
                        {"anchor_head_sha256": "e" * 64}):
            with self.subTest(changed=changed):
                with self.assertRaises(Q7WitnessError):
                    self._inspect(raw=self._signed({**self.claims, **changed}))

    def test_swapped_or_replayed_phase_receipt_rejected_even_if_resigned(self):
        with self.assertRaises(Q7WitnessError):
            self._inspect(start=self.anchor)
        with self.assertRaises(Q7WitnessError):
            self._inspect(pins=replace(self.pins, training_start_receipt_sha256="e" * 64))

    def test_unsigned_tamper_forged_signer_and_authority_flag_rejected(self):
        parsed = json.loads(self._signed())
        parsed["before_index"] += 1
        with self.assertRaises(Q7WitnessError):
            self._inspect(raw=_wire(parsed))
        foreign = Ed25519PrivateKey.generate()
        with self.assertRaises(Q7WitnessError):
            self._inspect(raw=self._signed(signer=foreign))
        with self.assertRaises(Q7WitnessError):
            self._inspect(raw=self._signed({**self.claims, "graduation_authorized": True}))

    def test_noncanonical_duplicate_json_field_and_positive_host_claim_rejected(self):
        raw = self._signed().replace(b'"host_id":"host.example",',
                                     b'"host_id":"host.example","host_id":"host.example",')
        with self.assertRaises(Q7WitnessError):
            self._inspect(raw=raw)
        with self.assertRaises(Q7WitnessError):
            self._inspect(raw=self._signed({**self.claims, "custody_verified": True}))
        with self.assertRaises(Q7WitnessError):
            self._inspect(raw=b'{}')

    def test_resigned_float_overflow_and_boolean_index_rejected(self):
        for index in (True, -1, 2**63, 41.5):
            with self.subTest(index=index), self.assertRaises(Q7WitnessError):
                self._inspect(raw=self._signed({**self.claims, "before_index": index}))
        raw = self._signed().replace(b'"before_index":41', b'"before_index":1e309')
        self.assertNotEqual(raw, self._signed())
        with self.assertRaises(Q7WitnessError):
            self._inspect(raw=raw)

    def test_resigned_nonfinite_precommit_flag_rejected_as_q7_witness_error(self):
        # A signed precommit with a malformed JSON number must never escape
        # the public Q7 API as an unrelated ValueError/OverflowError.
        for invalid in (b"1e309", b"NaN", b"Infinity", b"-Infinity"):
            with self.subTest(invalid=invalid):
                malformed = self.precommit.public_bytes.replace(
                    b'"external_anchor_verified":false',
                    b'"external_anchor_verified":' + invalid,
                )
                self.assertNotEqual(malformed, self.precommit.public_bytes)
                changed = replace(
                    self.precommit, public_bytes=malformed,
                    signature_hex=self.pre_signer.sign(
                        b"labyrinth.final-exam-pretraining-precommit.v1\0"
                        + malformed).hex(),
                )
                with self.assertRaises(Q7WitnessError):
                    self._inspect(precommit=changed)


if __name__ == "__main__":
    unittest.main()
