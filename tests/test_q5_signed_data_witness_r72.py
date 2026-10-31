"""A Q5 witness is source data, never executable caller-provided verifier code."""

from dataclasses import replace
from hashlib import sha256
import json
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from labyrinth.q5_signed_data_witness_r72 import (
    SignedQ5SourceWitness, Q5SignedWitnessError,
    screen_q5_with_signed_source_witness, verify_q5_signed_source_witness,
)
from tests.test_q5_evaluation_preregistration_r68 import _fixture


DOMAIN = b"labyrinth.q5-signed-preoutcome-source-witness.v1\0"
SCHEMA = "labyrinth.q5-signed-preoutcome-source-witness.v1"


def canonical(body):
    return (json.dumps(body, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


class Q5SignedDataWitnessR72Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outcomes, cls.precommit = _fixture()

    def setUp(self):
        self.signer = Ed25519PrivateKey.generate()
        self.public_key = self.signer.public_key().public_bytes(
            encoding=Encoding.Raw, format=PublicFormat.Raw)
        self.event_id = "q5-r72-campaign-1"
        self.body = {
            "schema": SCHEMA,
            "event_id": self.event_id,
            "precommit_sha256": self.precommit.precommit_sha256,
            "signer_public_key_sha256": sha256(self.public_key).hexdigest(),
            "declared_stage": "BEFORE_OUTCOMES",
        }

    def signed(self, body=None, *, body_bytes=None):
        public_bytes = canonical(body or self.body) if body_bytes is None else body_bytes
        return SignedQ5SourceWitness(public_bytes, self.signer.sign(DOMAIN + public_bytes).hex())

    def check(self, signed=None, **changes):
        kw = {
            "witness": signed or self.signed(),
            "pinned_signer_public_key": self.public_key,
            "expected_precommit_sha256": self.precommit.precommit_sha256,
            "expected_event_id": self.event_id,
            "consumed_event_sha256s": frozenset(),
        }
        kw.update(changes)
        return verify_q5_signed_source_witness(**kw)

    def test_signed_canonical_source_witness_is_not_custody_or_chronology(self):
        source = self.check()
        self.assertEqual(source.event_id, self.event_id)
        self.assertEqual(source.precommit_sha256, self.precommit.precommit_sha256)
        for name in ("external_chronology_verified", "independent_custody_verified",
                     "external_nonreplay_verified", "graduation_authorized"):
            self.assertIs(getattr(source, name), False)

    def test_signer_pin_and_source_pin_fail_closed(self):
        with self.assertRaises(Q5SignedWitnessError):
            self.check(pinned_signer_public_key=Ed25519PrivateKey.generate().public_key().public_bytes(
                encoding=Encoding.Raw, format=PublicFormat.Raw))
        with self.assertRaises(Q5SignedWitnessError):
            self.check(expected_precommit_sha256="f" * 64)
        with self.assertRaises(Q5SignedWitnessError):
            self.check(expected_event_id="another-event")

    def test_signed_retroactive_roster_does_not_match_external_precommit(self):
        changed = dict(self.body, precommit_sha256="f" * 64)
        with self.assertRaises(Q5SignedWitnessError):
            self.check(self.signed(changed))

    def test_duplicate_fields_and_noncanonical_encoding_fail_even_when_signed(self):
        duplicate = self.signed(body_bytes=(
            canonical(self.body).rstrip()[:-1] + b',"event_id":"other"}\n'))
        with self.assertRaises(Q5SignedWitnessError):
            self.check(duplicate)
        unordered = json.dumps(self.body, separators=(",", ":"),
                               ensure_ascii=True).encode("ascii") + b"\n"
        self.assertNotEqual(unordered, canonical(self.body))
        with self.assertRaises(Q5SignedWitnessError):
            self.check(self.signed(body_bytes=unordered))

    def test_nonfinite_json_tokens_fail_with_public_witness_error(self):
        # Python's json decoder accepts nonstandard NaN/Infinity by default.
        # Such signed bytes must fail *through this API's own error type*.
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                payload = json.dumps(self.body, sort_keys=True,
                                     separators=(",", ":"), ensure_ascii=True)
                payload = payload.replace('"event_id":"' + self.event_id + '"',
                                          '"event_id":' + token)
                with self.assertRaises(Q5SignedWitnessError):
                    self.check(self.signed(body_bytes=(payload + "\n").encode("ascii")))

    def test_overflowing_json_exponents_fail_with_public_witness_error(self):
        # parse_constant rejects NaN/Infinity tokens but does not intercept
        # exponent overflow, which Python decodes as +/- infinity.
        for token in ("1e9999", "-1e9999"):
            with self.subTest(token=token):
                payload = json.dumps(self.body, sort_keys=True,
                                     separators=(",", ":"), ensure_ascii=True)
                payload = payload.replace('"event_id":"' + self.event_id + '"',
                                          '"event_id":' + token)
                with self.assertRaises(Q5SignedWitnessError):
                    self.check(self.signed(body_bytes=(payload + "\n").encode("ascii")))

    def test_caller_supplied_spent_event_set_rejects_seen_event_but_is_not_host_replay_authority(self):
        event_digest = sha256((SCHEMA + "\0" + self.event_id).encode("ascii")).hexdigest()
        with self.assertRaisesRegex(Q5SignedWitnessError, "consumed"):
            self.check(consumed_event_sha256s=frozenset((event_digest,)))

    def test_q5_statistical_screen_consumes_signed_data_without_an_executable_caller_callback(self):
        result = screen_q5_with_signed_source_witness(
            precommit=self.precommit,
            external_expected_precommit_sha256=self.precommit.precommit_sha256,
            witness=self.signed(),
            pinned_signer_public_key=self.public_key,
            expected_event_id=self.event_id,
            consumed_event_sha256s=frozenset(),
            outcomes=self.outcomes,
        )
        self.assertTrue(result["source_screen_pass"])
        self.assertIs(result["signed_witness_source_signature_verified"], True)
        self.assertIs(result["independent_preoutcome_witness_custody_verified"], False)
        self.assertIs(result["graduation_authorized"], False)
        self.assertEqual(result["witness_source_record"].event_id, self.event_id)

    def test_q5_wrapper_rejects_forged_signature_and_mutated_source_roster(self):
        signed = self.signed()
        forged = replace(signed, signature_hex="00" * 64)
        with self.assertRaises(Q5SignedWitnessError):
            screen_q5_with_signed_source_witness(
                precommit=self.precommit,
                external_expected_precommit_sha256=self.precommit.precommit_sha256,
                witness=forged, pinned_signer_public_key=self.public_key,
                expected_event_id=self.event_id,
                consumed_event_sha256s=frozenset(), outcomes=self.outcomes)
        with self.assertRaises(Q5SignedWitnessError):
            screen_q5_with_signed_source_witness(
                precommit=self.precommit,
                external_expected_precommit_sha256="f" * 64,
                witness=signed, pinned_signer_public_key=self.public_key,
                expected_event_id=self.event_id,
                consumed_event_sha256s=frozenset(), outcomes=self.outcomes)


if __name__ == "__main__":
    unittest.main()
