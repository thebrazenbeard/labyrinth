"""Source-only cryptographic intake checks; these are not physical Q8 tests."""

import hashlib
import json
from dataclasses import replace
from unittest import TestCase

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from labyrinth.q8_host_authority_probe import (
    PinnedQ8Authorities, Q8EvidenceError, REQUIRED_CASES, SCHEMA,
    inspect_signed_q8_evidence,
)


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True) + "\n").encode("ascii")


def _public(key):
    return key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)


class Q8SourceIntakeTests(TestCase):
    def setUp(self):
        self.operator = Ed25519PrivateKey.generate()
        self.witness = Ed25519PrivateKey.generate()
        self.pins = PinnedQ8Authorities(
            host_id="independent-host.test", source_sha256="1" * 64,
            audit_bundle_sha256="2" * 64,
            operator_public_key=_public(self.operator), witness_public_key=_public(self.witness),
        )
        self.claims = {
            "schema": SCHEMA, "host_id": self.pins.host_id,
            "source_sha256": self.pins.source_sha256,
            "audit_bundle_sha256": self.pins.audit_bundle_sha256,
            "observations": [self._row(case) for case in REQUIRED_CASES],
        }

    @staticmethod
    def _row(case):
        outcome = ({"Q8-A5": "reconciled", "Q8-A9": "one_winner",
                    "Q8-A10": "fail_closed"}).get(case, "refused")
        changed = outcome in ("one_winner", "reconciled")
        return {"case_id": case, "outcome": outcome,
                "before_index": 8, "after_index": 9 if changed else 8,
                "before_head_sha256": "a" * 64,
                "after_head_sha256": "b" * 64 if changed else "a" * 64,
                "trace_sha256": hashlib.sha256(case.encode("ascii")).hexdigest()}

    def _signed(self, *, operator=None, witness=None, claims=None):
        body = self.claims if claims is None else claims
        payload = _canonical(body)
        result = dict(body)
        result["operator_signature_hex"] = (operator or self.operator).sign(
            b"labyrinth.q8.operator-observation.v1\0" + payload).hex()
        result["witness_signature_hex"] = (witness or self.witness).sign(
            b"labyrinth.q8.witness-observation.v1\0" + payload).hex()
        return _canonical(result)

    def test_genuine_distinct_signatures_validate_only_evidence_envelope(self):
        result = inspect_signed_q8_evidence(self._signed(), self.pins)
        self.assertEqual(set(result.observed_cases), set(REQUIRED_CASES))
        self.assertEqual(len(result.evidence_sha256), 64)
        self.assertEqual(result.classification, "SIGNED_SOURCE_ENVELOPE_NOT_PHYSICAL_Q8")
        self.assertFalse(hasattr(result, "graduation_authorized"))
        self.assertFalse(hasattr(result, "deployment_authorized"))
        with self.assertRaises(AttributeError):
            object.__setattr__(result, "classification", "Q8_PASS")

    def test_independent_cases_may_reset_to_same_witness_head(self):
        # Q8-A5 and Q8-A9 are separate experiments; neither this evidence
        # schema nor the Q8 test contract says they share one sequential CAS.
        observations = self.claims["observations"]
        a5 = next(row for row in observations if row["case_id"] == "Q8-A5")
        a9 = next(row for row in observations if row["case_id"] == "Q8-A9")
        for field in ("before_index", "before_head_sha256", "after_index",
                      "after_head_sha256"):
            self.assertEqual(a5[field], a9[field])
        self.assertNotEqual(a5["trace_sha256"], a9["trace_sha256"])
        result = inspect_signed_q8_evidence(self._signed(), self.pins)
        self.assertEqual(result.classification, "SIGNED_SOURCE_ENVELOPE_NOT_PHYSICAL_Q8")

    def test_resigned_cross_case_trace_reuse_rejected(self):
        # Valid signatures cannot turn one recorded trace into two independent
        # observations, regardless of whether the claimed results agree.
        for first, second in (("Q8-A1", "Q8-A2"),
                              ("Q8-A5", "Q8-A9"),
                              ("Q8-A1", "Q8-A9")):
            with self.subTest(first=first, second=second):
                observations = [dict(row) for row in self.claims["observations"]]
                source = next(row for row in observations if row["case_id"] == first)
                target = next(row for row in observations if row["case_id"] == second)
                target["trace_sha256"] = source["trace_sha256"]
                claims = dict(self.claims, observations=observations)
                with self.assertRaisesRegex(Q8EvidenceError, "duplicate.*trace"):
                    inspect_signed_q8_evidence(self._signed(claims=claims), self.pins)

    def test_resigned_ten_cases_cannot_all_reuse_one_trace(self):
        observations = [dict(row) for row in self.claims["observations"]]
        for row in observations:
            row["trace_sha256"] = "c" * 64
        claims = dict(self.claims, observations=observations)
        with self.assertRaisesRegex(Q8EvidenceError, "duplicate.*trace"):
            inspect_signed_q8_evidence(self._signed(claims=claims), self.pins)

    def test_cross_role_signature_swap_rejected(self):
        obj = json.loads(self._signed())
        obj["operator_signature_hex"], obj["witness_signature_hex"] = (
            obj["witness_signature_hex"], obj["operator_signature_hex"])
        with self.assertRaisesRegex(Q8EvidenceError, "signature"):
            inspect_signed_q8_evidence(_canonical(obj), self.pins)

    def test_foreign_signer_and_forged_own_pin_rejected(self):
        foreign = Ed25519PrivateKey.generate()
        with self.assertRaisesRegex(Q8EvidenceError, "signature"):
            inspect_signed_q8_evidence(self._signed(operator=foreign), self.pins)
        with self.assertRaisesRegex(Q8EvidenceError, "distinct"):
            replace(self.pins, witness_public_key=self.pins.operator_public_key)

    def test_post_creation_pin_mutation_cannot_collapse_distinct_roles(self):
        # frozen=True only guards normal setattr; a hostile caller can still
        # rewrite fields via object.__setattr__. Recheck a local snapshot at
        # verification entry, before accepting any signature as independent.
        object.__setattr__(self.pins, "witness_public_key", self.pins.operator_public_key)
        with self.assertRaisesRegex(Q8EvidenceError, "distinct"):
            inspect_signed_q8_evidence(self._signed(witness=self.operator), self.pins)

    def test_changed_source_audit_and_host_pins_fail(self):
        for name, fake in (("host_id", "other-host"),
                           ("source_sha256", "f" * 64),
                           ("audit_bundle_sha256", "e" * 64)):
            with self.subTest(field=name):
                claims = dict(self.claims, **{name: fake})
                with self.assertRaisesRegex(Q8EvidenceError, "pin mismatch"):
                    inspect_signed_q8_evidence(self._signed(claims=claims), self.pins)

    def test_payload_mutation_after_signing_fails(self):
        obj = json.loads(self._signed())
        obj["observations"][0]["trace_sha256"] = "d" * 64
        with self.assertRaisesRegex(Q8EvidenceError, "signature"):
            inspect_signed_q8_evidence(_canonical(obj), self.pins)

    def test_noncanonical_reserialization_cannot_rewrite_evidence_digest(self):
        valid = self._signed()
        with self.assertRaisesRegex(Q8EvidenceError, "canonical wire"):
            inspect_signed_q8_evidence(json.dumps(json.loads(valid), indent=2).encode(), self.pins)

    def test_missing_duplicate_and_unknown_case_rejected(self):
        for replacement in (self.claims["observations"][:-1],
                            self.claims["observations"][:-1] + [self._row("Q8-A1")],
                            self.claims["observations"][:-1] + [self._row("Q8-X")]):
            with self.subTest(case=replacement[-1]):
                claims = dict(self.claims, observations=replacement)
                with self.assertRaises(Q8EvidenceError):
                    inspect_signed_q8_evidence(self._signed(claims=claims), self.pins)

    def test_unknown_or_refused_cannot_advance_claimed_witness(self):
        for case, outcome in (("Q8-A1", "refused"), ("Q8-A5", "unknown"),
                              ("Q8-A10", "fail_closed")):
            with self.subTest(case=case):
                claims = dict(self.claims, observations=[dict(row) for row in self.claims["observations"]])
                row = next(x for x in claims["observations"] if x["case_id"] == case)
                row["outcome"] = outcome
                row["after_index"] += 1
                row["after_head_sha256"] = "f" * 64
                with self.assertRaisesRegex(Q8EvidenceError, "changed claimed"):
                    inspect_signed_q8_evidence(self._signed(claims=claims), self.pins)

    def test_race_cannot_claim_two_winners(self):
        claims = dict(self.claims, observations=[dict(row) for row in self.claims["observations"]])
        row = next(x for x in claims["observations"] if x["case_id"] == "Q8-A9")
        row["after_index"] += 1
        with self.assertRaisesRegex(Q8EvidenceError, "one claimed CAS"):
            inspect_signed_q8_evidence(self._signed(claims=claims), self.pins)

    def test_wrong_outcome_and_changed_head_without_counter_rejected(self):
        claims = dict(self.claims, observations=[dict(row) for row in self.claims["observations"]])
        row = claims["observations"][0]
        row["outcome"] = "reconciled"
        with self.assertRaisesRegex(Q8EvidenceError, "unacceptable claimed outcome"):
            inspect_signed_q8_evidence(self._signed(claims=claims), self.pins)
        row["outcome"] = "refused"
        row["after_head_sha256"] = "e" * 64
        with self.assertRaisesRegex(Q8EvidenceError, "changed claimed"):
            inspect_signed_q8_evidence(self._signed(claims=claims), self.pins)

    def test_duplicate_keys_and_nonfinite_json_rejected(self):
        for raw in (b'{"schema":"x","schema":"y"}', b'{"x":NaN}'):
            with self.assertRaises(Q8EvidenceError):
                inspect_signed_q8_evidence(raw, self.pins)

    def test_overflowed_exponent_in_signed_envelope_has_controlled_error(self):
        # json.loads permits exponent overflow (1e309 -> float('inf')) even
        # with parse_constant rejecting the literal Infinity and NaN tokens.
        # Reject that input under our documented verifier error contract.
        signed = self._signed()
        corrupted = signed.replace(b'"before_index":8', b'"before_index":1e309', 1)
        self.assertNotEqual(corrupted, signed)
        with self.assertRaisesRegex(Q8EvidenceError, "canonical"):
            inspect_signed_q8_evidence(corrupted, self.pins)

    def test_wrong_types_and_oversized_rejected(self):
        for raw in ("not-bytes", b"", b"x" * (128 * 1024 + 1)):
            with self.assertRaises(Q8EvidenceError):
                inspect_signed_q8_evidence(raw, self.pins)
        for bad_pin in (None, self.pins.operator_public_key, b"short"):
            with self.assertRaises(Q8EvidenceError):
                PinnedQ8Authorities("h", "a" * 64, "b" * 64,
                                   self.pins.operator_public_key, bad_pin)

    def test_boolean_and_negative_indexes_rejected(self):
        for value in (True, False, -1, 2**63):
            claims = dict(self.claims, observations=[dict(row) for row in self.claims["observations"]])
            claims["observations"][0]["before_index"] = value
            with self.subTest(value=value), self.assertRaises(Q8EvidenceError):
                inspect_signed_q8_evidence(self._signed(claims=claims), self.pins)
