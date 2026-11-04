"""Adversarial, source-only checks for the externally operated Q8 test ledger."""

from __future__ import annotations

import hashlib
import json
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from labyrinth.q8_external_acceptance_source_r68 import (
    Q8AcceptanceSourceError,
    Q8ExpectedWitnessPins,
    q8_case_signed_payload,
    validate_q8_acceptance_bundle,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _private(value: str) -> Ed25519PrivateKey:
    # Deterministic TEST-ONLY keys. These are not external provider credentials.
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(value.encode()).digest())


def _public(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


_OUTCOMES = {
    "Q8-A1": "RESTORED_TRAINER_REFUSED",
    "Q8-A2": "RESTORED_SERVICE_REFUSED",
    "Q8-A3": "CLONE_REFUSED",
    "Q8-A4": "REPLAY_REFUSED",
    "Q8-A5": "RECONCILED_COMMIT",
    "Q8-A6": "BINDING_MISMATCH",
    "Q8-A7": "PEER_UNAUTHORIZED",
    "Q8-A8": "ROTATION_STALE",
    "Q8-A9": "SINGLE_CAS_WINNER",
    "Q8-A10": "FAIL_CLOSED_UNAVAILABLE",
}


class ExternalQ8AcceptanceSourceTests(unittest.TestCase):
    def setUp(self):
        self.operator = _private("independent-operator-test")
        self.witness = _private("external-witness-test")
        self.pins = Q8ExpectedWitnessPins(
            provider_id="independent-provider", run_id="run-68",
            host_sha256=_sha("host"), service_sha256=_sha("service"),
            source_commit_sha="a" * 40, source_tree_sha="b" * 40,
            roster_sha256=_sha("roster"), policy_sha256=_sha("policy"),
            signer_generation=4, witness_generation=7,
            operator_public_key=_public(self.operator),
            witness_public_key=_public(self.witness),
        )
        self.statement = {
            "schema": "labyrinth.q8-external-acceptance-source.v1",
            "provider_id": self.pins.provider_id,
            "run_id": self.pins.run_id,
            "host_sha256": self.pins.host_sha256,
            "service_sha256": self.pins.service_sha256,
            "source_commit_sha": self.pins.source_commit_sha,
            "source_tree_sha": self.pins.source_tree_sha,
            "roster_sha256": self.pins.roster_sha256,
            "policy_sha256": self.pins.policy_sha256,
            "signer_generation": self.pins.signer_generation,
            "witness_generation": self.pins.witness_generation,
        }
        self.cases = []
        for index, (test_id, outcome) in enumerate(_OUTCOMES.items(), 1):
            before = 100 + 3 * index
            advanced = test_id in ("Q8-A5", "Q8-A9")
            case = {
                "test_id": test_id,
                "operation_nonce": "nonce-" + str(index),
                "before_index": before,
                "before_head_sha256": _sha("before-" + str(index)),
                "after_index": before + int(advanced),
                "after_head_sha256": _sha("after-" + str(index)) if advanced
                else _sha("before-" + str(index)),
                "refusal_code": outcome,
                "audit_sha256": _sha("audit-" + str(index)),
                "trace_sha256": _sha("trace-" + str(index)),
            }
            self.cases.append(case)

    def _bundle(self, *, cases=None, statement=None, signer=None, witness=None) -> bytes:
        statement = dict(self.statement if statement is None else statement)
        cases = [dict(c) for c in (self.cases if cases is None else cases)]
        signer = self.operator if signer is None else signer
        witness = self.witness if witness is None else witness
        for case in cases:
            case.pop("operator_signature_hex", None)
            case.pop("witness_signature_hex", None)
            payload = q8_case_signed_payload(statement, case)
            case["operator_signature_hex"] = signer.sign(payload).hex()
            case["witness_signature_hex"] = witness.sign(payload).hex()
        return (json.dumps({**statement, "cases": cases}, sort_keys=True,
                           separators=(",", ":")) + "\n").encode()

    def test_ten_independently_signed_source_cases_never_grant_authority(self):
        result = validate_q8_acceptance_bundle(self._bundle(), self.pins)
        self.assertEqual(result.case_ids, tuple(_OUTCOMES))
        self.assertEqual(result.status, "SOURCE_ONLY_UNATTESTED")
        self.assertFalse(result.q8_physically_qualified)
        self.assertFalse(result.can_graduate)
        self.assertFalse(result.can_merge)
        self.assertFalse(result.can_deploy)
        for name, value in (("status", "HOST_Q8_PASS"),
                            ("q8_physically_qualified", True),
                            ("can_merge", True), ("can_deploy", True),
                            ("case_ids", ())):
            with self.subTest(name=name), self.assertRaises((AttributeError, TypeError)):
                object.__setattr__(result, name, value)

    def test_missing_duplicate_or_out_of_order_case_rejected(self):
        for cases in (self.cases[:-1],
                      self.cases[:9] + [self.cases[0]],
                      list(reversed(self.cases))):
            with self.subTest(cases=[c["test_id"] for c in cases]):
                with self.assertRaises(Q8AcceptanceSourceError):
                    validate_q8_acceptance_bundle(self._bundle(cases=cases), self.pins)

    def test_after_head_forgery_and_invalid_outcomes_rejected_even_if_signed(self):
        mutations = (
            (0, "after_index", 105),  # stale trainer must not advance
            (1, "after_head_sha256", _sha("rollback")),
            (2, "refusal_code", "CLONE_ACCEPTED"),
            (3, "before_index", True),
            (4, "after_index", 106),  # A5 must be same or exactly +1
            (8, "after_index", 130),  # A9 only one CAS winner
            (9, "refusal_code", "LOCAL_FALLBACK"),
        )
        for index, field, replacement in mutations:
            cases = [dict(c) for c in self.cases]
            cases[index][field] = replacement
            with self.subTest(case=index, field=field), self.assertRaises(Q8AcceptanceSourceError):
                validate_q8_acceptance_bundle(self._bundle(cases=cases), self.pins)

    def test_unpinned_roster_generation_provider_and_keys_rejected(self):
        for field, value in (("roster_sha256", _sha("restored-roster")),
                             ("signer_generation", 3),
                             ("witness_generation", 6),
                             ("host_sha256", _sha("cloned-host")),
                             ("provider_id", "cloned-provider")):
            statement = dict(self.statement)
            statement[field] = value
            with self.subTest(field=field), self.assertRaises(Q8AcceptanceSourceError):
                validate_q8_acceptance_bundle(self._bundle(statement=statement), self.pins)
        with self.assertRaises(Q8AcceptanceSourceError):
            validate_q8_acceptance_bundle(
                self._bundle(signer=_private("different-signer")), self.pins)
        with self.assertRaises(Q8AcceptanceSourceError):
            validate_q8_acceptance_bundle(
                self._bundle(witness=_private("different-witness")), self.pins)

    def test_duplicate_nonce_trace_and_audit_rejected_even_with_new_signatures(self):
        for field in ("operation_nonce", "trace_sha256", "audit_sha256"):
            cases = [dict(c) for c in self.cases]
            cases[1][field] = cases[0][field]
            with self.subTest(field=field), self.assertRaises(Q8AcceptanceSourceError):
                validate_q8_acceptance_bundle(self._bundle(cases=cases), self.pins)

    def test_cross_field_audit_trace_digest_alias_rejected_even_if_resigned(self):
        # One SHA must never be reused as another row's different evidence class.
        for recipient, donor in (("trace_sha256", "audit_sha256"),
                                 ("audit_sha256", "trace_sha256")):
            cases = [dict(c) for c in self.cases]
            cases[1][recipient] = cases[0][donor]
            with self.subTest(recipient=recipient), self.assertRaises(Q8AcceptanceSourceError):
                validate_q8_acceptance_bundle(self._bundle(cases=cases), self.pins)

    def test_unexpected_fields_duplicate_json_and_untrusted_key_rejected(self):
        data = json.loads(self._bundle())
        data["graduation_authorized"] = True
        with self.assertRaises(Q8AcceptanceSourceError):
            validate_q8_acceptance_bundle(json.dumps(data).encode(), self.pins)
        cases = [dict(c) for c in self.cases]
        cases[0]["host_authorized"] = True
        with self.assertRaises(Q8AcceptanceSourceError):
            validate_q8_acceptance_bundle(self._bundle(cases=cases), self.pins)
        raw = self._bundle().replace(b'"provider_id":"independent-provider"',
            b'"provider_id":"independent-provider","provider_id":"independent-provider"', 1)
        with self.assertRaises(Q8AcceptanceSourceError):
            validate_q8_acceptance_bundle(raw, self.pins)
        bad_pins = self.pins._replace(witness_public_key=_public(self.operator))
        with self.assertRaises(Q8AcceptanceSourceError):
            validate_q8_acceptance_bundle(self._bundle(), bad_pins)

    def test_lost_ack_can_remain_unknown_without_new_write(self):
        cases = [dict(c) for c in self.cases]
        cases[4]["after_index"] = cases[4]["before_index"]
        cases[4]["after_head_sha256"] = cases[4]["before_head_sha256"]
        cases[4]["refusal_code"] = "REMAINS_UNKNOWN"
        result = validate_q8_acceptance_bundle(self._bundle(cases=cases), self.pins)
        self.assertFalse(result.q8_physically_qualified)

    def test_validly_shaped_post_signature_mutations_and_row_transplant_fail(self):
        data = json.loads(self._bundle())
        # A9 still describes a +1 winner and new head, but the witness did not sign it.
        data["cases"][8]["after_head_sha256"] = _sha("forged-cas-winner")
        with self.assertRaises(Q8AcceptanceSourceError):
            validate_q8_acceptance_bundle(json.dumps(data).encode(), self.pins)

        data = json.loads(self._bundle())
        # A1's signed proof cannot be copied to A2, even with matching key pair.
        data["cases"][1]["operator_signature_hex"] = data["cases"][0]["operator_signature_hex"]
        data["cases"][1]["witness_signature_hex"] = data["cases"][0]["witness_signature_hex"]
        with self.assertRaises(Q8AcceptanceSourceError):
            validate_q8_acceptance_bundle(json.dumps(data).encode(), self.pins)


if __name__ == "__main__":
    unittest.main()
