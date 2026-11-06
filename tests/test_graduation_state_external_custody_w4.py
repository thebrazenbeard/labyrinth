"""Adversarial source checks for a reported prior Minotaur graduation."""

from dataclasses import replace
from hashlib import sha256
import json
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from labyrinth.graduation_gate import GateState, GraduationPolicy, _policy_fingerprint
from labyrinth.graduation_state import _serialize
from labyrinth.graduation_state_external_custody_w4 import (
    _DOMAIN, _SCHEMA, ExternalGateStateSourceError, inspect_signed_source_state,
)


H = lambda digit: digit * 64


class ExternalGateStateSourceW4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = GraduationPolicy(
            objective_id="objective.alpha", trainee_sha256=H("a"),
            base_sha256=H("b"), session_nonce="c" * 32,
            expected_evaluator_sha256=H("d"), expected_conditions_sha256=H("e"))
        self.state = GateState(
            next_attempt=2, consumed_run_ids=frozenset({"run.1"}),
            graduated=True, policy_fingerprint=_policy_fingerprint(self.policy))
        self.private = Ed25519PrivateKey.generate()
        self.public = self.private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.head = H("f")
        self.version = 8

    def signed(self, record=None, *, raw=None, private=None):
        state_json = _serialize(self.state)
        body = {"schema": _SCHEMA, "provider_id": "exam.host", "version": self.version,
                "policy_fingerprint": _policy_fingerprint(self.policy),
                "state_json": state_json,
                "state_sha256": sha256(state_json.encode()).hexdigest(),
                "external_head_sha256": self.head}
        if record is not None:
            body.update(record)
        wire = raw if raw is not None else (
            json.dumps(body, sort_keys=True, separators=(",", ":"),
                       allow_nan=False, ensure_ascii=True) + "\n").encode("ascii")
        signed = (private or self.private).sign(_DOMAIN + wire).hex()
        return wire, signed

    def check(self, wire, signed, **overrides):
        values = dict(policy=self.policy, candidate_state=self.state,
                      signed_source_bytes=wire, signature_hex=signed,
                      pinned_public_key=self.public, pinned_provider_id="exam.host",
                      expected_state_sha256=sha256(_serialize(self.state).encode()).hexdigest(),
                      expected_version=self.version, expected_external_head_sha256=self.head)
        values.update(overrides)
        return inspect_signed_source_state(**values)

    def test_signed_prior_graduation_is_only_source_observation(self):
        result = self.check(*self.signed())
        self.assertTrue(result.reported_prior_graduated)
        self.assertTrue(result.source_bytes_and_signature_match)
        self.assertFalse(result.independent_host_custody_verified)
        self.assertFalse(result.fresh_final_exam_verified)
        self.assertFalse(result.graduation_authorized)
        self.assertFalse(result.deployment_authorized)
        self.assertFalse(hasattr(result, "state"))
        with self.assertRaises(AttributeError):
            object.__setattr__(result, "graduation_authorized", True)

    def test_signed_prior_graduation_report_bound_to_serialized_snapshot(self):
        self.state = replace(self.state, graduated=False)
        wire, signature = self.signed()
        original_serialize = _serialize

        def mutate_after_serialization(state):
            serialized = original_serialize(state)
            # Simulate an interleaving after exact bytes were bound, but
            # before the final result's reported_prior_graduated is read.
            object.__setattr__(self.state, "graduated", True)
            return serialized

        with patch("labyrinth.graduation_state_external_custody_w4._serialize",
                   side_effect=mutate_after_serialization):
            observation = self.check(wire, signature)
        self.assertFalse(observation.reported_prior_graduated)
        self.assertFalse(observation.graduation_authorized)

    def test_caller_created_graduated_state_without_external_signature_fails(self):
        with self.assertRaises(ExternalGateStateSourceError):
            self.check(b"{}\n", "0" * 128)

    def test_replacement_graduated_state_and_wrong_policy_are_rejected(self):
        wire, signature = self.signed()
        altered = replace(self.state, graduated=False)
        with self.assertRaises(ExternalGateStateSourceError):
            self.check(wire, signature, candidate_state=altered)
        other = replace(self.policy, trainee_sha256=H("1"))
        with self.assertRaises(ExternalGateStateSourceError):
            self.check(wire, signature, policy=other)

    def test_mutated_policy_cannot_satisfy_source_check_with_resigned_pins(self):
        # The previous implementation accepted a self-consistent signed
        # snapshot after object.__setattr__ bypassed constructor thresholds.
        # Re-signing a bad policy must not make its now-invalid state valid.
        for field, bad in (("min_accuracy", 0.0),
                           ("min_format_rate", 0.0),
                           ("min_gain_over_controls", 0.0),
                           ("max_attempts", 0),
                           ("required_slices", ("heldout",)),
                           ("min_accuracy", 10**309)):
            with self.subTest(field=field, bad=str(bad)[:40]):
                bad_policy = replace(self.policy)
                object.__setattr__(bad_policy, field, bad)
                bad_state = replace(self.state,
                                    policy_fingerprint=_policy_fingerprint(bad_policy))
                old_policy, old_state = self.policy, self.state
                try:
                    self.policy, self.state = bad_policy, bad_state
                    wire, signature = self.signed()
                    with self.assertRaises(ExternalGateStateSourceError):
                        self.check(wire, signature)
                finally:
                    self.policy, self.state = old_policy, old_state

    def test_attacker_chosen_signer_and_stale_pins_fail(self):
        foreign = Ed25519PrivateKey.generate()
        wire, signature = self.signed(private=foreign)
        with self.assertRaises(ExternalGateStateSourceError):
            self.check(wire, signature)
        wire, signature = self.signed()
        for changed in (dict(expected_version=9), dict(expected_external_head_sha256=H("1")),
                        dict(expected_state_sha256=H("2")), dict(pinned_provider_id="other")):
            with self.subTest(changed=changed), self.assertRaises(ExternalGateStateSourceError):
                self.check(wire, signature, **changed)

    def test_replay_does_not_claim_freshness_even_with_valid_signature(self):
        # Matching *old* pins cannot prove this was the latest protected head.
        old, signature = self.signed()
        result = self.check(old, signature)
        self.assertFalse(result.independent_host_custody_verified)
        self.assertFalse(result.graduation_authorized)

    def test_canonical_and_duplicate_field_attacks_are_controlled_errors(self):
        body = self.signed()[0].decode("ascii").strip()
        noncanonical = ("  " + body + "\n").encode("ascii")
        duplicate = body.replace('"version":8', '"version":8,"version":8').encode("ascii") + b"\n"
        for wire in (noncanonical, duplicate):
            with self.subTest(wire=wire[:30]), self.assertRaises(ExternalGateStateSourceError):
                self.check(*self.signed(raw=wire))

    def test_nonfinite_signed_numeric_value_maps_to_source_error(self):
        valid = self.signed()[0]
        bad = valid.replace(b'"version":8', b'"version":1e309')
        self.assertNotEqual(valid, bad)
        with self.assertRaises(ExternalGateStateSourceError):
            self.check(*self.signed(raw=bad))

    def test_untrusted_subclasses_and_invalid_state_shape_fail(self):
        wire, signature = self.signed()
        class PretendBytes(bytes):
            pass
        with self.assertRaises(ExternalGateStateSourceError):
            self.check(wire, signature, signed_source_bytes=PretendBytes(wire))
        malicious_state = replace(self.state, consumed_run_ids=frozenset({12}))
        with self.assertRaises(ExternalGateStateSourceError):
            self.check(wire, signature, candidate_state=malicious_state)


if __name__ == "__main__":
    unittest.main()
