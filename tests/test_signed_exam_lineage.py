"""A legacy signed FAIL must retain the identity of its scored development exam."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import hmac
import json
import unittest

from labyrinth.signed_exam_lineage import (
    _DOMAIN, SignedExamLineageBinding, sign_exam_lineage_binding,
    verify_exam_lineage_binding,
)
from labyrinth.hf_development_transition import development_case_commitment
from tests import test_blind_transition_bridge as bridge_fixtures


BINDING_KEY = b"x" * 32
SIGNER = "independent-lineage-operator"
NONCE = sha256(b"precommitted one-shot development binding").hexdigest()
_SCORER_DOMAIN = b"labyrinth.blind-development-result.v1\0"
EVALUATOR_KEY = bridge_fixtures.EVALUATOR_KEY
LINEAGE_KEY = bridge_fixtures.LINEAGE_KEY
TRANSITION_KEY = bridge_fixtures.TRANSITION_KEY


class SignedExamLineageTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing real MAC-sealed SQLite scorer + signed transition
        # fixture instead of mocking provenance at the trust boundary.
        self.fixture = bridge_fixtures.BlindTransitionBridgeTests(
            "test_authenticated_score_plus_lineage_produces_verified_development_fail")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.transition = self.fixture.bridge()
        raw, signature = self.fixture.lineage()
        self.options = {
            "scope": self.fixture.scope,
            "context": self.fixture.context,
            "transition": self.transition,
            "scored_receipt_bytes": self.fixture.scored.receipt_bytes,
            "scored_signature_hex": self.fixture.scored.signature_hex,
            "candidate_lineage_bytes": raw,
            "candidate_lineage_signature_hex": signature,
            "candidate_run_id": "candidate-run-0",
            "candidate_artifact_sha256": self.fixture.artifact_hash,
            "expected_evaluator_roster_sha256": self.fixture.public_precommit,
            "expected_development_roster_sha256": development_case_commitment(
                self.transition.development_public_roster),
            "failure_threshold_percent": 75,
            "binding_signer_id": SIGNER,
            "expected_nonce_sha256": NONCE,
            "independent_evaluator_key": EVALUATOR_KEY,
            "independent_lineage_key": LINEAGE_KEY,
            "independent_transition_key": TRANSITION_KEY,
            "independent_binding_key": BINDING_KEY,
        }

    def bind(self, **changes):
        return sign_exam_lineage_binding(**{**self.options, **changes})

    def verify(self, binding, **changes):
        return verify_exam_lineage_binding(binding, **{**self.options, **changes})

    def different_authenticated_scored_exam(self):
        # Different evaluator-signed aggregate for the exact same candidate,
        # still a valid failed development examination. The old transition
        # does not carry this difference by design.
        changed = json.loads(self.fixture.scored.receipt_bytes)
        group = next(group for group in changed["groups"]
                     if group["split"] == "heldout")
        group["correct"] = group["correct"] - 1
        raw = (json.dumps(changed, sort_keys=True, separators=(",", ":"))
               + "\n").encode("utf-8")
        sig = hmac.new(EVALUATOR_KEY, _SCORER_DOMAIN + raw, sha256).hexdigest()
        other_transition = replace(
            self.transition, scored_receipt_sha256=sha256(raw).hexdigest(),
        )
        return raw, sig, other_transition

    def test_exact_scored_exam_candidate_transition_and_nonce_verified(self):
        binding = self.bind()
        body = self.verify(binding)
        self.assertEqual(body["scored_receipt_sha256"],
                         sha256(self.fixture.scored.receipt_bytes).hexdigest())
        self.assertEqual(body["legacy_transition_sha256"],
                         sha256(self.transition.receipt_bytes).hexdigest())
        self.assertEqual(body["candidate_artifact_sha256"], self.fixture.artifact_hash)
        self.assertEqual(body["binding_signer_id"], SIGNER)
        self.assertEqual(body["binding_nonce_sha256"], NONCE)
        self.assertEqual(body["failure_threshold_percent"], 75)
        self.assertFalse(body["graduation_authorized"])
        self.assertFalse(body["model_execution_authenticated"])
        self.assertNotIn(b"A pleasant welcome", binding.binding_bytes)

    def test_distinct_authenticated_scored_exams_no_longer_have_same_binding(self):
        other_raw, other_sig, other_transition = self.different_authenticated_scored_exam()
        self.assertEqual(self.transition.receipt_bytes, other_transition.receipt_bytes)
        self.assertEqual(self.transition.signature_hex, other_transition.signature_hex)
        first = self.bind()
        second = self.bind(
            transition=other_transition,
            scored_receipt_bytes=other_raw,
            scored_signature_hex=other_sig,
        )
        self.assertNotEqual(first.binding_bytes, second.binding_bytes)
        self.assertNotEqual(first.signature_hex, second.signature_hex)
        self.verify(first)
        self.verify(second, transition=other_transition,
                    scored_receipt_bytes=other_raw,
                    scored_signature_hex=other_sig)
        with self.assertRaises(ValueError):
            self.verify(first, transition=other_transition,
                        scored_receipt_bytes=other_raw,
                        scored_signature_hex=other_sig)

    def test_signer_crossing_candidate_swap_and_nonce_replay_refused(self):
        binding = self.bind()
        for changes in (
            {"candidate_run_id": "candidate-run-1"},
            {"candidate_artifact_sha256": "f" * 64},
            {"binding_signer_id": "other-signer"},
            {"expected_nonce_sha256": "f" * 64},
            {"failure_threshold_percent": 100},
            {"independent_binding_key": b"y" * 32},
        ):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.verify(binding, **changes)

    def test_evaluator_lineage_and_transition_forgery_refused(self):
        for changes in (
            {"scored_signature_hex": "f" * 64},
            {"candidate_lineage_signature_hex": "f" * 64},
            {"transition": replace(self.transition, signature_hex="f" * 64)},
            {"expected_evaluator_roster_sha256": "f" * 64},
            {"expected_development_roster_sha256": "f" * 64},
            {"failure_threshold_percent": True},
            {"independent_binding_key": EVALUATOR_KEY},
            {"independent_lineage_key": BINDING_KEY},
        ):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.bind(**changes)

    def test_authenticated_score_cannot_authorize_inconsistent_weaknesses(self):
        changed = json.loads(self.fixture.scored.receipt_bytes)
        for group in changed["groups"]:
            group["correct"] = group["total"]
        raw = (json.dumps(changed, sort_keys=True, separators=(",", ":"))
               + "\n").encode("utf-8")
        sig = hmac.new(EVALUATOR_KEY, _SCORER_DOMAIN + raw, sha256).hexdigest()
        claimed = replace(self.transition,
                          scored_receipt_sha256=sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError, "weaknesses do not derive"):
            self.bind(transition=claimed, scored_receipt_bytes=raw,
                      scored_signature_hex=sig)

    def test_binding_authenticity_malformed_receipt_and_graduation_refused(self):
        binding = self.bind()
        for changed in (
            replace(binding, binding_bytes=binding.binding_bytes + b" "),
            replace(binding, signature_hex="0" * 64),
            replace(binding, graduation_authorized=True),
            replace(binding, model_execution_authenticated=True),
            replace(binding, binding_bytes=b"{}\n"),
        ):
            with self.subTest(case=changed), self.assertRaises(ValueError):
                self.verify(changed)

    def test_legacy_transition_without_companion_cannot_verify_provenance(self):
        # A valid legacy signature alone is deliberately not a valid object
        # for this stronger source verification path.
        with self.assertRaises(ValueError):
            self.verify(self.transition)

    def test_binding_domain_separation_and_signature_pinned(self):
        binding = self.bind()
        sig = hmac.new(BINDING_KEY, _DOMAIN + binding.binding_bytes,
                       sha256).hexdigest()
        self.assertEqual(sig, binding.signature_hex)
        self.assertNotEqual(
            hmac.new(BINDING_KEY, binding.binding_bytes, sha256).hexdigest(), sig)


if __name__ == "__main__":
    unittest.main()
