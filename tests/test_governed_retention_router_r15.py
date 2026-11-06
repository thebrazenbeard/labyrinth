"""Source-only governed arithmetic retention routing; test keys are NOT host custody."""

from dataclasses import replace
from hashlib import sha256
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from test_retention_admission_r12 import campaign, one_fact_roster
from labyrinth.arithmetic_retention_admission import arithmetic_training_inventory_sha256
from labyrinth.governed_retention_router_r15 import (
    SignedRetentionRoute, route_governed_retention,
)


def digest(value):
    return sha256(value.encode("utf-8")).hexdigest()


class GovernedRetentionRouterR15Tests(unittest.TestCase):
    def setUp(self):
        self.key = Ed25519PrivateKey.generate()
        self.pub = self.key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.args, self.opts = campaign()
        request, policy, roster, *_ = self.args
        self.route = SignedRetentionRoute(
            provider_id="operator-provider", route_id="review-r15",
            classification="single_addition_arithmetic", objective_id="arithmetic",
            scope_sha256=request.scope_fingerprint,
            development_exam_id=request.development_exam_id,
            prior_checkpoint_sha256=policy.before_checkpoint_sha256,
            candidate_checkpoint_sha256=request.candidate_checkpoint_sha256,
            policy_sha256=self.opts["expected_policy_sha256"],
            roster_sha256=self.opts["expected_roster_sha256"],
            semantic_inventory_sha256=self.opts["expected_semantic_inventory_sha256"],
            training_inventory_sha256=arithmetic_training_inventory_sha256(
                policy.excluded_training_prompts),
            taxonomy_signer_id=self.opts["expected_taxonomy_signer_id"],
            taxonomy_nonce_sha256=self.opts["expected_taxonomy_nonce_sha256"],
            host_epoch=23,
            signature_hex="0" * 128,
        )
        self.route = self.sign(self.route)
        self.live = self.route

    def sign(self, route):
        return replace(route, signature_hex=self.key.sign(route.signed_bytes()).hex())

    def run_gate(self, *, args=None, route=None, public_key=None,
                 reader=None, provider_id="operator-provider", **opts):
        kwargs = dict(self.opts)
        kwargs.update(opts)
        # The signed route determines the strict expected digests; callers
        # cannot rebind those from an untrusted candidate argument.
        for key in ("expected_policy_sha256", "expected_roster_sha256",
                    "expected_semantic_inventory_sha256",
                    "expected_taxonomy_signer_id", "expected_taxonomy_nonce_sha256",
                    "trusted_prior_checkpoint_sha256"):
            kwargs.pop(key)
        return route_governed_retention(
            *(self.args if args is None else args),
            route=self.route if route is None else route,
            pinned_provider_id=provider_id,
            pinned_ed25519_public_key=self.pub if public_key is None else public_key,
            read_host_route=(lambda _rid: self.live) if reader is None else reader,
            **kwargs,
        )

    def test_valid_source_route_still_never_graduates(self):
        with patch("labyrinth.graduation_state.decide_campaign",
                   side_effect=AssertionError("private graduation invoked")) as gate:
            result = self.run_gate()
        gate.assert_not_called()
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.admission.status, "PASS")
        self.assertEqual(result.provenance, "SIGNED_ROUTE_SOURCE_ONLY")
        self.assertFalse(result.graduation_authorized)
        self.assertFalse(result.private_exam_authorized)
        self.assertFalse(result.model_execution_authenticated)
        self.assertFalse(result.examiner_custody_verified)

    def test_64_distinct_family_aliases_of_same_arithmetic_fact_block(self):
        args, opts = campaign(one_fact_roster())
        _, policy, *_ = args
        poisoned = self.sign(replace(
            self.route,
            policy_sha256=opts["expected_policy_sha256"],
            roster_sha256=opts["expected_roster_sha256"],
            semantic_inventory_sha256=opts["expected_semantic_inventory_sha256"],
            training_inventory_sha256=arithmetic_training_inventory_sha256(
                policy.excluded_training_prompts),
        ))
        self.live = poisoned
        result = self.run_gate(args=args, route=poisoned)
        self.assertEqual(result.status, "BLOCKED")
        self.assertIsNone(result.admission if result.admission is None else result.admission.retention)

    def test_unsigned_missing_or_foreign_host_route_fails_closed(self):
        cases = (
            {"route": None},
            {"route": replace(self.route, signature_hex="0" * 128)},
            {"public_key": b"\x42" * 32},
            {"provider_id": "foreign-provider"},
            {"reader": lambda _: None},
            {"reader": lambda _: replace(self.route, host_epoch=22)},
            {"reader": lambda _: replace(self.route, route_id="foreign-route")},
        )
        # Explicitly pass non-route invalid values separately since omission
        # defaults to the known-good route in this test helper.
        sanitized = {key: value for key, value in self.opts.items()
                     if key not in {"trusted_prior_checkpoint_sha256",
                                    "expected_policy_sha256", "expected_roster_sha256",
                                    "expected_semantic_inventory_sha256",
                                    "expected_taxonomy_signer_id",
                                    "expected_taxonomy_nonce_sha256"}}
        self.assertEqual(route_governed_retention(
            *self.args, route=None, pinned_provider_id="operator-provider",
            pinned_ed25519_public_key=self.pub,
            read_host_route=lambda _: self.route, **sanitized).status, "BLOCKED")
        for case in cases[1:]:
            with self.subTest(case=tuple(case)):
                self.assertEqual(self.run_gate(**case).status, "BLOCKED")

    def test_unrecognized_or_ambiguous_classification_never_downgrades(self):
        for classification in ("general_reasoning", "unknown", "", None,
                               ["single_addition_arithmetic", "general_reasoning"]):
            with self.subTest(classification=classification):
                forged = replace(self.route, classification=classification)
                # Even if a cooperating test signer signs the wrong class,
                # the narrow route refuses all classifications but one.
                if type(classification) is str:
                    forged = self.sign(forged)
                self.live = forged
                self.assertEqual(self.run_gate(route=forged).status, "BLOCKED")

    def test_candidate_policy_inventory_and_exam_cannot_be_rebound(self):
        for changed in (
            replace(self.route, policy_sha256=digest("changed-policy")),
            replace(self.route, semantic_inventory_sha256=digest("changed-semantics")),
            replace(self.route, training_inventory_sha256=digest("changed-training")),
            replace(self.route, candidate_checkpoint_sha256=digest("other-candidate")),
            replace(self.route, development_exam_id="another-development"),
            replace(self.route, taxonomy_signer_id="other-signer"),
        ):
            with self.subTest(changed=changed):
                changed = self.sign(changed)
                self.live = changed
                self.assertEqual(self.run_gate(route=changed).status, "BLOCKED")

    def test_loss_of_live_route_after_review_fails_closed(self):
        calls = 0
        def read(_):
            nonlocal calls
            calls += 1
            return self.route if calls == 1 else replace(self.route, host_epoch=22)
        self.assertEqual(self.run_gate(reader=read).status, "BLOCKED")
        self.assertEqual(calls, 2)


if __name__ == "__main__":
    unittest.main()
