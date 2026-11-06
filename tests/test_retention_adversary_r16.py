"""R16 hostile probes of the opt-in signed retention router.

The signer and host readback callback are local fakes. Passing these tests
does not establish external signer custody, an immutable route head, or a
mandatory runtime entrypoint.
"""

from dataclasses import replace
from hashlib import sha256
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import test_governed_retention_router_r15 as retention_fixture
from labyrinth.governed_retention_router_r15 import SignedRetentionRoute
import labyrinth.governed_retention_router_r15 as router


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


class RetentionAdversaryR16Tests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = retention_fixture.GovernedRetentionRouterR15Tests(
            "test_valid_source_route_still_never_graduates")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture

    def test_every_signed_identity_mutation_denied_before_local_pass(self) -> None:
        f = self.fixture
        mutations = {
            "provider_id": "foreign-provider",
            "route_id": "changed-route",
            "classification": "general_reasoning",
            "objective_id": "empathy",
            "scope_sha256": _digest("changed-scope"),
            "development_exam_id": "changed-exam",
            "prior_checkpoint_sha256": _digest("changed-prior"),
            "candidate_checkpoint_sha256": _digest("changed-candidate"),
            "policy_sha256": _digest("changed-policy"),
            "roster_sha256": _digest("changed-roster"),
            "semantic_inventory_sha256": _digest("changed-semantic"),
            "training_inventory_sha256": _digest("changed-training"),
            "taxonomy_signer_id": "changed-signer",
            "taxonomy_nonce_sha256": _digest("changed-nonce"),
            "host_epoch": 24,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                forged = replace(f.route, **{field: value})
                f.live = forged
                result = f.run_gate(route=forged)
                self.assertEqual(result.status, "BLOCKED")
                self.assertFalse(result.graduation_authorized)

    def test_signed_rebinding_cannot_reuse_previous_host_head(self) -> None:
        f = self.fixture
        for field, value in (
            ("route_id", "reissued-route"),
            ("host_epoch", f.route.host_epoch + 1),
            ("provider_id", "alternate-provider"),
        ):
            with self.subTest(field=field):
                reissued = f.sign(replace(f.route, **{field: value}))
                # Even a real test key signature does not override the
                # host's different exact signed record.
                f.live = f.route
                self.assertEqual(f.run_gate(route=reissued).status, "BLOCKED")

    def test_epoch_type_boolean_or_float_rejected_even_if_resigned(self) -> None:
        f = self.fixture
        for epoch in (True, 1.0, -1, 0, 2**63, "23"):
            with self.subTest(epoch=epoch):
                altered = f.sign(replace(f.route, host_epoch=epoch))
                f.live = altered
                self.assertEqual(f.run_gate(route=altered).status, "BLOCKED")

    def test_second_host_readback_transport_failure_denies_without_graduation(self) -> None:
        f = self.fixture
        seen = []

        def host_read(route_id: str) -> SignedRetentionRoute:
            seen.append(route_id)
            if len(seen) > 1:
                raise ConnectionError("synthetic host readback loss")
            return f.route

        result = f.run_gate(reader=host_read)
        self.assertEqual(seen, [f.route.route_id, f.route.route_id])
        self.assertEqual(result.status, "BLOCKED")
        self.assertFalse(result.private_exam_authorized)
        self.assertFalse(result.graduation_authorized)

    def test_canonical_ed25519_domain_cannot_accept_cross_key_or_bitflip(self) -> None:
        f = self.fixture
        flipped = "0" if f.route.signature_hex[0] != "0" else "1"
        altered = replace(f.route, signature_hex=flipped + f.route.signature_hex[1:])
        f.live = altered
        self.assertEqual(f.run_gate(route=altered).status, "BLOCKED")
        wrong_key = Ed25519PrivateKey.generate().public_key()
        from cryptography.hazmat.primitives import serialization
        pub = wrong_key.public_bytes(serialization.Encoding.Raw,
                                     serialization.PublicFormat.Raw)
        f.live = f.route
        self.assertEqual(f.run_gate(public_key=pub).status, "BLOCKED")
        self.assertTrue(f.route.signed_bytes().startswith(
            b"labyrinth.governed-retention-route.v1\x00"))

    def test_mutated_frozen_route_after_signature_fails_validation(self) -> None:
        f = self.fixture
        route = replace(f.route)
        object.__setattr__(route, "classification", "program_reasoning")
        f.live = route
        self.assertEqual(f.run_gate(route=route).status, "BLOCKED")

    def test_caller_provided_static_readback_cannot_prove_external_freshness(self) -> None:
        """Known limitation: valid signed data + replaying fake host is not Q8."""
        f = self.fixture
        static_readback = lambda _route_id: f.route
        first = f.run_gate(reader=static_readback)
        second = f.run_gate(reader=static_readback)
        self.assertEqual((first.status, second.status), ("PASS", "PASS"))
        self.assertEqual(first.provenance, "SIGNED_ROUTE_SOURCE_ONLY")
        self.assertFalse(first.model_execution_authenticated)
        self.assertFalse(second.graduation_authorized)
        # No independent monotonic host receipt or one-shot epoch consumption
        # is obtained merely by calling this Python callback twice.

    def test_policy_candidate_mutated_during_admission_requires_postcheck(self) -> None:
        """An otherwise valid PASS must not outlive the policy it just used."""
        f = self.fixture
        original = router.evaluate_strict_retention_admission

        def mutate_after_admission(*args, **kwargs):
            verdict = original(*args, **kwargs)
            self.assertEqual(verdict.status, "PASS")
            policy = args[1]
            # Frozen Python dataclasses are still mutable by object.__setattr__
            # when an untrusted holder keeps a reference to the same object.
            object.__setattr__(policy, "after_checkpoint_sha256", _digest("swapped-model"))
            return verdict

        with patch.object(router, "evaluate_strict_retention_admission",
                          side_effect=mutate_after_admission):
            outcome = f.run_gate()
        self.assertEqual(outcome.status, "BLOCKED",
                         "mutated candidate policy accepted after local admission")


if __name__ == "__main__":
    unittest.main()
