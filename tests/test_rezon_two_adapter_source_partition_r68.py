"""Two distinct model adapters may share source observations, never authority."""

from dataclasses import asdict, replace
import linecache
import sys
import unittest

from labyrinth.evidence_pointer import (
    EXPERIMENT, QUALIFICATION, SCHEMA, ExpectedSourcePinsV1,
    SourceEvidencePointerV1, encode_source_evidence_pointer,
)
from labyrinth.rezon_pointer_mapping_r57 import (
    EXPECTED_ARMS, ExpectedRezonPinsV1, RezonSourceClaimsV1,
)
from labyrinth.rezon_two_adapter_source_partition_r68 import (
    AdapterSourceClaimV1, ExpectedAdapterSourceV1,
    AdapterSourcePartitionError, compare_two_model_adapter_sources,
)


class TwoAdapterSourcePartitionTests(unittest.TestCase):
    def setUp(self):
        self.lab_expected = ExpectedSourcePinsV1(
            run_id="lab-run-68", source_head="a" * 40,
            source_tree="b" * 40, plan_sha256="1" * 64,
            train_exposure_sha256="2" * 64,
            control_matrix_sha256="3" * 64,
            development_event_sha256="4" * 64,
            checkpoint_manifest_sha256="5" * 64,
            scope_sha256="6" * 64, rezon_source_head="c" * 40,
            rezon_receipt_bundle_sha256="7" * 64,
        )
        self.lab_payload = encode_source_evidence_pointer(SourceEvidencePointerV1(
            schema=SCHEMA, experiment=EXPERIMENT, qualification_state=QUALIFICATION,
            **asdict(self.lab_expected),
        ))
        self.rezon_claim = RezonSourceClaimsV1(
            run_id="rezon-run-68", rezon_source_head="c" * 40,
            labyrinth_source_head="a" * 40, receipt_bundle_sha256="7" * 64,
            maze_exposure_sha256="8" * 64, checkpoint_sha256="9" * 64,
            control_matrix_sha256="3" * 64,
            development_event_sha256="4" * 64,
            training_program="LABYRINTH_MINILM_FIVE_ARM",
            bridge_semantics="PROVENANCE_ONLY", control_arms=EXPECTED_ARMS,
        )
        self.rezon_expected = ExpectedRezonPinsV1(**asdict(self.rezon_claim))
        self.alpha = AdapterSourceClaimV1(
            adapter_id="model-alpha", model_source_head="d" * 40,
            model_artifact_sha256="a" * 64,
            adapter_contract_sha256="c" * 64,
        )
        self.beta = AdapterSourceClaimV1(
            adapter_id="model-beta", model_source_head="e" * 40,
            model_artifact_sha256="b" * 64,
            adapter_contract_sha256="c" * 64,
        )
        self.alpha_expected = ExpectedAdapterSourceV1(**asdict(self.alpha))
        self.beta_expected = ExpectedAdapterSourceV1(**asdict(self.beta))

    def compare(self, **overrides):
        data = dict(
            labyrinth_pointer_payload=self.lab_payload,
            expected_labyrinth=self.lab_expected,
            observed_rezon=self.rezon_claim,
            expected_rezon=self.rezon_expected,
            observed_alpha=self.alpha, expected_alpha=self.alpha_expected,
            observed_beta=self.beta, expected_beta=self.beta_expected,
        )
        data.update(overrides)
        return compare_two_model_adapter_sources(**data)

    def assert_rejects_post_validation_mutation(self, mutate, **overrides):
        """Deterministically inject in the post-check/pre-hash trace window."""
        fired = False
        code = compare_two_model_adapter_sources.__code__

        def tracer(frame, event, _arg):
            nonlocal fired
            if (not fired and event == "line" and frame.f_code is code
                    and linecache.getline(code.co_filename, frame.f_lineno).strip()
                    == "material = {"):
                fired = True
                mutate()
            return tracer

        previous_trace = sys.gettrace()
        sys.settrace(tracer)
        try:
            with self.assertRaises(AdapterSourcePartitionError):
                self.compare(**overrides)
        finally:
            sys.settrace(previous_trace)
        self.assertTrue(fired, "hostile post-validation trace hook did not execute")

    def test_two_distinct_pinned_models_share_only_untrusted_provenance(self):
        result = self.compare()
        self.assertEqual(len(result.source_pair_sha256), 64)
        self.assertEqual(result.qualification_state, "SOURCE_ONLY_TWO_ADAPTERS_UNATTESTED")
        self.assertEqual(result.unresolved_semantic_pairs, (
            ("train_exposure_sha256", "maze_exposure_sha256"),
            ("checkpoint_manifest_sha256", "checkpoint_sha256"),
        ))
        for flag in (
            "model_identity_verified", "source_receipts_verified",
            "host_execution_verified", "checkpoint_verified",
            "checkpoint_identity_verified", "agi_qualification_verified",
            "consent_verified", "permissions_verified", "principal_verified",
            "cross_model_grants_transferred", "cross_model_identity_transferred",
            "can_authorize_training", "can_authorize_graduation",
            "can_authorize_merge", "can_authorize_deployment",
        ):
            self.assertIs(getattr(result, flag), False, flag)
        for forbidden in ("adapter_id", "model_artifact_sha256", "principal", "consent", "permissions"):
            self.assertFalse(hasattr(result, forbidden), forbidden)

    def test_swapped_models_or_cross_model_replay_cannot_satisfy_distinct_expectations(self):
        for changes in (
            dict(observed_alpha=self.beta),
            dict(observed_beta=self.alpha),
            dict(observed_alpha=self.beta, observed_beta=self.alpha),
            dict(observed_alpha=replace(self.alpha, model_artifact_sha256=self.beta.model_artifact_sha256)),
            dict(observed_beta=replace(self.beta, model_source_head=self.alpha.model_source_head)),
        ):
            with self.subTest(changes=changes), self.assertRaises(AdapterSourcePartitionError):
                self.compare(**changes)

    def test_adapter_source_expected_pins_cannot_be_substituted(self):
        with self.assertRaises(AdapterSourcePartitionError):
            self.compare(expected_alpha=self.beta_expected)
        with self.assertRaises(AdapterSourcePartitionError):
            self.compare(expected_beta=replace(self.beta_expected,
                                               adapter_contract_sha256="f" * 64))

    def test_r57_labyrinth_rezon_source_gate_cannot_be_bypassed(self):
        with self.assertRaises(ValueError):
            self.compare(expected_labyrinth=replace(self.lab_expected,
                                                    source_tree="f" * 40))
        with self.assertRaises(ValueError):
            self.compare(observed_rezon=replace(self.rezon_claim,
                                                control_matrix_sha256="f" * 64))

    def test_trace_mutation_of_frozen_adapter_after_pin_check_is_rejected(self):
        offered = replace(self.alpha)
        self.assert_rejects_post_validation_mutation(
            lambda: object.__setattr__(offered, "model_artifact_sha256", "f" * 64),
            observed_alpha=offered,
        )

    def test_trace_mutation_of_frozen_rezon_claim_after_source_check_is_rejected(self):
        offered = replace(self.rezon_claim)
        self.assert_rejects_post_validation_mutation(
            lambda: object.__setattr__(offered, "control_matrix_sha256", "f" * 64),
            observed_rezon=offered,
        )

    def test_trace_mutation_of_expected_adapter_after_pin_check_is_rejected(self):
        expected = replace(self.beta_expected)
        self.assert_rejects_post_validation_mutation(
            lambda: object.__setattr__(expected, "model_source_head", "f" * 40),
            expected_beta=expected,
        )

    def test_same_model_or_same_adapter_key_cannot_masquerade_as_two_models(self):
        for beta in (
            replace(self.beta, model_artifact_sha256=self.alpha.model_artifact_sha256),
            replace(self.beta, adapter_id=self.alpha.adapter_id),
        ):
            with self.subTest(beta=beta), self.assertRaises(AdapterSourcePartitionError):
                self.compare(observed_beta=beta, expected_beta=ExpectedAdapterSourceV1(**asdict(beta)))

    def test_different_source_pairs_produce_different_digest_but_never_authority(self):
        initial = self.compare()
        changed_beta = replace(self.beta, model_artifact_sha256="f" * 64)
        another = self.compare(observed_beta=changed_beta,
                               expected_beta=ExpectedAdapterSourceV1(**asdict(changed_beta)))
        self.assertNotEqual(initial.source_pair_sha256, another.source_pair_sha256)
        self.assertIs(another.cross_model_identity_transferred, False)

    def test_malformed_model_and_adapter_source_fields_rejected(self):
        for field, value in (
            ("adapter_id", "../../alpha"),
            ("adapter_id", "Model-Alpha"),
            ("model_source_head", "A" * 40),
            ("model_artifact_sha256", True),
            ("adapter_contract_sha256", "c" * 63),
        ):
            altered = replace(self.alpha, **{field: value})
            with self.subTest(field=field), self.assertRaises(AdapterSourcePartitionError):
                self.compare(observed_alpha=altered,
                             expected_alpha=ExpectedAdapterSourceV1(**asdict(altered)))

    def test_permission_and_model_identity_are_not_constructor_fields(self):
        for forbidden in ("consent", "permissions", "principal", "graduation_authorized", "model_identity_verified"):
            with self.subTest(forbidden=forbidden), self.assertRaises(TypeError):
                AdapterSourceClaimV1(**dict(asdict(self.alpha), **{forbidden: True}))

    def test_observation_cannot_be_upgraded_via_object_setattr(self):
        report = self.compare()
        for field, forged in (
            ("consent_verified", True), ("permissions_verified", True),
            ("principal_verified", True), ("checkpoint_identity_verified", True),
            ("model_identity_verified", True),
            ("can_authorize_deployment", True),
            ("qualification_state", "AUTHORIZED"),
        ):
            with self.subTest(field=field), self.assertRaises((AttributeError, TypeError)):
                object.__setattr__(report, field, forged)
        with self.assertRaises(TypeError):
            type(report)(source_pair_sha256=report.source_pair_sha256,
                         can_authorize_training=True)

    def test_source_pair_is_sensitive_to_labyrinth_rezon_binding(self):
        original = self.compare()
        new_alpha = replace(self.alpha, adapter_contract_sha256="e" * 64)
        changed = self.compare(observed_alpha=new_alpha,
                               expected_alpha=ExpectedAdapterSourceV1(**asdict(new_alpha)))
        self.assertNotEqual(original.source_pair_sha256, changed.source_pair_sha256)


if __name__ == "__main__":
    unittest.main()
