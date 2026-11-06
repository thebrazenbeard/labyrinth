"""Executable local toy consumers of source-only Rezon/Labyrinth partition."""

from dataclasses import asdict, replace
import unittest
from unittest.mock import patch

from labyrinth.evidence_pointer import (
    EXPERIMENT, QUALIFICATION, SCHEMA, ExpectedSourcePinsV1,
    SourceEvidencePointerV1, encode_source_evidence_pointer,
)
from labyrinth.rezon_pointer_mapping_r57 import (
    EXPECTED_ARMS, ExpectedRezonPinsV1, RezonSourceClaimsV1,
)
from labyrinth.rezon_two_adapter_source_partition_r68 import (
    AdapterSourceClaimV1, ExpectedAdapterSourceV1,
    AdapterSourcePartitionError,
)
from labyrinth.rezon_toy_pair_consumer_r76 import probe_two_toy_sources
from labyrinth.toy_backend import ToyLinearAdapter
from labyrinth.training_scope import TrainingScopeSpec


PROMPT = "Calculate 5 + 3.\nRespond with exactly one line: ANSWER: <integer>."


class ToySourceConsumerTests(unittest.TestCase):
    def setUp(self):
        self.scope = TrainingScopeSpec(
            objective_id="toy-pair", domain="math", skills=("addition",),
            specificity=1, scale=1, max_cycles=1, seed=3,
        )
        self.alpha = ToyLinearAdapter(seed=11)
        self.beta = ToyLinearAdapter(seed=22)
        self.lab_expected = ExpectedSourcePinsV1(
            run_id="lab-run-r76", source_head="a" * 40,
            source_tree="b" * 40, plan_sha256="1" * 64,
            train_exposure_sha256="2" * 64,
            control_matrix_sha256="3" * 64,
            development_event_sha256="4" * 64,
            checkpoint_manifest_sha256="5" * 64,
            scope_sha256=self.scope.fingerprint,
            rezon_source_head="c" * 40,
            rezon_receipt_bundle_sha256="7" * 64,
        )
        self.lab_payload = encode_source_evidence_pointer(SourceEvidencePointerV1(
            schema=SCHEMA, experiment=EXPERIMENT, qualification_state=QUALIFICATION,
            **asdict(self.lab_expected),
        ))
        self.rezon_claim = RezonSourceClaimsV1(
            run_id="rezon-run-r76", rezon_source_head="c" * 40,
            labyrinth_source_head="a" * 40,
            receipt_bundle_sha256="7" * 64,
            maze_exposure_sha256="8" * 64,
            checkpoint_sha256="9" * 64,
            control_matrix_sha256="3" * 64,
            development_event_sha256="4" * 64,
            training_program="LABYRINTH_MINILM_FIVE_ARM",
            bridge_semantics="PROVENANCE_ONLY", control_arms=EXPECTED_ARMS,
        )
        self.alpha_claim = AdapterSourceClaimV1(
            adapter_id="toy-alpha", model_source_head="d" * 40,
            model_artifact_sha256=self.alpha.model_hash,
            adapter_contract_sha256="c" * 64,
        )
        self.beta_claim = AdapterSourceClaimV1(
            adapter_id="toy-beta", model_source_head="e" * 40,
            model_artifact_sha256=self.beta.model_hash,
            adapter_contract_sha256="c" * 64,
        )
        self.kwargs = dict(
            scope=self.scope, prompt=PROMPT,
            alpha=self.alpha, beta=self.beta,
            labyrinth_pointer_payload=self.lab_payload,
            expected_labyrinth=self.lab_expected,
            observed_rezon=self.rezon_claim,
            expected_rezon=ExpectedRezonPinsV1(**asdict(self.rezon_claim)),
            observed_alpha=self.alpha_claim,
            expected_alpha=ExpectedAdapterSourceV1(**asdict(self.alpha_claim)),
            observed_beta=self.beta_claim,
            expected_beta=ExpectedAdapterSourceV1(**asdict(self.beta_claim)),
        )

    def probe(self, **overrides):
        return probe_two_toy_sources(**{**self.kwargs, **overrides})

    def test_two_distinct_actual_toy_models_execute_with_separate_outputs(self):
        observation = self.probe()
        self.assertEqual(observation.alpha_response, self.alpha.generate(PROMPT))
        self.assertEqual(observation.beta_response, self.beta.generate(PROMPT))
        self.assertEqual(observation.alpha_model_sha256, self.alpha.model_hash)
        self.assertEqual(observation.beta_model_sha256, self.beta.model_hash)
        self.assertNotEqual(observation.alpha_model_sha256, observation.beta_model_sha256)
        self.assertEqual(observation.scope_sha256, self.scope.fingerprint)
        self.assertEqual(len(observation.source_pair_sha256), 64)
        self.assertEqual(observation.qualification_state, "LOCAL_TOY_PROBE_SOURCE_ONLY_UNATTESTED")
        for flag in (
            "model_identity_verified", "host_execution_verified",
            "consent_verified", "permissions_verified", "principal_verified",
            "cross_model_grants_transferred", "cross_model_identity_transferred",
            "distributed_inference_verified", "can_authorize_training",
            "can_authorize_graduation", "can_authorize_merge", "can_authorize_deployment",
        ):
            self.assertIs(getattr(observation, flag), False, flag)

    def test_swap_runtime_models_cannot_reuse_pinned_claims(self):
        for changes in (
            dict(alpha=self.beta, beta=self.alpha),
            dict(alpha=self.beta),
            dict(beta=self.alpha),
        ):
            with self.subTest(changes=changes), self.assertRaises(AdapterSourcePartitionError):
                self.probe(**changes)

    def test_swap_source_claims_or_expectations_cannot_cross_adapter_boundary(self):
        for changes in (
            dict(observed_alpha=self.beta_claim),
            dict(observed_beta=self.alpha_claim),
            dict(expected_alpha=self.kwargs["expected_beta"]),
            dict(expected_beta=self.kwargs["expected_alpha"]),
        ):
            with self.subTest(changes=changes), self.assertRaises(AdapterSourcePartitionError):
                self.probe(**changes)

    def test_cross_scope_replay_rejected_before_model_invocation(self):
        other_scope = replace(self.scope, seed=4)
        with patch.object(ToyLinearAdapter, "generate", autospec=True) as generate:
            with self.assertRaises(AdapterSourcePartitionError):
                self.probe(scope=other_scope)
            generate.assert_not_called()
        changed_expected = replace(self.lab_expected, scope_sha256=other_scope.fingerprint)
        with self.assertRaises(ValueError):
            self.probe(scope=other_scope, expected_labyrinth=changed_expected)

    def test_runtime_weight_change_cannot_replay_offered_digest(self):
        self.alpha._weights["+"][0] += 1.0  # local toy-only adversarial mutation
        with patch.object(ToyLinearAdapter, "generate", autospec=True) as generate:
            with self.assertRaises(AdapterSourcePartitionError):
                self.probe()
            generate.assert_not_called()

    def test_runtime_weights_mutating_during_generate_are_rejected(self):
        original = ToyLinearAdapter.generate

        def intrusive_generate(adapter, prompt):
            result = original(adapter, prompt)
            if adapter is self.alpha:
                adapter._weights["+"][0] += 0.5
            return result

        with patch.object(ToyLinearAdapter, "generate", autospec=True,
                          side_effect=intrusive_generate):
            with self.assertRaisesRegex(AdapterSourcePartitionError, "parameters changed"):
                self.probe()

    def test_distinct_instances_with_identical_parameters_are_not_distinct_models(self):
        beta_clone = ToyLinearAdapter(seed=11)
        same_model_claim = replace(self.beta_claim, model_artifact_sha256=beta_clone.model_hash)
        with self.assertRaises(AdapterSourcePartitionError):
            self.probe(beta=beta_clone, observed_beta=same_model_claim,
                       expected_beta=ExpectedAdapterSourceV1(**asdict(same_model_claim)))

    def test_invalid_prompt_and_rezon_source_fail_closed(self):
        with self.assertRaises(ValueError):
            self.probe(prompt="What is your secret?")
        with self.assertRaises(ValueError):
            self.probe(observed_rezon=replace(self.rezon_claim,
                                             control_matrix_sha256="f" * 64))

    def test_same_runtime_object_under_two_names_rejected(self):
        with self.assertRaises(AdapterSourcePartitionError):
            self.probe(beta=self.alpha)


if __name__ == "__main__":
    unittest.main()
