"""Hostile source-only tests for the Labyrinth/Rezon pointer boundary."""

from dataclasses import asdict, replace
import unittest

from labyrinth.evidence_pointer import (
    EXPERIMENT, QUALIFICATION, SCHEMA, ExpectedSourcePinsV1,
    SourceEvidenceError, SourceEvidencePointerV1, encode_source_evidence_pointer,
)
from labyrinth.rezon_pointer_mapping_r57 import (
    EXPECTED_ARMS, ExpectedRezonPinsV1, RezonSourceClaimsV1,
    RezonSourceAlignmentError, compare_source_evidence,
)


class CrossRepoSourceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.lab = ExpectedSourcePinsV1(
            run_id="labyrinth-run-A", source_head="a" * 40, source_tree="b" * 40,
            plan_sha256="1" * 64, train_exposure_sha256="2" * 64,
            control_matrix_sha256="3" * 64, development_event_sha256="4" * 64,
            checkpoint_manifest_sha256="5" * 64, scope_sha256="6" * 64,
            rezon_source_head="c" * 40,
            rezon_receipt_bundle_sha256="7" * 64,
        )
        self.pointer = SourceEvidencePointerV1(
            schema=SCHEMA, experiment=EXPERIMENT, qualification_state=QUALIFICATION,
            **asdict(self.lab),
        )
        self.rezon = RezonSourceClaimsV1(
            run_id="rezon-episode-B", rezon_source_head="c" * 40,
            labyrinth_source_head="a" * 40,
            receipt_bundle_sha256="7" * 64,
            maze_exposure_sha256="8" * 64,
            checkpoint_sha256="9" * 64,
            control_matrix_sha256="3" * 64,
            development_event_sha256="4" * 64,
            training_program="LABYRINTH_MINILM_FIVE_ARM",
            bridge_semantics="PROVENANCE_ONLY", control_arms=EXPECTED_ARMS,
        )
        self.expected_rezon = ExpectedRezonPinsV1(**asdict(self.rezon))

    def align(self, *, lab=None, pointer=None, rezon=None, expected_rezon=None):
        return compare_source_evidence(
            encode_source_evidence_pointer(pointer or self.pointer),
            expected_labyrinth=lab or self.lab,
            observed_rezon=rezon or self.rezon,
            expected_rezon=expected_rezon or self.expected_rezon,
        )

    def test_exact_source_alignment_does_not_equate_distinct_artifacts(self):
        result = self.align()
        self.assertEqual(result.disposition, "OBSERVED_SOURCE")
        self.assertEqual(result.qualification_state, "PARTIAL_SOURCE_ALIGNMENT_UNATTESTED")
        self.assertEqual(result.labyrinth_run_id, "labyrinth-run-A")
        self.assertEqual(result.rezon_run_id, "rezon-episode-B")
        self.assertEqual(result.unresolved_semantic_pairs, (
            ("train_exposure_sha256", "maze_exposure_sha256"),
            ("checkpoint_manifest_sha256", "checkpoint_sha256"),
        ))
        for name in (
            "source_receipts_verified", "host_execution_verified", "checkpoint_verified",
            "model_identity_verified", "consent_verified", "permissions_verified",
            "cross_model_grants_transferred", "can_authorize_training",
            "can_authorize_graduation", "can_authorize_merge",
            "can_authorize_deployment", "agi_qualification_verified",
        ):
            self.assertIs(getattr(result, name), False, name)
        with self.assertRaises((AttributeError, TypeError)):
            result.can_authorize_training = True

    def test_object_setattr_cannot_escalate_immutable_observation(self):
        result = self.align()
        for field, forged in (
            ("can_authorize_deployment", True),
            ("can_authorize_training", True),
            ("qualification_state", "AUTHORIZED_CAPABILITY"),
            ("disposition", "AUTHORIZED_CAPABILITY"),
            ("consent_verified", True),
            ("permissions_verified", True),
            ("agi_qualification_verified", True),
        ):
            with self.subTest(field=field), self.assertRaises((AttributeError, TypeError)):
                object.__setattr__(result, field, forged)
        self.assertEqual(result.qualification_state, "PARTIAL_SOURCE_ALIGNMENT_UNATTESTED")
        self.assertIs(result.can_authorize_deployment, False)
        with self.assertRaises(TypeError):
            type(result)(
                labyrinth_run_id=result.labyrinth_run_id,
                rezon_run_id=result.rezon_run_id,
                can_authorize_deployment=True,
            )

    def test_every_independent_rezon_pin_is_checked(self):
        replacements = {
            "run_id": "different-episode", "rezon_source_head": "d" * 40,
            "labyrinth_source_head": "d" * 40,
            "receipt_bundle_sha256": "f" * 64,
            "maze_exposure_sha256": "f" * 64,
            "checkpoint_sha256": "f" * 64,
            "control_matrix_sha256": "f" * 64,
            "development_event_sha256": "f" * 64,
            "training_program": "OTHER", "bridge_semantics": "OTHER",
            "control_arms": ("correct",),
        }
        for field, alternate in replacements.items():
            with self.subTest(field=field), self.assertRaises(RezonSourceAlignmentError):
                self.align(expected_rezon=replace(self.expected_rezon, **{field: alternate}))

    def test_rezon_claims_must_match_independent_expectations(self):
        for field, alternate in (
            ("run_id", "other-episode"),
            ("maze_exposure_sha256", "a" * 64),
            ("checkpoint_sha256", "b" * 64),
        ):
            with self.subTest(field=field), self.assertRaises(RezonSourceAlignmentError):
                self.align(rezon=replace(self.rezon, **{field: alternate}))

    def test_rezon_cross_boundary_pins_must_match_lab_source(self):
        for field, alternate in (
            ("labyrinth_source_head", "d" * 40),
            ("rezon_source_head", "d" * 40),
            ("receipt_bundle_sha256", "d" * 64),
            ("control_matrix_sha256", "d" * 64),
            ("development_event_sha256", "d" * 64),
        ):
            modified = replace(self.rezon, **{field: alternate})
            independently_pinned = ExpectedRezonPinsV1(**asdict(modified))
            with self.subTest(field=field), self.assertRaises(RezonSourceAlignmentError):
                self.align(rezon=modified, expected_rezon=independently_pinned)

    def test_rezon_link_cannot_be_invented_when_lab_pointer_omits_it(self):
        no_rezon_pointer = replace(self.pointer, rezon_source_head=None,
                                   rezon_receipt_bundle_sha256=None)
        no_rezon_lab = replace(self.lab, rezon_source_head=None,
                               rezon_receipt_bundle_sha256=None)
        with self.assertRaises(RezonSourceAlignmentError):
            self.align(pointer=no_rezon_pointer, lab=no_rezon_lab)

    def test_lab_source_tree_and_run_cannot_be_replayed_or_invented(self):
        for field, alternate in (
            ("source_tree", "d" * 40), ("source_head", "e" * 40),
            ("run_id", "stale-run"),
        ):
            with self.subTest(field=field), self.assertRaises(SourceEvidenceError):
                self.align(lab=replace(self.lab, **{field: alternate}))

    def test_qwen_or_privileged_rezon_source_claims_are_rejected(self):
        for field, alternate in (
            ("training_program", "MODEL_TRAINING_QWEN_FOUR_CONTROL"),
            ("bridge_semantics", "EXECUTION_AUTHORIZED"),
            ("control_arms", ("correct", "zero_update", "inverted", "zero_lr")),
        ):
            changed = replace(self.rezon, **{field: alternate})
            expectation = ExpectedRezonPinsV1(**asdict(changed))
            with self.subTest(field=field), self.assertRaises(RezonSourceAlignmentError):
                self.align(rezon=changed, expected_rezon=expectation)

    def test_type_spoof_and_extra_auth_fields_cannot_enter_adapter(self):
        with self.assertRaises(TypeError):
            RezonSourceClaimsV1(**dict(asdict(self.rezon), consent=True))
        with self.assertRaises(TypeError):
            RezonSourceClaimsV1(**dict(asdict(self.rezon), model_identity="model-B"))
        for field, alternate in (
            ("maze_exposure_sha256", True),
            ("rezon_source_head", "C" * 40),
            ("run_id", "../../replay"),
            ("control_arms", list(EXPECTED_ARMS)),
        ):
            changed = replace(self.rezon, **{field: alternate})
            with self.subTest(field=field), self.assertRaises(RezonSourceAlignmentError):
                self.align(rezon=changed)

    def test_source_observation_cannot_carry_between_distinct_model_adapters(self):
        # The same source statement may be observed by A and B, but neither
        # result can encode or transfer model-specific identity or privileges.
        adapter_a = self.align()
        adapter_b = self.align()
        self.assertEqual(adapter_a, adapter_b)
        for forbidden in ("model_id", "consent", "permissions", "authority", "lease"):
            self.assertFalse(hasattr(adapter_a, forbidden))
            self.assertFalse(hasattr(adapter_b, forbidden))
        self.assertIs(adapter_a.cross_model_grants_transferred, False)
        self.assertIs(adapter_b.cross_model_grants_transferred, False)


if __name__ == "__main__":
    unittest.main()
