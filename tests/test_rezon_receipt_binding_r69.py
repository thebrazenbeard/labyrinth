"""Hostile tests for opaque Rezon bundle *byte* identity, never host custody."""

from dataclasses import asdict, replace
from hashlib import sha256
import unittest

from labyrinth.evidence_pointer import (
    EXPERIMENT, QUALIFICATION, SCHEMA, ExpectedSourcePinsV1,
    SourceEvidencePointerV1, encode_source_evidence_pointer,
)
from labyrinth.rezon_pointer_mapping_r57 import (
    EXPECTED_ARMS, ExpectedRezonPinsV1, RezonSourceAlignmentError,
    RezonSourceClaimsV1,
)
from labyrinth.rezon_receipt_binding_r69 import (
    MAX_BUNDLE_BYTES, compare_bundle_bytes,
)


class CrossRepoByteBindingR69Tests(unittest.TestCase):
    def setUp(self):
        self.bundle = b'{"host":"fixture","role":"effect","status":"SOURCE_ONLY"}\n'
        self.digest = sha256(self.bundle).hexdigest()
        self.lab = ExpectedSourcePinsV1(
            run_id="labyrinth-run-A", source_head="a" * 40, source_tree="b" * 40,
            plan_sha256="1" * 64, train_exposure_sha256="2" * 64,
            control_matrix_sha256="3" * 64, development_event_sha256="4" * 64,
            checkpoint_manifest_sha256="5" * 64, scope_sha256="6" * 64,
            rezon_source_head="c" * 40,
            rezon_receipt_bundle_sha256=self.digest,
        )
        self.pointer = SourceEvidencePointerV1(
            schema=SCHEMA, experiment=EXPERIMENT, qualification_state=QUALIFICATION,
            **asdict(self.lab),
        )
        self.rezon = RezonSourceClaimsV1(
            run_id="rezon-episode-B", rezon_source_head="c" * 40,
            labyrinth_source_head="a" * 40,
            receipt_bundle_sha256=self.digest,
            maze_exposure_sha256="8" * 64, checkpoint_sha256="9" * 64,
            control_matrix_sha256="3" * 64,
            development_event_sha256="4" * 64,
            training_program="LABYRINTH_MINILM_FIVE_ARM",
            bridge_semantics="PROVENANCE_ONLY", control_arms=EXPECTED_ARMS,
        )
        self.expected_rezon = ExpectedRezonPinsV1(**asdict(self.rezon))

    def observe(self, *, bundle=None, lab=None, pointer=None, rezon=None,
                expected_rezon=None):
        return compare_bundle_bytes(
            encode_source_evidence_pointer(pointer or self.pointer),
            self.bundle if bundle is None else bundle,
            expected_labyrinth=self.lab if lab is None else lab,
            observed_rezon=self.rezon if rezon is None else rezon,
            expected_rezon=(self.expected_rezon if expected_rezon is None
                            else expected_rezon),
        )

    def test_actual_bundle_bytes_match_both_independent_pin_sets(self):
        observed = self.observe()
        self.assertEqual(observed.bundle_sha256, self.digest)
        self.assertEqual((observed.labyrinth_run_id, observed.rezon_run_id),
                         ("labyrinth-run-A", "rezon-episode-B"))
        self.assertEqual(observed.disposition,
                         "SOURCE_BUNDLE_BYTES_MATCH_UNATTESTED")
        self.assertIs(observed.source_bundle_bytes_matched, True)

    def test_byte_mutation_and_old_generation_replay_are_rejected(self):
        for altered in (self.bundle + b" ", self.bundle[:-1],
                        self.bundle.replace(b"effect", b"lease")):
            with self.subTest(altered=altered[:12]), self.assertRaisesRegex(
                    RezonSourceAlignmentError, "bundle bytes differ"):
                self.observe(bundle=altered)
        # Even valid *old* bytes cannot match new independent expected source
        # pins. This is not host-protected monotonicity: old pins can be cloned.
        current = b'{"generation":2,"status":"SOURCE_ONLY"}\n'
        fresh_digest = sha256(current).hexdigest()
        fresh_lab = replace(self.lab, rezon_receipt_bundle_sha256=fresh_digest)
        fresh_pointer = replace(self.pointer,
                                rezon_receipt_bundle_sha256=fresh_digest)
        fresh_rezon = replace(self.rezon, receipt_bundle_sha256=fresh_digest)
        fresh_expected = ExpectedRezonPinsV1(**asdict(fresh_rezon))
        with self.assertRaisesRegex(RezonSourceAlignmentError, "bundle bytes differ"):
            self.observe(bundle=self.bundle, lab=fresh_lab, pointer=fresh_pointer,
                         rezon=fresh_rezon, expected_rezon=fresh_expected)
        self.assertEqual(self.observe(bundle=current, lab=fresh_lab,
                                      pointer=fresh_pointer, rezon=fresh_rezon,
                                      expected_rezon=fresh_expected).bundle_sha256,
                         fresh_digest)

    def test_source_mismatch_rejected_even_if_bundle_bytes_match(self):
        with self.assertRaises(RezonSourceAlignmentError):
            self.observe(rezon=replace(self.rezon,
                                      labyrinth_source_head="d" * 40))
        with self.assertRaises(RezonSourceAlignmentError):
            self.observe(expected_rezon=replace(self.expected_rezon,
                                               checkpoint_sha256="a" * 64))

    def test_bundle_must_be_exact_bounded_nonempty_bytes(self):
        class MaliciousBytes(bytes):
            def __eq__(self, other):
                return True

        for bogus in (b"", "string", bytearray(self.bundle),
                      MaliciousBytes(self.bundle), b"X" * (MAX_BUNDLE_BYTES + 1)):
            with self.subTest(kind=type(bogus).__name__), self.assertRaisesRegex(
                    RezonSourceAlignmentError, "bounded exact"):
                self.observe(bundle=bogus)

    def test_matching_source_bundle_cannot_grant_authority_or_receipt_validity(self):
        observed = self.observe()
        for name in ("source_receipts_verified", "host_execution_verified",
                     "replay_exclusion_verified", "model_identity_verified",
                     "consent_verified", "permissions_verified",
                     "can_authorize_training", "can_authorize_graduation",
                     "can_authorize_merge", "can_authorize_deployment"):
            self.assertIs(getattr(observed, name), False, name)
            with self.subTest(name=name), self.assertRaises(
                    (TypeError, AttributeError)):
                object.__setattr__(observed, name, True)
        with self.assertRaises(TypeError):
            type(observed)(*observed, can_authorize_deployment=True)


if __name__ == "__main__":
    unittest.main()
