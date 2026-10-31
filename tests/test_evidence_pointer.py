"""Source-only provenance pointers never become authority grants."""

from dataclasses import asdict, replace
import json
import unittest

from labyrinth.evidence_pointer import (
    EXPERIMENT,
    QUALIFICATION,
    SCHEMA,
    ExpectedSourcePinsV1,
    SourceEvidenceError,
    SourceEvidenceMatchV1,
    SourceEvidencePointerV1,
    encode_source_evidence_pointer,
    verify_source_evidence_pointer,
)


class SourceEvidencePointerTests(unittest.TestCase):
    def setUp(self):
        self.expected = ExpectedSourcePinsV1(
            run_id="minilm-run-20261031-a",
            source_head="a" * 40,
            source_tree="b" * 40,
            plan_sha256="1" * 64,
            train_exposure_sha256="2" * 64,
            control_matrix_sha256="3" * 64,
            development_event_sha256="4" * 64,
            checkpoint_manifest_sha256="5" * 64,
            scope_sha256="6" * 64,
            rezon_source_head="c" * 40,
            rezon_receipt_bundle_sha256="7" * 64,
        )
        self.pointer = SourceEvidencePointerV1(
            schema=SCHEMA,
            experiment=EXPERIMENT,
            qualification_state=QUALIFICATION,
            **asdict(self.expected),
        )

    def _verified(self, pointer=None, expected=None):
        return verify_source_evidence_pointer(
            encode_source_evidence_pointer(pointer or self.pointer),
            expected=expected or self.expected,
        )

    def test_exact_expected_pins_pass_with_no_authority(self):
        result = self._verified()
        self.assertEqual(result.pointer, self.pointer)
        self.assertEqual(result.qualification_state, "SOURCE_ONLY_UNATTESTED")
        for name in (
            "host_execution_verified", "independent_checkpoint_verified",
            "model_identity_verified", "consent_verified", "permissions_verified",
            "can_authorize_training", "can_authorize_graduation",
            "can_authorize_merge", "can_authorize_deployment",
        ):
            self.assertIs(getattr(result, name), False, name)
        with self.assertRaises((AttributeError, TypeError)):
            result.can_authorize_deployment = True
        with self.assertRaises(TypeError):
            SourceEvidenceMatchV1(pointer=self.pointer, can_authorize_deployment=True)

    def test_every_required_pin_must_match_independent_expectation(self):
        for name in (
            "run_id", "source_head", "source_tree", "plan_sha256",
            "train_exposure_sha256", "control_matrix_sha256",
            "development_event_sha256", "checkpoint_manifest_sha256",
            "scope_sha256", "rezon_source_head", "rezon_receipt_bundle_sha256",
        ):
            current = getattr(self.expected, name)
            different = ("other-run" if name == "run_id" else
                         ("f" if len(current) == 40 else "e") * len(current))
            with self.subTest(name=name), self.assertRaises(SourceEvidenceError):
                self._verified(expected=replace(self.expected, **{name: different}))

    def test_source_commit_and_generation_cannot_be_replayed_after_pin_change(self):
        stale = self.pointer
        new_source = replace(self.expected, source_head="d" * 40,
                             source_tree="e" * 40)
        with self.assertRaises(SourceEvidenceError):
            self._verified(pointer=stale, expected=new_source)

    def test_qwen_four_control_experiment_substitution_is_rejected(self):
        with self.assertRaises(SourceEvidenceError):
            encode_source_evidence_pointer(replace(
                self.pointer, experiment="model-training.qwen-four-control.v1"))

    def test_unknown_schema_status_and_unauthorized_claims_are_rejected(self):
        for field, value in (
            ("schema", "labyrinth.source-evidence-pointer.v2"),
            ("qualification_state", "GRADUATED"),
        ):
            with self.subTest(field=field), self.assertRaises(SourceEvidenceError):
                encode_source_evidence_pointer(replace(self.pointer, **{field: value}))
        data = json.loads(encode_source_evidence_pointer(self.pointer))
        for field, value in (
            ("can_authorize_training", True), ("host_execution_verified", True),
            ("identity", "model-A"), ("consent", True), ("permissions", ["write"]),
            ("signature", "caller-claimed"),
        ):
            hostile = dict(data, **{field: value})
            with self.subTest(field=field), self.assertRaises(SourceEvidenceError):
                verify_source_evidence_pointer(self._json(hostile), expected=self.expected)

    def test_missing_optional_rezon_pins_must_be_jointly_absent(self):
        for key in ("rezon_source_head", "rezon_receipt_bundle_sha256"):
            with self.subTest(key=key), self.assertRaises(SourceEvidenceError):
                encode_source_evidence_pointer(replace(self.pointer, **{key: None}))
        neither = replace(self.pointer, rezon_source_head=None,
                          rezon_receipt_bundle_sha256=None)
        expected = replace(self.expected, rezon_source_head=None,
                           rezon_receipt_bundle_sha256=None)
        self.assertEqual(self._verified(pointer=neither, expected=expected).pointer, neither)

    def test_invalid_digests_ids_and_noncanonical_source_are_rejected(self):
        for name, value in (
            ("plan_sha256", "z" * 64), ("scope_sha256", "A" * 64),
            ("source_head", "a" * 64), ("run_id", "../../secret"),
            ("run_id", ""), ("run_id", "x" * 129),
        ):
            with self.subTest(name=name), self.assertRaises(SourceEvidenceError):
                encode_source_evidence_pointer(replace(self.pointer, **{name: value}))
        raw = encode_source_evidence_pointer(self.pointer)
        for malformed in (raw[:-1], b" " + raw, raw + b"\n", raw + b" " * 5000,
                          raw.replace(b'"schema":', b'"schema":"forged","schema":', 1),
                          b'{}\n', b'[]\n', b'not json'):
            with self.subTest(malformed=malformed[:40]), self.assertRaises(SourceEvidenceError):
                verify_source_evidence_pointer(malformed, expected=self.expected)

    def test_invalid_expected_pins_fail_closed_even_if_equal_to_payload(self):
        invalid = replace(self.expected, checkpoint_manifest_sha256="invalid")
        with self.assertRaises(SourceEvidenceError):
            verify_source_evidence_pointer(encode_source_evidence_pointer(self.pointer),
                                           expected=invalid)

    @staticmethod
    def _json(value):
        return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True) + "\n").encode("ascii")


if __name__ == "__main__":
    unittest.main()
