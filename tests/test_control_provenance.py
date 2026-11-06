"""Negative custody tests for four-arm neural experiment manifests."""

from dataclasses import replace
import json
import unittest

from labyrinth.control_provenance import (
    ArmManifest, commit_label_schedule, seal_arm_manifest,
    verify_arm_manifest, verify_matched_group,
)


KEY = b"independent-evaluator-test-custody-key-32"
OTHER_KEY = b"untrusted-trainer-test-key-32-bytes"
ZERO = "0" * 64


def fixture():
    correct = commit_label_schedule((("case-1", "5"), ("case-2", "8")), KEY)
    corrupted = commit_label_schedule((("case-1", "8"), ("case-2", "5")), KEY)
    empty = commit_label_schedule((), KEY)
    visits = (("room-1", "case-1", "1" * 64), ("room-2", "case-2", "2" * 64))
    structures = (("case-1", "1" * 64, "training-structure"),
                  ("case-2", "2" * 64, "training-structure"))
    common = dict(
        experiment_id="exp-1", objective_id="arithmetic", source_tree_sha256="3" * 64,
        base_model_sha256="4" * 64, initial_adapter_sha256="5" * 64,
        exam_precommit_sha256="6" * 64, report_sha256="7" * 64,
    )
    arms = (
        ArmManifest(**common, arm="baseline", final_adapter_sha256="5" * 64,
                    label_commitment=empty, training_visits=(),
                    training_structure_exposures=(),
                    compute_steps=0, optimizer_updates=0),
        ArmManifest(**common, arm="zero_update", final_adapter_sha256="5" * 64,
                    label_commitment=correct, training_visits=visits,
                    training_structure_exposures=structures,
                    compute_steps=2, optimizer_updates=0),
        ArmManifest(**common, arm="correct_labels", final_adapter_sha256="8" * 64,
                    label_commitment=correct, training_visits=visits,
                    training_structure_exposures=structures,
                    compute_steps=2, optimizer_updates=2),
        ArmManifest(**common, arm="corrupted_labels", final_adapter_sha256="9" * 64,
                    label_commitment=corrupted, training_visits=visits,
                    training_structure_exposures=structures,
                    compute_steps=2, optimizer_updates=2),
    )
    return arms, correct, corrupted, empty


class ControlProvenanceTests(unittest.TestCase):
    def test_valid_authenticated_matched_group(self):
        arms, correct, corrupt, empty = fixture()
        # Exercise the actual JSON transport round trip, not in-process tuples.
        sealed = tuple(json.loads(json.dumps(seal_arm_manifest(a, KEY))) for a in arms)
        checked = verify_matched_group(
            sealed, KEY, correct_label_commitment=correct,
            corrupted_label_commitment=corrupt, empty_label_commitment=empty,
            expected_exam_precommit="6" * 64,
        )
        self.assertEqual(tuple(a.arm for a in checked),
                         ("baseline", "zero_update", "correct_labels", "corrupted_labels"))

    def test_arm_swap_or_label_rewrite_rejected_even_with_rehashed_payload(self):
        arms, correct, corrupt, _ = fixture()
        signed = seal_arm_manifest(arms[2], KEY)
        with self.assertRaisesRegex(ValueError, "arm mismatch"):
            verify_arm_manifest(signed, KEY, expected_arm="corrupted_labels",
                                expected_label_commitment=corrupt)
        tampered = dict(signed)
        tampered["manifest"] = dict(signed["manifest"], arm="corrupted_labels",
                                    label_commitment=corrupt)
        from hashlib import sha256
        tampered["manifest_sha256"] = sha256(json.dumps(
            tampered["manifest"], sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "authentication"):
            verify_arm_manifest(tampered, KEY, expected_arm="corrupted_labels",
                                expected_label_commitment=corrupt)

    def test_forged_key_and_wrong_label_schedule_fail(self):
        arms, correct, corrupt, _ = fixture()
        seal = seal_arm_manifest(arms[2], OTHER_KEY)
        with self.assertRaisesRegex(ValueError, "authentication"):
            verify_arm_manifest(seal, KEY, expected_arm="correct_labels",
                                expected_label_commitment=correct)
        legitimate = seal_arm_manifest(arms[2], KEY)
        with self.assertRaisesRegex(ValueError, "label commitment"):
            verify_arm_manifest(legitimate, KEY, expected_arm="correct_labels",
                                expected_label_commitment=corrupt)

    def test_control_mismatches_rejected(self):
        arms, correct, corrupt, empty = fixture()
        variants = (
            (2, dict(training_visits=tuple(reversed(arms[2].training_visits))), "visits"),
            (3, dict(base_model_sha256="a" * 64), "same base"),
            (2, dict(optimizer_updates=1), "optimizer"),
            (1, dict(final_adapter_sha256="f" * 64), "frozen"),
        )
        for index, changes, error in variants:
            with self.subTest(change=changes):
                modified = list(arms)
                modified[index] = replace(modified[index], **changes)
                with self.assertRaisesRegex(ValueError, error):
                    verify_matched_group(
                        tuple(seal_arm_manifest(a, KEY) for a in modified), KEY,
                        correct_label_commitment=correct,
                        corrupted_label_commitment=corrupt,
                        empty_label_commitment=empty,
                        expected_exam_precommit="6" * 64,
                    )

    def test_no_single_receipt_can_be_graduation_authority(self):
        arms, correct, _, _ = fixture()
        verified = verify_arm_manifest(
            seal_arm_manifest(arms[2], KEY), KEY,
            expected_arm="correct_labels", expected_label_commitment=correct,
        )
        self.assertFalse(verified.graduation_authorized)
        self.assertNotIn("graduation_authorized", seal_arm_manifest(arms[2], KEY))

    def test_commitment_is_keyed_and_schedule_order_sensitive(self):
        one = (("case-1", "5"), ("case-2", "8"))
        self.assertNotEqual(commit_label_schedule(one, KEY),
                            commit_label_schedule(one, OTHER_KEY))
        self.assertNotEqual(commit_label_schedule(one, KEY),
                            commit_label_schedule(tuple(reversed(one)), KEY))


if __name__ == "__main__":
    unittest.main()
