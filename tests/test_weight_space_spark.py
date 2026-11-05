"""Hostile metadata-only tests: this module must never open or edit a model."""

from __future__ import annotations

from copy import deepcopy
import json
import unittest

from labyrinth.weight_space_spark import (
    SCHEMA, create_sparse_edit_proposal, verify_sparse_edit_proposal,
    w0_manifest_sha256,
)


MODEL = "a" * 64
MANIFEST = [
    {"name": "layer.0.weight", "shape": [100], "dtype": "float16",
     "tensor_sha256": "b" * 64},
    {"name": "layer.1.weight", "shape": [100], "dtype": "float32",
     "tensor_sha256": "c" * 64},
]
EDIT = {"tensor": "layer.0.weight", "index": [3], "old": 0.0, "new": 0.25}


def proposal(*, edits=None, manifest=None, nonce="d" * 64, budget=10000):
    return create_sparse_edit_proposal(
        source_model_sha256=MODEL, w0_manifest=deepcopy(MANIFEST if manifest is None else manifest),
        proposal_nonce_sha256=nonce, proposer_receipt_sha256="e" * 64,
        edits=deepcopy([EDIT] if edits is None else edits), budget_ppm=budget,
    )


def verify(artifact, *, manifest=None, model=MODEL, nonces=(), digests=()):
    return verify_sparse_edit_proposal(
        artifact, expected_source_model_sha256=model,
        expected_w0_manifest=MANIFEST if manifest is None else manifest,
        seen_proposal_nonces=nonces, seen_proposal_sha256=digests,
    )


class WeightSpaceSparkTests(unittest.TestCase):
    def test_opt_in_proposal_is_deterministic_non_executable_and_json_only(self):
        a = proposal()
        b = proposal()
        self.assertEqual(a, b)
        self.assertEqual(a["schema"], SCHEMA)
        self.assertIs(a["execution_authorized"], False)
        self.assertEqual(a["provenance"]["input_boundary"], "w0_metadata_only")
        self.assertEqual(a["rollback_receipt"]["status"], "NOT_EXECUTED")
        self.assertEqual(a["paired_control_receipt"]["status"], "NOT_EVALUATED")
        self.assertEqual(a["w0_manifest_sha256"], w0_manifest_sha256(MANIFEST))
        self.assertEqual(a["proposal_sha256"], verify(a))
        self.assertEqual(a, json.loads(json.dumps(a, allow_nan=False)))
        self.assertNotIn("gold", str(a).lower())
        for forbidden in ("train_case_ids", "heldout", "exam_prompts", "answers"):
            self.assertNotIn(forbidden, a)

    def test_fails_closed_on_nonfinite_wrong_w0_bounds_repeat_and_budget(self):
        second = {"tensor": "layer.0.weight", "index": [4], "old": 0.0, "new": 0.5}
        third = {"tensor": "layer.1.weight", "index": [5], "old": 1.0, "new": 1.5}
        for name, kwargs in (
            ("nan", {"edits": [{**EDIT, "new": float("nan") }]}),
            ("infinity", {"edits": [{**EDIT, "new": float("inf") }]}),
            ("dtype_overflow", {"edits": [{**EDIT, "new": 70_000.0}]}),
            ("huge_integer", {"edits": [{**EDIT, "new": 10**1000}]}),
            ("negative_index", {"edits": [{**EDIT, "index": [-1]}]}),
            ("out_of_bounds", {"edits": [{**EDIT, "index": [100]}]}),
            ("boolean_index", {"edits": [{**EDIT, "index": [True]}]}),
            ("unknown_tensor", {"edits": [{**EDIT, "tensor": "unlisted"}]}),
            ("duplicate_coordinate", {"edits": [EDIT, EDIT]}),
            ("too_many_edits", {"edits": [EDIT, second, third]}),
            ("too_high_budget", {"budget": 10_001}),
            ("no_edit", {"edits": [{**EDIT, "new": 0.0}]}),
            ("manifest_collision", {"manifest": [MANIFEST[0], MANIFEST[0]]}),
            ("unordered_manifest", {"manifest": list(reversed(MANIFEST))}),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                proposal(**kwargs)

    def test_collisions_are_coordinates_not_numeric_flat_indexes(self):
        # Identical index in different tensor is NOT a collision, but two
        # changes to one tensor coordinate are forbidden.
        distinct = [{**EDIT}, {**EDIT, "tensor": "layer.1.weight"}]
        artifact = proposal(edits=distinct)
        self.assertEqual(len(artifact["edits"]), 2)
        self.assertTrue(verify(artifact))
        with self.assertRaisesRegex(ValueError, "duplicate tensor coordinate"):
            proposal(edits=[EDIT, {**EDIT, "new": 0.75}])

    def test_rejects_wrong_source_or_manifest_and_tampering(self):
        artifact = proposal()
        changed_manifest = deepcopy(MANIFEST)
        changed_manifest[0]["tensor_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "wrong W0 model"):
            verify(artifact, model="f" * 64)
        with self.assertRaisesRegex(ValueError, "manifest does not match"):
            verify(artifact, manifest=changed_manifest)
        for name, tamper in (
            ("number", lambda item: item["edits"][0].__setitem__("new", 0.75)),
            ("index", lambda item: item["edits"][0].__setitem__("index", [4])),
            ("tensor_digest", lambda item: item["w0_manifest"][0].__setitem__(
                "tensor_sha256", "f" * 64)),
            ("manifest_digest", lambda item: item.__setitem__(
                "w0_manifest_sha256", "f" * 64)),
            ("rollback", lambda item: item["rollback_receipt"].__setitem__(
                "status", "RESTORED")),
            ("control", lambda item: item["paired_control_receipt"].__setitem__(
                "status", "EVALUATED")),
            ("execution", lambda item: item.__setitem__("execution_authorized", True)),
            ("heldout_leak", lambda item: item.__setitem__("heldout_answer", "4")),
            ("train_leak", lambda item: item["provenance"].__setitem__("train_ids", ["x"])),
            ("digest", lambda item: item.__setitem__("proposal_sha256", "f" * 64)),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                tampered = deepcopy(artifact)
                tamper(tampered)
                verify(tampered)

    def test_replay_nonce_collision_and_digest_registry(self):
        a = proposal()
        b = proposal(edits=[{**EDIT, "new": 0.5}])
        self.assertNotEqual(a["proposal_sha256"], b["proposal_sha256"])
        self.assertEqual(a["proposal_nonce_sha256"], b["proposal_nonce_sha256"])
        with self.assertRaisesRegex(ValueError, "replayed"):
            verify(a, digests=(a["proposal_sha256"],))
        with self.assertRaisesRegex(ValueError, "replayed"):
            verify(b, nonces=(a["proposal_nonce_sha256"],))
        self.assertEqual(verify(b), b["proposal_sha256"])


if __name__ == "__main__":
    unittest.main()
