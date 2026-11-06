"""SOURCE_ONLY physical direct weight edits are bounded and never self-certified."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

from labyrinth.campaign_full_state_v1 import (
    capture_campaign_state, save_campaign_state_once, verify_campaign_state,
)
from labyrinth.evolutionary_population_v1 import (
    Candidate, EvaluationPlan, MutationKind, deterministic_seed, prepare_population,
)
from labyrinth.weight_space_spark_v1 import (
    WeightSpaceBoundaryError, WeightSpaceEditPolicy, WeightSpaceEditProposal,
    _edited_tensor_file, create_weight_space_candidate,
)


def _tensor(values: tuple[float, ...]) -> bytes:
    header = json.dumps({"proj.weight": {
        "dtype": "F32", "shape": [2, 4],
        "data_offsets": [0, 8 * 4],
    }}, separators=(",", ":")).encode()
    return struct.pack("<Q", len(header)) + header + struct.pack("<8f", *values)


def _read_weights(path: Path) -> tuple[float, ...]:
    data = path.read_bytes()
    hsize = struct.unpack_from("<Q", data)[0]
    return struct.unpack_from("<8f", data, 8 + hsize)


def _capture(folder: Path, parent=None):
    result = capture_campaign_state(
        base_dir=folder / "base", adapter_dir=None,
        optimizer_dir=folder / "optimizer",
        scheduler_dir=folder / "scheduler",
        rng_dir=folder / "rng",
        data_cursor_path=folder / "cursor.json",
        family="hf_safetensors_full", mutation_kind="full_weight_edit",
        parent_manifest=parent,
    )
    path = folder / "manifest.json"
    save_campaign_state_once(result, path)
    return result, path


class WeightSpaceSparkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.origin = self.root / "w0"
        for role in ("base", "optimizer", "scheduler", "rng"):
            (self.origin / role).mkdir(parents=True)
        (self.origin / "base" / "config.json").write_text('{"model_type":"toy"}')
        (self.origin / "base" / "tokenizer.json").write_text('{"tokens":["A"]}')
        self.values = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        self.parent_weights = self.origin / "base" / "model.safetensors"
        self.parent_weights.write_bytes(_tensor(self.values))
        (self.origin / "optimizer" / "state.pt").write_bytes(b"unrestored optimizer")
        (self.origin / "scheduler" / "state.pt").write_bytes(b"scheduler")
        (self.origin / "rng" / "state.pt").write_bytes(b"rng")
        (self.origin / "cursor.json").write_text('{"step":0}')
        self.w0, self.w0_path = _capture(self.origin)
        self.policy = WeightSpaceEditPolicy(
            ("model.safetensors",), ("proj.weight",),
            maximum_modified_fraction=.25, maximum_relative_norm=.15,
        )
        self.proposal = WeightSpaceEditProposal(
            "removal_a", self.w0["state_sha256"], "f" * 64,
            "model.safetensors", "proj.weight", (0,),
            (1.0, 0.0), .5, "Remove a suspected excess direction from TRAIN evidence.",
        )

    def test_direct_edit_changes_only_approved_float_and_preserves_complete_parent(self):
        before = hashlib.sha256(self.parent_weights.read_bytes()).hexdigest()
        result = create_weight_space_candidate(
            parent_manifest_path=self.w0_path,
            candidate_root=self.root / "candidate-a",
            proposal=self.proposal, policy=self.policy,
        )
        self.assertTrue(result.parent_unchanged)
        self.assertEqual(result.evidence_class,
                         "SOURCE_ONLY_DIRECT_WEIGHT_EDIT_NO_NEURAL_GAIN_PROOF")
        self.assertEqual(result.changed_element_count, 1)
        self.assertEqual(result.tensor_element_count, 8)
        self.assertLess(result.relative_edit_norm, .15)
        self.assertEqual(result.parent_state_sha256, self.w0["state_sha256"])
        self.assertNotEqual(result.child_state_sha256, self.w0["state_sha256"])
        self.assertEqual(hashlib.sha256(self.parent_weights.read_bytes()).hexdigest(), before)
        self.assertEqual(_read_weights(self.parent_weights), self.values)
        candidate = self.root / "candidate-a" / "base" / "model.safetensors"
        updated = _read_weights(candidate)
        self.assertEqual(updated[0], .5)
        self.assertEqual(updated[1:], self.values[1:])
        child = json.loads(result.child_manifest_path.read_text())
        self.assertEqual(child["parent_state_sha256"], self.w0["state_sha256"])
        self.assertFalse(child["neural_update_proven"])
        self.assertFalse(child["restorable_training_state_proven"])
        self.assertFalse(child["graduation_authorized"])
        self.assertEqual(verify_campaign_state(result.child_manifest_path, fresh_process=True)
                         ["state_sha256"], result.child_state_sha256)
        for role in ("optimizer", "scheduler", "rng"):
            self.assertEqual(child[role]["files"], self.w0[role]["files"])
        with self.assertRaisesRegex(WeightSpaceBoundaryError, "cannot claim"):
            replace(result, evidence_class="NEURAL_UPDATE_VERIFIED")

    def test_model_proposals_cannot_escape_operator_whitelist_or_edit_budget(self):
        cases = (
            replace(self.proposal, tensor_name="lm_head.weight"),
            replace(self.proposal, shard_name="../model.safetensors"),
            replace(self.proposal, selected_columns=(0, 1, 2)),
            replace(self.proposal, selected_columns=(0, 0)),
            replace(self.proposal, selected_columns=(-1,)),
            replace(self.proposal, unit_direction=(0., 0.)),
            replace(self.proposal, unit_direction=(1.,)),
            replace(self.proposal, gain=0.0),
            replace(self.proposal, gain=float("nan")),
            replace(self.proposal, public_train_source_sha256="secret-private-test"),
            replace(self.proposal, parent_state_sha256="0" * 64),
        )
        for index, proposal in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaises(WeightSpaceBoundaryError):
                    create_weight_space_candidate(
                        parent_manifest_path=self.w0_path,
                        candidate_root=self.root / f"denied-{index}",
                        proposal=proposal, policy=self.policy,
                    )
                self.assertFalse((self.root / f"denied-{index}").exists())
        # A legal direction with an unsafe relative update must still be refused.
        with self.assertRaisesRegex(WeightSpaceBoundaryError, "norm"):
            create_weight_space_candidate(
                parent_manifest_path=self.w0_path,
                candidate_root=self.root / "denied-norm",
                proposal=replace(self.proposal, selected_columns=(3,), gain=2.0),
                policy=replace(self.policy, maximum_relative_norm=.01),
            )
        self.assertFalse((self.root / "denied-norm").exists())

    def test_output_collision_and_non_f32_tensor_are_rejected_without_parent_write(self):
        occupied = self.root / "existing"
        occupied.mkdir()
        marker = occupied / "user-owned.txt"
        marker.write_text("PRESERVE")
        with self.assertRaisesRegex(WeightSpaceBoundaryError, "fresh candidate"):
            create_weight_space_candidate(
                parent_manifest_path=self.w0_path, candidate_root=occupied,
                proposal=self.proposal, policy=self.policy,
            )
        self.assertEqual(marker.read_text(), "PRESERVE")
        with self.assertRaisesRegex(WeightSpaceBoundaryError, "overlaps"):
            create_weight_space_candidate(
                parent_manifest_path=self.w0_path,
                candidate_root=self.origin / "base" / "recursive-child",
                proposal=self.proposal, policy=self.policy,
            )
        self.assertFalse((self.origin / "base" / "recursive-child").exists())
        # I32 is structurally valid safetensors but not the bounded F32 pilot.
        new = _tensor(self.values).replace(b'"F32"', b'"I32"')
        source = self.root / "notf32"
        shutil.copytree(self.origin, source)
        (source / "manifest.json").unlink()
        (source / "base" / "model.safetensors").write_bytes(new)
        integer_parent, integer_parent_path = _capture(source)
        with self.assertRaisesRegex(WeightSpaceBoundaryError, "F32 rank-two"):
            create_weight_space_candidate(
                parent_manifest_path=integer_parent_path,
                candidate_root=self.root / "narrow-i32-test",
                proposal=replace(self.proposal,
                                 parent_state_sha256=integer_parent["state_sha256"]),
                policy=self.policy,
            )
        self.assertFalse((self.root / "narrow-i32-test").exists())

    def test_two_distinct_direct_edit_siblings_pass_population_byte_seams_only(self):
        left = create_weight_space_candidate(
            parent_manifest_path=self.w0_path, candidate_root=self.root / "left",
            proposal=self.proposal, policy=self.policy,
        )
        right = create_weight_space_candidate(
            parent_manifest_path=self.w0_path, candidate_root=self.root / "right",
            proposal=replace(self.proposal, proposal_id="removal_b", gain=1.0),
            policy=self.policy,
        )
        self.assertNotEqual(left.child_state_sha256, right.child_state_sha256)
        control = self.root / "control"
        shutil.copytree(self.origin, control)
        (control / "manifest.json").unlink()
        _, control_path = _capture(control, self.w0)
        candidates = tuple(Candidate(
            candidate_id=name, manifest_path=result.child_manifest_path,
            mutation_kind=MutationKind.FULL_MODEL_EDIT, trainer_id="direct_editor",
            trainer_receipt_sha256=result.proposal_sha256,
            preupdate_parent_state_sha256=self.w0["state_sha256"],
            seed=deterministic_seed(self.w0["state_sha256"], 1, name),
        ) for name, result in (("left", left), ("right", right)))
        population = prepare_population(
            development_generation=1, origin_manifest_path=self.w0_path,
            parent_manifest_path=self.w0_path, control_manifest_path=control_path,
            candidates=candidates,
            plan=EvaluationPlan("minotaur_development", "a"*64, "b"*64, "c"*64, "d"*64),
        )
        self.assertEqual(len(population.candidates), 2)
        self.assertEqual(population.qualification,
                         "SOURCE_ONLY_UNAUTHENTICATED_EXAMINER_FIXTURE")

    def test_overlapping_safetensors_entries_cannot_alias_approved_weights(self):
        """An unapproved alias cannot be modified through approved tensor bytes."""
        header = json.dumps({
            "proj.weight": {"dtype": "F32", "shape": [2, 4],
                            "data_offsets": [0, 32]},
            "unapproved.weight": {"dtype": "F32", "shape": [2, 4],
                                  "data_offsets": [0, 32]},
        }, separators=(",", ":")).encode()
        aliased = (struct.pack("<Q", len(header)) + header
                   + struct.pack("<8f", *self.values))
        with self.assertRaisesRegex(WeightSpaceBoundaryError, "overlap"):
            _edited_tensor_file(aliased, self.proposal, self.policy)
        # The higher-level campaign manifest also rejects aliased shards.
        malicious_parent = self.root / "aliased-parent"
        shutil.copytree(self.origin, malicious_parent)
        (malicious_parent / "manifest.json").unlink()
        (malicious_parent / "base" / "model.safetensors").write_bytes(aliased)
        # The parent-state validator already refuses an aliased physical
        # shard; this new narrow editor check is defense in depth against
        # bypassing or later weakening that upstream boundary.
        with self.assertRaisesRegex(ValueError, "overlapping tensor data"):
            _capture(malicious_parent)


if __name__ == "__main__":
    unittest.main()
