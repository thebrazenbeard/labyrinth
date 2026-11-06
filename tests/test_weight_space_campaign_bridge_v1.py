"""Physical W0/control/W1, Ariadne WAL, and Minotaur population integration."""

from dataclasses import replace
import json
from pathlib import Path
import struct
import tempfile
import unittest

from labyrinth.ariadne_recovery_v1 import AriadneJournal, UnsupportedJournalPlatform
from labyrinth.campaign_full_state_v1 import verify_campaign_state
from labyrinth.evolutionary_population_v1 import EvaluationPlan, prepare_population
from labyrinth.weight_space_campaign_bridge_v1 import (
    PublicProposalContext, WeightSpaceCampaignError, run_weight_space_generation,
)
from labyrinth.weight_space_spark_v1 import (
    WeightSpaceBoundaryError, WeightSpaceEditPolicy, WeightSpaceEditProposal,
)
from tests.test_weight_space_spark_v1 import _capture, _read_weights, _tensor


class WeightSpaceCampaignBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        w0_root = self.root / "w0"
        for role in ("base", "optimizer", "scheduler", "rng"):
            (w0_root / role).mkdir(parents=True)
        (w0_root / "base" / "config.json").write_text('{"model_type":"toy"}')
        (w0_root / "base" / "tokenizer.json").write_text('{"tokens":["A"]}')
        self.weights = w0_root / "base" / "model.safetensors"
        self.weights.write_bytes(_tensor((1., 2., 3., 4., 5., 6., 7., 8.)))
        for role in ("optimizer", "scheduler", "rng"):
            (w0_root / role / "state.bin").write_bytes(role.encode())
        (w0_root / "optimizer" / "padding.bin").write_bytes(b"p" * 2048)
        (w0_root / "cursor.json").write_text('{"step":0}')
        self.w0, self.w0_path = _capture(w0_root)
        self.policy = WeightSpaceEditPolicy(
            ("model.safetensors",), ("proj.weight",),
            maximum_modified_fraction=.25, maximum_relative_norm=.15,
        )
        self.plan = EvaluationPlan("minotaur_development", "a"*64, "b"*64, "c"*64, "d"*64)
        self.observed: list[PublicProposalContext] = []

    def propose(self, ctx: PublicProposalContext) -> WeightSpaceEditProposal:
        self.observed.append(ctx)
        return WeightSpaceEditProposal(
            proposal_id=ctx.candidate_id,
            parent_state_sha256=ctx.parent_state_sha256,
            public_train_source_sha256=ctx.public_train_source_sha256,
            shard_name="model.safetensors", tensor_name="proj.weight",
            selected_columns=(0,), unit_direction=(1., 0.),
            gain=.5 if ctx.candidate_id == "left" else 1.,
            rationale="Source-only projection hypothesis; no expected test outcomes.",
        )

    def run_campaign(self):
        return run_weight_space_generation(
            w0_manifest_path=self.w0_path,
            campaign_root=self.root / "generation_1",
            candidate_ids=("left", "right"),
            public_train_source_sha256="f"*64,
            propose=self.propose, policy=self.policy,
            evaluation_plan=self.plan,
        )

    def test_real_journal_full_control_and_distinct_physical_sibling_population(self):
        try:
            result = self.run_campaign()
        except UnsupportedJournalPlatform:
            self.skipTest("Ariadne requires Linux/WSL for crash-consistent WAL")
        self.assertEqual(result.qualification,
                         "SOURCE_ONLY_JOURNALED_WEIGHT_EDIT_POPULATION_NO_NEURAL_GAIN_PROOF")
        self.assertEqual(len(self.observed), 2)
        self.assertEqual([ctx.candidate_id for ctx in self.observed], ["left", "right"])
        self.assertTrue(all(ctx.public_train_source_sha256 == "f"*64
                            and ctx.development_generation == 1 for ctx in self.observed))
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(len(result.population.candidates), 2)
        self.assertEqual(result.population.qualification,
                         "SOURCE_ONLY_UNAUTHENTICATED_EXAMINER_FIXTURE")
        self.assertNotEqual(result.population.candidate_states[0][1],
                            result.population.candidate_states[1][1])
        self.assertEqual(result.population.parent_state_sha256, self.w0["state_sha256"])
        self.assertEqual(verify_campaign_state(result.origin_manifest_path)["state_sha256"],
                         self.w0["state_sha256"])
        self.assertEqual(verify_campaign_state(result.control_manifest_path)["state_sha256"],
                         self.w0["state_sha256"])
        control = json.loads(result.control_manifest_path.read_text())
        self.assertNotEqual(control["base"]["root"], self.w0["base"]["root"])
        self.assertEqual(control["base"]["files"], self.w0["base"]["files"])
        self.assertEqual(control["optimizer"]["files"], self.w0["optimizer"]["files"])
        self.assertEqual(control["rng"]["files"], self.w0["rng"]["files"])
        self.assertEqual(control["data_cursor"]["sha256"], self.w0["data_cursor"]["sha256"])
        self.assertEqual(struct.unpack_from("<Q", self.weights.read_bytes())[0],
                         struct.unpack_from("<Q", (self.root / "generation_1" / "control"
                                                   / "base" / "model.safetensors").read_bytes())[0])
        self.assertEqual(self.weights.read_bytes(),
                         (self.root / "generation_1" / "control" / "base"
                          / "model.safetensors").read_bytes())
        # The two siblings change different physical tensor values;
        # distinct receipt IDs or manifest paths alone do not establish this.
        self.assertEqual(_read_weights(self.weights), (1., 2., 3., 4., 5., 6., 7., 8.))
        self.assertEqual(_read_weights(self.root / "generation_1" / "candidate_left"
                                       / "base" / "model.safetensors"),
                         (.5, 2., 3., 4., 5., 6., 7., 8.))
        self.assertEqual(_read_weights(self.root / "generation_1" / "candidate_right"
                                       / "base" / "model.safetensors"),
                         (0., 2., 3., 4., 5., 6., 7., 8.))
        self.assertTrue(all(not json.loads(item.child_manifest_path.read_text())
                            ["neural_update_proven"] for item in result.candidates))
        journal = AriadneJournal.open(result.journal_root)
        events = journal.events()
        self.assertEqual(len(events), 7)  # genesis + 2 * (branch, intent, commit)
        self.assertEqual([event["kind"] for event in events],
                         ["GENESIS", "BRANCH_EXPERIENCE", "TRAIN_PREPARED",
                          "TRAIN_COMMITTED", "BRANCH_EXPERIENCE", "TRAIN_PREPARED",
                          "TRAIN_COMMITTED"])
        descriptor = journal.continuation_descriptor(expected_head_sha256=result.ariadne_head_sha256)
        self.assertEqual(descriptor["status"], "READY_FOR_EXTERNAL_FULL_STATE_LOAD")
        self.assertEqual({item["fate"] for item in descriptor["candidates"]}, {"UNEVALUATED"})
        self.assertFalse(descriptor["neural_update_proven"])
        self.assertFalse(descriptor["graduation_authorized"])
        # Independent tests are not run here. Re-prepare physical population only.
        duplicate_readback = prepare_population(
            development_generation=1, origin_manifest_path=result.origin_manifest_path,
            parent_manifest_path=result.origin_manifest_path,
            control_manifest_path=result.control_manifest_path,
            candidates=result.population.candidates, plan=self.plan,
        )
        self.assertEqual(duplicate_readback, result.population)

    def test_unbound_public_train_proposal_fails_before_any_campaign_artifact(self):
        def injected(ctx):
            return replace(self.propose(ctx), public_train_source_sha256="e"*64)

        with self.assertRaisesRegex(WeightSpaceCampaignError, "public bound"):
            run_weight_space_generation(
                w0_manifest_path=self.w0_path, campaign_root=self.root / "denied",
                candidate_ids=("left", "right"), public_train_source_sha256="f"*64,
                propose=injected, policy=self.policy, evaluation_plan=self.plan,
            )
        self.assertFalse((self.root / "denied").exists())

    def test_duplicate_edits_and_self_exam_plan_fail_before_genome_write(self):
        def duplicate(ctx):
            return replace(self.propose(ctx), gain=.5)

        with self.assertRaisesRegex(WeightSpaceCampaignError, "duplicate physical"):
            run_weight_space_generation(
                w0_manifest_path=self.w0_path, campaign_root=self.root / "duplicates",
                candidate_ids=("left", "right"), public_train_source_sha256="f"*64,
                propose=duplicate, policy=self.policy, evaluation_plan=self.plan,
            )
        self.assertFalse((self.root / "duplicates").exists())
        with self.assertRaisesRegex(ValueError, "trainer cannot be its own examiner"):
            run_weight_space_generation(
                w0_manifest_path=self.w0_path, campaign_root=self.root / "bad_exam",
                candidate_ids=("left", "right"), public_train_source_sha256="f"*64,
                propose=self.propose, policy=self.policy,
                evaluation_plan=replace(self.plan, examiner_id="weight_space_spark"),
            )
        self.assertFalse((self.root / "bad_exam").exists())

    def test_whole_genome_copy_budget_fails_before_any_effect(self):
        with self.assertRaisesRegex(WeightSpaceCampaignError, "disk budget"):
            run_weight_space_generation(
                w0_manifest_path=self.w0_path, campaign_root=self.root / "over_budget",
                candidate_ids=("left", "right"), public_train_source_sha256="f"*64,
                propose=self.propose, policy=self.policy, evaluation_plan=self.plan,
                max_copied_bytes=4096,
            )
        self.assertFalse((self.root / "over_budget").exists())
        self.assertEqual(self.observed, [])

    def test_callback_context_excludes_exam_fields_and_no_auto_selection(self):
        try:
            result = self.run_campaign()
        except UnsupportedJournalPlatform:
            self.skipTest("Ariadne requires Linux/WSL")
        self.assertNotIn("suite_sha256", PublicProposalContext.__dataclass_fields__)
        self.assertNotIn("examiner_id", PublicProposalContext.__dataclass_fields__)
        self.assertFalse(hasattr(result, "selected_candidate_id"))
        self.assertEqual(len(AriadneJournal.open(result.journal_root)
                             .continuation_descriptor()["selected_lineage"]), 0)

    def test_maximum_length_candidate_identifier_does_not_overflow_ariadne_keys(self):
        long_id = "a" * 64

        def propose_long(ctx):
            return replace(self.propose(ctx), gain=.5 if ctx.candidate_id == long_id else 1.)

        try:
            result = run_weight_space_generation(
                w0_manifest_path=self.w0_path, campaign_root=self.root / "long_names",
                candidate_ids=(long_id, "right"), public_train_source_sha256="f"*64,
                propose=propose_long, policy=self.policy, evaluation_plan=self.plan,
            )
        except UnsupportedJournalPlatform:
            self.skipTest("Ariadne requires Linux/WSL")
        self.assertEqual({candidate.candidate_id for candidate in result.population.candidates},
                         {long_id, "right"})
        self.assertEqual(len(AriadneJournal.open(result.journal_root).events()), 7)

    def test_model_invalid_selected_tensor_keeps_wal_indeterminate(self):
        # Legal-looking but out-of-shape column fails after PREPARED intent.
        def out_of_shape(ctx):
            return replace(self.propose(ctx), selected_columns=(9,))

        try:
            with self.assertRaises(WeightSpaceBoundaryError):
                run_weight_space_generation(
                    w0_manifest_path=self.w0_path, campaign_root=self.root / "failed_effect",
                    candidate_ids=("left", "right"), public_train_source_sha256="f"*64,
                    propose=out_of_shape, policy=self.policy, evaluation_plan=self.plan,
                )
        except UnsupportedJournalPlatform:
            self.skipTest("Ariadne requires Linux/WSL")
        journal = AriadneJournal.open(self.root / "failed_effect" / "ariadne")
        descriptor = journal.continuation_descriptor()
        self.assertEqual(descriptor["status"], "INDETERMINATE_TRAINER_EFFECT")
        self.assertEqual(descriptor["pending_training"]["candidate_id"], "left")
        self.assertFalse((self.root / "failed_effect" / "candidate_left").exists())
        self.assertEqual(verify_campaign_state(self.w0_path)["state_sha256"],
                         self.w0["state_sha256"])


if __name__ == "__main__":
    unittest.main()
