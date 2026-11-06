"""Adversarial SOURCE_ONLY population policy tests with physical fake HF files."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

from labyrinth.campaign_full_state_v1 import capture_campaign_state, save_campaign_state_once
from labyrinth.evolutionary_population_v1 import (
    Candidate, EvaluationPlan, Examination, Metrics, MutationKind,
    PopulationError, deterministic_seed, prepare_population, select_generation,
    verify_selected_entry,
)


def _hash(char: str) -> str:
    return char * 64


def _tensor(value: bytes) -> bytes:
    header = json.dumps({"weight": {"dtype": "U8", "shape": [len(value)],
                                     "data_offsets": [0, len(value)]}}).encode()
    return struct.pack("<Q", len(header)) + header + value


class PopulationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = EvaluationPlan("minotaur", _hash("a"), _hash("b"),
                                   _hash("c"), _hash("d"))
        self.w0_paths = self._files("w0", b"W0")
        self.w0, self.w0_file = self._capture("w0", self.w0_paths)

    def _files(self, name: str, adapter_bytes: bytes):
        root = self.root / name
        paths = {role: root / role for role in
                 ("base", "adapter", "optimizer", "scheduler", "rng")}
        for path in paths.values():
            path.mkdir(parents=True)
        (paths["base"] / "config.json").write_text('{"model_type":"qwen3"}')
        (paths["base"] / "tokenizer.json").write_text('{"tokens":["A"]}')
        (paths["base"] / "model.safetensors").write_bytes(_tensor(b"BASE"))
        (paths["adapter"] / "adapter_config.json").write_text('{"r":8}')
        (paths["adapter"] / "adapter_model.safetensors").write_bytes(_tensor(adapter_bytes))
        (paths["optimizer"] / "state.pt").write_bytes(b"optimizer-" + adapter_bytes)
        (paths["scheduler"] / "state.pt").write_bytes(b"scheduler")
        (paths["rng"] / "state.pt").write_bytes(b"rng")
        paths["cursor"] = root / "cursor.json"
        paths["cursor"].write_text('{"step":0}')
        return paths

    def _capture(self, name, paths, parent=None):
        manifest = capture_campaign_state(
            base_dir=paths["base"], adapter_dir=paths["adapter"],
            optimizer_dir=paths["optimizer"], scheduler_dir=paths["scheduler"],
            rng_dir=paths["rng"], data_cursor_path=paths["cursor"],
            family="hf_safetensors_lora", mutation_kind="lora_adapter",
            parent_manifest=parent,
        )
        destination = self.root / f"{name}-manifest.json"
        save_campaign_state_once(manifest, destination)
        return manifest, destination

    def _generation(self, development_generation=1, parent=None,
                    parent_file=None, origin_file=None, names=("left", "right")):
        parent = parent or self.w0
        parent_file = parent_file or self.w0_file
        origin_file = origin_file or self.w0_file
        parent_root = Path(parent["adapter"]["root"]).parent
        control_paths = self.root / f"g{development_generation}-control"
        shutil.copytree(parent_root, control_paths)
        control = {role: control_paths / role for role in
                   ("base", "adapter", "optimizer", "scheduler", "rng")}
        control["cursor"] = control_paths / "cursor.json"
        previous = self.w0 if parent is not self.w0 else None
        # The unchanged control is independently captured with identical content.
        control_manifest, control_file = self._capture(
            f"g{development_generation}-control", control, previous)
        self.assertEqual(control_manifest["state_sha256"], parent["state_sha256"])
        candidates = []
        manifests = {}
        for idx, name in enumerate(names):
            folder = f"g{development_generation}-{name}"
            paths = self.root / folder
            shutil.copytree(parent_root, paths)
            pieces = {role: paths / role for role in
                      ("base", "adapter", "optimizer", "scheduler", "rng")}
            pieces["cursor"] = paths / "cursor.json"
            changed = f"{folder}-updated".encode()
            (pieces["adapter"] / "adapter_model.safetensors").write_bytes(_tensor(changed))
            (pieces["optimizer"] / "state.pt").write_bytes(b"optimizer-" + changed)
            (pieces["cursor"]).write_text('{"step":1}')
            manifest, path = self._capture(folder, pieces, parent)
            manifests[name] = (manifest, path)
            candidates.append(Candidate(
                candidate_id=name, manifest_path=path,
                mutation_kind=MutationKind.ADAPTER_DELTA, trainer_id="trainer",
                trainer_receipt_sha256=_hash(str(idx + 1)),
                preupdate_parent_state_sha256=parent["state_sha256"],
                seed=deterministic_seed(parent["state_sha256"],
                                        development_generation, name)))
        population = prepare_population(
            development_generation=development_generation,
            origin_manifest_path=origin_file, parent_manifest_path=parent_file,
            control_manifest_path=control_file,
            candidates=tuple(candidates), plan=self.plan)
        return population, manifests

    def _exams(self, pop, scores=None):
        default = {
            "parent": Metrics(.50, .50, .99, .99, .02),
            "control": Metrics(.51, .50, .99, .99, .02),
            "left": Metrics(.67, .66, .99, .99, .01),
            "right": Metrics(.73, .72, .99, .99, .01),
        }
        if scores:
            default.update(scores)
        states = {"parent": pop.parent_state_sha256,
                  "control": pop.parent_state_sha256,
                  **{name: state for name, state, _ in pop.candidate_states}}
        return tuple(Examination(
            subject_id=name, state_sha256=state,
            examiner_id=self.plan.examiner_id,
            suite_sha256=self.plan.suite_sha256,
            sealed_case_commitment=self.plan.sealed_case_commitment,
            preregistration_sha256=self.plan.preregistration_sha256,
            private_cohort_commitment=self.plan.private_cohort_commitment,
            examiner_run_sha256=f"{index + 100:064x}", metrics=default[name],
        ) for index, (name, state) in enumerate(states.items()))

    def test_adapter_metadata_only_change_is_not_neural_weight_mutation(self):
        population, _ = self._generation()
        copied = self.root / "metadata-only"
        shutil.copytree(Path(self.w0["adapter"]["root"]).parent, copied)
        paths = {role: copied / role for role in
                 ("base", "adapter", "optimizer", "scheduler", "rng")}
        paths["cursor"] = copied / "cursor.json"
        (paths["adapter"] / "adapter_config.json").write_text('{"r":16}')
        (paths["optimizer"] / "state.pt").write_bytes(b"changed-optimizer-only")
        _, manifest_path = self._capture("metadata-only", paths, self.w0)
        mutants = (replace(population.candidates[0], manifest_path=manifest_path),
                   population.candidates[1])
        with self.assertRaisesRegex(PopulationError, "mutate adapter tensor weights"):
            prepare_population(
                development_generation=1,
                origin_manifest_path=self.w0_file,
                parent_manifest_path=self.w0_file,
                control_manifest_path=population.control_manifest_path,
                candidates=mutants,
                plan=self.plan,
            )

    def test_sibling_winner_exact_physical_entry_then_two_generations(self):
        pop, first = self._generation()
        self.assertEqual(pop.qualification, "SOURCE_ONLY_UNAUTHENTICATED_EXAMINER_FIXTURE")
        one = select_generation(pop, self._exams(pop))
        self.assertEqual(one.winner_id, "right")
        self.assertEqual(one.rejected_candidate_ids, ("left",))
        self.assertEqual(one.next_development_generation, 2)
        selected, selected_file = first["right"]
        next_paths_root = self.root / "next-entry"
        shutil.copytree(Path(selected["adapter"]["root"]).parent, next_paths_root)
        next_paths = {role: next_paths_root / role for role in
                      ("base", "adapter", "optimizer", "scheduler", "rng")}
        next_paths["cursor"] = next_paths_root / "cursor.json"
        entry, entry_path = self._capture("gen2-entry", next_paths, selected)
        receipt = verify_selected_entry(pop, one, entry_path)
        self.assertFalse(receipt["neural_update_proven"])

        # Generation 2 forks *from the selected genotype*, not from W0 or
        # the losing sibling. Saga cycles and development generations differ.
        two_pop, _ = self._generation(2, entry, entry_path,
                                      names=("left", "right"))
        self.assertEqual(two_pop.parent_state_sha256, one.selected_state_sha256)
        two = select_generation(two_pop, self._exams(two_pop, {
            "left": Metrics(.75, .77, .99, .99, .01),
            "right": Metrics(.73, .72, .99, .99, .01),
        }))
        self.assertEqual(two.winner_id, "left")  # rank reversal
        self.assertNotEqual(two.selected_state_sha256, one.selected_state_sha256)
        self.assertEqual(two.next_development_generation, 3)

    def test_forged_self_grading_mismatched_suite_and_candidate_state_rejected(self):
        pop, _ = self._generation()
        exams = self._exams(pop)
        tampered = [replace(exams[2], producer_role="TRAINER")]
        with self.assertRaisesRegex(PopulationError, "forged/self-graded"):
            select_generation(pop, exams[:2] + tuple(tampered) + exams[3:])
        for bad in (replace(exams[2], state_sha256=_hash("f")),
                    replace(exams[2], suite_sha256=_hash("f")),
                    replace(exams[2], private_cohort_commitment=_hash("f"))):
            with self.assertRaisesRegex(PopulationError, "forged/self-graded"):
                select_generation(pop, exams[:2] + (bad,) + exams[3:])
        with self.assertRaisesRegex(PopulationError, "trainer cannot"):
            prepare_population(
                development_generation=1, origin_manifest_path=pop.origin_manifest_path,
                parent_manifest_path=pop.parent_manifest_path,
                control_manifest_path=pop.control_manifest_path,
                candidates=pop.candidates,
                plan=replace(self.plan, examiner_id="trainer"))
        with self.assertRaisesRegex(PopulationError, "trainer cannot"):
            prepare_population(
                development_generation=1, origin_manifest_path=pop.origin_manifest_path,
                parent_manifest_path=pop.parent_manifest_path,
                control_manifest_path=pop.control_manifest_path,
                candidates=pop.candidates,
                plan=replace(self.plan, examiner_id="left"))

    def test_missing_control_or_parent_exam_rejected(self):
        pop, _ = self._generation()
        exams = self._exams(pop)
        for missing in ("control", "parent"):
            with self.assertRaisesRegex(PopulationError, "parent, control"):
                select_generation(pop, tuple(e for e in exams if e.subject_id != missing))
        with self.assertRaisesRegex(PopulationError, "positive"):
            select_generation(pop, exams, min_holdout_gain=0)

    def test_regression_identity_retention_and_no_winner_keep_parent(self):
        pop, _ = self._generation()
        scores = {
            "left": Metrics(.90, .90, .90, .99, .01),
            "right": Metrics(.90, .90, .99, .90, .30),
        }
        verdict = select_generation(pop, self._exams(pop, scores))
        self.assertIsNone(verdict.winner_id)
        self.assertEqual(verdict.selected_manifest_path, pop.parent_manifest_path)
        self.assertEqual(verdict.selected_state_sha256, pop.parent_state_sha256)
        with self.assertRaisesRegex(PopulationError, "selected physical child"):
            verify_selected_entry(pop, verdict, pop.control_manifest_path)

    def test_duplicate_sibling_parent_and_wrong_training_seed_rejected(self):
        pop, _ = self._generation()
        with self.assertRaisesRegex(PopulationError, "distinct paths|duplicate"):
            prepare_population(
                development_generation=pop.development_generation,
                origin_manifest_path=pop.origin_manifest_path,
                parent_manifest_path=pop.parent_manifest_path,
                control_manifest_path=pop.control_manifest_path,
                candidates=(pop.candidates[0], pop.candidates[0]), plan=pop.plan)
        with self.assertRaisesRegex(PopulationError, "seed mismatch"):
            prepare_population(
                development_generation=pop.development_generation,
                origin_manifest_path=pop.origin_manifest_path,
                parent_manifest_path=pop.parent_manifest_path,
                control_manifest_path=pop.control_manifest_path,
                candidates=(replace(pop.candidates[0], seed=42), pop.candidates[1]),
                plan=pop.plan)
        # A manifest with a changed child's declared ancestry still has to
        # pass full-state digest/readback; plain claims cannot rewrite bytes.
        manifest = json.loads(pop.candidates[0].manifest_path.read_text())
        manifest["parent_state_sha256"] = _hash("f")
        pop.candidates[0].manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(PopulationError, "readback"):
            select_generation(pop, self._exams(pop))

    def test_parent_control_behavioral_reversal_disqualifies_candidate(self):
        pop, _ = self._generation()
        exam = self._exams(pop, {"control": Metrics(.95, .95, .99, .99, .02)})
        self.assertIsNone(select_generation(pop, exam).winner_id)

    def test_higher_parent_identity_and_retention_baseline_must_not_be_lost(self):
        pop, _ = self._generation()
        exams = self._exams(pop, {
            "parent": Metrics(.50, .50, 1.0, 1.0, .02),
            "control": Metrics(.51, .50, .95, .95, .02),
            "left": Metrics(.90, .90, .98, .98, .01),
            "right": Metrics(.90, .90, .98, .98, .01),
        })
        self.assertIsNone(select_generation(pop, exams).winner_id)

    def test_full_weight_mutation_is_real_weight_bytes_not_a_generic_hash(self):
        def capture_full(paths, name, parent=None):
            manifest = capture_campaign_state(
                base_dir=paths["base"], adapter_dir=None,
                optimizer_dir=paths["optimizer"], scheduler_dir=paths["scheduler"],
                rng_dir=paths["rng"], data_cursor_path=paths["cursor"],
                family="hf_safetensors_full", mutation_kind="full_weight_edit",
                parent_manifest=parent)
            dest = self.root / f"{name}-full.json"
            save_campaign_state_once(manifest, dest)
            return manifest, dest

        parent, parent_file = capture_full(self.w0_paths, "w0")
        control_paths = self._files("full-control", b"W0")
        control, control_file = capture_full(control_paths, "control")
        self.assertEqual(control["state_sha256"], parent["state_sha256"])
        variants = []
        for index, name in enumerate(("left", "right")):
            paths = self._files(f"full-{name}", b"W0")
            (paths["base"] / "model.safetensors").write_bytes(
                _tensor(f"changed-base-{name}".encode()))
            candidate, candidate_file = capture_full(paths, name, parent)
            self.assertNotEqual(candidate["state_sha256"], parent["state_sha256"])
            variants.append(Candidate(
                name, candidate_file, MutationKind.FULL_MODEL_EDIT, "trainer",
                _hash(str(index + 1)), parent["state_sha256"],
                deterministic_seed(parent["state_sha256"], 1, name)))
        full = prepare_population(
            development_generation=1, origin_manifest_path=parent_file,
            parent_manifest_path=parent_file, control_manifest_path=control_file,
            candidates=tuple(variants), plan=self.plan)
        self.assertEqual(len(full.candidate_states), 2)
        # Full weights must not sneak a modified tokenizer through the family.
        tokenizer = self.root / "full-left" / "base" / "tokenizer.json"
        tokenizer.write_text('{"tokens":["not-A"]}')
        with self.assertRaisesRegex(PopulationError, "readback"):
            select_generation(full, self._exams(full))

    def test_physical_optimizer_tamper_after_preparation_rejected(self):
        pop, _ = self._generation()
        child_root = pop.candidates[0].manifest_path
        child = json.loads(child_root.read_text())
        path = Path(child["optimizer"]["root"]) / "state.pt"
        path.write_bytes(b"post-preparation-tamper")
        with self.assertRaisesRegex(PopulationError, "readback"):
            select_generation(pop, self._exams(pop))

    def test_deceptive_duplicate_parent_candidate_rejected(self):
        pop, _ = self._generation()
        # Repoint a candidate to a perfectly valid unchanged-control file.
        fake = replace(pop.candidates[0], manifest_path=pop.control_manifest_path)
        another_control = self.root / "another-control-manifest.json"
        shutil.copyfile(pop.control_manifest_path, another_control)
        with self.assertRaisesRegex(PopulationError, "parent/origin descendant"):
            prepare_population(
                development_generation=pop.development_generation,
                origin_manifest_path=pop.origin_manifest_path,
                parent_manifest_path=pop.parent_manifest_path,
                control_manifest_path=another_control,
                candidates=(fake, pop.candidates[1]), plan=pop.plan)


if __name__ == "__main__":
    unittest.main()
