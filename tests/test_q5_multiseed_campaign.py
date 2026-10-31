"""No-neural-dependency regressions for the nonauthorizing Q5 campaign driver."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.maze_conditioned_minilm_cpu_experiment import (
    PREREG_ARMS, PREREG_SCHEMA, build_training_cycles, default_scope,
    preregistered_randomized_labels, select_training_batch, training_batch_labels,
)
from labyrinth.minilm_cpu_experiment import MODEL_ID
from scripts.experiments.q5_multiseed_campaign import (
    CampaignError, aggregate, campaign, parse_seeds,
    read_report_json, verify_checkpoint_files, verify_seed_report,
)


def digest(label):
    return sha256(label.encode("ascii")).hexdigest()


GOOD_SEEDS = tuple(range(20261030, 20261036)) + (20261037, 20261038)


def fixture(seed=20261030, steps=4):
    scope = replace(default_scope(), seed=seed)
    cycles = build_training_cycles(scope)
    report = {
        "schema": PREREG_SCHEMA, "qualification": "PUBLIC_DEVELOPMENT_ONLY",
        "private_final_exam": "NOT_RUN", "graduation_authorized": False,
        "control_arms": list(PREREG_ARMS), "steps_per_cycle": steps,
        "model_id": MODEL_ID, "pretrained_base_sha256": digest("base"),
        "harness_sha256": digest("source"), "scope": scope.to_dict(),
        "scope_fingerprint": scope.fingerprint, "cycle_one_weak_skill_hint": [],
        "weakness_feedback_provenance": "OPERATOR_DECLARED_NOT_MEASURED",
        "arms": {},
    }
    for arm in PREREG_ARMS:
        before = digest("initial trainable")
        lora_before = digest("initial LoRA")
        results = []
        for index, cycle in enumerate(cycles):
            randomized = (preregistered_randomized_labels(cycle, seed)
                          if arm == "randomized_labels" else None)
            labels = training_batch_labels(cycle.examples, arm, randomized)
            assigned = {
                row.exposure_id: label for row, label in zip(cycle.examples, labels)
            }
            label_sha = sha256(json.dumps(assigned, sort_keys=True,
                                          separators=(",", ":")).encode("ascii")).hexdigest()
            visits = [row.exposure_id for step in range(steps)
                      for row in select_training_batch(cycle, step)]
            frozen = arm in ("zero_update", "zero_lr")
            after = before if frozen else digest(f"{arm}-{index}-trained")
            lora_after = lora_before if frozen else digest(f"{arm}-{index}-LoRA")
            results.append({
                "cycle": index, "trainable_sha256_before": before,
                "trainable_sha256_after": after,
                "lora_sha256_before": lora_before,
                "lora_sha256_after": lora_after,
                "adapter_file_sha256": digest(f"{arm}-{index}-adapter-file"),
                "head_file_sha256": digest(f"{arm}-{index}-head-file"),
                "pretrained_base_sha256": digest("base"),
                "training_visit_exposure_ids": visits,
                "label_assignment_sha256": label_sha,
                "training_visits": 4 * steps, "forward_steps": steps,
                "backward_steps": steps,
                "optimizer_step_invocations": 0 if arm == "zero_update" else steps,
                "public_development_accuracy": .9 if arm == "correct" else .5,
                "public_transfer_accuracy": .8 if arm == "correct" else .4,
            })
            before, lora_before = after, lora_after
        report["arms"][arm] = {
            "initial_trainable_sha256": digest("initial trainable"),
            "initial_lora_sha256": digest("initial LoRA"),
            "cycles": results,
        }
    return report


class Q5CampaignTests(unittest.TestCase):
    def test_reject_bad_seed_sets(self):
        for raw in ("1,2,3", "1,2,3,4,5,6,7,7", "-1,2,3,4,5,6,7,8",
                    "1,2,3,4,5,6,7,2147483648", "1, 2,3,4,5,6,7,8"):
            with self.subTest(raw=raw), self.assertRaises(CampaignError):
                parse_seeds(raw)
        self.assertEqual(parse_seeds("1,2,3,4,5,6,7,8"), tuple(range(1, 9)))

    def test_source_consistent_five_arm_report(self):
        result = verify_seed_report(fixture(), seed=20261030, steps=4)
        self.assertEqual(result["seed"], 20261030)
        self.assertEqual(result["scores"]["public_transfer"]["correct"], .8)

    def test_reject_identity_exposure_labels_and_continuity_forgery(self):
        mutators = (
            lambda r: r["arms"].pop("randomized_labels"),
            lambda r: r["arms"]["correct"].__setitem__(
                "initial_trainable_sha256", digest("other")),
            lambda r: r["arms"]["correct"]["cycles"][0][
                "training_visit_exposure_ids"].reverse(),
            lambda r: r["arms"]["inverted"]["cycles"][1].__setitem__(
                "label_assignment_sha256", digest("fake labels")),
            lambda r: r["arms"]["zero_lr"]["cycles"][0].__setitem__(
                "trainable_sha256_after", digest("mutated sham")),
            lambda r: r["arms"]["correct"]["cycles"][1].__setitem__(
                "trainable_sha256_before", digest("forgot prior maze")),
            lambda r: r["scope"].__setitem__("seed", 77),
        )
        for index, alter in enumerate(mutators):
            report = fixture()
            alter(report)
            with self.subTest(index=index), self.assertRaises(CampaignError):
                verify_seed_report(report, seed=20261030, steps=4)

    def test_lora_initial_regeneration_and_sham_continuity_are_mandatory(self):
        mutators = (
            lambda r: r["arms"]["correct"]["cycles"][0].__setitem__(
                "lora_sha256_before", digest("different initial adapter")),
            lambda r: r["arms"]["correct"]["cycles"][1].__setitem__(
                "lora_sha256_before", digest("lost learned adapter")),
            lambda r: r["arms"]["zero_update"]["cycles"][0].__setitem__(
                "lora_sha256_after", digest("sham mutated adapter")),
            lambda r: r["arms"]["zero_lr"]["cycles"][1].__setitem__(
                "lora_sha256_after", digest("zero lr mutated adapter")),
        )
        for index, mutate in enumerate(mutators):
            report = fixture()
            mutate(report)
            with self.subTest(index=index), self.assertRaises(CampaignError):
                verify_seed_report(report, seed=20261030, steps=4)

    def test_oversized_accuracy_int_is_a_campaign_error(self):
        report = fixture()
        report["arms"]["correct"]["cycles"][1]["public_transfer_accuracy"] = 10**309
        with self.assertRaises(CampaignError):
            verify_seed_report(report, seed=20261030, steps=4)

    def test_artifact_bytes_mismatch_fails(self):
        report = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in PREREG_ARMS:
                for index in range(2):
                    checkpoint = root / arm / f"cycle-{index}"
                    (checkpoint / "adapter").mkdir(parents=True)
                    for name, field in (("adapter/adapter_model.safetensors", "adapter_file_sha256"),
                                        ("head.safetensors", "head_file_sha256")):
                        contents = f"{arm}-{index}-{name}".encode("ascii")
                        (checkpoint / name).write_bytes(contents)
                        report["arms"][arm]["cycles"][index][field] = sha256(contents).hexdigest()
            verify_checkpoint_files(report, root)
            (root / "correct" / "cycle-1" / "head.safetensors").write_bytes(b"bad")
            with self.assertRaises(CampaignError):
                verify_checkpoint_files(report, root)

    def test_strict_report_json_rejects_duplicate_and_nan_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            for payload in (b'{"schema": 1,"schema": 2}',
                            b'{"score": NaN}', b'["not an object"]'):
                path.write_bytes(payload)
                with self.subTest(payload=payload), self.assertRaises(CampaignError):
                    read_report_json(path)
            path.write_bytes(b'{"schema":"source", "score":0.75}')
            self.assertEqual(read_report_json(path)["score"], .75)

    def test_aggregate_eight_seed_controls_does_not_graduate(self):
        seeds = GOOD_SEEDS
        results = [verify_seed_report(fixture(seed), seed=seed, steps=4)
                   for seed in seeds]
        aggregated = aggregate(results, seeds)
        self.assertEqual(aggregated["seed_count"], 8)
        self.assertEqual(aggregated["comparisons"]["public_transfer"]
                         ["inverted"]["positive_seed_count"], 8)
        self.assertFalse(aggregated["independent_learning_qualified"])
        self.assertFalse(aggregated["graduation_authorized"])
        with self.assertRaises(CampaignError):
            aggregate(results[:-1], seeds)

    def test_dry_run_and_failed_child_do_not_fabricate_experiment(self):
        seeds = GOOD_SEEDS
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            planned = campaign(seeds, output_parent=parent, name="q5-dry",
                               steps=4, timeout=60, execute=False)
            self.assertFalse((parent / "q5-dry").exists())
            self.assertEqual(len(planned["plans"]), 8)
            with patch("scripts.experiments.q5_multiseed_campaign.subprocess.run") as runner:
                runner.return_value.returncode = 3
                failed = campaign(seeds, output_parent=parent, name="q5-failed",
                                  steps=4, timeout=60, execute=True)
            self.assertEqual(failed["state"], "INCOMPLETE")
            self.assertFalse(failed["graduation_authorized"])
            self.assertEqual(failed["traces"][0]["state"], "CHILD_FAILED")
            self.assertTrue((parent / "q5-failed" / "campaign-report.json").is_file())
            self.assertFalse((parent / "q5-failed" / f"seed-{seeds[0]}" / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
