"""Source-only, multi-seed/budget public MiniLM campaign acceptance tests."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from test_physical_report_bridge import physical_report, digest

from labyrinth.preregistered_minilm_campaign import (
    CampaignRun,
    PublicCase,
    ReportReceipt,
    build_public_minilm_campaign,
    score_public_minilm_campaign,
)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def synthetic_run_report(run_id: str, seed: int, steps: int) -> dict:
    report = physical_report()
    report.update({"run_id": run_id, "seed": seed, "steps_per_cycle": steps,
                   "learning_rate": 0.002})
    for name, arm in report["arms"].items():
        cycle = arm["cycles"][0]
        if name == "baseline":
            continue
        original = deepcopy(cycle["training_batches"])
        batches = []
        for step in range(steps):
            batch = deepcopy(original[step % len(original)])
            batch["step"] = step
            batches.append(batch)
        cycle["training_batches"] = batches
        cycle["training_prompt_sha256"] = [
            row["prompt_sha256"] for batch in batches
            for row in batch["examples"]
        ]
        cycle["optimizer_step_calls"] = steps
        cycle["compute_steps"] = steps
        cycle["forward_steps"] = steps
        cycle["backward_steps"] = steps
        cycle["training_forward_steps"] = steps
        cycle["training_backward_steps"] = steps
        cycle["optimizer_updates"] = 0 if name == "zero_update" else steps
        cycle["train_example_visits"] = steps * 4
    return report


def schedule_digest(report: dict) -> str:
    schedules = [cycle["training_batches"] for cycle in
                 report["arms"]["correct_labels"]["cycles"]]
    return sha256(canonical(schedules)).hexdigest()


class PreregisteredMiniLMCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.report_template = physical_report()
        cycles = self.report_template["arms"]["correct_labels"]["cycles"]
        self.heldout = tuple(PublicCase(row["case_id"], row["prompt_sha256"],
                                        row["expected_label"])
                             for row in cycles[0]["heldout_predictions"])
        self.transfer = tuple(PublicCase(row["case_id"], row["prompt_sha256"],
                                         row["expected_label"])
                              for row in cycles[0]["transfer_predictions"])
        self.runs = tuple(
            CampaignRun(
                run_id=f"seed-{seed}-budget-{steps}", seed=seed,
                steps_per_cycle=steps, learning_rate=0.002,
                training_schedule_sha256=schedule_digest(
                    synthetic_run_report(f"seed-{seed}-budget-{steps}", seed, steps)),
            )
            for seed in (101, 202) for steps in (8, 16)
        )
        self.campaign = self.build(self.runs)

    def build(self, runs):
        return build_public_minilm_campaign(
            campaign_id="campaign-20261031", source_tree_sha256=digest("source"),
            pretrained_model_sha256=digest("pretrained"),
            tokenizer_sha256=digest("tokenizer"),
            harness_sha256=digest("harness"),
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            runs=runs, heldout_cases=self.heldout, transfer_cases=self.transfer,
            training_batch_size=4,
        )

    def receipts(self):
        result = []
        for run in self.runs:
            report = synthetic_run_report(run.run_id, run.seed, run.steps_per_cycle)
            content = canonical(report)
            path = self.root / f"{run.run_id}.json"
            path.write_bytes(content)
            result.append(ReportReceipt(run.run_id, path, sha256(content).hexdigest()))
        return tuple(result)

    def score(self, receipts, *, campaign=None):
        campaign = campaign or self.campaign
        return score_public_minilm_campaign(
            campaign, expected_campaign_sha256=self.campaign.sha256,
            reports=receipts,
        )

    def test_all_preregistered_seeds_and_budgets_score_without_graduation(self):
        result = self.score(self.receipts())
        self.assertEqual(result.status, "PROVISIONAL_PUBLIC_DEVELOPMENT")
        self.assertEqual(len(result.run_results), 4)
        self.assertEqual({(run.seed, run.steps_per_cycle)
                          for run in result.run_results},
                         {(101, 8), (101, 16), (202, 8), (202, 16)})
        self.assertFalse(result.physical_model_verified)
        self.assertFalse(result.fresh_process_verified)
        self.assertFalse(result.evaluator_custody_verified)
        self.assertFalse(result.graduation_authorized)
        self.assertEqual({run.heldout_correct_minus_zero
                          for run in result.run_results}, {0.0})

    def test_missing_losing_seed_or_budget_cannot_be_postselected_away(self):
        receipts = self.receipts()
        with self.assertRaisesRegex(ValueError, "all preregistered runs"):
            self.score(receipts[:-1])

    def test_duplicate_or_unplanned_run_rejected(self):
        receipts = self.receipts()
        with self.assertRaisesRegex(ValueError, "all preregistered runs"):
            self.score(receipts[:-1] + receipts[:1])
        with self.assertRaisesRegex(ValueError, "all preregistered runs"):
            self.score(receipts + (ReportReceipt("invented", receipts[0].report_path,
                                               receipts[0].report_sha256),))

    def test_duplicate_seed_or_missing_grid_cell_rejected_before_outcomes(self):
        duplicate_seed = tuple(self.runs[:3]) + (
            CampaignRun("duplicate-seed-budget", 101, 8, 0.002,
                        self.runs[0].training_schedule_sha256),)
        with self.assertRaisesRegex(ValueError, "seed.*budget|grid"):
            self.build(duplicate_seed)
        with self.assertRaisesRegex(ValueError, "seed.*budget|grid"):
            self.build(self.runs[:-1])

    def test_post_hoc_manifest_mutation_rejected_against_external_digest(self):
        changed = self.build(self.runs)
        object.__setattr__(changed, "runs", changed.runs[:-1])
        with self.assertRaisesRegex(ValueError, "campaign.*digest|manifest.*mismatch"):
            self.score(self.receipts(), campaign=changed)

    def test_unpinned_report_bytes_and_changed_schedule_rejected(self):
        receipts = self.receipts()
        path = receipts[0].report_path
        report = json.loads(path.read_text(encoding="utf-8"))
        report["seed"] += 1
        path.write_bytes(canonical(report))
        with self.assertRaisesRegex(ValueError, "report digest"):
            self.score(receipts)

        receipts = self.receipts()
        path = receipts[0].report_path
        report = json.loads(path.read_text(encoding="utf-8"))
        for arm in ("correct_labels", "zero_update", "corrupted_labels"):
            for batch in report["arms"][arm]["cycles"][0]["training_batches"]:
                for row in batch["examples"]:
                    row["case_id"] = "substituted-" + row["case_id"]
        path.write_bytes(canonical(report))
        changed = (ReportReceipt(receipts[0].run_id, path,
                                 sha256(path.read_bytes()).hexdigest()),) + receipts[1:]
        with self.assertRaisesRegex(ValueError, "schedule.*digest|training schedule"):
            self.score(changed)

    def test_private_or_model_execution_claim_cannot_be_upgraded(self):
        receipts = self.receipts()
        path = receipts[0].report_path
        report = json.loads(path.read_text(encoding="utf-8"))
        report["graduation_authorized"] = True
        path.write_bytes(canonical(report))
        changed = (ReportReceipt(receipts[0].run_id, path,
                                 sha256(path.read_bytes()).hexdigest()),) + receipts[1:]
        with self.assertRaisesRegex(ValueError, "graduation"):
            self.score(changed)


if __name__ == "__main__":
    unittest.main()
