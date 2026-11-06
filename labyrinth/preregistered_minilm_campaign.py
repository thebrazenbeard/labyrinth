"""Precommitted, source-only public MiniLM development campaign controls.

The manifest intentionally fixes *all* seeds and optimizer-step budgets before
outcomes. The scorer is deterministic and separate from the trainer, but it
only checks trainer-origin report bytes; it is NOT an independent examiner.
An external, immutable record of the manifest digest and its creation time is
required to establish actual temporal preregistration. This module cannot do
that, attest model weights or grant graduation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from .control_provenance import ARMS
from .physical_report_bridge import inspect_public_minilm_report


SCHEMA = "labyrinth.public-minilm-campaign.v1"
STOPPING_RULE = "EXHAUST_FULL_SEED_BUDGET_GRID_NO_EARLY_STOP"
SELECTION_RULE = "REPORT_ALL_CELLS_NO_BEST_STEP_POSTSELECTION"
SUCCESS_RULE = "CORRECT_STRICTLY_EXCEEDS_EACH_CONTROL_ON_BOTH_SPLITS_IN_EVERY_CELL"
_SHA = re.compile(r"[a-f0-9]{64}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)


def _digest(value: Any, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(f"invalid {label} digest")
    return value


def _identifier(value: Any, label: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")
    return value


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON report field")
        result[key] = value
    return result


@dataclass(frozen=True)
class PublicCase:
    case_id: str
    prompt_sha256: str
    expected_label: int


@dataclass(frozen=True)
class CampaignRun:
    run_id: str
    seed: int
    steps_per_cycle: int
    learning_rate: float
    training_schedule_sha256: str


@dataclass(frozen=True)
class PublicMiniLMCampaign:
    campaign_id: str
    source_tree_sha256: str
    pretrained_model_sha256: str
    tokenizer_sha256: str
    harness_sha256: str
    model_id: str
    training_batch_size: int
    runs: tuple[CampaignRun, ...]
    heldout_cases: tuple[PublicCase, ...]
    transfer_cases: tuple[PublicCase, ...]
    canonical_bytes: bytes
    sha256: str
    physical_model_verified: bool = False
    evaluator_custody_verified: bool = False
    graduation_authorized: bool = False


@dataclass(frozen=True)
class ReportReceipt:
    run_id: str
    report_path: str | Path
    report_sha256: str


@dataclass(frozen=True)
class PublicRunScore:
    run_id: str
    seed: int
    steps_per_cycle: int
    report_sha256: str
    heldout_correct_minus_zero: float
    transfer_correct_minus_zero: float
    heldout_accuracy_by_arm: tuple[tuple[str, float], ...]
    transfer_accuracy_by_arm: tuple[tuple[str, float], ...]
    meets_predeclared_public_comparison: bool


@dataclass(frozen=True)
class PublicCampaignScore:
    status: str
    campaign_sha256: str
    run_results: tuple[PublicRunScore, ...]
    all_cells_meet_predeclared_public_comparison: bool
    limitations: tuple[str, ...]
    physical_model_verified: bool = False
    fresh_process_verified: bool = False
    evaluator_custody_verified: bool = False
    graduation_authorized: bool = False


def _cases(value: Iterable[PublicCase], split: str) -> tuple[PublicCase, ...]:
    cases = tuple(value)
    if not 1 <= len(cases) <= 4096:
        raise ValueError(f"invalid {split} case count")
    ids: set[str] = set()
    prompts: set[str] = set()
    for case in cases:
        if type(case) is not PublicCase:
            raise ValueError(f"invalid {split} case")
        _identifier(case.case_id, f"{split} case ID")
        _digest(case.prompt_sha256, f"{split} prompt")
        if type(case.expected_label) is not int or case.expected_label not in (0, 1):
            raise ValueError(f"invalid {split} label")
        if case.case_id in ids or case.prompt_sha256 in prompts:
            raise ValueError(f"duplicate {split} case identity")
        ids.add(case.case_id)
        prompts.add(case.prompt_sha256)
    return cases


def _run_grid(runs: Iterable[CampaignRun]) -> tuple[CampaignRun, ...]:
    normalized = tuple(runs)
    if not 4 <= len(normalized) <= 128:
        raise ValueError("seed budget grid must have at least four cells")
    cells: set[tuple[int, int]] = set()
    ids: set[str] = set()
    seeds: set[int] = set()
    budgets: set[int] = set()
    for run in normalized:
        if type(run) is not CampaignRun:
            raise ValueError("invalid seed budget run")
        _identifier(run.run_id, "run ID")
        if type(run.seed) is not int or not 0 <= run.seed <= 2**32 - 1:
            raise ValueError("invalid seed")
        if type(run.steps_per_cycle) is not int or not 1 <= run.steps_per_cycle <= 32:
            raise ValueError("invalid optimizer step budget")
        if (type(run.learning_rate) not in (int, float)
                or not math.isfinite(run.learning_rate) or not 0 < run.learning_rate <= 1):
            raise ValueError("invalid learning rate")
        _digest(run.training_schedule_sha256, "training schedule")
        if run.run_id in ids or (run.seed, run.steps_per_cycle) in cells:
            raise ValueError("duplicate seed budget grid cell or run identity")
        ids.add(run.run_id)
        cells.add((run.seed, run.steps_per_cycle))
        seeds.add(run.seed)
        budgets.add(run.steps_per_cycle)
    if len(seeds) < 2 or len(budgets) < 2 or len(cells) != len(seeds) * len(budgets):
        raise ValueError("incomplete seed budget Cartesian grid")
    return tuple(sorted(normalized, key=lambda run: (run.seed, run.steps_per_cycle)))


def _payload(campaign: PublicMiniLMCampaign) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "campaign_id": campaign.campaign_id,
        "source_tree_sha256": campaign.source_tree_sha256,
        "pretrained_model_sha256": campaign.pretrained_model_sha256,
        "tokenizer_sha256": campaign.tokenizer_sha256,
        "harness_sha256": campaign.harness_sha256,
        "model_id": campaign.model_id,
        "training_batch_size": campaign.training_batch_size,
        "arms": list(ARMS),
        "runs": [{"run_id": run.run_id, "seed": run.seed,
                  "steps_per_cycle": run.steps_per_cycle,
                  "learning_rate": run.learning_rate,
                  "training_schedule_sha256": run.training_schedule_sha256}
                 for run in campaign.runs],
        "heldout_cases": [{"case_id": case.case_id,
                           "prompt_sha256": case.prompt_sha256,
                           "expected_label": case.expected_label}
                          for case in campaign.heldout_cases],
        "transfer_cases": [{"case_id": case.case_id,
                            "prompt_sha256": case.prompt_sha256,
                            "expected_label": case.expected_label}
                           for case in campaign.transfer_cases],
        "stopping_rule": STOPPING_RULE,
        "selection_rule": SELECTION_RULE,
        "success_rule": SUCCESS_RULE,
        "qualification": "PUBLIC_DEVELOPMENT_ONLY",
        "graduation_authorized": False,
    }


def build_public_minilm_campaign(*, campaign_id: str, source_tree_sha256: str,
                                 pretrained_model_sha256: str, tokenizer_sha256: str,
                                 harness_sha256: str, model_id: str,
                                 runs: Iterable[CampaignRun],
                                 heldout_cases: Iterable[PublicCase],
                                 transfer_cases: Iterable[PublicCase],
                                 training_batch_size: int = 4) -> PublicMiniLMCampaign:
    """Produce content-addressable precommit bytes; caller stores digest externally."""
    _identifier(campaign_id, "campaign ID")
    for name, value in (("source tree", source_tree_sha256),
                        ("pretrained model", pretrained_model_sha256),
                        ("tokenizer", tokenizer_sha256), ("harness", harness_sha256)):
        _digest(value, name)
    if type(model_id) is not str or not 1 <= len(model_id) <= 256:
        raise ValueError("invalid model identity")
    if type(training_batch_size) is not int or not 2 <= training_batch_size <= 64:
        raise ValueError("invalid training batch size")
    ordered_runs = _run_grid(runs)
    heldout = _cases(heldout_cases, "heldout")
    transfer = _cases(transfer_cases, "transfer")
    if ({c.case_id for c in heldout} & {c.case_id for c in transfer}
            or {c.prompt_sha256 for c in heldout} & {c.prompt_sha256 for c in transfer}):
        raise ValueError("heldout and transfer identities overlap")
    initial = PublicMiniLMCampaign(campaign_id, source_tree_sha256,
                                   pretrained_model_sha256, tokenizer_sha256,
                                   harness_sha256, model_id, training_batch_size,
                                   ordered_runs, heldout, transfer, b"", "")
    content = _canonical(_payload(initial))
    return PublicMiniLMCampaign(campaign_id, source_tree_sha256,
                                pretrained_model_sha256, tokenizer_sha256,
                                harness_sha256, model_id, training_batch_size,
                                ordered_runs, heldout, transfer, content,
                                sha256(content).hexdigest())


def _case_identity(rows: Any, label: str) -> tuple[tuple[str, str, int], ...]:
    if type(rows) is not list:
        raise ValueError(f"invalid {label} predictions")
    return tuple((row["case_id"], row["prompt_sha256"], row["expected_label"])
                 for row in rows)


def _score_report(campaign: PublicMiniLMCampaign, run: CampaignRun,
                  receipt: ReportReceipt) -> PublicRunScore:
    # This bridge enforces cross-arm, step, exposure, checkpoint, label-flip,
    # per-case, and aggregate internal coherence before campaign-level scoring.
    readback = inspect_public_minilm_report(
        receipt.report_path, expected_report_sha256=receipt.report_sha256,
        expected_harness_sha256=campaign.harness_sha256,
        expected_run_id=run.run_id, expected_model_id=campaign.model_id,
    )
    if readback.status != "PROVISIONAL" or readback.graduation_authorized:
        raise ValueError("nonprovisional development report or graduation claim")
    # Independently read back and hash exactly the report the scorer uses.
    # Source-only checks cannot guarantee concurrent filesystem immutability.
    with Path(receipt.report_path).open("rb") as stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("report exceeds byte budget during scoring")
    if sha256(raw).hexdigest() != receipt.report_sha256:
        raise ValueError("report digest mismatch during scoring")
    report = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError(f"nonfinite JSON {value}")))
    if report.get("seed") != run.seed or type(report.get("seed")) is not int:
        raise ValueError("reported seed differs from preregistered seed")
    if (report.get("steps_per_cycle") != run.steps_per_cycle
            or type(report.get("steps_per_cycle")) is not int):
        raise ValueError("reported step budget differs from preregistered budget")
    if (type(report.get("learning_rate")) not in (int, float)
            or report["learning_rate"] != run.learning_rate):
        raise ValueError("learning rate differs from preregistered setting")
    if (report.get("physical_training_batch_size") != campaign.training_batch_size
            or type(report.get("physical_training_batch_size")) is not int):
        raise ValueError("training batch size differs from preregistered batch size")
    arms = report["arms"]
    correct_cycles = arms["correct_labels"]["cycles"]
    schedule = [cycle["training_batches"] for cycle in correct_cycles]
    if sha256(_canonical(schedule)).hexdigest() != run.training_schedule_sha256:
        raise ValueError("training schedule digest differs from precommit")
    expected_heldout = tuple((case.case_id, case.prompt_sha256, case.expected_label)
                             for case in campaign.heldout_cases)
    expected_transfer = tuple((case.case_id, case.prompt_sha256, case.expected_label)
                              for case in campaign.transfer_cases)
    for arm in ARMS:
        for cycle in arms[arm]["cycles"]:
            if _case_identity(cycle["heldout_predictions"], "heldout") != expected_heldout:
                raise ValueError("heldout identity/label differs from precommit")
            if _case_identity(cycle["transfer_predictions"], "transfer") != expected_transfer:
                raise ValueError("transfer identity/label differs from precommit")
    train_ids = set()
    train_prompts = set()
    for cycle in correct_cycles:
        for batch in cycle["training_batches"]:
            for case in batch["examples"]:
                train_ids.add(case["case_id"])
                train_prompts.add(case["prompt_sha256"])
    if (train_ids & {item[0] for item in expected_heldout + expected_transfer}
            or train_prompts & {item[1] for item in expected_heldout + expected_transfer}):
        raise ValueError("preregistered train/evaluation identities overlap")

    def independently_scored(split: str) -> dict[str, float]:
        final = {}
        for arm in ARMS:
            predictions = arms[arm]["cycles"][-1][f"{split}_predictions"]
            final[arm] = sum(row["predicted_label"] == row["expected_label"]
                             for row in predictions) / len(predictions)
        return final

    heldout = independently_scored("heldout")
    transfer = independently_scored("transfer")
    public_success = all(
        accuracies["correct_labels"] > accuracies[arm]
        for accuracies in (heldout, transfer)
        for arm in ARMS if arm != "correct_labels"
    )
    return PublicRunScore(
        run_id=run.run_id, seed=run.seed, steps_per_cycle=run.steps_per_cycle,
        report_sha256=readback.report_sha256,
        heldout_correct_minus_zero=heldout["correct_labels"] - heldout["zero_update"],
        transfer_correct_minus_zero=transfer["correct_labels"] - transfer["zero_update"],
        heldout_accuracy_by_arm=tuple((name, heldout[name]) for name in ARMS),
        transfer_accuracy_by_arm=tuple((name, transfer[name]) for name in ARMS),
        meets_predeclared_public_comparison=public_success,
    )


def score_public_minilm_campaign(campaign: PublicMiniLMCampaign, *,
                                 expected_campaign_sha256: str,
                                 reports: Iterable[ReportReceipt]) -> PublicCampaignScore:
    """Reject incomplete/postselected grids, then score every pinned report.

    ``expected_campaign_sha256`` MUST come from a distinct pre-outcome record.
    Passing back a digest computed from observed reports defeats preregistration.
    The all-cell success comparison is descriptive only, not graduation.
    """
    if type(campaign) is not PublicMiniLMCampaign:
        raise ValueError("invalid campaign manifest")
    _digest(expected_campaign_sha256, "expected campaign")
    if (campaign.sha256 != expected_campaign_sha256
            or _canonical(_payload(campaign)) != campaign.canonical_bytes
            or sha256(campaign.canonical_bytes).hexdigest() != expected_campaign_sha256):
        raise ValueError("campaign manifest digest mismatch from preregistered record")
    if (campaign.physical_model_verified or campaign.evaluator_custody_verified
            or campaign.graduation_authorized):
        raise ValueError("campaign may not assert physical model/graduation verification")
    expected = {run.run_id: run for run in campaign.runs}
    supplied = tuple(reports)
    if (len(supplied) != len(expected)
            or any(type(item) is not ReportReceipt or item.run_id not in expected
                   for item in supplied)
            or len({item.run_id for item in supplied}) != len(supplied)):
        raise ValueError("all preregistered runs must be supplied exactly once")
    receipts = {item.run_id: item for item in supplied}
    if set(receipts) != set(expected):
        raise ValueError("all preregistered runs must be supplied exactly once")
    results = tuple(_score_report(campaign, run, receipts[run.run_id])
                    for run in campaign.runs)
    return PublicCampaignScore(
        status="PROVISIONAL_PUBLIC_DEVELOPMENT",
        campaign_sha256=expected_campaign_sha256,
        run_results=results,
        all_cells_meet_predeclared_public_comparison=all(
            item.meets_predeclared_public_comparison for item in results),
        limitations=(
            "Manifest digest has no independent timestamp or immutability attestation here",
            "Trainer-origin public reports do not establish physical model or fresh inference",
            "No independent examiner, private transfer custody, formal verification, or graduation",
        ),
    )
