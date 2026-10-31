"""Reload and re-audit the opt-in FIVE-arm public MiniLM campaign.

This is an offline SOURCE audit of *already saved* reports and checkpoint
files. It is NOT the separate FOUR-arm protocol in
``preregistered_minilm_campaign`` and deliberately does not translate between
their labels, controls, or report schemas.

A caller must supply an independently recorded, pre-outcome plan digest and
its own witness verifier. This module checks callback output and file bytes;
it cannot establish the verifier's custody, the witness's trusted time, the
origin of pretrained neural weights, or private examination independence.
In-process test callbacks NEVER become a physical or graduation receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from scripts import qualify_maze_multiseed_protocol_cpu as protocol


_SHA = re.compile(r"[a-f0-9]{64}\Z", re.ASCII)
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_ARTIFACTS = 4096


class FiveArmCampaignError(ValueError):
    """An offline campaign input or source audit failed closed."""


@dataclass(frozen=True)
class FiveArmCampaignEvidence:
    status: str
    plan_sha256: str
    control_arms: tuple[str, ...]
    run_count: int
    checkpoint_files_checked: int
    all_cells_correct_exceed_every_control: bool
    public_aggregate_json: bytes
    public_aggregate_sha256: str
    limitations: tuple[str, ...]
    physical_model_verified: bool = False
    evaluator_custody_verified: bool = False
    graduation_authorized: bool = False


def _sha(value: Any, name: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise FiveArmCampaignError(f"invalid {name} SHA256")
    return value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise FiveArmCampaignError("duplicate JSON field in child report")
        result[name] = value
    return result


def _read_regular(path: Path, *, limit: int) -> bytes:
    if (path.is_symlink() or
            (hasattr(path, "is_junction") and path.is_junction()) or
            not path.is_file()):
        raise FiveArmCampaignError("required campaign artifact missing or symlinked")
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise FiveArmCampaignError("campaign JSON artifact exceeds bounded size")
    return content


def _check_root(root: Path) -> None:
    if (not isinstance(root, Path) or not root.is_absolute()
            or root != root.resolve(strict=False) or not root.is_dir()):
        raise FiveArmCampaignError("campaign must have a canonical absolute root")
    for parent in (root, *root.parents):
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise FiveArmCampaignError("campaign root traverses a symlink/junction")
    count = 0
    for item in root.rglob("*"):
        count += 1
        if count > _MAX_ARTIFACTS:
            raise FiveArmCampaignError("campaign has too many artifacts")
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise FiveArmCampaignError("campaign has symlinked artifacts")


def audit_fivearm_campaign_from_disk(
    *,
    spec: protocol.ProtocolSpec,
    campaign_root: Path,
    expected_plan_sha256: str,
    expected_runner_sha256: str,
    expected_protocol_sha256: str,
    expected_pretrained_base_sha256: str,
    expected_report_sha256_by_run: Mapping[str, str],
    preoutcome_witness: bytes,
    verify_external_preoutcome_witness: Callable[[bytes, str], bool],
) -> FiveArmCampaignEvidence:
    """Replay EVERY pinned child/aggregate check without starting a trainer.

    The injected witness verifier must live in independently governed host
    infrastructure. A true result here merely permits a conditional public
    SOURCE audit, not a factual claim that such infrastructure exists.
    """
    if type(spec) is not protocol.ProtocolSpec or spec.preregistered_controls is not True:
        raise FiveArmCampaignError("explicit five-arm protocol required")
    try:
        spec.__post_init__()
        plan = protocol.build_plan(spec)
    except (TypeError, ValueError, OSError) as exc:
        raise FiveArmCampaignError("invalid five-arm source protocol") from exc
    active_arms = tuple(plan["manifest"]["arms"])
    if active_arms != tuple(protocol.PREREG_ARMS) or len(active_arms) != 5:
        raise FiveArmCampaignError("source protocol does not supply exactly five arms")

    plan_hash = _sha(expected_plan_sha256, "externally pinned plan")
    runner_hash = _sha(expected_runner_sha256, "runner")
    protocol_hash = _sha(expected_protocol_sha256, "protocol")
    base_hash = _sha(expected_pretrained_base_sha256, "pretrained base")
    if (sha256(protocol._canonical(plan)).hexdigest() != plan_hash
            or plan["manifest"]["runner_sha256"] != runner_hash
            or plan["manifest"]["protocol_sha256"] != protocol_hash):
        raise FiveArmCampaignError("campaign plan/source pin differs from precommit")
    if (type(preoutcome_witness) is not bytes
            or not 16 <= len(preoutcome_witness) <= 4096
            or not callable(verify_external_preoutcome_witness)
            or verify_external_preoutcome_witness(preoutcome_witness, plan_hash) is not True):
        raise FiveArmCampaignError("missing or unaccepted external pre-outcome witness")

    jobs = plan["manifest"]["jobs"]
    expected_runs = {item["run_name"] for item in jobs}
    if (type(expected_report_sha256_by_run) is not dict
            or set(expected_report_sha256_by_run) != expected_runs
            or len(expected_report_sha256_by_run) != len(jobs)):
        raise FiveArmCampaignError("incomplete five-arm seed-budget grid")
    for run_name, digest in expected_report_sha256_by_run.items():
        if type(run_name) is not str:
            raise FiveArmCampaignError("invalid run identity")
        _sha(digest, f"run {run_name} report")

    _check_root(campaign_root)
    if _read_regular(campaign_root / "plan.json", limit=_MAX_JSON_BYTES) != protocol._canonical(plan):
        raise FiveArmCampaignError("persisted plan changed after precommit")
    saved_runs = campaign_root / "runs"
    if (not saved_runs.is_dir() or saved_runs.is_symlink()
            or {item.name for item in saved_runs.iterdir()} != expected_runs):
        raise FiveArmCampaignError("saved run roots do not match complete grid")

    audited: list[dict[str, Any]] = []
    for job in jobs:
        run_name = job["run_name"]
        run_root = saved_runs / run_name
        if not run_root.is_dir() or run_root.is_symlink():
            raise FiveArmCampaignError("run root missing or symlinked")
        report_path = run_root / "report.json"
        original = _read_regular(report_path, limit=_MAX_JSON_BYTES)
        if sha256(original).hexdigest() != expected_report_sha256_by_run[run_name]:
            raise FiveArmCampaignError("child report hash differs from pinned receipt")
        try:
            json.loads(original.decode("utf-8"), object_pairs_hook=_unique_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(
                           FiveArmCampaignError("nonfinite JSON in child report")))
            row = protocol.audit_child(job, plan, run_root)
        except (ValueError, TypeError, KeyError, OSError, UnicodeError,
                IndexError, RecursionError) as exc:
            raise FiveArmCampaignError(f"child {run_name} audit refused report or checkpoint") from exc
        # Audit is a separate read of the report and all checkpoint file hashes.
        # If the file changed between reads, the pinned receipt must no longer pass.
        if _read_regular(report_path, limit=_MAX_JSON_BYTES) != original:
            raise FiveArmCampaignError("child report changed during source re-audit")
        if row["pretrained_base_sha256"] != base_hash:
            raise FiveArmCampaignError("pretrained base digest differs from pinned model")
        audited.append(row)

    # A checkpoint read for an early job can become stale while later jobs
    # are audited. Reopen and rehash every child after the entire grid is
    # examined, and compare the same source-derived evidence before sealing
    # the aggregate. This is still NOT an atomic host-owned filesystem snapshot.
    _check_root(campaign_root)
    for job, prior in zip(jobs, audited, strict=True):
        run_name = job["run_name"]
        run_root = saved_runs / run_name
        try:
            second = protocol.audit_child(job, plan, run_root)
        except (ValueError, TypeError, KeyError, OSError, UnicodeError,
                IndexError, RecursionError) as exc:
            raise FiveArmCampaignError(
                f"child {run_name} checkpoint changed during campaign audit"
            ) from exc
        if (second != prior or
                sha256(_read_regular(run_root / "report.json",
                                     limit=_MAX_JSON_BYTES)).hexdigest()
                != expected_report_sha256_by_run[run_name]):
            raise FiveArmCampaignError(
                f"child {run_name} changed during campaign audit"
            )

    try:
        summary = protocol.summarize_public_controls(audited, spec)
    except (ValueError, TypeError, KeyError) as exc:
        raise FiveArmCampaignError("incomplete or mismatched public control scores") from exc
    saved_result = {
        "schema": protocol.SCHEMA,
        "plan_sha256": protocol._digest(plan),
        "qualification": "PUBLIC_MULTISEED_DEVELOPMENT_ONLY",
        "results": audited,
        "complete_jobs": len(audited),
        "public_aggregate": summary,
        "private_final_exam": "NOT_RUN",
        "prior_skill_retention": "NOT_RUN",
        "independent_model_and_evaluator_custody": "NOT_ESTABLISHED",
        "crash_resume": "NOT_SUPPORTED",
        "graduation_authorized": False,
        "preregistered_controls": "FIVE_ARM_PUBLIC_UNSEALED_NOT_INDEPENDENT",
    }
    if (_read_regular(campaign_root / "result.json", limit=_MAX_JSON_BYTES)
            != protocol._canonical(saved_result)):
        raise FiveArmCampaignError("persisted result differs from complete offline re-audit")
    all_positive = all(
        item["public_correct_advantage_all_controls"][split]
        for item in audited for split in protocol.PUBLIC_SPLITS
    )
    return FiveArmCampaignEvidence(
        status="FIVE_ARM_PUBLIC_SOURCE_AUDIT_ONLY",
        plan_sha256=plan_hash,
        control_arms=active_arms,
        run_count=len(audited),
        checkpoint_files_checked=sum(item["checkpoint_files_checked"] for item in audited),
        all_cells_correct_exceed_every_control=all_positive,
        public_aggregate_json=protocol._canonical(summary),
        public_aggregate_sha256=sha256(protocol._canonical(summary)).hexdigest(),
        limitations=(
            "Pre-outcome witness callback is not independently attested here",
            "Five-arm checkpoint files are rehashed, not physically reopened as neural tensors",
            "Trainer-origin public scores and model base digests remain self-reported",
            "No private examination, independent semantic transfer, anti-rollback or graduation",
        ),
    )
