"""Strict source-level bridge from arm custody to independent holdout scoring.

Run ONLY inside an independently controlled evaluator host. Caller-supplied
artifact digests are expected to come from separate byte readback, but this
function cannot prove how they were acquired. Likewise, HMAC authentication
cannot prove key custody or physical model execution. No graduation authority
is returned or delegated to the training-side Minotaur.
"""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import re
from typing import Mapping, Any

from .control_provenance import ARMS, verify_matched_group
from .independent_holdout import (
    ArmRun, ExamCase, HoldoutVerdict, TrainingExposure,
    case_commitment, evaluate_holdout,
)


_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)


@dataclass(frozen=True)
class BoundEvaluation:
    verdict: HoldoutVerdict
    sealed_manifest_sha256: tuple[tuple[str, str], ...]
    graduation_authorized: bool = False
    physical_model_verified: bool = False


def _sha(value: Any, label: str) -> None:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")


def _id(value: Any, label: str) -> None:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")


def bind_and_evaluate(
    *,
    sealed_arms: tuple[dict[str, Any], ...],
    arm_runs: tuple[ArmRun, ...],
    training_exposure: TrainingExposure,
    cases: tuple[ExamCase, ...],
    evaluator_precommitment: str,
    examiner_gold_seal: str,
    retention_before: Mapping[str, str],
    custody_key: bytes,
    examiner_key: bytes,
    correct_label_commitment: str,
    corrupted_label_commitment: str,
    empty_label_commitment: str,
    observed_artifacts: Mapping[str, tuple[str, str]],
    expected_experiment_id: str,
    expected_source_tree_sha256: str,
    expected_objective_id: str,
    expected_structure_keys: tuple[str, ...],
) -> BoundEvaluation:
    """Bind four source-level claims to the evaluator's immutable inputs.

    ``observed_artifacts`` gives `(adapter_digest, report_digest)` for every
    arm and must be independently derived from actual files. This code only
    compares the supplied identities; physical readback is an external gate.

    Require separate control-custody and private-gold keys. The source cannot
    enforce that the host keeps either secret from the trainer; it fails if a
    caller accidentally supplies the same bytes for both responsibilities.
    """
    if (type(custody_key) is not bytes or type(examiner_key) is not bytes
            or len(custody_key) < 32 or len(examiner_key) < 32):
        raise ValueError("distinct evaluator-held 32-byte keys required")
    if hmac.compare_digest(custody_key, examiner_key):
        raise ValueError("distinct control-custody and examiner-gold keys required")
    _id(expected_experiment_id, "experiment ID")
    _id(expected_objective_id, "objective ID")
    _sha(expected_source_tree_sha256, "source-tree identity")
    _sha(evaluator_precommitment, "exam precommitment")
    if case_commitment(cases) != evaluator_precommitment:
        raise ValueError("exam differs from evaluator precommitment")

    authenticated = verify_matched_group(
        sealed_arms, custody_key,
        correct_label_commitment=correct_label_commitment,
        corrupted_label_commitment=corrupted_label_commitment,
        empty_label_commitment=empty_label_commitment,
        expected_exam_precommit=evaluator_precommitment,
    )
    if any(m.experiment_id != expected_experiment_id for m in authenticated):
        raise ValueError("experiment identity mismatch")
    if any(m.objective_id != expected_objective_id for m in authenticated):
        raise ValueError("objective identity mismatch")
    if any(m.source_tree_sha256 != expected_source_tree_sha256 for m in authenticated):
        raise ValueError("source tree identity mismatch")

    if (type(arm_runs) is not tuple or len(arm_runs) != len(ARMS)
            or any(type(a) is not ArmRun for a in arm_runs)):
        raise ValueError("four evaluated model arm runs required")
    for run in arm_runs:
        run.validate()
    by_name = {run.name: run for run in arm_runs}
    if len(by_name) != len(ARMS) or set(by_name) != set(ARMS):
        raise ValueError("missing/duplicate evaluated arm")
    for manifest in authenticated:
        run = by_name[manifest.arm]
        if (run.base_sha256 != manifest.base_model_sha256
                or run.initial_adapter_sha256 != manifest.initial_adapter_sha256
                or run.final_adapter_sha256 != manifest.final_adapter_sha256):
            raise ValueError("arm checkpoint identity mismatch")
        if (run.optimizer_updates != manifest.optimizer_updates
                or run.compute_steps != manifest.compute_steps
                or run.visited_case_ids != tuple(v[1] for v in manifest.training_visits)):
            raise ValueError("arm training/optimizer evidence mismatch")

    if type(training_exposure) is not TrainingExposure:
        raise ValueError("invalid independent training exposure")
    training_exposure.validate()
    visits = authenticated[2].training_visits
    if (set(training_exposure.case_ids) != {visit[1] for visit in visits}
            or set(training_exposure.prompt_sha256) != {visit[2] for visit in visits}):
        raise ValueError("training exposure differs from authenticated visits")
    if (tuple(zip(training_exposure.case_ids, training_exposure.prompt_sha256,
                  training_exposure.structure_keys))
            != authenticated[2].training_structure_exposures):
        raise ValueError("authenticated training structure provenance mismatch")
    if (type(expected_structure_keys) is not tuple
            or not expected_structure_keys
            or training_exposure.structure_keys != expected_structure_keys):
        raise ValueError("training structure custody mismatch")
    for structure in expected_structure_keys:
        _id(structure, "training structure")

    if (not isinstance(observed_artifacts, Mapping)
            or set(observed_artifacts) != set(ARMS)):
        raise ValueError("complete independent artifact readback required")
    for manifest in authenticated:
        observed = observed_artifacts[manifest.arm]
        if type(observed) is not tuple or len(observed) != 2:
            raise ValueError("invalid artifact readback pair")
        _sha(observed[0], "adapter artifact")
        _sha(observed[1], "report artifact")
        if (observed[0] != manifest.final_adapter_sha256
                or observed[1] != manifest.report_sha256):
            raise ValueError("artifact readback does not match authenticated manifest")

    verdict = evaluate_holdout(
        cases, evaluator_precommitment, training_exposure, arm_runs,
        retention_before, examiner_gold_seal=examiner_gold_seal,
        examiner_key=examiner_key,
    )
    if verdict.graduation_authorized:
        raise ValueError("independent development evaluator must not graduate")
    sealed_digests = {
        seal["manifest"]["arm"]: seal["manifest_sha256"] for seal in sealed_arms
    }
    return BoundEvaluation(
        verdict=verdict,
        sealed_manifest_sha256=tuple(
            (manifest.arm, sealed_digests[manifest.arm]) for manifest in authenticated
        ),
    )
