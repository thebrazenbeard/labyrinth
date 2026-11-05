"""Immutable source precommit for matched *development* neural controls.

This module constructs a reproducible experiment PLAN, not training evidence.
It cannot know whether a declared family is semantically independent, whether
the specified optimizer was run, or whether a model actually consumed a case.
Public development labels are visible; private final-exam cases are not.

Final-exam roster commitments can be signed by a *separately administered*
evaluator, but source verification never releases private prompts and never
authorizes graduation. Real custody, pre-experiment timestamps, a single-use
ledger, and mandatory execution hooks require an external deployment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
import json
import random
import re
from typing import Any

from .retention_family_registry import canonical_family_id


SCHEMA = "labyrinth.preregistered-control-matrix.v1"
FINAL_SCHEMA = "labyrinth.sealed-final-roster-placeholder.v1"
DOMAIN = b"labyrinth.preregistered-control-matrix.v1\0"
FINAL_DOMAIN = b"labyrinth.sealed-final-roster-placeholder.v1\0"
HEX = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
ARMS = ("correct", "zero_lr", "permuted", "randomized")
BASELINES = ("majority_train_only", "tfidf_bow_train_only",
             "numeric_rule_feasibility_oracle")
REPRESENTATIONS = ("qualitative", "quantitative")


class PrecommitError(ValueError):
    """The proposed experimental roster cannot be preregistered safely."""


@dataclass(frozen=True)
class CaseRef:
    case_id: str
    family_id: str
    prompt_sha256: str
    label: int
    representation: str
    seed: int


@dataclass(frozen=True)
class ArmRef:
    name: str
    optimizer: str
    learning_rate: float
    initial_weights_sha256: str
    model_seed: int
    training_case_visits: tuple[str, ...]
    label_transform: str
    corruption_seed: int | None = None
    permutation: tuple[int, ...] = ()
    optimizer_step_required: bool = True


@dataclass(frozen=True)
class FinalExamPlaceholder:
    """Only the commitment and independently signed metadata, never cases."""

    roster_sha256: str
    signer_id: str
    evaluator_host_id: str
    trainer_host_id: str
    nonce_sha256: str
    signature_hex: str
    graduation_authorized: bool = False


@dataclass(frozen=True)
class PrecommittedMatrix:
    """Canonical immutable bytes; source digest is NOT an external timestamp."""

    canonical_bytes: bytes
    sha256: str
    graduation_authorized: bool = False
    independent_exam_authorized: bool = False


def _canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _hex(value: Any) -> bool:
    return type(value) is str and HEX.fullmatch(value) is not None


def _identifier(value: Any) -> bool:
    return type(value) is str and IDENTIFIER.fullmatch(value) is not None


def _validate_cases(cases: tuple[CaseRef, ...], label_count: int,
                    description: str) -> None:
    if (type(cases) is not tuple or not 2 <= len(cases) <= 4096
            or any(type(case) is not CaseRef for case in cases)):
        raise PrecommitError(f"bounded {description} case roster required")
    for case in cases:
        if (not _identifier(case.case_id) or not _hex(case.prompt_sha256)
                or type(case.label) is not int or not 0 <= case.label < label_count
                or case.representation not in REPRESENTATIONS
                or type(case.seed) is not int or not 0 <= case.seed < 2**63):
            raise PrecommitError(f"invalid {description} case commitment")
        try:
            canonical = canonical_family_id(case.family_id)
        except ValueError as exc:
            raise PrecommitError("invalid declared semantic family ID") from exc
        if canonical != case.family_id:
            raise PrecommitError("family ID must be stored in canonical form")
    if (len({case.case_id for case in cases}) != len(cases)
            or len({case.prompt_sha256 for case in cases}) != len(cases)
            or len({case.seed for case in cases}) != len(cases)):
        raise PrecommitError(f"reused {description} case/prompt/seed")
    counts = [sum(case.label == label for case in cases)
              for label in range(label_count)]
    if not all(counts) or max(counts) != min(counts):
        raise PrecommitError(f"{description} gold labels are not class balanced")
    # Whole-roster balance is insufficient when representation predicts gold.
    # Every representation used in a partition needs each class equally often;
    # otherwise a classifier can exploit the modality instead of the task.
    for representation in {case.representation for case in cases}:
        stratum_counts = [
            sum(case.representation == representation and case.label == label
                for case in cases)
            for label in range(label_count)
        ]
        if not all(stratum_counts) or max(stratum_counts) != min(stratum_counts):
            raise PrecommitError(
                f"{description} representation/label strata are confounded")


def _validate_arms(arms: tuple[ArmRef, ...], cases: tuple[CaseRef, ...],
                   label_count: int) -> dict[str, tuple[int, ...]]:
    if (type(arms) is not tuple or len(arms) != 4
            or any(type(arm) is not ArmRef or type(arm.name) is not str
                   for arm in arms)
            or {arm.name for arm in arms} != set(ARMS)):
        raise PrecommitError("exact four matched neural control arms required")
    reference = arms[0]
    original_visits = tuple(case.case_id for case in cases)
    if (len(set(reference.training_case_visits)) != len(cases)
            or set(reference.training_case_visits) != set(original_visits)):
        raise PrecommitError("visits must cover all TRAIN case IDs exactly once")
    transforms = {
        "correct": "identity",
        "zero_lr": "identity",
        "permuted": "fixed_permutation",
        "randomized": "random_balanced_shuffling",
    }
    derived: dict[str, tuple[int, ...]] = {}
    case_by_id = {case.case_id: case for case in cases}
    for arm in arms:
        if (arm.optimizer != "AdamW"
                or type(arm.learning_rate) is not float
                or not 0 <= arm.learning_rate <= 0.01
                or not _hex(arm.initial_weights_sha256)
                or type(arm.model_seed) is not int
                or not 0 <= arm.model_seed < 2**63
                or arm.training_case_visits != reference.training_case_visits
                or arm.initial_weights_sha256 != reference.initial_weights_sha256
                or arm.model_seed != reference.model_seed
                or arm.label_transform != transforms[arm.name]
                or arm.optimizer_step_required is not True):
            raise PrecommitError("optimizer, initial weights, seed, visits or label transform mismatched")
        if (arm.learning_rate == 0.0) != (arm.name == "zero_lr"):
            raise PrecommitError("zero-learning-rate control must still call optimizer.step")
        labels = [case_by_id[case_id].label for case_id in arm.training_case_visits]
        if arm.name in ("correct", "zero_lr"):
            if arm.permutation or arm.corruption_seed is not None:
                raise PrecommitError("identity arm cannot declare corruption metadata")
        elif arm.name == "permuted":
            permutation = arm.permutation
            if (type(permutation) is not tuple or len(permutation) != label_count
                    or sorted(permutation) != list(range(label_count))
                    or permutation == tuple(range(label_count))
                    or arm.corruption_seed is not None):
                raise PrecommitError("nonidentity balanced label permutation required")
            labels = [permutation[label] for label in labels]
        else:
            if (arm.permutation or type(arm.corruption_seed) is not int
                    or not 0 <= arm.corruption_seed < 2**63
                    or arm.corruption_seed == arm.model_seed):
                raise PrecommitError("separate randomized corruption seed required")
            random.Random(arm.corruption_seed).shuffle(labels)
            if labels == [case_by_id[case_id].label
                          for case_id in arm.training_case_visits]:
                raise PrecommitError("random corruption produced identical labels")
        derived[arm.name] = tuple(labels)
    if (derived["permuted"] == derived["correct"]
            or derived["randomized"] == derived["correct"]):
        raise PrecommitError("corrupted controls must change at least one label")
    return derived


def build_precommitted_matrix(
    *, objective_id: str, scope_sha256: str, pretrained_model_sha256: str,
    nonce_sha256: str, training_cases: tuple[CaseRef, ...],
    public_development_cases: tuple[CaseRef, ...],
    arms: tuple[ArmRef, ...], label_count: int,
    representation_policy: str, final_placeholder: FinalExamPlaceholder,
    independent_evaluator_key: bytes, trainer_key: bytes,
    expected_final_signer_id: str, expected_evaluator_host_id: str,
    expected_trainer_host_id: str,
    baseline_roster: tuple[str, ...] = BASELINES,
) -> PrecommittedMatrix:
    """Freeze a reproducible experiment before any TRAIN work begins."""
    if (not _identifier(objective_id) or not _hex(scope_sha256)
            or not _hex(pretrained_model_sha256) or not _hex(nonce_sha256)
            or type(label_count) is not int or not 2 <= label_count <= 16):
        raise PrecommitError("bounded declared objective/model/nonce required")
    _validate_cases(training_cases, label_count, "TRAIN")
    _validate_cases(public_development_cases, label_count, "public development")
    if (len({case.case_id for case in training_cases + public_development_cases})
            != len(training_cases) + len(public_development_cases)
            or len({case.prompt_sha256 for case in training_cases + public_development_cases})
            != len(training_cases) + len(public_development_cases)
            or len({case.seed for case in training_cases + public_development_cases})
            != len(training_cases) + len(public_development_cases)):
        raise PrecommitError("TRAIN/public exam case, prompt or seed reused")
    if ({case.family_id for case in training_cases}
            & {case.family_id for case in public_development_cases}):
        raise PrecommitError("declared TRAIN/public semantic families overlap")
    train_repr = {case.representation for case in training_cases}
    development_repr = {case.representation for case in public_development_cases}
    if (representation_policy == "same_representation"
            and train_repr != development_repr
            or representation_policy == "cross_representation"
            and not train_repr.isdisjoint(development_repr)
            or representation_policy not in ("same_representation", "cross_representation")):
        raise PrecommitError("undeclared train/development representation mixing")
    if (type(baseline_roster) is not tuple or baseline_roster != BASELINES):
        raise PrecommitError("preregistered learner-free baseline roster required")
    labels_by_arm = _validate_arms(arms, training_cases, label_count)
    if (type(final_placeholder) is not FinalExamPlaceholder
            or not _hex(final_placeholder.roster_sha256)
            or not _hex(final_placeholder.nonce_sha256)
            or not _hex(final_placeholder.signature_hex)
            or not _identifier(final_placeholder.signer_id)
            or not _identifier(final_placeholder.evaluator_host_id)
            or not _identifier(final_placeholder.trainer_host_id)
            or final_placeholder.graduation_authorized is not False):
        raise PrecommitError("sealed final exam metadata required; answers prohibited")
    if final_placeholder.evaluator_host_id == final_placeholder.trainer_host_id:
        raise PrecommitError("declared evaluator/trainer host identities must differ")
    if not hmac.compare_digest(final_placeholder.nonce_sha256, nonce_sha256):
        raise PrecommitError("final roster nonce differs from pinned experiment nonce")
    # A 64-hex signature shape is not authenticated evidence. Verify against
    # the caller's independently provisioned evaluator key before recording
    # the placeholder. Possession/custody of that key is NOT proved here.
    source_final_authentication = verify_final_placeholder(
        final_placeholder,
        evaluator_key=independent_evaluator_key,
        trainer_key=trainer_key,
        expected_signer_id=expected_final_signer_id,
        expected_evaluator_host_id=expected_evaluator_host_id,
        expected_trainer_host_id=expected_trainer_host_id,
    )
    payload = {
        "schema": SCHEMA,
        "objective_id": objective_id,
        "scope_sha256": scope_sha256,
        "pretrained_model_sha256": pretrained_model_sha256,
        "nonce_sha256": nonce_sha256,
        "label_count": label_count,
        "representation_policy": representation_policy,
        "training_cases": [asdict(case) for case in training_cases],
        "public_development_cases": [asdict(case) for case in public_development_cases],
        "arms": [asdict(arm) for arm in arms],
        "effective_training_labels_by_arm": {
            arm: list(labels) for arm, labels in sorted(labels_by_arm.items())
        },
        "baseline_roster": baseline_roster,
        "numeric_rule_baseline_role": "TASK_FEASIBILITY_ONLY_NOT_NEURAL_COMPETITOR",
        "final_exam_placeholder": asdict(final_placeholder),
        "final_exam_release": "BLOCKED_PENDING_INDEPENDENT_CUSTODY",
        "final_exam_source_authentication": source_final_authentication,
        "final_exam_key_custody": "CALLER_ASSERTED_NOT_EXTERNALLY_ATTESTED",
        "semantic_family_independence": "UNVERIFIED_EXTERNAL_REVIEW_REQUIRED",
        "precommit_timestamp_authority": "NONE_SOURCE_DIGEST_ONLY",
        "model_execution_authenticated": False,
        "graduation_authorized": False,
    }
    canonical = _canonical(payload)
    if len(canonical) > 2_000_000:
        raise PrecommitError("bounded serialized control plan exceeded")
    return PrecommittedMatrix(canonical, sha256(DOMAIN + canonical).hexdigest())


def _final_body(placeholder: FinalExamPlaceholder) -> dict[str, Any]:
    return {
        "schema": FINAL_SCHEMA,
        "roster_sha256": placeholder.roster_sha256,
        "signer_id": placeholder.signer_id,
        "evaluator_host_id": placeholder.evaluator_host_id,
        "trainer_host_id": placeholder.trainer_host_id,
        "nonce_sha256": placeholder.nonce_sha256,
        "graduation_authorized": False,
    }


def signed_final_placeholder(
    *, roster_sha256: str, signer_id: str, evaluator_host_id: str,
    trainer_host_id: str, nonce_sha256: str,
    evaluator_key: bytes,
) -> FinalExamPlaceholder:
    """Source signer helper; only an external custodian can make it independent."""
    preliminary = FinalExamPlaceholder(roster_sha256, signer_id,
                                       evaluator_host_id, trainer_host_id,
                                       nonce_sha256, "0" * 64)
    if (not all(_hex(value) for value in (roster_sha256, nonce_sha256))
            or not all(_identifier(value) for value in
                       (signer_id, evaluator_host_id, trainer_host_id))
            or evaluator_host_id == trainer_host_id
            or type(evaluator_key) is not bytes or len(evaluator_key) < 32):
        raise PrecommitError("invalid independently sealable final roster metadata")
    signature = hmac.new(evaluator_key, FINAL_DOMAIN + _canonical(
        _final_body(preliminary)), sha256).hexdigest()
    return FinalExamPlaceholder(roster_sha256, signer_id, evaluator_host_id,
                                trainer_host_id, nonce_sha256, signature)


def verify_final_placeholder(
    placeholder: FinalExamPlaceholder, *, evaluator_key: bytes,
    trainer_key: bytes, expected_signer_id: str,
    expected_evaluator_host_id: str, expected_trainer_host_id: str,
) -> str:
    """Authenticate source metadata, but NEVER issue exam-release authority."""
    if (type(placeholder) is not FinalExamPlaceholder
            or placeholder.graduation_authorized is not False
            or not all(_hex(value) for value in (
                placeholder.roster_sha256, placeholder.nonce_sha256,
                placeholder.signature_hex))
            or not all(_identifier(value) for value in (
                placeholder.signer_id, placeholder.evaluator_host_id,
                placeholder.trainer_host_id, expected_signer_id,
                expected_evaluator_host_id, expected_trainer_host_id))
            or type(evaluator_key) is not bytes or len(evaluator_key) < 32
            or type(trainer_key) is not bytes or len(trainer_key) < 32
            or hmac.compare_digest(evaluator_key, trainer_key)
            or placeholder.signer_id != expected_signer_id
            or placeholder.evaluator_host_id != expected_evaluator_host_id
            or placeholder.trainer_host_id != expected_trainer_host_id
            or placeholder.evaluator_host_id == placeholder.trainer_host_id):
        raise PrecommitError("final exam source custody metadata invalid")
    expected = hmac.new(evaluator_key, FINAL_DOMAIN + _canonical(
        _final_body(placeholder)), sha256).hexdigest()
    if not hmac.compare_digest(expected, placeholder.signature_hex):
        raise PrecommitError("final roster placeholder signature invalid")
    return "SOURCE_SIGNATURE_VERIFIED_FINAL_EXAM_STILL_BLOCKED"
