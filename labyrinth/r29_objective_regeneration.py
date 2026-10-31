"""R29 public-toy Minotaur corridor regeneration after a demonstrated shortcut.

All hashes, observations, template identities, and rights/reference commitments
are public caller declarations, not authenticated evidence. This creates NO
private exam content, neural update, graduation result, or promotion authority.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import re

from .r28_reward_hacking import (
    ChallengeObservation, FrozenChallengePlan, audit_reward_shortcuts,
    build_public_toy_challenge, _plan_commitment,
)
from .trainingday_objective_metamorphic import (
    FormalObjectiveContract, _formal_fingerprint,
)


_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_SEED = re.compile(r"[A-Za-z0-9_.:-]{1,32}\Z", re.ASCII)
_SHORTCUTS = frozenset((
    "PROMPT_HINT_LEAKAGE", "GRADING_TEMPLATE_IMITATION", "TRAINING_EXEMPLAR_REUSE",
))
_CASE_LIMIT = 512
_MAX_GENERATION = 1024


def _sha(value: object) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _commit(domain: bytes, payload: dict[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
    return sha256(domain + b"\0" + data).hexdigest()


@dataclass(frozen=True)
class PublicRegenerationBoundary:
    objective_fingerprint: str
    rights_sha256: str
    reference_sha256: str
    evaluator_sha256: str
    conditions_sha256: str
    public_context_sha256: str
    commitment_sha256: str


@dataclass(frozen=True)
class PublicTemplateFamily:
    skill_id: str
    development_sha256: str
    challenge_sha256: str


@dataclass(frozen=True)
class PublicRegenerationReceipt:
    boundary: PublicRegenerationBoundary
    generation: int
    seed: str
    per_skill: int
    source_plan_commitment_sha256: str
    source_training_prompt_sha256: tuple[str, ...]
    predecessor_commitment_sha256: str | None
    attack_findings: tuple[str, ...]
    template_families: tuple[PublicTemplateFamily, ...]
    development_plan: FrozenChallengePlan
    untouched_challenge_plan: FrozenChallengePlan
    commitment_sha256: str
    graduation_authorized: bool = False
    promotion_allowed: bool = False
    independent_exam_verified: bool = False


@dataclass(frozen=True)
class PublicRegenerationVerdict:
    status: str
    reasons: tuple[str, ...]
    graduation_authorized: bool = False
    promotion_allowed: bool = False
    independent_exam_verified: bool = False


def _boundary_commitment(boundary: PublicRegenerationBoundary) -> str:
    return _commit(b"labyrinth.r29.public-boundary.v1", {
        "objective": boundary.objective_fingerprint,
        "rights": boundary.rights_sha256,
        "reference": boundary.reference_sha256,
        "evaluator": boundary.evaluator_sha256,
        "conditions": boundary.conditions_sha256,
        "context": boundary.public_context_sha256,
    })


def freeze_public_regeneration_boundary(
    contract: FormalObjectiveContract,
    rights_sha256: str,
    reference_sha256: str,
) -> PublicRegenerationBoundary:
    """Freeze *declared* rights/reference and exact objective semantics."""
    fingerprint = _formal_fingerprint(contract)
    if fingerprint is None or not _sha(rights_sha256) or not _sha(reference_sha256):
        raise ValueError("malformed public regeneration boundary")
    boundary = PublicRegenerationBoundary(
        fingerprint, rights_sha256, reference_sha256,
        contract.evaluator_sha256, contract.conditions_sha256,
        contract.public_context_sha256, "0" * 64,
    )
    return PublicRegenerationBoundary(
        **{**vars(boundary), "commitment_sha256": _boundary_commitment(boundary)}
    )


def _boundary_matches(contract: FormalObjectiveContract,
                      boundary: object) -> bool:
    fingerprint = _formal_fingerprint(contract)
    return (fingerprint is not None
            and type(boundary) is PublicRegenerationBoundary
            and all(_sha(getattr(boundary, key)) for key in (
                "objective_fingerprint", "rights_sha256", "reference_sha256",
                "evaluator_sha256", "conditions_sha256",
                "public_context_sha256", "commitment_sha256",
            ))
            and boundary.objective_fingerprint == fingerprint
            and boundary.evaluator_sha256 == contract.evaluator_sha256
            and boundary.conditions_sha256 == contract.conditions_sha256
            and boundary.public_context_sha256 == contract.public_context_sha256
            and boundary.commitment_sha256 == _boundary_commitment(boundary))


def _templates_valid(contract: FormalObjectiveContract, value: object) -> bool:
    if type(value) is not tuple or len(value) != len(contract.skills):
        return False
    if not all(type(x) is PublicTemplateFamily and type(x.skill_id) is str
               and _sha(x.development_sha256) and _sha(x.challenge_sha256)
               for x in value):
        return False
    ids = [x.skill_id for x in value]
    hashes = [h for x in value for h in (x.development_sha256, x.challenge_sha256)]
    return (set(ids) == {x.skill_id for x in contract.skills}
            and len(ids) == len(set(ids))
            and len(hashes) == len(set(hashes)))


def _seed_for(receipt: PublicRegenerationReceipt, split: str) -> str:
    namespace = _commit(b"labyrinth.r29.public-toy-generation.v1", {
        "boundary": receipt.boundary.commitment_sha256,
        "source": receipt.source_plan_commitment_sha256,
        "predecessor": receipt.predecessor_commitment_sha256,
        "generation": receipt.generation,
        "seed": receipt.seed,
        "split": split,
        # Regeneration must respond to what failed. Two distinct public
        # shortcuts or template constraints cannot silently share case IDs.
        "shortcut_findings": list(receipt.attack_findings),
        "template_families": [vars(x) for x in receipt.template_families],
    })[:32]
    return f"r29-{split}-{namespace}"


def _receipt_payload(receipt: PublicRegenerationReceipt) -> dict[str, object]:
    return {
        "boundary": receipt.boundary.commitment_sha256,
        "generation": receipt.generation,
        "seed": receipt.seed,
        "per_skill": receipt.per_skill,
        "source_plan": receipt.source_plan_commitment_sha256,
        "source_training": list(receipt.source_training_prompt_sha256),
        "predecessor": receipt.predecessor_commitment_sha256,
        "attack_findings": list(receipt.attack_findings),
        "templates": [vars(x) for x in receipt.template_families],
        "development_plan": receipt.development_plan.commitment_sha256,
        "challenge_plan": receipt.untouched_challenge_plan.commitment_sha256,
        "graduation_authorized": receipt.graduation_authorized,
        "promotion_allowed": receipt.promotion_allowed,
        "independent_exam_verified": receipt.independent_exam_verified,
    }


def _plan_shape(plan: object) -> bool:
    if type(plan) is not FrozenChallengePlan:
        return False
    if (not all(_sha(getattr(plan, x)) for x in (
            "objective_fingerprint", "evaluator_sha256", "conditions_sha256",
            "commitment_sha256"))
            or type(plan.objective_id) is not str
            or type(plan.cases) is not tuple
            or not 1 <= len(plan.cases) <= _CASE_LIMIT
            or type(plan.training_prompt_sha256) is not tuple
            or len(plan.training_prompt_sha256) > 4096
            or not all(_sha(x) for x in plan.training_prompt_sha256)):
        return False
    from .r28_reward_hacking import ChallengeSpec
    return all(type(x) is ChallengeSpec and type(x.case_id) is str
               and type(x.skill_id) is str and _sha(x.prompt_sha256)
               for x in plan.cases)


def _verify_public_receipt_local(
    contract: FormalObjectiveContract,
    receipt: PublicRegenerationReceipt,
) -> PublicRegenerationVerdict:
    """Check one receipt without claiming predecessor-chain continuity."""
    invalid = PublicRegenerationVerdict(
        "PUBLIC_TOY_RECEIPT_INVALID", ("UNATTESTED_OR_INCONSISTENT_TOY_EVIDENCE",)
    )
    if (type(receipt) is not PublicRegenerationReceipt
            or not _boundary_matches(contract, receipt.boundary)
            or type(receipt.generation) is not int
            or not 1 <= receipt.generation <= _MAX_GENERATION
            or type(receipt.seed) is not str or _SEED.fullmatch(receipt.seed) is None
            or type(receipt.per_skill) is not int
            or not 1 <= receipt.per_skill <= 256
            or receipt.per_skill * len(contract.skills) > _CASE_LIMIT
            or any(receipt.per_skill < x.min_cases for x in contract.skills)
            or not _sha(receipt.source_plan_commitment_sha256)
            or type(receipt.source_training_prompt_sha256) is not tuple
            or len(receipt.source_training_prompt_sha256) > 4096
            or not all(_sha(x) for x in receipt.source_training_prompt_sha256)
            or len(set(receipt.source_training_prompt_sha256))
            != len(receipt.source_training_prompt_sha256)
            or (receipt.predecessor_commitment_sha256 is not None
                and not _sha(receipt.predecessor_commitment_sha256))
            or (receipt.generation == 1) != (receipt.predecessor_commitment_sha256 is None)
            or type(receipt.attack_findings) is not tuple
            or not receipt.attack_findings
            or tuple(sorted(set(receipt.attack_findings))) != receipt.attack_findings
            or not set(receipt.attack_findings) <= _SHORTCUTS
            or not _templates_valid(contract, receipt.template_families)
            or not _plan_shape(receipt.development_plan)
            or not _plan_shape(receipt.untouched_challenge_plan)
            or not _sha(receipt.commitment_sha256)
            or receipt.graduation_authorized is not False
            or receipt.promotion_allowed is not False
            or receipt.independent_exam_verified is not False):
        return invalid
    try:
        expected_development = build_public_toy_challenge(
            contract, _seed_for(receipt, "dev"), receipt.per_skill,
            training_prompt_sha256=receipt.source_training_prompt_sha256,
        )
        expected_challenge = build_public_toy_challenge(
            contract, _seed_for(receipt, "holdout"), receipt.per_skill,
            training_prompt_sha256=tuple(sorted(
                x.prompt_sha256 for x in expected_development.cases
            )),
        )
        if (receipt.development_plan != expected_development
                or receipt.untouched_challenge_plan != expected_challenge):
            return invalid
        development = expected_development.cases
        challenge = expected_challenge.cases
        if (set(x.case_id for x in development).intersection(x.case_id for x in challenge)
                or set(x.prompt_sha256 for x in development).intersection(
                    x.prompt_sha256 for x in challenge)
                or set(receipt.source_training_prompt_sha256).intersection(
                    x.prompt_sha256 for x in development)):
            return invalid
        commitment = _commit(
            b"labyrinth.r29.public-toy-receipt.v1", _receipt_payload(receipt)
        )
        if receipt.commitment_sha256 != commitment:
            return invalid
    except (TypeError, ValueError, AttributeError):
        return invalid
    return PublicRegenerationVerdict(
        "PUBLIC_TOY_RECEIPT_VALID", ("DECLARED_PUBLIC_TOY_ONLY_NO_PRIVATE_EXAM",)
    )


def _transition_valid(previous: PublicRegenerationReceipt,
                      current: PublicRegenerationReceipt,
                      contract: FormalObjectiveContract) -> bool:
    """Require independently supplied, exact predecessor custody at each hop."""
    prior_templates = {
        digest for family in previous.template_families
        for digest in (family.development_sha256, family.challenge_sha256)
    }
    next_templates = {
        digest for family in current.template_families
        for digest in (family.development_sha256, family.challenge_sha256)
    }
    prior_counts = Counter(x.skill_id for x in previous.untouched_challenge_plan.cases)
    return (current.generation == previous.generation + 1
            and current.predecessor_commitment_sha256 == previous.commitment_sha256
            and current.boundary == previous.boundary
            and current.seed != previous.seed
            and current.source_plan_commitment_sha256
            == previous.untouched_challenge_plan.commitment_sha256
            and current.source_training_prompt_sha256
            == previous.untouched_challenge_plan.training_prompt_sha256
            and not prior_templates.intersection(next_templates)
            and all(current.per_skill >= prior_counts[x.skill_id] for x in contract.skills))


def verify_public_regeneration_receipt(
    contract: FormalObjectiveContract,
    receipt: PublicRegenerationReceipt,
    *,
    predecessor: PublicRegenerationReceipt | None = None,
    predecessor_ancestors: tuple[PublicRegenerationReceipt, ...] = (),
) -> PublicRegenerationVerdict:
    """Check local toy receipts plus externally supplied exact lineage.

    A G2+ receipt cannot self-certify predecessor custody: callers must
    supply G1..G(N-2) in ``predecessor_ancestors`` and G(N-1) as
    ``predecessor``. No authentication or private exam is implied.
    """
    invalid = PublicRegenerationVerdict(
        "PUBLIC_TOY_RECEIPT_INVALID", ("MISSING_OR_INCONSISTENT_PREDECESSOR_CUSTODY",)
    )
    if (type(predecessor_ancestors) is not tuple
            or type(receipt) is not PublicRegenerationReceipt
            or type(receipt.generation) is not int):
        return invalid
    if receipt.generation == 1:
        if predecessor is not None or predecessor_ancestors:
            return invalid
        return _verify_public_receipt_local(contract, receipt)
    if (receipt.generation < 2
            or type(predecessor) is not PublicRegenerationReceipt
            or len(predecessor_ancestors) != receipt.generation - 2
            or not all(type(x) is PublicRegenerationReceipt for x in predecessor_ancestors)):
        return invalid
    chain = (*predecessor_ancestors, predecessor, receipt)
    for item in chain:
        if _verify_public_receipt_local(contract, item).status != "PUBLIC_TOY_RECEIPT_VALID":
            return invalid
    for older, newer in zip(chain, chain[1:]):
        if not _transition_valid(older, newer, contract):
            return invalid
    return PublicRegenerationVerdict(
        "PUBLIC_TOY_RECEIPT_VALID", ("DECLARED_PUBLIC_TOY_LINEAGE_NO_PRIVATE_EXAM",)
    )


def regenerate_from_public_shortcut(
    contract: FormalObjectiveContract,
    failed_plan: FrozenChallengePlan,
    observations: tuple[ChallengeObservation, ...],
    selected_ids: tuple[str, ...],
    *,
    boundary: PublicRegenerationBoundary,
    generation: int,
    seed: str,
    per_skill: int,
    template_families: tuple[PublicTemplateFamily, ...],
    predecessor: PublicRegenerationReceipt | None = None,
    predecessor_ancestors: tuple[PublicRegenerationReceipt, ...] = (),
) -> PublicRegenerationReceipt:
    """Regenerate two distinct answer-free public corridors after shortcut proof.

    The caller must supply an R28 public shortcut failure on the *exact* prior
    challenge plan. ``untouched_challenge_plan`` is a separate synthetic set
    of identities, never automatically scored or added to training.
    """
    if (not _boundary_matches(contract, boundary)
            or not _templates_valid(contract, template_families)
            or not _plan_shape(failed_plan)):
        raise ValueError("objective, rights, reference, or template boundary changed")
    if (type(generation) is not int or not 1 <= generation <= _MAX_GENERATION
            or type(seed) is not str or _SEED.fullmatch(seed) is None
            or type(per_skill) is not int or not 1 <= per_skill <= 256
            or per_skill * len(contract.skills) > _CASE_LIMIT
            or any(per_skill < skill.min_cases for skill in contract.skills)):
        raise ValueError("invalid public generation size or seed")
    previous_commitment = None
    if predecessor is None:
        if generation != 1 or predecessor_ancestors:
            raise ValueError("missing predecessor generation")
    else:
        if (type(predecessor_ancestors) is not tuple
                or verify_public_regeneration_receipt(
                    contract, predecessor,
                    predecessor=(predecessor_ancestors[-1]
                                 if predecessor_ancestors else None),
                    predecessor_ancestors=predecessor_ancestors[:-1],
                ).status
                != "PUBLIC_TOY_RECEIPT_VALID"
                or generation != predecessor.generation + 1
                or seed == predecessor.seed
                or boundary != predecessor.boundary
                or failed_plan != predecessor.untouched_challenge_plan):
            raise ValueError("stale, forked, or unverified predecessor receipt")
        prior_templates = {
            digest for family in predecessor.template_families
            for digest in (family.development_sha256, family.challenge_sha256)
        }
        next_templates = {
            digest for family in template_families
            for digest in (family.development_sha256, family.challenge_sha256)
        }
        if prior_templates.intersection(next_templates):
            raise ValueError("predecessor development/challenge template family reused")
        previous_commitment = predecessor.commitment_sha256
    audit = audit_reward_shortcuts(contract, failed_plan, observations, selected_ids)
    if audit.status != "REWARD_HACK_BLOCKED":
        raise ValueError("no demonstrated public shortcut on frozen source plan")
    # A stale case-ID replay can also produce REWARD_HACK_BLOCKED. A genuine
    # shortcut needs case identities and split intact, and an explicit flag.
    if ("CASE_IDENTITY_OR_SPLIT_FORGERY" in audit.findings
            or "DUPLICATE_OR_RESELECTED_OBSERVATIONS" in audit.findings):
        raise ValueError("source observation identities or split replayed")
    findings = tuple(sorted(set(audit.findings).intersection(_SHORTCUTS)))
    if not findings:
        raise ValueError("source failure lacks a demonstrated shortcut")
    source_counts = Counter(x.skill_id for x in failed_plan.cases)
    if any(per_skill < source_counts[skill.skill_id] for skill in contract.skills):
        raise ValueError("regeneration would downgrade public curriculum case count")
    if (len(failed_plan.training_prompt_sha256) > 4096
            or len(set(failed_plan.training_prompt_sha256))
            != len(failed_plan.training_prompt_sha256)
            or failed_plan.commitment_sha256 != _plan_commitment(failed_plan)):
        raise ValueError("invalid source plan commitment")
    draft = PublicRegenerationReceipt(
        boundary, generation, seed, per_skill, failed_plan.commitment_sha256,
        failed_plan.training_prompt_sha256, previous_commitment, findings,
        tuple(sorted(template_families, key=lambda item: item.skill_id)),
        failed_plan, failed_plan, "0" * 64,
    )
    development = build_public_toy_challenge(
        contract, _seed_for(draft, "dev"), per_skill,
        training_prompt_sha256=failed_plan.training_prompt_sha256,
    )
    challenge = build_public_toy_challenge(
        contract, _seed_for(draft, "holdout"), per_skill,
        training_prompt_sha256=tuple(sorted(x.prompt_sha256 for x in development.cases)),
    )
    receipt = PublicRegenerationReceipt(
        boundary, generation, seed, per_skill, failed_plan.commitment_sha256,
        failed_plan.training_prompt_sha256, previous_commitment, findings,
        draft.template_families, development, challenge, "0" * 64,
    )
    receipt = PublicRegenerationReceipt(**{
        **vars(receipt),
        "commitment_sha256": _commit(
            b"labyrinth.r29.public-toy-receipt.v1", _receipt_payload(receipt),
        ),
    })
    if verify_public_regeneration_receipt(
        contract, receipt, predecessor=predecessor,
        predecessor_ancestors=predecessor_ancestors,
    ).status != "PUBLIC_TOY_RECEIPT_VALID":
        raise ValueError("regeneration did not yield a locally consistent receipt")
    return receipt
