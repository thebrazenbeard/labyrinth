"""Opt-in, independent custody of retention structure-family aliases.

This development-only companion cannot attest semantic independence, physical
model execution, custodian independence, or graduation. It binds a frozen
canonicalized overlap inventory to an independent signer via HMAC. Raw private
prompts are never included in its receipt.

The normalization catches common *surface* aliases, not every confusable or
paraphrase; genuine held-out independence requires external semantic review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
import json
import re
import unicodedata
from typing import Any

from .retention_qualification import (
    RetentionPair, RetentionPolicy, RetentionDecision, SignedRetentionEvidence,
    _campaign, decide_retention, retention_roster_sha256,
)


TAXONOMY_VERSION = "labyrinth.retention-family-taxonomy.v1"
FAMILY_ALIAS_VERSION = "labyrinth.family-nfkc-casefold-separator.v1"
PROMPT_ALIAS_VERSION = "labyrinth.prompt-nfkc-casefold-hyphen.v1"
_TAXONOMY_DOMAIN = b"labyrinth.retention-family-taxonomy.v1\0"
_POLICY_DOMAIN = b"labyrinth.family-taxonomy-policy.v1\0"
_FAMILY_INVENTORY_DOMAIN = b"labyrinth.family-taxonomy-inventory.v1\0"
_PROMPT_INVENTORY_DOMAIN = b"labyrinth.prompt-alias-inventory.v1\0"
_PROMPT_ALIAS_DOMAIN = b"labyrinth.prompt-nfkc-casefold-hyphen.v1\0"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_CANONICAL_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RECEIPT = 1_048_576


class FamilyTaxonomyError(ValueError):
    """A required independent, versioned taxonomy assertion is invalid."""


@dataclass(frozen=True)
class SignedFamilyTaxonomy:
    receipt_bytes: bytes
    signature_hex: str
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False


def _canonical(data: Any) -> bytes:
    return (json.dumps(data, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _reject_invisibles(value: str) -> None:
    # Format, private/unassigned, surrogate and modifier marks can hide
    # visually relevant distinctions. Reject rather than silently erase.
    for character in value:
        category = unicodedata.category(character)
        if category in {"Cf", "Cs", "Co", "Cn", "Mn", "Me", "Zl", "Zp"}:
            raise ValueError("invisible or unassigned Unicode character forbidden")
        if category == "Cc" and character not in "\t\n\r":
            raise ValueError("control character forbidden")


def _fold(value: str) -> str:
    _reject_invisibles(value)
    folded = unicodedata.normalize("NFKC", value.casefold())
    folded = unicodedata.normalize("NFKC", folded.casefold())
    _reject_invisibles(folded)
    return folded


def _hyphen_alias(character: str) -> bool:
    # Unicode hyphens (including U+2010) and minus sign, but not arbitrary
    # punctuation whose semantic use might differ.
    return unicodedata.category(character) == "Pd" or character == "\u2212"


def canonical_family_id(identifier: str) -> str:
    """One conservative, versioned ASCII ID for common visual family aliases."""
    if type(identifier) is not str or not 1 <= len(identifier) <= 256:
        raise ValueError("bounded structure-family identifier required")
    folded = _fold(identifier)
    chunks: list[str] = []
    for char in folded:
        if "a" <= char <= "z" or "0" <= char <= "9":
            chunks.append(char)
        elif char in "_.:" or char.isspace() or _hyphen_alias(char):
            chunks.append("-")
        else:
            raise ValueError("ambiguous or unsupported family punctuation/homoglyph")
    result = re.sub(r"-+", "-", "".join(chunks)).strip("-")
    if not _CANONICAL_ID.fullmatch(result) or len(result) > 128:
        raise ValueError("invalid canonical structure-family identifier")
    return result


def canonical_prompt_alias_sha256(prompt_text: str) -> str:
    """Additional surface-alias detector; NOT a semantic-equivalence proof."""
    if type(prompt_text) is not str or not 1 <= len(prompt_text) <= 8192:
        raise ValueError("bounded raw prompt required for alias commitment")
    folded = _fold(prompt_text)
    normalized = " ".join("".join(
        "-" if _hyphen_alias(character) else character for character in folded
    ).split())
    if not normalized or len(normalized.encode("utf-8")) > 8192:
        raise ValueError("empty or oversized canonical prompt alias")
    return sha256(_PROMPT_ALIAS_DOMAIN + normalized.encode("utf-8")).hexdigest()


def _sha(domain: bytes, data: Any) -> str:
    return sha256(domain + _canonical(data)).hexdigest()


def _taxonomy_body(policy: RetentionPolicy, roster: tuple[RetentionPair, ...],
                   signer_id: str, nonce_sha256: str) -> dict[str, Any]:
    _campaign(policy, roster)  # First retain ALL existing frozen development gates.
    if type(signer_id) is not str or not _ID.fullmatch(signer_id):
        raise ValueError("invalid external taxonomy signer identifier")
    if type(nonce_sha256) is not str or not _HEX.fullmatch(nonce_sha256):
        raise ValueError("invalid precommitted taxonomy nonce SHA-256")

    training_families = [canonical_family_id(value)
                         for value in policy.excluded_training_structure_family_ids]
    review_families = [canonical_family_id(pair.structure_family_id) for pair in roster]
    if len(set(training_families)) != len(training_families):
        raise FamilyTaxonomyError("duplicate canonical training family alias")
    if len(set(review_families)) != len(review_families):
        raise FamilyTaxonomyError("duplicate canonical review family alias")
    if set(training_families).intersection(review_families):
        raise FamilyTaxonomyError("training and retention structure-family alias overlap")

    training_prompts = [canonical_prompt_alias_sha256(item)
                        for item in policy.excluded_training_prompts]
    review_prompts = [canonical_prompt_alias_sha256(prompt)
                      for pair in roster for prompt in
                      (pair.before_prompt_text, pair.after_prompt_text)]
    if len(set(training_prompts)) != len(training_prompts):
        raise FamilyTaxonomyError("duplicate training prompt alias")
    if len(set(review_prompts)) != len(review_prompts):
        raise FamilyTaxonomyError("duplicate retention prompt alias")
    if set(training_prompts).intersection(review_prompts):
        raise FamilyTaxonomyError("training/retention prompt alias overlap")

    return {
        "schema": TAXONOMY_VERSION,
        "family_alias_version": FAMILY_ALIAS_VERSION,
        "prompt_alias_version": PROMPT_ALIAS_VERSION,
        "objective_id": policy.objective_id,
        "scope_sha256": policy.scope_sha256,
        "policy_sha256": _sha(_POLICY_DOMAIN, asdict(policy)),
        "roster_sha256": retention_roster_sha256(roster),
        "training_family_inventory_sha256": _sha(
            _FAMILY_INVENTORY_DOMAIN, sorted(training_families)),
        "retention_family_inventory_sha256": _sha(
            _FAMILY_INVENTORY_DOMAIN, sorted(review_families)),
        "training_prompt_alias_inventory_sha256": _sha(
            _PROMPT_INVENTORY_DOMAIN, sorted(training_prompts)),
        "retention_prompt_alias_inventory_sha256": _sha(
            _PROMPT_INVENTORY_DOMAIN, sorted(review_prompts)),
        "training_family_count": len(training_families),
        "retention_family_count": len(review_families),
        "training_prompt_count": len(training_prompts),
        "retention_prompt_count": len(review_prompts),
        "signer_id": signer_id,
        "precommitted_nonce_sha256": nonce_sha256,
        "graduation_authorized": False,
        "model_execution_authenticated": False,
    }


def _keys(taxonomy_key: bytes, evaluator_key: bytes) -> None:
    if (type(taxonomy_key) is not bytes or not 32 <= len(taxonomy_key) <= 4096
            or type(evaluator_key) is not bytes or not 32 <= len(evaluator_key) <= 4096):
        raise ValueError("independent bounded taxonomy/evaluator HMAC keys required")
    if hmac.compare_digest(taxonomy_key, evaluator_key):
        raise ValueError("taxonomy and evaluator HMAC keys must be distinct")


def sign_family_taxonomy(policy: RetentionPolicy, roster: tuple[RetentionPair, ...], *,
                         signer_id: str, precommitted_nonce_sha256: str,
                         independent_taxonomy_key: bytes,
                         independent_evaluator_key: bytes) -> SignedFamilyTaxonomy:
    """Custodian-side source receipt; no independent custodian is provisioned."""
    _keys(independent_taxonomy_key, independent_evaluator_key)
    body = _taxonomy_body(policy, roster, signer_id, precommitted_nonce_sha256)
    raw = _canonical(body)
    if len(raw) > _MAX_RECEIPT:
        raise FamilyTaxonomyError("oversized taxonomy receipt")
    return SignedFamilyTaxonomy(raw, hmac.new(
        independent_taxonomy_key, _TAXONOMY_DOMAIN + raw, sha256).hexdigest())


def verify_family_taxonomy(signed: SignedFamilyTaxonomy,
                           policy: RetentionPolicy, roster: tuple[RetentionPair, ...], *,
                           expected_signer_id: str, expected_nonce_sha256: str,
                           independent_taxonomy_key: bytes,
                           independent_evaluator_key: bytes) -> dict[str, Any]:
    """Fail closed when signer, nonce, key, policy, or alias inventory changes."""
    _keys(independent_taxonomy_key, independent_evaluator_key)
    if (type(signed) is not SignedFamilyTaxonomy
            or signed.graduation_authorized is not False
            or signed.model_execution_authenticated is not False
            or type(signed.receipt_bytes) is not bytes
            or not 0 < len(signed.receipt_bytes) <= _MAX_RECEIPT
            or type(signed.signature_hex) is not str
            or not _HEX.fullmatch(signed.signature_hex)):
        raise FamilyTaxonomyError("invalid taxonomy receipt or claimed authority")
    calculated = hmac.new(independent_taxonomy_key,
                          _TAXONOMY_DOMAIN + signed.receipt_bytes, sha256).hexdigest()
    if not hmac.compare_digest(calculated, signed.signature_hex):
        raise FamilyTaxonomyError("taxonomy HMAC signature not authentic")
    expected_bytes = _canonical(_taxonomy_body(
        policy, roster, expected_signer_id, expected_nonce_sha256))
    if not hmac.compare_digest(expected_bytes, signed.receipt_bytes):
        raise FamilyTaxonomyError("taxonomy policy, signer, nonce or inventory mismatch")
    return json.loads(signed.receipt_bytes)


def decide_retention_with_family_custody(
    policy: RetentionPolicy, roster: tuple[RetentionPair, ...],
    evidence: SignedRetentionEvidence, signed_taxonomy: SignedFamilyTaxonomy, *,
    expected_signer_id: str, expected_nonce_sha256: str,
    independent_taxonomy_key: bytes, independent_evaluator_key: bytes,
) -> RetentionDecision:
    """Require distinct taxonomic custody before legacy development-only veto."""
    verify_family_taxonomy(
        signed_taxonomy, policy, roster,
        expected_signer_id=expected_signer_id,
        expected_nonce_sha256=expected_nonce_sha256,
        independent_taxonomy_key=independent_taxonomy_key,
        independent_evaluator_key=independent_evaluator_key)
    return decide_retention(policy, roster, evidence,
                            independent_evaluator_key=independent_evaluator_key)
