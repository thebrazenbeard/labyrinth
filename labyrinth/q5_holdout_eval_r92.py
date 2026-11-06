"""Offline, one-shot examiner-side Q5 scoring of *raw* five-arm predictions.

The private bank, HMAC key, evaluator signing key, immutable precommitment,
external signer pins, and SQLite ledger belong to an independent examiner.
The trainer gets only prompts and eventually a signed aggregate report; it
never receives gold or per-case correctness. Signed *source claims* cannot
prove when a host ran, whether weights really changed, or key independence.
This module cannot qualify Q7/Q8 or authorize graduation/deployment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
from hashlib import sha256
from hmac import compare_digest, new as hmac_new
import json
import math
import re
import sqlite3
import unicodedata
from contextlib import closing
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

from .five_arm_experiment_contract import (
    ArmClaim, FrozenFiveArmPlan, PriorCaseObservation, audit_five_arm_claims,
)
from .five_arm_statistical_screen import (
    CONTROLS, PairedCase, SeedOutcome, _case_digest, _exact_sign_tail,
    screen_five_arm_seed_outcomes,
)
from .independent_holdout import _custodian_db_file
from .maze_conditioned_minilm_cpu_experiment import PREREG_ARMS
from .minilm_cpu_experiment import (
    TRAIN as PUBLIC_TRAIN, HELDOUT as PUBLIC_HELDOUT,
    TRANSFER as PUBLIC_TRANSFER,
)


SCHEMA = "labyrinth.q5-private-raw-holdout.r92.v1"
REPORT_SCHEMA = "labyrinth.q5-private-aggregate-signed.r92.v1"
PRECOMMIT_SCHEMA = "labyrinth.q5-exam-precommit.r92.v1"
RETENTION_SCHEMA = "labyrinth.q5-prior-retention-batch.r92.v1"
COMPLETION_SCHEMA = "labyrinth.q5-train-completion.r92.v1"
PREDICTION_SCHEMA = "labyrinth.q5-inference-batch.r92.v1"
SPLITS = ("development", "transfer", "retention")
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_HEX = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_SIGNATURE = re.compile(r"[0-9a-f]{128}\Z", re.ASCII)
_DOMAINS = {
    PRECOMMIT_SCHEMA: b"labyrinth.q5.r92.precommit\0",
    RETENTION_SCHEMA: b"labyrinth.q5.r92.prior-retention\0",
    COMPLETION_SCHEMA: b"labyrinth.q5.r92.training-complete\0",
    PREDICTION_SCHEMA: b"labyrinth.q5.r92.predictions\0",
    REPORT_SCHEMA: b"labyrinth.q5.r92.aggregate-report\0",
}
_PUBLIC_PROMPTS = frozenset(" ".join(p.casefold().split()) for p, _ in
                             tuple(row for cycle in PUBLIC_TRAIN for row in cycle)
                             + PUBLIC_HELDOUT + PUBLIC_TRANSFER)
# These non-Cf default-ignorable code points/ranges can make a copied
# public prompt appear to have novel bytes without visibly new content.
# Do not erase them and risk claiming two genuinely different texts equal.
_DEFAULT_IGNORABLE_RANGES = (
    (0x034F, 0x034F), (0x115F, 0x1160), (0x17B4, 0x17B5),
    (0x180B, 0x180F), (0x2060, 0x206F), (0x3164, 0x3164),
    (0xFE00, 0xFE0F), (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8),
    (0xE0000, 0xE0FFF),
)


class Q5HoldoutError(ValueError):
    """Reject contaminated, uncommitted or unverifiable source evidence."""


@dataclass(frozen=True)
class PrivateCase:
    run_id: str
    case_id: str
    split: str
    prompt: str
    gold: int  # examiner-only; NEVER included in the public report
    structure_family: str


@dataclass(frozen=True)
class SignedBytes:
    public_bytes: bytes
    signature_hex: str


@dataclass(frozen=True)
class RawArmPredictions:
    run_id: str
    arm: str
    process_id: str
    final_trainable_sha256: str
    responses: tuple[tuple[str, str], ...]
    signed_inference: SignedBytes


@dataclass(frozen=True)
class PrivateSeed:
    plan: FrozenFiveArmPlan
    arm_claims: tuple[ArmClaim, ...]
    cases: tuple[PrivateCase, ...]
    predictions: tuple[RawArmPredictions, ...]
    # The predecessor's raw, prior-skill inference; never correctness bits.
    retention_before: tuple[tuple[str, str], ...]
    # Independently signed and pinned BEFORE Q5 training, not a self-assertion.
    signed_retention: SignedBytes | None = None


@dataclass(frozen=True)
class Q5TrustPins:
    examiner_public_key: bytes
    training_completion_public_key: bytes
    inference_public_key: bytes
    expected_exam_sha256: str
    expected_training_completion_sha256: str
    expected_evaluator_key_sha256: str
    expected_inference_sha256s: tuple[tuple[str, str, str], ...]
    # Independently held prior exposure roster, not derived from submissions.
    excluded_case_ids: frozenset[str]
    excluded_prompt_sha256s: frozenset[str]
    retention_public_key: bytes
    expected_retention_sha256s: tuple[tuple[str, str], ...]


def _id(x: object) -> bool:
    return type(x) is str and _IDENT.fullmatch(x) is not None


def _hex(x: object) -> bool:
    return type(x) is str and _HEX.fullmatch(x) is not None


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise Q5HoldoutError(message)


def _canonical(body: object) -> bytes:
    try:
        return (json.dumps(body, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    except (ValueError, TypeError, UnicodeError, OverflowError) as exc:
        raise Q5HoldoutError("noncanonical JSON data") from exc


def _parse_exact(data: bytes, schema: str, fields: frozenset[str]) -> dict:
    _require(type(data) is bytes and 0 < len(data) <= 262144, "bounded signed bytes required")

    def unique(pairs: list[tuple[str, object]]) -> dict:
        out = {}
        for k, v in pairs:
            _require(k not in out, "duplicate signed JSON field")
            out[k] = v
        return out

    def invalid_constant(_: str) -> None:
        raise Q5HoldoutError("nonfinite signed number")

    try:
        body = json.loads(data, object_pairs_hook=unique, parse_constant=invalid_constant)
    except (ValueError, UnicodeError, TypeError) as exc:
        raise Q5HoldoutError("invalid signed JSON") from exc
    _require(type(body) is dict and body.keys() == fields and body.get("schema") == schema,
             "signed schema changed")
    _require(_canonical(body) == data, "noncanonical signed JSON")
    return body


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _normalize(prompt: str) -> str:
    _require(type(prompt) is str and 1 <= len(prompt) <= 8192, "invalid private prompt")
    _require(not any(
        unicodedata.category(ch) in ("Cf", "Cn", "Cs") or
        any(low <= ord(ch) <= high for low, high in _DEFAULT_IGNORABLE_RANGES)
        for ch in prompt), "private prompt contains default-ignorable Unicode")
    normalized = " ".join(prompt.casefold().split())
    _require(bool(normalized), "empty normalized private prompt")
    return normalized


def _prompt_hash(prompt: str) -> str:
    return _digest(_normalize(prompt).encode("utf-8"))


def _public_key(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)


def sign_source_bytes(public_bytes: bytes, private_key: Ed25519PrivateKey,
                      schema: str) -> SignedBytes:
    """Examiner/test helper; never manufacture trusted pins from this result."""
    _require(schema in _DOMAINS and isinstance(private_key, Ed25519PrivateKey),
             "exact Ed25519 signer and source domain required")
    _require(type(public_bytes) is bytes and bool(public_bytes), "signed source bytes required")
    return SignedBytes(public_bytes, private_key.sign(_DOMAINS[schema] + public_bytes).hex())


def _verify_signed(envelope: SignedBytes, key: bytes, schema: str,
                   expected_sha: str | None, fields: frozenset[str]) -> dict:
    _require(type(envelope) is SignedBytes and type(key) is bytes and len(key) == 32
             and type(envelope.signature_hex) is str
             and _SIGNATURE.fullmatch(envelope.signature_hex) is not None,
             "malformed signature or external key pin")
    if expected_sha is not None:
        _require(_hex(expected_sha) and compare_digest(_digest(envelope.public_bytes), expected_sha),
                 "external signed-byte digest pin mismatch")
    try:
        Ed25519PublicKey.from_public_bytes(key).verify(
            bytes.fromhex(envelope.signature_hex), _DOMAINS[schema] + envelope.public_bytes)
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise Q5HoldoutError("invalid independently pinned signature") from exc
    return _parse_exact(envelope.public_bytes, schema, fields)


def _bank(seeds: tuple[PrivateSeed, ...]) -> tuple[PrivateCase, ...]:
    _require(type(seeds) is tuple and 8 <= len(seeds) <= 64 and
             all(type(s) is PrivateSeed for s in seeds), "8..64 exact private seed outcomes required")
    cases: list[PrivateCase] = []
    ids: set[str] = set()
    prompts: set[str] = set()
    # A seed's private exam must also avoid EVERY OTHER seed's TRAIN inventory.
    # Only checking its own plan lets a relabeled cross-seed training prompt
    # masquerade as an independent holdout.
    all_training_ids: set[str] = set()
    all_training_prompts: set[str] = set()
    for seed in seeds:
        _require(type(seed.plan) is FrozenFiveArmPlan, "frozen source plan required")
        seed.plan.validate()
        all_training_ids.update(seed.plan.train_visits)
        all_training_prompts.update(seed.plan.train_prompt_hashes)
    for seed in seeds:
        _require(type(seed.cases) is tuple and all(type(c) is PrivateCase for c in seed.cases),
                 "typed private cases required")
        tally = {split: 0 for split in SPLITS}
        for c in seed.cases:
            _require(type(c.run_id) is str and c.run_id == seed.plan.run_id and
                     _id(c.case_id) and type(c.split) is str and c.split in SPLITS and
                     type(c.gold) is int and c.gold in (0, 1) and
                     _id(c.structure_family), "malformed private case")
            normalized = _normalize(c.prompt)
            digest = _digest(normalized.encode("utf-8"))
            _require(c.case_id not in ids and digest not in prompts,
                     "duplicate case or prompt across all seeds/splits")
            _require(normalized not in _PUBLIC_PROMPTS,
                     "repeated public development/transfer fixture is not private")
            _require(c.case_id not in all_training_ids and
                     digest not in all_training_prompts,
                     "private case overlaps TRAIN identity")
            if c.split == "transfer":
                _require(c.structure_family not in seed.plan.generated_family_ids,
                         "transfer reuses TRAIN structural family")
            ids.add(c.case_id)
            prompts.add(digest)
            tally[c.split] += 1
            cases.append(c)
        _require(20 <= tally["development"] <= 512 and
                 20 <= tally["transfer"] <= 512 and
                 tally["retention"] == len(seed.plan.retention_case_ids) and
                 20 <= tally["retention"] <= 512 and
                 set(c.case_id for c in seed.cases if c.split == "retention")
                 == set(seed.plan.retention_case_ids),
                 "incomplete precommitted development/transfer/retention bank")
    return tuple(cases)


def bank_public_commitment(seeds: tuple[PrivateSeed, ...]) -> str:
    """Opaque bank identity; no gold or individual public answer hash."""
    return _digest(_canonical([(c.run_id, c.case_id, c.split, _prompt_hash(c.prompt),
                                c.structure_family) for c in _bank(seeds)]))


def bank_private_gold_seal(seeds: tuple[PrivateSeed, ...], examiner_hmac_key: bytes) -> str:
    _require(type(examiner_hmac_key) is bytes and len(examiner_hmac_key) >= 32,
             "independent 32-byte private gold key required")
    payload = _canonical([(c.run_id, c.case_id, _prompt_hash(c.prompt), c.gold)
                          for c in _bank(seeds)])
    return hmac_new(examiner_hmac_key, b"labyrinth.q5.r92.private-gold\0" + payload,
                    sha256).hexdigest()


def _retention_cases(seed: PrivateSeed) -> tuple[PrivateCase, ...]:
    retention = tuple(c for c in seed.cases if c.split == "retention")
    _require(type(seed.retention_before) is tuple and
             all(type(item) is tuple and len(item) == 2 for item in seed.retention_before) and
             tuple(cid for cid, _ in seed.retention_before) ==
             tuple(c.case_id for c in retention),
             "raw predecessor retention predictions absent or reordered")
    prediction_commitment(seed.retention_before)
    return retention


def retention_bytes(*, run_id: str, plan_sha256: str, predictions_sha256: str,
                    retention_sequence: int) -> bytes:
    """Separate predecessor-signer input; sequence is NOT a host time proof."""
    _require(_id(run_id) and _hex(plan_sha256) and _hex(predictions_sha256) and
             type(retention_sequence) is int and 0 <= retention_sequence < 2**63 - 1,
             "invalid predecessor retention source identity")
    return _canonical({
        "schema": RETENTION_SCHEMA, "phase": "PRIOR_SKILL_BEFORE_Q5_TRAINING",
        "run_id": run_id, "plan_sha256": plan_sha256,
        "predictions_sha256": predictions_sha256,
        "retention_sequence": retention_sequence,
    })


def _retention_precommitments(seeds: tuple[PrivateSeed, ...]) -> list[list[str]]:
    commitments = []
    for seed in seeds:
        _retention_cases(seed)
        _require(type(seed.signed_retention) is SignedBytes and
                 type(seed.signed_retention.public_bytes) is bytes and
                 type(seed.signed_retention.signature_hex) is str,
                 "signed predecessor retention receipt required")
        commitments.append([seed.plan.run_id,
                            prediction_commitment(seed.retention_before),
                            _digest(seed.signed_retention.public_bytes)])
    return commitments


def precommit_bytes(seeds: tuple[PrivateSeed, ...], *, exam_id: str,
                    examiner_hmac_key: bytes, pretraining_sequence: int) -> bytes:
    """Prepare for independent signing BEFORE training; source has no clock."""
    _require(_id(exam_id) and type(pretraining_sequence) is int and
             0 <= pretraining_sequence <= 2**63 - 2, "invalid pretraining identity/sequence")
    return _canonical({
        "schema": PRECOMMIT_SCHEMA, "phase": "BEFORE_TRAINING", "exam_id": exam_id,
        "pretraining_sequence": pretraining_sequence,
        "bank_public_sha256": bank_public_commitment(seeds),
        "bank_private_hmac": bank_private_gold_seal(seeds, examiner_hmac_key),
        "plan_sha256s": [[s.plan.run_id, s.plan.source_contract_sha256] for s in seeds],
        "retention_precommitments": _retention_precommitments(seeds),
        "decision_rule": "8_seed_sign_tests_holm_familywise_0.05_min_gain_0.10",
        "maximum_attempts": 1,
    })


def _claims_digest(seeds: tuple[PrivateSeed, ...]) -> str:
    return _digest(_canonical([[s.plan.run_id, [asdict(c) for c in s.arm_claims]]
                               for s in seeds]))


def training_completion_bytes(*, exam_id: str, exam_sha256: str,
                              claims_sha256: str, training_end_sequence: int) -> bytes:
    _require(_id(exam_id) and _hex(exam_sha256) and _hex(claims_sha256) and
             type(training_end_sequence) is int and 0 <= training_end_sequence <= 2**63 - 2,
             "invalid training completion identity")
    return _canonical({"schema": COMPLETION_SCHEMA, "phase": "TRAINING_COMPLETED",
                       "exam_id": exam_id, "exam_sha256": exam_sha256,
                       "claims_sha256": claims_sha256,
                       "training_end_sequence": training_end_sequence})


def prediction_commitment(responses: tuple[tuple[str, str], ...]) -> str:
    _require(type(responses) is tuple and 1 <= len(responses) <= 1536,
             "raw response tuple required")
    for item in responses:
        _require(type(item) is tuple and len(item) == 2 and _id(item[0]) and
                 type(item[1]) is str and len(item[1]) <= 8192,
                 "malformed raw prediction")
    return _digest(b"labyrinth.q5.r92.raw-predictions\0" + _canonical(responses))


def inference_bytes(*, exam_sha256: str, completion_sha256: str, run_id: str,
                    arm: str, process_id: str, final_trainable_sha256: str,
                    predictions_sha256: str, inference_sequence: int) -> bytes:
    _require(all(_hex(x) for x in (exam_sha256, completion_sha256,
                                   final_trainable_sha256, predictions_sha256)) and
             _id(run_id) and _id(process_id) and type(arm) is str and arm in PREREG_ARMS and
             type(inference_sequence) is int and 0 <= inference_sequence <= 2**63 - 1,
             "invalid inference metadata")
    return _canonical({"schema": PREDICTION_SCHEMA, "phase": "POST_TRAINING_INFERENCE",
                       "exam_sha256": exam_sha256, "completion_sha256": completion_sha256,
                       "run_id": run_id, "arm": arm, "process_id": process_id,
                       "final_trainable_sha256": final_trainable_sha256,
                       "predictions_sha256": predictions_sha256,
                       "inference_sequence": inference_sequence})


def _validate_pins(pins: Q5TrustPins, evaluator_private_key: Ed25519PrivateKey) -> tuple[dict, dict]:
    _require(type(pins) is Q5TrustPins and isinstance(evaluator_private_key, Ed25519PrivateKey),
             "exact examiner pins and signing key required")
    keys = (pins.examiner_public_key, pins.training_completion_public_key,
            pins.inference_public_key, pins.retention_public_key)
    _require(all(type(k) is bytes and len(k) == 32 for k in keys) and
             len(set(keys)) == 4, "distinct externally pinned signer roles required")
    _require(_hex(pins.expected_exam_sha256) and _hex(pins.expected_training_completion_sha256)
             and _hex(pins.expected_evaluator_key_sha256) and
             compare_digest(_digest(keys[0]), pins.expected_evaluator_key_sha256) and
             compare_digest(_public_key(evaluator_private_key), keys[0]),
             "evaluator private key differs from independent public pin")
    _require(type(pins.excluded_case_ids) is frozenset and
             all(_id(x) for x in pins.excluded_case_ids) and
             type(pins.excluded_prompt_sha256s) is frozenset and
             all(_hex(x) for x in pins.excluded_prompt_sha256s),
             "independently excluded training/previous case inventory required")
    entries = pins.expected_inference_sha256s
    _require(type(entries) is tuple and len(entries) >= 40 and
             all(type(e) is tuple and len(e) == 3 and _id(e[0]) and
                 type(e[1]) is str and e[1] in PREREG_ARMS and _hex(e[2]) for e in entries),
             "externally pinned per-arm inference receipts required")
    mapping = {(a, b): c for a, b, c in entries}
    _require(len(mapping) == len(entries), "duplicate independent prediction pin")
    retentions = pins.expected_retention_sha256s
    _require(type(retentions) is tuple and 8 <= len(retentions) <= 64 and
             all(type(row) is tuple and len(row) == 2 and _id(row[0]) and _hex(row[1])
                 for row in retentions), "externally pinned predecessor retention receipts required")
    retention_mapping = dict(retentions)
    _require(len(retention_mapping) == len(retentions),
             "duplicate independent predecessor retention pin")
    return mapping, retention_mapping


def _verify_retention_sources(seeds: tuple[PrivateSeed, ...], pretraining_sequence: int,
                              pins: Q5TrustPins, retention_pins: dict) -> None:
    """Verify every predecessor source before scoring even the first private case."""
    _require(set(retention_pins) == {s.plan.run_id for s in seeds},
             "independent predecessor retention roster changed")
    for seed in seeds:
        _retention_cases(seed)
        expected = retention_pins[seed.plan.run_id]
        body = _verify_signed(seed.signed_retention, pins.retention_public_key,
                              RETENTION_SCHEMA, expected,
                              frozenset({"schema", "phase", "run_id", "plan_sha256",
                                         "predictions_sha256", "retention_sequence"}))
        _require(type(body["retention_sequence"]) is int and
                 0 <= body["retention_sequence"] < pretraining_sequence and
                 body == json.loads(retention_bytes(
                     run_id=seed.plan.run_id, plan_sha256=seed.plan.source_contract_sha256,
                     predictions_sha256=prediction_commitment(seed.retention_before),
                     retention_sequence=body["retention_sequence"])),
                 "signed predecessor retention receipt does not bind prior raw predictions")


def _score_seeds(seeds: tuple[PrivateSeed, ...], exam_sha: str, completion_sha: str,
                 completion_sequence: int, pins: Q5TrustPins,
                 prediction_pins: dict) -> tuple[SeedOutcome, ...]:
    outcome: list[SeedOutcome] = []
    seen_processes: set[str] = set()
    for seed in seeds:
        try:
            audit_five_arm_claims(seed.plan, seed.arm_claims)
        except ValueError as exc:
            raise Q5HoldoutError("unmatched source TRAIN budget/label/checkpoint claims") from exc
        _require(type(seed.predictions) is tuple and len(seed.predictions) == len(PREREG_ARMS)
                 and all(type(p) is RawArmPredictions for p in seed.predictions) and
                 tuple(p.arm for p in seed.predictions) == PREREG_ARMS,
                 "five ordered raw prediction arms required")
        reference = {c.case_id: c for c in seed.cases}
        ordered_ids = tuple(c.case_id for c in seed.cases)
        scores: dict[str, dict[str, bool]] = {}
        for prediction, claim in zip(seed.predictions, seed.arm_claims):
            _require(type(prediction.run_id) is str and
                     prediction.run_id == seed.plan.run_id and
                     type(prediction.arm) is str and prediction.arm == claim.arm and
                     _id(prediction.process_id) and
                     prediction.process_id not in seen_processes and
                     prediction.final_trainable_sha256 == claim.final_trainable_sha256,
                     "prediction process/model binding absent or duplicated")
            seen_processes.add(prediction.process_id)
            _require(tuple(cid for cid, _ in prediction.responses) == ordered_ids,
                     "missing/reordered/duplicate/foreign raw predictions")
            digest = prediction_commitment(prediction.responses)
            envelope = prediction.signed_inference
            expected_digest = prediction_pins.get((prediction.run_id, prediction.arm))
            _require(expected_digest is not None, "missing independent inference pin")
            body = _verify_signed(envelope, pins.inference_public_key, PREDICTION_SCHEMA,
                                  expected_digest, frozenset({
                                      "schema", "phase", "exam_sha256", "completion_sha256",
                                      "run_id", "arm", "process_id", "final_trainable_sha256",
                                      "predictions_sha256", "inference_sequence"}))
            _require(body == json.loads(inference_bytes(
                exam_sha256=exam_sha, completion_sha256=completion_sha,
                run_id=prediction.run_id, arm=prediction.arm,
                process_id=prediction.process_id,
                final_trainable_sha256=prediction.final_trainable_sha256,
                predictions_sha256=digest, inference_sequence=completion_sequence + 1)),
                "signed inference does not bind post-training raw outputs")
            scores[prediction.arm] = {cid: (raw == str(reference[cid].gold))
                                      for cid, raw in prediction.responses}
        rows: dict[str, tuple[PairedCase, ...]] = {}
        for split in ("development", "transfer"):
            rows[split] = tuple(PairedCase(
                _digest(b"labyrinth.q5.r92.case\0" +
                        _canonical([c.case_id, _prompt_hash(c.prompt)])),
                tuple((arm, scores[arm][c.case_id]) for arm in PREREG_ARMS))
                for c in seed.cases if c.split == split)
        retention = _retention_cases(seed)
        prior = tuple(PriorCaseObservation(
            c.case_id, raw == str(c.gold), scores["correct"][c.case_id])
            for c, (_, raw) in zip(retention, seed.retention_before))
        outcome.append(SeedOutcome(seed.plan, _case_digest(rows["development"]),
                                   _case_digest(rows["transfer"]),
                                   rows["development"], rows["transfer"], prior))
    _require(set(prediction_pins) == {(s.plan.run_id, a)
                                      for s in seeds for a in PREREG_ARMS},
             "independent inference roster changed")
    return tuple(outcome)


def _reserve_once(db_path: str | Path, exam_id: str, cases: tuple[PrivateCase, ...]) -> None:
    """Consume the exam before scoring; failed scoring never permits a retry."""
    try:
        target = _custodian_db_file(db_path)
        with closing(sqlite3.connect(target, timeout=10)) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("CREATE TABLE IF NOT EXISTS q5_exams (exam_id TEXT PRIMARY KEY)")
                connection.execute("CREATE TABLE IF NOT EXISTS q5_case_registry ("
                                   "identity_kind TEXT NOT NULL, digest TEXT NOT NULL, "
                                   "exam_id TEXT NOT NULL, PRIMARY KEY(identity_kind,digest))")
                connection.execute("INSERT INTO q5_exams(exam_id) VALUES(?)", (exam_id,))
                for c in cases:
                    connection.execute("INSERT INTO q5_case_registry VALUES(?,?,?)",
                                       ("id", _digest(c.case_id.encode("utf-8")), exam_id))
                    connection.execute("INSERT INTO q5_case_registry VALUES(?,?,?)",
                                       ("prompt", _prompt_hash(c.prompt), exam_id))
    except (ValueError, sqlite3.IntegrityError, sqlite3.OperationalError) as exc:
        raise Q5HoldoutError("exam or case/prompt already consumed, or secure ledger unavailable") from exc


def evaluate_private_q5(*, seeds: tuple[PrivateSeed, ...], exam: SignedBytes,
                        completion: SignedBytes, pins: Q5TrustPins,
                        examiner_hmac_key: bytes, evaluator_private_key: Ed25519PrivateKey,
                        custodian_db_path: str | Path) -> SignedBytes:
    """Return signed aggregate only, with no trainer-visible labels/case rows.

    All three signer pins and the expected signed-message digests must be
    independently configured. Phases/sequences validate *claims*, NOT host
    chronology. A protected host must attest pretraining custody and true
    inference separately. SQLite must live outside trainer write authority.
    """
    # Snapshot caller-held source material and externally configured pins at
    # the examiner boundary. `frozen=True` alone is insufficient:
    # object.__setattr__ can change a signing-role pin after its distinct-key
    # check, or a PrivateCase.gold value after the signed bank HMAC check but
    # before scoring. All subsequent reads use these local snapshots.
    _require(type(pins) is Q5TrustPins, "exact examiner pins required")
    pins = replace(pins)
    _require(type(seeds) is tuple and 8 <= len(seeds) <= 64 and
             all(type(s) is PrivateSeed for s in seeds),
             "8..64 exact private seed outcomes required")
    seeds = deepcopy(seeds)
    prediction_pins, retention_pins = _validate_pins(pins, evaluator_private_key)
    cases = _bank(seeds)
    _require(not {c.case_id for c in cases} & pins.excluded_case_ids and
             not {_prompt_hash(c.prompt) for c in cases} & pins.excluded_prompt_sha256s,
             "evaluation reused externally excluded TRAIN or prior public fixtures")
    exam_body = _verify_signed(exam, pins.examiner_public_key, PRECOMMIT_SCHEMA,
                               pins.expected_exam_sha256, frozenset({
                                   "schema", "phase", "exam_id", "pretraining_sequence",
                                   "bank_public_sha256", "bank_private_hmac", "plan_sha256s",
                                   "retention_precommitments",
                                   "decision_rule", "maximum_attempts"}))
    _require(_id(exam_body["exam_id"]) and type(exam_body["pretraining_sequence"]) is int and
             0 <= exam_body["pretraining_sequence"] < 2**63 - 1 and
             exam_body == json.loads(precommit_bytes(
                 seeds, exam_id=exam_body["exam_id"], examiner_hmac_key=examiner_hmac_key,
                 pretraining_sequence=exam_body["pretraining_sequence"])),
             "private bank/labels/plan differ from evaluator pretraining precommit")
    completion_body = _verify_signed(completion, pins.training_completion_public_key,
                                     COMPLETION_SCHEMA,
                                     pins.expected_training_completion_sha256,
                                     frozenset({"schema", "phase", "exam_id", "exam_sha256",
                                                "claims_sha256", "training_end_sequence"}))
    _require(type(completion_body["training_end_sequence"]) is int and
             exam_body["pretraining_sequence"] < completion_body["training_end_sequence"] < 2**63 - 1,
             "training completion must follow exam precommit before inference")
    _require(completion_body == json.loads(training_completion_bytes(
        exam_id=exam_body["exam_id"], exam_sha256=pins.expected_exam_sha256,
        claims_sha256=_claims_digest(seeds),
        training_end_sequence=completion_body["training_end_sequence"])),
        "signed training completion does not bind exact five-arm claims")
    # Burn the entire private exam BEFORE comparing any raw answer to gold.
    # Even an invalid prediction batch must not enable an adaptive retry.
    _reserve_once(custodian_db_path, exam_body["exam_id"], cases)
    # Independent predecessor signatures are mandatory before any scoring.
    _verify_retention_sources(seeds, exam_body["pretraining_sequence"],
                              pins, retention_pins)
    # Verify raw-output signatures and complete source budgets before reporting.
    outcomes = _score_seeds(seeds, pins.expected_exam_sha256,
                            pins.expected_training_completion_sha256,
                            completion_body["training_end_sequence"], pins,
                            prediction_pins)
    try:
        screen = screen_five_arm_seed_outcomes(outcomes)
    except ValueError as exc:
        raise Q5HoldoutError("frozen eight-comparison rule rejected private outcomes") from exc
    report = {
        "schema": REPORT_SCHEMA,
        "status": "SIGNED_AGGREGATE_SOURCE_SCORING_HOST_UNATTESTED",
        "exam_sha256": pins.expected_exam_sha256,
        "training_completion_sha256": pins.expected_training_completion_sha256,
        "evaluator_public_key_sha256": pins.expected_evaluator_key_sha256,
        "seed_count": screen["seed_count"],
        "case_count_by_split": {split: sum(c.split == split for c in cases) for split in SPLITS},
        "decision_rule": screen["method"],
        "source_screen_pass": screen["source_screen_pass"],
        "comparisons": screen["comparisons"],
        "host_training_and_inference_authenticated": False,
        "pretraining_temporal_custody_authenticated": False,
        "semantic_transfer_independently_established": False,
        "q7_final_exam_performed": False,
        "graduation_authorized": False,
    }
    return sign_source_bytes(_canonical(report), evaluator_private_key, REPORT_SCHEMA)


def verify_public_aggregate(report: SignedBytes, *, pinned_evaluator_public_key: bytes,
                            expected_report_sha256: str,
                            expected_exam_sha256: str) -> dict:
    """Validate a public report's exact signature/schema, never infer physical PASS."""
    fields = frozenset({"schema", "status", "exam_sha256", "training_completion_sha256",
                        "evaluator_public_key_sha256", "seed_count", "case_count_by_split",
                        "decision_rule", "source_screen_pass", "comparisons",
                        "host_training_and_inference_authenticated",
                        "pretraining_temporal_custody_authenticated",
                        "semantic_transfer_independently_established",
                        "q7_final_exam_performed", "graduation_authorized"})
    body = _verify_signed(report, pinned_evaluator_public_key, REPORT_SCHEMA,
                          expected_report_sha256, fields)
    _require(_hex(expected_exam_sha256) and body["exam_sha256"] == expected_exam_sha256
             and _hex(body["training_completion_sha256"])
             and body["evaluator_public_key_sha256"] == _digest(pinned_evaluator_public_key)
             and body["status"] == "SIGNED_AGGREGATE_SOURCE_SCORING_HOST_UNATTESTED"
             and type(body["source_screen_pass"]) is bool
             and type(body["seed_count"]) is int and 8 <= body["seed_count"] <= 64
             and type(body["case_count_by_split"]) is dict and
             set(body["case_count_by_split"]) == set(SPLITS)
             and all(type(n) is int and n >= 0 for n in body["case_count_by_split"].values())
             and body["decision_rule"] == "one_sided_seed_level_exact_sign_holm_8_comparisons"
             and type(body["comparisons"]) is list and len(body["comparisons"]) == 8
             and all(body[field] is False for field in (
                 "host_training_and_inference_authenticated",
                 "pretraining_temporal_custody_authenticated",
                 "semantic_transfer_independently_established",
                 "q7_final_exam_performed", "graduation_authorized")),
             "forged or malformed public report fields")
    for split in ("development", "transfer", "retention"):
        count = body["case_count_by_split"][split]
        _require(20 * body["seed_count"] <= count <= 512 * body["seed_count"],
                 "aggregate case coverage malformed")
    expected_pairs = [(split, control) for split in ("development", "transfer")
                      for control in CONTROLS]
    _require(all(type(row) is dict and row.keys() == {
        "split", "control", "seed_wins", "seed_losses", "seed_ties",
        "mean_gain", "unadjusted_p", "holm_pass"}
        for row in body["comparisons"]), "unexpected metric/private-label fields")
    for row, pair in zip(body["comparisons"], expected_pairs):
        wins, losses, ties = (row[k] for k in ("seed_wins", "seed_losses", "seed_ties"))
        gain, p = row["mean_gain"], row["unadjusted_p"]
        _require((row["split"], row["control"]) == pair
                 and all(type(x) is int and 0 <= x <= body["seed_count"]
                         for x in (wins, losses, ties))
                 and wins + losses + ties == body["seed_count"]
                 and type(gain) is float and math.isfinite(gain) and -1 <= gain <= 1
                 and type(p) is float and math.isfinite(p) and 0 <= p <= 1
                 and abs(p - _exact_sign_tail(wins, losses)) < 1e-12
                 and type(row["holm_pass"]) is bool,
                 "reported statistical metrics contradict prespecified rule")
    order = sorted(range(8), key=lambda i: body["comparisons"][i]["unadjusted_p"])
    still_eligible = True
    for rank, index in enumerate(order):
        row = body["comparisons"][index]
        expected_pass = (still_eligible and
                         row["unadjusted_p"] <= 0.05 / (8 - rank) and
                         row["mean_gain"] >= 0.10)
        _require(row["holm_pass"] is expected_pass,
                 "forged Holm comparison decision")
        still_eligible = expected_pass
    _require(body["source_screen_pass"] is all(row["holm_pass"]
                                               for row in body["comparisons"]),
             "forged aggregate decision")
    return body
