"""Opt-in operator-only synthetic cognitive TRAIN data export (source-only).

This module intentionally materializes *labeled TRAIN* examples for an
operator-controlled supervised trainer. Unlike the answerless trainee views,
these bytes contain the synthetic choice key: never send them to a learner
under evaluation or to a protected examination pipeline. Canonical SHA-256
digests are local provenance checks, NOT host signatures, independent
attestations, checkpoint updates, or graduation authority.

No disk I/O, network access, ML weights, or GPU actions occur here.
"""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

from .approved_cognitive_empathy_train_r90 import PROVIDER_ID, RUBRIC_ID
from .cognitive_objective_training_run import CognitiveObjectiveTrainingRun
from .journey import JourneyState
from .objective_contract import ObjectiveContract


SCHEMA_ID = "labyrinth.cognitive-supervised-train-example.v1"
MAX_DATASET_BYTES = 262_144


class CognitiveTrainDataset(NamedTuple):
    """Immutable primitive export; not trusted as a signed evidence receipt."""

    jsonl_bytes: bytes
    sha256: str
    case_count: int
    objective_id: str
    contract_fingerprint: str
    cycle_index: int
    provider_id: str

    @property
    def authorizes_host_execution(self) -> bool:
        return False

    @property
    def authorizes_training(self) -> bool:
        return False

    @property
    def authorizes_exam(self) -> bool:
        return False

    @property
    def authorizes_graduation(self) -> bool:
        return False

    @property
    def independent_exam_evidence(self) -> bool:
        return False


def export_cognitive_train_jsonl(
    contract: ObjectiveContract,
    journey: JourneyState,
    *,
    provider_id: str = PROVIDER_ID,
    forbidden_semantic_sha256: tuple[str, ...] | frozenset[str] = (),
) -> CognitiveTrainDataset:
    """Export one approved cognitive TRAIN cycle's operator-only labels.

    The approved runtime validates the current journey, prior cycle source
    commitments, and operator supplied semantic exclusions before any labeled
    example is produced. This does not replace independently hosted exposure
    CAS, externally held-out data, or physical training permissions.
    """
    runtime = CognitiveObjectiveTrainingRun(
        contract, journey, provider_id=provider_id,
        forbidden_semantic_sha256=forbidden_semantic_sha256,
    )
    # The runtime validated and deep-copied this contract. Never reread the
    # caller's mutable-by-object.__setattr__ input after that snapshot.
    approved_contract = runtime._contract
    # All private challenge records remain on the trusted exporter side.
    # Even if a future refactor changes regeneration, never export a room
    # unless its original provider binding and approved gold score revalidate.
    rooms = runtime._regenerated_roster()
    if not rooms or len(rooms) != runtime.feedback_state.total_rooms:
        raise ValueError("approved cognitive TRAIN roster count changed")

    lines: list[bytes] = []
    for room in rooms:
        challenge = room.prepared.challenge
        target_choice = challenge.answer
        view = room.trainee_view
        if room.score_train_choice(target_choice) != 100:
            raise ValueError("approved cognitive TRAIN gold did not revalidate")
        if room.provider_id != PROVIDER_ID or room.training_rubric_id != RUBRIC_ID:
            raise ValueError("unsupported synthetic cognitive TRAIN provider or rubric")
        row = {
            "schema": SCHEMA_ID,
            "split": "train",
            "provider_id": PROVIDER_ID,
            "training_rubric_id": RUBRIC_ID,
            "objective_id": approved_contract.scope.objective_id,
            "objective_contract_sha256": approved_contract.fingerprint,
            "cycle_index": runtime.feedback_state.cycle_index,
            "room_id": view.room_id,
            "case_id": view.challenge_id,
            "prompt": view.prompt,
            "prompt_sha256": view.prompt_sha256,
            "semantic_sha256": room.semantic_sha256,
            "source_binding_sha256": room._source_binding_sha256,
            "target_choice": target_choice,
        }
        lines.append(json.dumps(row, sort_keys=True, ensure_ascii=True,
                                separators=(",", ":")).encode("utf-8") + b"\n")
    payload = b"".join(lines)
    if len(payload) > MAX_DATASET_BYTES:
        raise ValueError("approved cognitive TRAIN export exceeds byte limit")
    return CognitiveTrainDataset(
        jsonl_bytes=payload, sha256=hashlib.sha256(payload).hexdigest(),
        case_count=len(rooms), objective_id=approved_contract.scope.objective_id,
        contract_fingerprint=approved_contract.fingerprint,
        cycle_index=runtime.feedback_state.cycle_index,
        provider_id=PROVIDER_ID,
    )
