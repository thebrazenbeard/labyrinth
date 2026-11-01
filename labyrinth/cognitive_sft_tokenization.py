"""Deterministic operator-side tokenization of synthetic cognitive TRAIN labels.

This source-only bridge never loads a model, invokes an optimizer, opens a
network connection, or authorizes host execution.  It re-generates and checks
the canonical labeled TRAIN dataset before giving target-choice tokens to an
injected, operator-controlled tokenizer.  These labels are sensitive TRAIN
gold; a Python import is *not* an authenticated data-access boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, NamedTuple

from .cognitive_train_dataset_export import (
    CognitiveTrainDataset, SCHEMA_ID, export_cognitive_train_jsonl,
)
from .journey import JourneyState
from .objective_contract import ObjectiveContract


IGNORE_LABEL = -100
MAX_SEQUENCE_TOKENS = 8192


class CognitiveSftExample(NamedTuple):
    """Detached token values, with prompt and instruction labels masked."""

    case_id: str
    prompt_sha256: str
    semantic_sha256: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]


class CognitiveSftBatch(NamedTuple):
    """A local TRAIN batch, never a proof of actual gradient updates."""

    dataset_sha256: str
    objective_id: str
    cycle_index: int
    samples: tuple[CognitiveSftExample, ...]

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


def _token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    if not isinstance(encoded, Mapping):
        raise ValueError("tokenizer must return an input_ids mapping")
    values = encoded.get("input_ids")
    if (type(values) not in (list, tuple) or not values
            or any(type(item) is not int or item < 0 or item > 2**31 - 1
                   for item in values)):
        raise ValueError("tokenizer returned invalid flat token IDs")
    return tuple(values)


def tokenize_cognitive_train_dataset(
    contract: ObjectiveContract,
    journey: JourneyState,
    dataset: CognitiveTrainDataset,
    *,
    tokenizer: Any,
    max_tokens: int = 2048,
    forbidden_semantic_sha256: tuple[str, ...] | frozenset[str] = (),
) -> CognitiveSftBatch:
    """Prepare a private SFT batch from *exactly* an approved TRAIN export.

    Prompt and final ``Answer:`` instruction IDs get label ``-100``. The
    response choice and one EOS ID are the only supervised tokens. The
    tokenizer runs in-process and must be trusted by the operator; an
    untrusted callback could introspect labels.  A true Qwen/LoRA trainer,
    host authority, and independent exam are deliberately separate.
    """
    if type(dataset) is not CognitiveTrainDataset:
        raise ValueError("exact approved cognitive TRAIN dataset required")
    if type(max_tokens) is not int or not 16 <= max_tokens <= MAX_SEQUENCE_TOKENS:
        raise ValueError("invalid source-only SFT sequence token budget")
    if not callable(tokenizer):
        raise ValueError("operator-controlled tokenizer must be callable")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if type(eos_id) is not int or not 0 <= eos_id <= 2**31 - 1:
        raise ValueError("tokenizer must define a valid EOS token ID")

    canonical = export_cognitive_train_jsonl(
        contract, journey, forbidden_semantic_sha256=forbidden_semantic_sha256,
    )
    if (dataset != canonical
            or hashlib.sha256(dataset.jsonl_bytes).hexdigest() != dataset.sha256):
        raise ValueError("labeled TRAIN export disagrees with approved source")

    # A local tokenizer may silently alias otherwise distinct labels. Pin all
    # three completions once, before any example is constructed, so an
    # unlearnable or stateful completion mapping fails closed in this batch.
    choices = {
        choice: _token_ids(tokenizer, " " + choice, add_special_tokens=False)
        for choice in "ABC"
    }
    if (len(set(choices.values())) != 3
            or any(eos_id in value for value in choices.values())):
        raise ValueError("ambiguous or prematurely ended cognitive choice tokens")

    examples: list[CognitiveSftExample] = []
    ids: set[str] = set()
    semantics: set[str] = set()
    for line in canonical.jsonl_bytes.splitlines():
        row = json.loads(line)
        if (row["schema"] != SCHEMA_ID or row["split"] != "train"
                or row["target_choice"] not in ("A", "B", "C")
                or row["case_id"] in ids
                or row["semantic_sha256"] in semantics):
            raise ValueError("duplicate or unsupported labeled TRAIN row")
        ids.add(row["case_id"])
        semantics.add(row["semantic_sha256"])
        # Keep the answer instruction on the masked side of the supervision
        # boundary; only the choice and EOS train the next-token objective.
        prompt_and_instruction = row["prompt"] + "\nAnswer:"
        prefix = _token_ids(tokenizer, prompt_and_instruction,
                            add_special_tokens=True)
        if eos_id in prefix:
            raise ValueError("tokenizer inserted EOS before the supervised choice")
        completion = choices[row["target_choice"]]
        # Subword tokenization is context sensitive. Independently encoding
        # the masked prefix and target may not equal tokenizing their actual
        # concatenated text (even when A/B/C choices are unique). Never pass
        # such a synthetic sequence off as model-valid next-token labels.
        contextual = _token_ids(
            tokenizer, prompt_and_instruction + " " + row["target_choice"],
            add_special_tokens=True,
        )
        if contextual != prefix + completion:
            raise ValueError("full-context tokenization disagrees at answer boundary")
        sequence = prefix + completion + (eos_id,)
        if len(sequence) > max_tokens:
            raise ValueError("cognitive TRAIN example exceeds fixed token budget")
        labels = (IGNORE_LABEL,) * len(prefix) + completion + (eos_id,)
        examples.append(CognitiveSftExample(
            case_id=row["case_id"], prompt_sha256=row["prompt_sha256"],
            semantic_sha256=row["semantic_sha256"], input_ids=sequence,
            attention_mask=(1,) * len(sequence), labels=labels,
        ))
    if not examples or len(examples) != canonical.case_count:
        raise ValueError("empty or partial cognitive TRAIN batch")
    return CognitiveSftBatch(
        dataset_sha256=canonical.sha256, objective_id=canonical.objective_id,
        cycle_index=canonical.cycle_index, samples=tuple(examples),
    )
