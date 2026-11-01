"""Fail-closed Qwen3.5 chat-template formatting of canonical cognitive TRAIN rows.

This source-only operator bridge checks the *actual chat-template token sequence*
provided by a locally controlled tokenizer. It never fetches weights, performs
gradients, admits data rights, or grants training/host/evaluation authority.
Caller-supplied source identity is not an independent tokenizer attestation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, NamedTuple

from .cognitive_sft_tokenization import (
    IGNORE_LABEL, MAX_SEQUENCE_TOKENS, CognitiveSftBatch, CognitiveSftExample,
    _token_ids,
)
from .cognitive_train_dataset_export import (
    CognitiveTrainDataset, SCHEMA_ID, export_cognitive_train_jsonl,
)
from .journey import JourneyState
from .objective_contract import ObjectiveContract


OFFICIAL_QWEN35_MODEL_ID = "Qwen/Qwen3.5-4B"
OFFICIAL_QWEN35_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
OFFICIAL_QWEN35_TOKENIZER_JSON_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
OFFICIAL_QWEN35_EOS_ID = 248046


class Qwen35ChatSourceBinding(NamedTuple):
    """Operator-supplied public source pins; NOT a host-issued attestation."""

    model_id: str
    revision: str
    tokenizer_json_sha256: str
    chat_template_sha256: str
    enable_thinking: bool


def _chat_ids(tokenizer: Any, messages: list[dict[str, str]], *, generate: bool) -> tuple[int, ...]:
    """Obtain flat IDs from the official HF chat-template public interface."""
    applied = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=generate,
        enable_thinking=False,
    )
    if isinstance(applied, Mapping):
        applied = applied.get("input_ids")
    if (type(applied) not in (list, tuple) or not applied
            or any(type(value) is not int or value < 0 or value > 2**31 - 1
                   for value in applied)):
        raise ValueError("Qwen chat template returned invalid flat token IDs")
    return tuple(applied)


def _require_public_source_binding(tokenizer: Any, binding: Qwen35ChatSourceBinding) -> int:
    if type(binding) is not Qwen35ChatSourceBinding:
        raise ValueError("explicit Qwen3.5 public source binding required")
    if (binding.model_id != OFFICIAL_QWEN35_MODEL_ID
            or binding.revision != OFFICIAL_QWEN35_REVISION
            or binding.tokenizer_json_sha256 != OFFICIAL_QWEN35_TOKENIZER_JSON_SHA256
            or binding.enable_thinking is not False):
        raise ValueError("unsupported Qwen3.5 source revision or template mode")
    template = getattr(tokenizer, "chat_template", None)
    if (type(template) is not str or not template
            or type(binding.chat_template_sha256) is not str
            or len(binding.chat_template_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in binding.chat_template_sha256)
            or hashlib.sha256(template.encode("utf-8")).hexdigest()
            != binding.chat_template_sha256):
        raise ValueError("chat template content disagrees with source binding")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if type(eos_id) is not int or eos_id != OFFICIAL_QWEN35_EOS_ID:
        raise ValueError("wrong Qwen3.5 EOS token ID")
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise ValueError("Qwen tokenizer lacks chat-template API")
    return eos_id


def tokenize_qwen35_cognitive_chat_dataset(
    contract: ObjectiveContract,
    journey: JourneyState,
    dataset: CognitiveTrainDataset,
    *,
    tokenizer: Any,
    binding: Qwen35ChatSourceBinding,
    max_tokens: int = 2048,
    forbidden_semantic_sha256: tuple[str, ...] | frozenset[str] = (),
) -> CognitiveSftBatch:
    """Prepare masked TRAIN labels only if generation and full-chat IDs agree.

    The tokenizer is an operator-controlled in-process object; bindings and
    successful formatting alone cannot admit rights, verify protected heldout
    exclusions, prove real pretrained loading, or authorize CUDA execution.
    """
    if type(dataset) is not CognitiveTrainDataset:
        raise ValueError("exact canonical cognitive TRAIN dataset required")
    if type(max_tokens) is not int or not 16 <= max_tokens <= MAX_SEQUENCE_TOKENS:
        raise ValueError("invalid Qwen3.5 SFT sequence token budget")
    if not callable(tokenizer):
        raise ValueError("operator-controlled tokenizer must be callable")
    eos_id = _require_public_source_binding(tokenizer, binding)
    canonical = export_cognitive_train_jsonl(
        contract, journey, forbidden_semantic_sha256=forbidden_semantic_sha256,
    )
    if (dataset != canonical
            or hashlib.sha256(dataset.jsonl_bytes).hexdigest() != dataset.sha256):
        raise ValueError("labeled TRAIN export disagrees with approved source")

    choices = {
        # The pinned Qwen3.5 assistant template renders content with `|trim`.
        # A leading space would train different subword IDs than those shown
        # by the actual full chat template for the same A/B/C label.
        letter: _token_ids(tokenizer, letter, add_special_tokens=False)
        for letter in "ABC"
    }
    if (len(set(choices.values())) != 3
            or any(eos_id in values for values in choices.values())):
        raise ValueError("ambiguous or prematurely ended cognitive choice tokens")

    samples: list[CognitiveSftExample] = []
    case_ids: set[str] = set()
    semantic_ids: set[str] = set()
    prefixes: dict[tuple[int, ...], str] = {}
    for line in canonical.jsonl_bytes.splitlines():
        row = json.loads(line)
        if (row["schema"] != SCHEMA_ID or row["split"] != "train"
                or row["target_choice"] not in choices
                or row["case_id"] in case_ids
                or row["semantic_sha256"] in semantic_ids):
            raise ValueError("duplicate or unsupported labeled TRAIN row")
        case_ids.add(row["case_id"])
        semantic_ids.add(row["semantic_sha256"])
        user_message = {"role": "user", "content": row["prompt"]}
        prefix = _chat_ids(tokenizer, [user_message], generate=True)
        # A chat template may contain EOS between the user's turn and the
        # assistant header. It may not terminate the *assistant prefix* itself.
        if prefix[-1] == eos_id:
            raise ValueError("chat template inserted EOS before the supervised choice")
        prior = prefixes.setdefault(prefix, row["prompt"])
        if prior != row["prompt"]:
            raise ValueError("distinct TRAIN prompts share Qwen generation prefix IDs")
        completion = choices[row["target_choice"]]
        full = _chat_ids(
            tokenizer,
            [user_message, {"role": "assistant", "content": row["target_choice"]}],
            generate=False,
        )
        if full[:len(prefix)] != prefix:
            raise ValueError("Qwen generation prefix differs from completed chat prefix")
        if full[len(prefix):] != completion + (eos_id,):
            raise ValueError("completed chat must supervise exactly choice and EOS")
        if len(full) > max_tokens:
            raise ValueError("cognitive Qwen3.5 TRAIN example exceeds fixed token budget")
        samples.append(CognitiveSftExample(
            case_id=row["case_id"], prompt_sha256=row["prompt_sha256"],
            semantic_sha256=row["semantic_sha256"], input_ids=full,
            attention_mask=(1,) * len(full),
            labels=(IGNORE_LABEL,) * len(prefix) + completion + (eos_id,),
        ))
    if not samples or len(samples) != canonical.case_count:
        raise ValueError("empty or partial cognitive Qwen3.5 TRAIN batch")
    return CognitiveSftBatch(
        dataset_sha256=canonical.sha256, objective_id=canonical.objective_id,
        cycle_index=canonical.cycle_index, samples=tuple(samples),
    )
