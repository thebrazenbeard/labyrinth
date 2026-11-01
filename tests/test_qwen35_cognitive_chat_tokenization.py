"""Fail-closed source checks for a versioned Qwen3.5 chat-template SFT path."""

import hashlib
import json
from unittest import TestCase

from labyrinth.cognitive_train_dataset_export import export_cognitive_train_jsonl
from labyrinth.journey import begin_journey
from labyrinth.qwen35_cognitive_chat_tokenization import (
    OFFICIAL_QWEN35_REVISION,
    OFFICIAL_QWEN35_TOKENIZER_JSON_SHA256,
    Qwen35ChatSourceBinding,
    tokenize_qwen35_cognitive_chat_dataset,
)
from tests.test_cognitive_objective_training_run import cognitive_contract


class TemplateTokenizer:
    eos_token_id = 248046
    chat_template = "synthetic-template-source-fixture"

    def __init__(self):
        self.bad_prefix = False
        self.bad_suffix = False
        self.early_eos = False
        self.seen_modes = []
        self.completed_contents = []

    def __call__(self, text, *, add_special_tokens):
        if add_special_tokens or not text or text[-1] not in "ABC" or text not in (
            "A", "B", "C", " A", " B", " C",
        ):
            raise AssertionError("unexpected completion encoding")
        # Official Jinja `render_content(..., true)|trim` removes a leading
        # space from the assistant content. These must be distinct token IDs.
        return {"input_ids": [400 + ord(text[-1]) + (1000 if text.startswith(" ") else 0)]}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        self.seen_modes.append((tokenize, add_generation_prompt, enable_thinking))
        assert tokenize and enable_thinking is False
        prompt = messages[0]["content"]
        # The official chat style can include the same EOS token at the end
        # of a *masked user turn*; only an EOS after the assistant header is bad.
        prefix = (
            [101] + [ord(c) + 1000 for c in "[user]" + prompt]
            + [self.eos_token_id]
            + [ord(c) + 1000 for c in "[assistant]"]
        )
        if self.early_eos:
            prefix.append(self.eos_token_id)
        if add_generation_prompt:
            assert len(messages) == 1
            return prefix
        assert len(messages) == 2 and messages[-1]["role"] == "assistant"
        self.completed_contents.append(messages[-1]["content"])
        rendered_content = messages[-1]["content"].strip()
        assert rendered_content in "ABC"
        choice = rendered_content
        if self.bad_prefix:
            prefix = [999] + prefix[1:]
        output = prefix + [400 + ord(choice)] + [self.eos_token_id]
        if self.bad_suffix:
            output.append(777)
        return output


class Qwen35CognitiveChatTokenizationTests(TestCase):
    def setUp(self):
        self.contract = cognitive_contract(seed=16135)
        self.journey = begin_journey(self.contract.scope, checkpoint_sha256="1" * 64)
        self.dataset = export_cognitive_train_jsonl(self.contract, self.journey)
        self.tokenizer = TemplateTokenizer()
        self.binding = Qwen35ChatSourceBinding(
            model_id="Qwen/Qwen3.5-4B",
            revision=OFFICIAL_QWEN35_REVISION,
            tokenizer_json_sha256=OFFICIAL_QWEN35_TOKENIZER_JSON_SHA256,
            chat_template_sha256=hashlib.sha256(
                self.tokenizer.chat_template.encode("utf-8")
            ).hexdigest(),
            enable_thinking=False,
        )

    def _encode(self, **kwargs):
        return tokenize_qwen35_cognitive_chat_dataset(
            self.contract, self.journey, self.dataset,
            tokenizer=self.tokenizer, binding=self.binding,
            max_tokens=4096, **kwargs,
        )

    def test_private_supervision_is_only_choice_and_eos(self):
        batch = self._encode()
        self.assertEqual(len(batch.samples), 15)
        self.assertEqual(batch.dataset_sha256, self.dataset.sha256)
        self.assertFalse(batch.authorizes_training)
        self.assertFalse(batch.authorizes_host_execution)
        self.assertFalse(batch.authorizes_exam)
        self.assertFalse(batch.authorizes_graduation)
        for sample in batch.samples:
            self.assertEqual(sample.attention_mask, (1,) * len(sample.input_ids))
            self.assertEqual(sample.input_ids[-1], 248046)
            self.assertEqual(sample.labels[-1], 248046)
            self.assertEqual(sum(v != -100 for v in sample.labels), 2)
            self.assertEqual(sample.labels[-2:], sample.input_ids[-2:])
        self.assertEqual(
            set(self.tokenizer.seen_modes),
            {(True, True, False), (True, False, False)},
        )
        self.assertEqual(set(self.tokenizer.completed_contents), set("ABC"))

    def test_official_template_trimming_changes_completion_token_ids(self):
        self.assertNotEqual(
            self.tokenizer(" A", add_special_tokens=False)["input_ids"],
            self.tokenizer("A", add_special_tokens=False)["input_ids"],
        )
        # If the adapter trains the space-prefixed ID rather than the
        # template-rendered trimmed choice, full-context parity must fail.
        batch = self._encode()
        self.assertEqual(len(batch.samples), 15)

    def test_chat_prefix_mismatch_and_extra_template_suffix_fail_closed(self):
        for flag, expected in (
            ("bad_prefix", "generation prefix"),
            ("bad_suffix", "choice and EOS"),
            ("early_eos", "EOS before"),
        ):
            setattr(self.tokenizer, flag, True)
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, expected):
                self._encode()
            setattr(self.tokenizer, flag, False)

    def test_unsupported_identity_template_and_thinking_mode_fail_closed(self):
        for field, value in (
            ("model_id", "rodrigomt/Qwen3.5-4B-Uncensored-Aggressive"),
            ("revision", "0" * 40),
            ("tokenizer_json_sha256", "0" * 64),
            ("chat_template_sha256", "0" * 64),
            ("enable_thinking", True),
        ):
            binding = self.binding._replace(**{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                tokenize_qwen35_cognitive_chat_dataset(
                    self.contract, self.journey, self.dataset,
                    tokenizer=self.tokenizer, binding=binding, max_tokens=4096,
                )

    def test_forged_source_bytes_cannot_create_labels(self):
        rows = [json.loads(line) for line in self.dataset.jsonl_bytes.splitlines()]
        rows[0]["target_choice"] = next(c for c in "ABC" if c != rows[0]["target_choice"])
        payload = b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for row in rows
        )
        self.dataset = self.dataset._replace(
            jsonl_bytes=payload, sha256=hashlib.sha256(payload).hexdigest()
        )
        with self.assertRaisesRegex(ValueError, "labeled TRAIN export"):
            self._encode()
