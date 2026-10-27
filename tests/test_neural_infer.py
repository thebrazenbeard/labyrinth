"""Inference receives public prompts and rejects any private answer fields."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from labyrinth.neural_infer import read_public_cases, render_public_prompt, generation_configuration


class _TemplateTokenizer:
    chat_template = "chat-template-v1"

    def __init__(self):
        self.last_call = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.last_call = (messages, tokenize, add_generation_prompt)
        return "[USER]" + messages[0]["content"] + "[/USER][ASSISTANT]"


class PublicPromptRenderingTests(unittest.TestCase):
    def test_raw_mode_preserves_historical_newline_convention(self):
        self.assertEqual(render_public_prompt(_TemplateTokenizer(), "Compute 2+2", mode="raw"),
                         "Compute 2+2\n")

    def test_chat_template_encodes_public_prompt_only(self):
        tok = _TemplateTokenizer()
        result = render_public_prompt(tok, "Compute 2+2", mode="chat")
        self.assertEqual(result, "[USER]Compute 2+2[/USER][ASSISTANT]")
        self.assertEqual(tok.last_call, ([{"role": "user", "content": "Compute 2+2"}],
                                         False, True))

    def test_missing_or_invalid_template_fails_closed(self):
        tok = _TemplateTokenizer()
        tok.chat_template = None
        with self.assertRaises(ValueError):
            render_public_prompt(tok, "Compute 2+2", mode="chat")
        with self.assertRaises(ValueError):
            render_public_prompt(tok, "Compute 2+2", mode="other")

    def test_generation_commitment_distinguishes_rendering_and_template(self):
        tok = _TemplateTokenizer()
        raw = generation_configuration(tok, max_new_tokens=48, mode="raw")
        chat = generation_configuration(tok, max_new_tokens=48, mode="chat")
        self.assertEqual(raw["prompt_mode"], "raw")
        self.assertEqual(chat["prompt_mode"], "chat")
        self.assertNotEqual(raw, chat)
        self.assertEqual(len(chat["chat_template_sha256"]), 64)
        tok.chat_template = "another-template"
        self.assertNotEqual(chat, generation_configuration(tok, max_new_tokens=48, mode="chat"))


class PublicCasesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "public.json"
        self.item = {"case_id": "final-1", "prompt": "Compute 2 + 2",
                     "prompt_sha256": hashlib.sha256(b"Compute 2 + 2").hexdigest()}

    def read(self, examples):
        self.path.write_text(json.dumps(examples))
        return read_public_cases(self.path)

    def test_valid_case(self):
        self.assertEqual(self.read([self.item]), [self.item])

    def test_no_gold_or_hidden_fields_allowed(self):
        for mutation in (
            {**self.item, "answer": "4"},
            {**self.item, "prompt_sha256": "0" * 64},
            {**self.item, "case_id": "../../escape"},
        ):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.read([mutation])
        with self.assertRaises(ValueError):
            self.read([self.item, self.item])


if __name__ == "__main__":
    unittest.main()
