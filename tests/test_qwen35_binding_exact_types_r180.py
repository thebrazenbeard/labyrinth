"""R180 synthetic hostile tests for Qwen3.5 source-pin equality spoofing."""
import hashlib
from unittest import TestCase
from labyrinth.qwen35_cognitive_chat_tokenization import (
    OFFICIAL_QWEN35_REVISION, OFFICIAL_QWEN35_TOKENIZER_JSON_SHA256,
    Qwen35ChatSourceBinding, tokenize_qwen35_cognitive_chat_dataset,
)
from labyrinth.cognitive_train_dataset_export import export_cognitive_train_jsonl
from labyrinth.journey import begin_journey
from tests.test_cognitive_objective_training_run import cognitive_contract
from tests.test_qwen35_cognitive_chat_tokenization import TemplateTokenizer

class EqualSpoof(str):
    def __eq__(self, other):
        return True
    def __ne__(self, other):
        return False

class ExactSourcePinTests(TestCase):
    def setUp(self):
        self.contract = cognitive_contract(seed=16135)
        self.journey = begin_journey(self.contract.scope, checkpoint_sha256="1" * 64)
        self.dataset = export_cognitive_train_jsonl(self.contract, self.journey)
        self.tokenizer = TemplateTokenizer()
        self.binding = Qwen35ChatSourceBinding(
            "Qwen/Qwen3.5-4B", OFFICIAL_QWEN35_REVISION,
            OFFICIAL_QWEN35_TOKENIZER_JSON_SHA256,
            hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest(),
            False)
    def test_equal_spoofed_source_pins_rejected(self):
        for field in ("model_id", "revision", "tokenizer_json_sha256"):
            with self.subTest(field=field):
                forged = self.binding._replace(**{field: EqualSpoof("FORGED")})
                with self.assertRaisesRegex(ValueError, "source revision|source binding"):
                    tokenize_qwen35_cognitive_chat_dataset(
                        self.contract, self.journey, self.dataset,
                        tokenizer=self.tokenizer, binding=forged, max_tokens=4096)
    def test_valid_exact_builtin_pins_still_pass(self):
        batch = tokenize_qwen35_cognitive_chat_dataset(
            self.contract, self.journey, self.dataset,
            tokenizer=self.tokenizer, binding=self.binding, max_tokens=4096)
        self.assertEqual(len(batch.samples), 15)
        self.assertFalse(batch.authorizes_training)
    def test_subclass_even_with_matching_text_rejected(self):
        for field, value in (
            ("model_id", "Qwen/Qwen3.5-4B"),
            ("revision", OFFICIAL_QWEN35_REVISION),
            ("tokenizer_json_sha256", OFFICIAL_QWEN35_TOKENIZER_JSON_SHA256),
        ):
            with self.subTest(field=field):
                forged = self.binding._replace(**{field: EqualSpoof(value)})
                with self.assertRaises(ValueError):
                    tokenize_qwen35_cognitive_chat_dataset(
                        self.contract, self.journey, self.dataset,
                        tokenizer=self.tokenizer, binding=forged, max_tokens=4096)
