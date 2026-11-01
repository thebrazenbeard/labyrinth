"""Source-only, model-agnostic cognitive TRAIN supervision boundaries."""

import importlib
import hashlib
import json
from dataclasses import replace
from unittest import TestCase

from labyrinth.cognitive_train_dataset_export import export_cognitive_train_jsonl
from labyrinth.cognitive_objective_training_run import CognitiveObjectiveTrainingRun
from labyrinth.journey import begin_journey, complete_maze
from tests.test_cognitive_objective_training_run import (
    cognitive_contract, development_fail_for_reroll,
)

from labyrinth.cognitive_sft_tokenization import tokenize_cognitive_train_dataset


class CharTokenizer:
    """Deterministic fake of the HF-like operator-controlled tokenizer seam."""

    eos_token_id = 2

    def __call__(self, content, *, add_special_tokens):
        values = [ord(character) + 1000 for character in content]
        if add_special_tokens:
            values.insert(0, 101)
        return {"input_ids": values}


class CognitiveSftTokenizationTests(TestCase):
    @staticmethod
    def _approved(seed):
        contract = cognitive_contract(seed=seed)
        journey = begin_journey(contract.scope, checkpoint_sha256="1" * 64)
        dataset = export_cognitive_train_jsonl(contract, journey)
        return contract, journey, dataset

    def test_canonical_train_labels_only_supervise_target_and_eos(self):
        # The test resolves the module through the public import seam so a
        # missing implementation is a useful failed assertion, not ImportError.
        try:
            target = importlib.import_module("labyrinth.cognitive_sft_tokenization")
        except ImportError:
            self.fail("approved cognitive TRAIN tokenizer adapter is unavailable")

        contract = cognitive_contract(seed=1151)
        journey = begin_journey(contract.scope, checkpoint_sha256="1" * 64)
        dataset = export_cognitive_train_jsonl(contract, journey)
        batch = target.tokenize_cognitive_train_dataset(
            contract, journey, dataset, tokenizer=CharTokenizer(),
            max_tokens=4096,
        )
        self.assertEqual(len(batch.samples), 15)
        self.assertEqual(batch.dataset_sha256, dataset.sha256)
        first = json.loads(dataset.jsonl_bytes.splitlines()[0])
        prefix = (101,) + tuple(ord(c) + 1000 for c in first["prompt"] + "\nAnswer:")
        supervised = tuple(ord(c) + 1000 for c in " " + first["target_choice"]) + (2,)
        self.assertEqual(batch.samples[0].input_ids, prefix + supervised)
        self.assertEqual(batch.samples[0].labels, (-100,) * len(prefix) + supervised)
        self.assertEqual(batch.samples[0].attention_mask, (1,) * (len(prefix) + len(supervised)))
        self.assertEqual(batch.samples[0].case_id, first["case_id"])
        self.assertEqual(batch.samples[0].semantic_sha256, first["semantic_sha256"])
        self.assertFalse(batch.authorizes_host_execution)
        self.assertFalse(batch.authorizes_training)
        self.assertFalse(batch.authorizes_exam)
        self.assertFalse(batch.authorizes_graduation)

    def test_forged_split_gold_duplicate_rows_and_rehashed_bytes_are_rejected(self):
        contract, journey, dataset = self._approved(1152)
        rows = [json.loads(line) for line in dataset.jsonl_bytes.splitlines()]
        for mutation in ("wrong_gold", "exam", "duplicate_case", "duplicate_semantic"):
            changed = [dict(row) for row in rows]
            if mutation == "wrong_gold":
                changed[0]["target_choice"] = next(
                    choice for choice in "ABC" if choice != rows[0]["target_choice"]
                )
            if mutation == "exam":
                changed[0]["split"] = "heldout"
            if mutation == "duplicate_case":
                changed[1]["case_id"] = changed[0]["case_id"]
            if mutation == "duplicate_semantic":
                changed[1]["semantic_sha256"] = changed[0]["semantic_sha256"]
            payload = b"".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                for row in changed
            )
            forged = dataset._replace(
                jsonl_bytes=payload, sha256=hashlib.sha256(payload).hexdigest()
            )
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                tokenize_cognitive_train_dataset(
                    contract, journey, forged, tokenizer=CharTokenizer(),
                    max_tokens=4096,
                )

    def test_forged_provenance_scope_and_eval_phase_cannot_create_supervision(self):
        contract, journey, dataset = self._approved(1153)
        forged = dataset._replace(contract_fingerprint="a" * 64)
        with self.assertRaises(ValueError):
            tokenize_cognitive_train_dataset(
                contract, journey, forged, tokenizer=CharTokenizer(), max_tokens=4096,
            )
        with self.assertRaises(ValueError):
            tokenize_cognitive_train_dataset(
                contract, replace(journey, phase="awaiting_exam"), dataset,
                tokenizer=CharTokenizer(), max_tokens=4096,
            )
        other = cognitive_contract(seed=1153, domain="reasoning")
        with self.assertRaises(ValueError):
            tokenize_cognitive_train_dataset(
                other, journey, dataset, tokenizer=CharTokenizer(), max_tokens=4096,
            )

    def test_prior_semantics_and_two_cycle_nonoverlap_fail_closed(self):
        contract, journey, dataset0 = self._approved(1154)
        batch0 = tokenize_cognitive_train_dataset(
            contract, journey, dataset0, tokenizer=CharTokenizer(), max_tokens=4096,
        )
        first_semantic = batch0.samples[0].semantic_sha256
        with self.assertRaises(ValueError):
            tokenize_cognitive_train_dataset(
                contract, journey, dataset0, tokenizer=CharTokenizer(),
                max_tokens=4096, forbidden_semantic_sha256=(first_semantic,),
            )
        runtime = CognitiveObjectiveTrainingRun(contract, journey)
        for _ in range(15):
            runtime.run_training_chamber(lambda _view: "A")
        waiting = complete_maze(journey, runtime.self_reported_completion_receipt(
            checkpoint_after_sha256="2" * 64,
        ))
        second = development_fail_for_reroll(waiting)
        dataset1 = export_cognitive_train_jsonl(contract, second)
        batch1 = tokenize_cognitive_train_dataset(
            contract, second, dataset1, tokenizer=CharTokenizer(), max_tokens=4096,
        )
        self.assertEqual((len(batch0.samples), len(batch1.samples)), (15, 15))
        self.assertFalse({x.case_id for x in batch0.samples}.intersection(
            x.case_id for x in batch1.samples))
        self.assertFalse({x.semantic_sha256 for x in batch0.samples}.intersection(
            x.semantic_sha256 for x in batch1.samples))
        self.assertEqual((batch0.cycle_index, batch1.cycle_index), (0, 1))

    def test_malformed_tokenizer_empty_completion_and_missing_eos_fail_closed(self):
        contract, journey, dataset = self._approved(1155)

        class MissingEos(CharTokenizer):
            eos_token_id = None

        class EmptyCompletion(CharTokenizer):
            def __call__(self, content, *, add_special_tokens):
                if not add_special_tokens:
                    return {"input_ids": []}
                return super().__call__(content, add_special_tokens=add_special_tokens)

        class NestedIds(CharTokenizer):
            def __call__(self, content, *, add_special_tokens):
                return {"input_ids": [[5, 6]]}

        class BooleanIds(CharTokenizer):
            def __call__(self, content, *, add_special_tokens):
                return {"input_ids": [True]}

        class EosInAnswer(CharTokenizer):
            def __call__(self, content, *, add_special_tokens):
                if not add_special_tokens:
                    return {"input_ids": [2]}
                return super().__call__(content, add_special_tokens=add_special_tokens)

        for bad in (MissingEos(), EmptyCompletion(), NestedIds(), BooleanIds(), EosInAnswer()):
            with self.subTest(tokenizer=type(bad).__name__), self.assertRaises(ValueError):
                tokenize_cognitive_train_dataset(
                    contract, journey, dataset, tokenizer=bad, max_tokens=4096,
                )

    def test_fixed_max_token_budget_rejects_truncation_and_all_masked_examples(self):
        contract, journey, dataset = self._approved(1156)
        for budget in (0, True, 8, 32, 8193):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                tokenize_cognitive_train_dataset(
                    contract, journey, dataset, tokenizer=CharTokenizer(), max_tokens=budget,
                )
        batch = tokenize_cognitive_train_dataset(
            contract, journey, dataset, tokenizer=CharTokenizer(), max_tokens=4096,
        )
        for sample in batch.samples:
            self.assertEqual(len(sample.input_ids), len(sample.labels))
            self.assertEqual(len(sample.input_ids), len(sample.attention_mask))
            self.assertEqual(sample.labels[-1], 2)
            self.assertIn(-100, sample.labels)
            self.assertTrue(all(token == -100 for token in
                                sample.labels[:sample.labels.index(sample.labels[-3])]))
            self.assertTrue(all(token != -100 for token in sample.labels[-3:]))
        with self.assertRaises((AttributeError, TypeError)):
            batch.samples[0].labels = ()

    def test_tokenizer_mutating_caller_contract_cannot_change_verified_provenance(self):
        contract, journey, dataset = self._approved(1157)
        objective_id = contract.scope.objective_id

        class MutatingTokenizer(CharTokenizer):
            def __init__(self):
                self.called = False

            def __call__(self, content, *, add_special_tokens):
                if not self.called:
                    self.called = True
                    object.__setattr__(contract.scope, "objective_id", "caller-fraud-after-validation")
                return super().__call__(content, add_special_tokens=add_special_tokens)

        batch = tokenize_cognitive_train_dataset(
            contract, journey, dataset, tokenizer=MutatingTokenizer(),
            max_tokens=4096,
        )
        self.assertEqual(batch.objective_id, objective_id)
        self.assertEqual(batch.dataset_sha256, dataset.sha256)
        self.assertEqual(len(batch.samples), 15)

    def test_tokenizer_inserting_eos_into_masked_prompt_is_rejected(self):
        contract, journey, dataset = self._approved(1158)

        class EarlyEosTokenizer(CharTokenizer):
            def __call__(self, content, *, add_special_tokens):
                encoded = super().__call__(content, add_special_tokens=add_special_tokens)
                if add_special_tokens:
                    encoded["input_ids"].append(self.eos_token_id)
                return encoded

        with self.assertRaises(ValueError):
            tokenize_cognitive_train_dataset(
                contract, journey, dataset,
                tokenizer=EarlyEosTokenizer(), max_tokens=4096,
            )

    def test_tokenizer_collapsing_all_choice_labels_is_rejected(self):
        contract, journey, dataset = self._approved(1159)

        class CollapsedChoicesTokenizer(CharTokenizer):
            def __call__(self, content, *, add_special_tokens):
                if not add_special_tokens and content in (" A", " B", " C"):
                    return {"input_ids": [300]}
                return super().__call__(content, add_special_tokens=add_special_tokens)

        with self.assertRaises(ValueError):
            tokenize_cognitive_train_dataset(
                contract, journey, dataset,
                tokenizer=CollapsedChoicesTokenizer(), max_tokens=4096,
            )

    def test_context_sensitive_merge_across_answer_boundary_is_rejected(self):
        contract, journey, dataset = self._approved(1160)

        class ContextMergingTokenizer(CharTokenizer):
            def __call__(self, content, *, add_special_tokens):
                encoded = super().__call__(content, add_special_tokens=add_special_tokens)
                # A tokenizer may merge the trailing space and choice with
                # preceding context. Standalone choice IDs remain distinct.
                if add_special_tokens and "\nAnswer: " in content:
                    encoded["input_ids"] = (
                        encoded["input_ids"][:-2]
                        + [700_000 + ord(content[-1])]
                    )
                return encoded

        with self.assertRaisesRegex(ValueError, "full-context tokenization"):
            tokenize_cognitive_train_dataset(
                contract, journey, dataset,
                tokenizer=ContextMergingTokenizer(), max_tokens=4096,
            )

    def test_distinct_train_prompts_cannot_collapse_to_bos_only_prefix(self):
        contract, journey, dataset = self._approved(1161)

        class LossyPrefixTokenizer(CharTokenizer):
            def __call__(self, content, *, add_special_tokens):
                if not add_special_tokens and content in (" A", " B", " C"):
                    return {"input_ids": [500 + ord(content[-1])]}
                if add_special_tokens and content.endswith("\nAnswer:"):
                    return {"input_ids": [101]}
                if add_special_tokens and "\nAnswer: " in content:
                    return {"input_ids": [101, 500 + ord(content[-1])]}
                return super().__call__(content, add_special_tokens=add_special_tokens)

        # Full-context parity still holds for every answer; nevertheless every
        # distinct canonical prompt is reduced to the same noninformative BOS.
        with self.assertRaisesRegex(ValueError, "distinct TRAIN prompts share prefix IDs"):
            tokenize_cognitive_train_dataset(
                contract, journey, dataset,
                tokenizer=LossyPrefixTokenizer(), max_tokens=4096,
            )
