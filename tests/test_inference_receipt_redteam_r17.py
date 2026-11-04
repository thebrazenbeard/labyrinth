"""R17 adversarial source-only evaluator inference boundary tests.

The synthetic Python drivers are NOT independent evaluators or neural proof.
"""

from dataclasses import replace
from hashlib import sha256
import hmac
import json
import unittest

import test_pinned_model_inference_receipt as fixture_module
from labyrinth.pinned_model_inference_receipt import (
    InferenceSourceError, PrivateInferenceCase,
)


class InferenceReceiptRedteamR17(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.PinnedModelInferenceReceiptTests(
            "test_logits_are_derived_inside_evaluator_source_and_signed")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_second_private_prompt_mutation_during_first_prediction_is_denied(self):
        """A signed roster must remain the roster ACTUALLY fed to the driver."""
        f = self.fixture
        expected = tuple(PrivateInferenceCase(c.case_id, c.prompt)
                         for c in f.cases)

        class MutatingDriver:
            def __init__(self):
                self.calls = 0

            def predict_logits(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    object.__setattr__(f.cases[1], "prompt", "positive substituted prompt")
                return (0.0, 3.0) if "positive" in prompt else (3.0, 0.0)

        with self.assertRaisesRegex(InferenceSourceError, "roster|prompt|changed"):
            f.perform(evaluator_selected_driver_factory=lambda _: MutatingDriver())
        # A verifier with an independently held original roster must not
        # have to infer whether the runner used the same private prompts.
        self.assertEqual(expected[1].prompt, "negative synthetic prompt")

    def test_private_verifier_source_only_cannot_attest_actual_prompt_use(self):
        """A source MAC cannot prove an opaque runtime used the signed prompt."""
        f = self.fixture
        pristine = tuple(PrivateInferenceCase(c.case_id, c.prompt)
                         for c in f.cases)

        class IgnoringDriver:
            def predict_logits(self, _prompt):
                # The opaque driver ignores the supplied prompt altogether.
                # Source signatures cannot attest private runtime semantics.
                return (0.0, 3.0)

        receipt, _ = f.perform(evaluator_selected_driver_factory=lambda _: IgnoringDriver())
        # This is an explicit, bounded negative finding, not an authorization:
        # the signed record has the correct case IDs, but the source verifier
        # cannot know which text an untrusted runtime actually consumed.
        self.assertEqual(f.verify(receipt, expected_cases=pristine),
                         (("case01", 1), ("case02", 1)))
        self.assertFalse(receipt.physical_execution_authenticated)
        self.assertFalse(receipt.independent_evaluator_host_verified)

    def test_intent_mutation_during_logits_cannot_be_signed_as_original(self):
        """Immutable Python dataclasses do not protect operator pins from mutation."""
        f = self.fixture

        class MutatingDriver:
            def predict_logits(self, _prompt):
                object.__setattr__(f.intent, "candidate_artifact_sha256", "e" * 64)
                return (0.0, 1.0)

        with self.assertRaisesRegex(InferenceSourceError, "intent|candidate|changed"):
            f.perform(evaluator_selected_driver_factory=lambda _: MutatingDriver())

    def test_validly_resigned_duplicate_json_key_is_not_canonical(self):
        f = self.fixture
        receipt, _ = f.perform()
        body = receipt.private_bytes
        altered = body.replace(b'"case_count":', b'"case_count":', 1)
        # Insert duplicate valid key while preserving valid signed bytes; a
        # parser that drops duplicate keys must still refuse noncanonical JSON.
        marker = b'"exam_id":"exam-source",'
        self.assertIn(marker, altered)
        altered = altered.replace(marker, marker + marker, 1)
        public = json.loads(receipt.public_bytes)
        public["private_prediction_record_sha256"] = sha256(altered).hexdigest()
        public_bytes = (json.dumps(public, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True) + "\n").encode("ascii")
        forged = replace(
            receipt, private_bytes=altered, public_bytes=public_bytes,
            private_mac_hex=hmac.new(fixture_module.EVALUATOR_KEY,
                b"labyrinth.evaluator-source-inference.private.v1\0" + altered,
                sha256).hexdigest(),
            public_mac_hex=hmac.new(fixture_module.EVALUATOR_KEY,
                b"labyrinth.evaluator-source-inference.public.v1\0" + public_bytes,
                sha256).hexdigest(),
        )
        with self.assertRaises(InferenceSourceError):
            f.verify(forged)

    def test_repeated_receipt_remains_source_only_not_one_shot_proof(self):
        f = self.fixture
        first, _ = f.perform()
        second, _ = f.perform()
        self.assertEqual(first, second)
        self.assertEqual(f.verify(first), f.verify(second))
        self.assertFalse(first.physical_execution_authenticated)
        self.assertFalse(first.graduation_authorized)

    def test_symlinked_component_ancestor_is_denied_before_driver(self):
        f = self.fixture
        linked = f.root / "linked"
        try:
            linked.symlink_to(f.root, target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("symlink unavailable on this platform")
        model = replace(f.model, adapter=replace(
            f.model.adapter, path=linked / "adapter.safetensors"))
        seen = []
        with self.assertRaisesRegex(InferenceSourceError, "linked"):
            f.perform(model=model,
                      evaluator_selected_driver_factory=lambda _: seen.append("called"))
        self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
