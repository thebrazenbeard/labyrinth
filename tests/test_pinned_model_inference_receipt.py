"""SOURCE-only private logits-custody tests, not an external model exam."""

from dataclasses import replace
from hashlib import sha256
import hmac
import json
from pathlib import Path
import tempfile
import unittest

from labyrinth.pinned_model_inference_receipt import (
    EvaluatorInferenceIntent, InferenceSourceError, PinnedComponent,
    PrivateInferenceCase, SignedSourceInference, ThreeComponentModel,
    evaluator_roster_commitment, model_component_set_sha256,
    candidate_component_binding_sha256,
    run_source_pinned_evaluator_inference,
    verify_source_inference_receipt,
)
from labyrinth.sealed_semantic_exam_protocol import PredictionSubmission


EVALUATOR_KEY = b"evaluator-only-fixture-key-32-bytes-000"
ROSTER_KEY = b"private-bank-roster-fixture-key-32-bytes"


class DeterministicLogitsDriver:
    def __init__(self):
        self.received: list[str] = []

    def predict_logits(self, prompt):
        self.received.append(prompt)
        return (0.0, 2.5) if "positive" in prompt else (2.5, 0.0)


class PinnedModelInferenceReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        files = {}
        for name, contents in (("pretrained", b"source-only-base-model-bytes"),
                               ("adapter", b"adapter-test-weights"),
                               ("head", b"classifier-test-weights")):
            path = self.root / (name + ".safetensors")
            path.write_bytes(contents)
            files[name] = PinnedComponent(path, sha256(contents).hexdigest())
        self.model = ThreeComponentModel(**files)
        self.component_set = model_component_set_sha256(self.model)
        self.cases = (
            PrivateInferenceCase("case01", "positive synthetic prompt"),
            PrivateInferenceCase("case02", "negative synthetic prompt"),
        )
        self.roster = evaluator_roster_commitment(self.cases, roster_key=ROSTER_KEY)
        self.intent = EvaluatorInferenceIntent(
            exam_id="exam-source", event_id="event-source", run_id="run-A",
            arm="correct", seed=7, steps_per_cycle=4, stage="after-training",
            candidate_artifact_sha256="c" * 64,
            prior_skill_checkpoint_sha256="b" * 64,
            scope_sha256="d" * 64, source_sha256="f" * 64,
            roster_commitment=self.roster,
            component_set_sha256=self.component_set,
        )
        self.candidate_binding = candidate_component_binding_sha256(
            self.intent.candidate_artifact_sha256, self.component_set)

    def perform(self, **changes):
        driver = DeterministicLogitsDriver()
        values = {
            "intent": self.intent, "cases": self.cases,
            "model": self.model,
            "externally_pinned_roster_commitment": self.roster,
            "externally_pinned_component_set_sha256": self.component_set,
            "externally_pinned_candidate_component_binding_sha256": (
                self.candidate_binding),
            "evaluator_key": EVALUATOR_KEY,
            "roster_key": ROSTER_KEY,
            "evaluator_selected_driver_factory": lambda _pins: driver,
        }
        values.update(changes)
        return run_source_pinned_evaluator_inference(**values), driver

    def verify(self, receipt, **changes):
        values = {
            "intent": self.intent,
            "evaluator_key": EVALUATOR_KEY,
            "expected_cases": self.cases,
            "roster_key": ROSTER_KEY,
            "externally_pinned_roster_commitment": self.roster,
            "externally_pinned_candidate_component_binding_sha256": (
                self.candidate_binding),
        }
        values.update(changes)
        return verify_source_inference_receipt(receipt, **values)

    def test_logits_are_derived_inside_evaluator_source_and_signed(self):
        receipt, driver = self.perform()
        self.assertEqual(driver.received, [row.prompt for row in self.cases])
        self.assertEqual(self.verify(receipt),
            (("case01", 1), ("case02", 0)))
        self.assertEqual(len(receipt.public_mac_hex), 64)
        self.assertEqual(len(receipt.private_mac_hex), 64)
        public = json.loads(receipt.public_bytes)
        private = json.loads(receipt.private_bytes)
        self.assertEqual(public["private_prediction_record_sha256"],
                         sha256(receipt.private_bytes).hexdigest())
        self.assertEqual(public["case_count"], 2)
        self.assertEqual(private["adapter_sha256"], self.model.adapter.sha256)
        self.assertEqual(private["head_sha256"], self.model.head.sha256)
        self.assertEqual(private["pretrained_sha256"], self.model.pretrained.sha256)
        for fragment in ("positive synthetic prompt", "negative synthetic prompt",
                         "gold_label", "case01"):
            self.assertNotIn(fragment.encode(), receipt.public_bytes)
        for fragment in ("positive synthetic prompt", "negative synthetic prompt",
                         "gold_label"):
            self.assertNotIn(fragment.encode(), receipt.private_bytes)
        for key in ("physical_execution_authenticated", "independent_evaluator_host_verified",
                    "graduation_authorized"):
            self.assertIs(public[key], False)
            self.assertIs(private[key], False)

    def test_trainer_gold_derived_prediction_submission_is_not_inference_input(self):
        # Existing sealed scorer permits trainer-signed prediction maps for
        # source tests. Here such maps cannot be supplied instead of prompts.
        fake = PredictionSubmission(
            arm="correct", seed=7, steps_per_cycle=4, run_id="run-A",
            initial_trainable_sha256="a" * 64,
            training_visits_sha256="b" * 64,
            candidate_artifact_sha256="c" * 64,
            predictions=(("case01", 1), ("case02", 0)),
            retention_before_predictions=(),
            prior_skill_checkpoint_sha256="b" * 64,
        )
        called = []
        with self.assertRaisesRegex(InferenceSourceError, "typed evaluator prompt"):
            self.perform(cases=(fake,), evaluator_selected_driver_factory=(
                lambda _: called.append("loaded")))
        self.assertEqual(called, [])

    def test_unpinned_or_reordered_private_prompt_roster_refused_before_load(self):
        for changes in (
            {"cases": self.cases[::-1]},
            {"cases": self.cases + (self.cases[0],)},
            {"externally_pinned_roster_commitment": "a" * 64},
            {"intent": replace(self.intent, roster_commitment="a" * 64)},
        ):
            loaded = []
            with self.subTest(changes=repr(changes)[:70]), self.assertRaises(
                    InferenceSourceError):
                self.perform(**changes, evaluator_selected_driver_factory=(
                    lambda _: loaded.append("loaded")))
            self.assertEqual(loaded, [])

    def test_wrong_component_digest_and_hardlink_alias_fail_before_inference(self):
        with self.assertRaisesRegex(InferenceSourceError, "component set differs"):
            self.perform(model=replace(self.model, adapter=replace(
                self.model.adapter, sha256="a" * 64)))
        aliased = replace(self.model, head=self.model.adapter)
        alias_digest = model_component_set_sha256(aliased)
        alias_binding = candidate_component_binding_sha256(
            self.intent.candidate_artifact_sha256, alias_digest)
        with self.assertRaisesRegex(InferenceSourceError, "distinct file"):
            self.perform(model=aliased,
                         intent=replace(self.intent, component_set_sha256=alias_digest),
                         externally_pinned_component_set_sha256=alias_digest,
                         externally_pinned_candidate_component_binding_sha256=(
                             alias_binding))

    def test_model_component_set_requires_separate_external_precommit(self):
        with self.assertRaisesRegex(InferenceSourceError, "external precommit"):
            self.perform(externally_pinned_component_set_sha256="a" * 64)
        with self.assertRaisesRegex(InferenceSourceError, "external precommit"):
            self.perform(intent=replace(self.intent, component_set_sha256="a" * 64))
        # Even a correctly typed but misleading individual file hash may not
        # silently replace the fixed candidate component inventory.
        with self.assertRaisesRegex(InferenceSourceError, "external precommit"):
            self.perform(model=replace(self.model, adapter=replace(
                self.model.adapter, sha256="a" * 64)))

    def test_foreign_candidate_cannot_rebind_same_pinned_model_components(self):
        foreign = replace(self.intent, candidate_artifact_sha256="a" * 64)
        loaded = []
        with self.assertRaisesRegex(InferenceSourceError,
                                    "candidate-to-components map"):
            self.perform(intent=foreign,
                         evaluator_selected_driver_factory=(
                             lambda _: loaded.append("loaded")))
        self.assertEqual(loaded, [])

        # A party holding the evaluator MAC key may re-sign both public and
        # private metadata. That must not let it invent a new candidate with
        # the same component files against the unchanged EXTERNAL map pin.
        receipt, _ = self.perform()
        private = json.loads(receipt.private_bytes)
        public = json.loads(receipt.public_bytes)
        binding = candidate_component_binding_sha256(
            foreign.candidate_artifact_sha256, self.component_set)
        for envelope in (private, public):
            envelope["candidate_artifact_sha256"] = foreign.candidate_artifact_sha256
            envelope["candidate_component_binding_sha256"] = binding
        canonical = lambda body: (json.dumps(
            body, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
        private_raw = canonical(private)
        public["private_prediction_record_sha256"] = sha256(private_raw).hexdigest()
        public_raw = canonical(public)
        forged = replace(
            receipt, public_bytes=public_raw, private_bytes=private_raw,
            public_mac_hex=hmac.new(
                EVALUATOR_KEY,
                b"labyrinth.evaluator-source-inference.public.v1\0" +
                public_raw, sha256).hexdigest(),
            private_mac_hex=hmac.new(
                EVALUATOR_KEY,
                b"labyrinth.evaluator-source-inference.private.v1\0" +
                private_raw, sha256).hexdigest(),
        )
        with self.assertRaisesRegex(InferenceSourceError,
                                    "mapping not independently pinned"):
            self.verify(forged, intent=foreign)

    def test_artifact_mutation_during_predict_refused_after_inference(self):
        path = self.model.adapter.path
        class TamperDriver:
            def predict_logits(self, _prompt):
                path.write_bytes(b"different weights")
                return (0.0, 1.0)
        with self.assertRaisesRegex(InferenceSourceError, "changed or differs"):
            self.perform(evaluator_selected_driver_factory=lambda _: TamperDriver())

    def test_only_detached_two_float_finite_nontied_logits_accepted(self):
        class Driver:
            def __init__(self, output): self.output = output
            def predict_logits(self, _prompt): return self.output
        bad = ([0.0, 1.0], (0, 1), (0.0, float("nan")),
               (1.0, 1.0), (float("inf"), 0.0), (0.0,),
               (0.0, 1.0, 2.0), ((0.0, 1.0),))
        for output in bad:
            with self.subTest(output=repr(output)), self.assertRaisesRegex(
                    InferenceSourceError, "detached float logits"):
                self.perform(evaluator_selected_driver_factory=(
                    lambda _, item=output: Driver(item)))

    def test_tampered_receipt_mac_or_private_prediction_refused(self):
        receipt, _ = self.perform()
        for changed in (
            replace(receipt, public_mac_hex="0" * 64),
            replace(receipt, private_bytes=receipt.private_bytes + b" "),
            replace(receipt, private_mac_hex="0" * 64),
        ):
            with self.subTest(changed=repr(changed)[:50]), self.assertRaises(
                    InferenceSourceError):
                self.verify(changed)

    def test_validly_resigned_foreign_or_reordered_case_rejected(self):
        # Hostile counterexample: a party with the evaluator MAC key can
        # resign a substituted prediction without altering the header's
        # roster commitment. Verification MUST consult a separately pinned,
        # evaluator-held ordered prompt roster, not just receipt identity.
        receipt, _ = self.perform()
        canonical = lambda body: (json.dumps(
            body, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
        for mutation in ("foreign", "reverse"):
            with self.subTest(mutation=mutation):
                private = json.loads(receipt.private_bytes)
                rows = private["predictions_derived_from_logits"]
                if mutation == "foreign":
                    rows[0]["case_id"] = "alien-case"
                else:
                    rows.reverse()
                private_raw = canonical(private)
                public = json.loads(receipt.public_bytes)
                public["private_prediction_record_sha256"] = sha256(
                    private_raw).hexdigest()
                public_raw = canonical(public)
                forged = replace(
                    receipt, public_bytes=public_raw, private_bytes=private_raw,
                    public_mac_hex=hmac.new(
                        EVALUATOR_KEY,
                        b"labyrinth.evaluator-source-inference.public.v1\0" +
                        public_raw, sha256).hexdigest(),
                    private_mac_hex=hmac.new(
                        EVALUATOR_KEY,
                        b"labyrinth.evaluator-source-inference.private.v1\0" +
                        private_raw, sha256).hexdigest(),
                )
                with self.assertRaisesRegex(InferenceSourceError,
                                            "pinned ordered roster"):
                    self.verify(forged)

    def test_mismatched_intent_and_graduation_flag_refused(self):
        receipt, _ = self.perform()
        with self.assertRaisesRegex(InferenceSourceError, "pinned intent"):
            self.verify(receipt, intent=replace(self.intent, arm="zero_update"))
        for changed in (replace(receipt, graduation_authorized=True),
                        replace(receipt, physical_execution_authenticated=True),
                        replace(receipt, independent_evaluator_host_verified=True)):
            with self.assertRaisesRegex(InferenceSourceError, "elevated"):
                self.verify(changed)

    def test_source_receipt_never_qualifies_host_execution(self):
        receipt, _ = self.perform()
        self.assertFalse(receipt.physical_execution_authenticated)
        self.assertFalse(receipt.independent_evaluator_host_verified)
        self.assertFalse(receipt.graduation_authorized)
        self.assertIn(b"SOURCE_SELECTED_IN_PROCESS_NOT_HOST_ATTESTED",
                      receipt.public_bytes)


if __name__ == "__main__":
    unittest.main()
