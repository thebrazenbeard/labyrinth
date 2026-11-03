"""Hostile source-only evidence tests; no host/device/model execution."""

from dataclasses import replace
import unittest

from labyrinth.cross_training_provenance import (
    IncompatibleTrainingEvidence, PR59, PR62, R18, PROFILES,
    SCHEMA, SOURCE_ONLY, assert_relation, make_envelope, verify_json,
)

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64


def valid(campaign=R18, control="correct", method="mezo_lora_zo"):
    if campaign == PR59:
        return make_envelope(PR59, method_id="source_only_corpus")
    if campaign == PR62:
        return make_envelope(PR62, method_id="adamw_lora", control_id=control,
                             training_plan_sha256=A,
                             baseline_checkpoint_sha256=B,
                             result_checkpoint_sha256=C)
    return make_envelope(R18, method_id=method, control_id=control,
                         training_plan_sha256=A,
                         baseline_checkpoint_sha256=B,
                         result_checkpoint_sha256=C,
                         eval_custody_sha256=D)


class TrainingProvenanceBridgeTests(unittest.TestCase):
    def rejects(self, candidate):
        with self.assertRaises(IncompatibleTrainingEvidence):
            candidate.validate()

    def test_distinct_valid_campaigns(self):
        self.assertEqual(SCHEMA, "labyrinth.cross-training-source-provenance.v1")
        for candidate in (valid(PR59), valid(PR62), valid(R18),
                          valid(R18, "frozen", "qlora_nf4")):
            with self.subTest(campaign=candidate.campaign):
                self.assertEqual(candidate.evidence_level, SOURCE_ONLY)
                self.assertFalse(candidate.graduation_authorized)
                self.assertEqual(verify_json(candidate.to_json()), candidate)
                self.assertEqual(len(candidate.digest), 64)

    def test_pr59_corpus_is_legitimate_r18_input_not_a_learner(self):
        self.assertEqual(assert_relation(valid(PR59), valid(R18),
                                        "R16_CORPUS_INPUT"),
                         "SOURCE_INPUT_PROVENANCE_ONLY")
        self.assertEqual(valid(PR59).model_id, None)
        self.assertIsNone(valid(PR59).baseline_checkpoint_sha256)
        self.rejects(replace(valid(PR59), result_checkpoint_sha256=C))

    def test_pr62_sentiment_is_not_r18_qwen(self):
        with self.assertRaises(IncompatibleTrainingEvidence):
            assert_relation(valid(PR62), valid(R18), "MATCHED_CONTROL_METADATA")
        with self.assertRaises(IncompatibleTrainingEvidence):
            assert_relation(valid(PR62), valid(R18), "R16_CORPUS_INPUT")
        with self.assertRaises(IncompatibleTrainingEvidence):
            assert_relation(valid(PR59), valid(PR62), "R16_CORPUS_INPUT")

    def test_swapped_corpus_refs_and_manifest(self):
        r18 = valid(R18)
        self.rejects(replace(r18, corpus_head=PROFILES[PR62].source_head))
        self.rejects(replace(r18, corpus_manifest_sha256=C))
        self.rejects(replace(r18, corpus_repo="thebrazenbeard/model-training"))
        self.rejects(replace(r18, source_head=PROFILES[PR59].source_head))
        self.rejects(replace(valid(PR62), corpus_manifest_sha256=
                             PROFILES[PR59].corpus_manifest_sha256))

    def test_swapped_model_task_and_objective(self):
        self.rejects(replace(valid(R18), model_id=PROFILES[PR62].model_id))
        self.rejects(replace(valid(PR62), model_id=PROFILES[R18].model_id))
        self.rejects(replace(valid(R18), task_id=PROFILES[PR62].task_id))
        self.rejects(replace(valid(R18), objective_family=
                             PROFILES[PR62].objective_family))
        self.rejects(replace(valid(R18), model_revision="other"))
        self.rejects(replace(valid(R18), eval_source=PROFILES[PR62].eval_source))

    def test_control_set_cross_contamination(self):
        self.rejects(replace(valid(R18), control_set_id=PROFILES[PR62].control_set_id))
        self.rejects(replace(valid(PR62), control_set_id=PROFILES[R18].control_set_id))
        self.rejects(replace(valid(R18), control_set=PROFILES[PR62].control_set))
        self.rejects(replace(valid(PR62), control_set=PROFILES[R18].control_set))
        self.rejects(replace(valid(R18), control_id="zero_lr"))
        self.rejects(replace(valid(PR62), control_id="corrupt"))
        self.rejects(replace(valid(R18), method_id="adamw_lora"))
        self.rejects(replace(valid(PR62), method_id="mezo_lora_zo"))

    def test_unbound_plan_checkpoint_and_eval(self):
        self.rejects(replace(valid(R18), training_plan_sha256=None))
        self.rejects(replace(valid(R18), baseline_checkpoint_sha256=None))
        self.rejects(replace(valid(R18), baseline_checkpoint_sha256="deadbeef"))
        self.rejects(replace(valid(R18), result_checkpoint_sha256="DEADBEEF"))
        self.rejects(replace(valid(R18), eval_custody_sha256=None))
        self.rejects(replace(valid(PR62), eval_custody_sha256=D))

    def test_never_claims_authority_or_exam(self):
        for flag in ("host_authority_attested", "independent_exam_pass",
                     "graduation_authorized"):
            with self.subTest(flag=flag):
                self.rejects(replace(valid(R18), **{flag: True}))
                self.rejects(replace(valid(PR62), **{flag: 1}))

    def test_valid_matched_controls_same_method_plan_base_eval(self):
        a, b = valid(R18, "correct"), valid(R18, "zero")
        self.assertEqual(assert_relation(a, b, "MATCHED_CONTROL_METADATA"),
                         "SOURCE_CONTROL_METADATA_ONLY")
        self.assertEqual(assert_relation(valid(PR62), valid(PR62, "zero_lr"),
                                        "MATCHED_CONTROL_METADATA"),
                         "SOURCE_CONTROL_METADATA_ONLY")
        for adversarial in (replace(b, training_plan_sha256=C),
                            replace(b, baseline_checkpoint_sha256=C),
                            replace(b, eval_custody_sha256=C),
                            valid(R18, "zero", "qlora_nf4"), a):
            with self.assertRaises(IncompatibleTrainingEvidence):
                assert_relation(a, adversarial, "MATCHED_CONTROL_METADATA")

    def test_foreign_or_malformed_json_fails(self):
        pristine = valid(R18).to_json()
        for mutation in (dict(pristine, extra="graduated"),
                         {k: v for k, v in pristine.items() if k != "source_head"},
                         dict(pristine, control_set=PROFILES[PR62].control_set),
                         dict(pristine, control_set=["correct", 1]),
                         dict(pristine, graduation_authorized=1),
                         dict(pristine, model_id=PROFILES[PR62].model_id)):
            with self.subTest(value=mutation):
                with self.assertRaises(IncompatibleTrainingEvidence):
                    verify_json(mutation)

    def test_digest_binds_identity_and_result(self):
        source = valid(R18)
        self.assertEqual(source.digest, valid(R18).digest)
        self.assertNotEqual(source.digest, valid(R18, "zero").digest)
        self.assertNotEqual(source.digest,
                            replace(source, result_checkpoint_sha256=B).digest)
        self.assertNotEqual(source.digest, valid(PR62).digest)

    def test_unknown_relation_never_implicitly_promotes(self):
        with self.assertRaises(IncompatibleTrainingEvidence):
            assert_relation(valid(R18), valid(R18, "zero"), "GRADUATION")

    def test_hostile_equality_object_cannot_spoof_pins_or_encode(self):
        class Ghost:
            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return False

        class GhostStr(str):
            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return False

        for field, attacker_value in (
            ("source_head", Ghost()), ("model_id", Ghost()),
            ("source_head", GhostStr("deadbeef")),
            ("model_id", GhostStr("foreign-model")),
        ):
            with self.subTest(field=field, value_type=type(attacker_value).__name__):
                candidate = replace(valid(R18), **{field: attacker_value})
                with self.assertRaises(IncompatibleTrainingEvidence):
                    candidate.validate()
                with self.assertRaises(IncompatibleTrainingEvidence):
                    candidate.digest
                with self.assertRaises(IncompatibleTrainingEvidence):
                    candidate.to_json()
                forged = valid(R18).to_json()
                forged[field] = attacker_value
                with self.assertRaises(IncompatibleTrainingEvidence):
                    verify_json(forged)
                with self.assertRaises(IncompatibleTrainingEvidence):
                    assert_relation(valid(PR59), candidate, "R16_CORPUS_INPUT")

    def test_exact_envelope_attribute_types_precede_equality(self):
        class GhostStr(str):
            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return False

        source = valid(R18)
        for field in (
            "schema", "campaign", "source_repo", "source_head",
            "corpus_repo", "corpus_head", "task_id", "objective_family",
            "model_id", "model_revision", "method_id", "control_set_id",
            "control_id", "eval_source", "evidence_level",
            "training_plan_sha256", "baseline_checkpoint_sha256",
            "result_checkpoint_sha256", "eval_custody_sha256",
            "corpus_manifest_sha256",
        ):
            with self.subTest(field=field):
                self.rejects(replace(source, **{field: GhostStr("spoof")}))
        self.rejects(replace(source, control_set=(GhostStr("frozen"), "zero", "correct", "corrupt")))
        self.rejects(replace(source, control_set=["frozen", "zero", "correct", "corrupt"]))
        self.rejects(replace(source, host_authority_attested=GhostStr("")))
        self.rejects(replace(source, independent_exam_pass=0))
        self.rejects(replace(source, graduation_authorized=0))
        self.rejects(replace(valid(PR59), model_id=GhostStr("")))
        with self.assertRaises(IncompatibleTrainingEvidence):
            assert_relation(valid(PR59), valid(R18), GhostStr("anything"))

    def test_json_dictionary_keys_must_be_exact_strings(self):
        class GhostStr(str):
            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return False

            __hash__ = str.__hash__

        candidate = valid(R18).to_json()
        candidate[GhostStr("source_head")] = candidate.pop("source_head")
        with self.assertRaises(IncompatibleTrainingEvidence):
            verify_json(candidate)


if __name__ == "__main__":
    unittest.main()
