"""Synthetic source-only custody tests. No genuinely private final exam here."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import hmac
import json
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from labyrinth.final_exam_precommit import (
    SCHEMA, FinalExamPrecommitError, SignedFinalExamPrecommit,
    create_pretraining_precommit, verify_pretraining_precommit,
)
from labyrinth.retention_family_registry import canonical_prompt_alias_sha256
from labyrinth.sealed_semantic_exam_protocol import (
    SealedSituation, SignedSealedPlan, TrainingOverlapInventory,
    build_sealed_plan, training_inventory_sha256,
)


EVALUATOR = b"eval-final-precommit-development-fixture-0001"
BANK = b"bank-final-precommit-development-fixture-0001"
TRAINER = b"trainer-final-precommit-development-fixture-1"


class FinalExamPrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Public synthetic bank only. All strings intentionally identifiable.
        bank = tuple(SealedSituation(
            case_id=f"secret-case-{family:02d}-{i:02d}",
            family_id=(f"transfer-family-{family:02d}" if family < 10
                       else f"retention-family-{family:02d}"),
            split="transfer" if family < 10 else "retention",
            after_prompt=f"SECRET_AFTER_{family}_{i} select polarity",
            before_prompt=(None if family < 10 else
                           f"SECRET_BEFORE_{family}_{i} select polarity"),
            gold_label=i % 2,
            semantic_key=f"secret-semantic-{family:02d}-{i:02d}",
        ) for family in range(20) for i in range(20))
        inventory = TrainingOverlapInventory(
            family_ids=("training-family",),
            prompt_alias_sha256s=(canonical_prompt_alias_sha256("train prompt"),),
            semantic_keys=("training-semantic",),
        )
        cls.plan = build_sealed_plan(
            cases=bank, training_inventory=inventory,
            externally_pinned_training_inventory_sha256=training_inventory_sha256(inventory),
            exam_id="exam-20261031", event_id="event-20261031",
            objective_id="sentiment.binary", scope_sha256="a" * 64,
            pretrained_model_sha256="b" * 64, tokenizer_sha256="c" * 64,
            optimizer_config_sha256="d" * 64, source_sha256="e" * 64,
            prior_skill_checkpoint_sha256="f" * 64,
            seeds=(20261031, 20261101, 20261102, 20261103, 20261104),
            step_budgets=(8,), evaluator_key=EVALUATOR,
            bank_key=BANK, trainer_key=TRAINER)
        cls.private_signer = Ed25519PrivateKey.generate()
        cls.public_key = cls.private_signer.public_key().public_bytes(
            encoding=Encoding.Raw, format=PublicFormat.Raw)
        cls.precommit = create_pretraining_precommit(
            cls.plan, evaluator_key=EVALUATOR, signing_key=cls.private_signer)

    def verify(self, *, precommit=None, key=None, consumed=frozenset(),
               event="event-20261031", scope="a" * 64, plan_digest=None):
        return verify_pretraining_precommit(
            self.precommit if precommit is None else precommit,
            pinned_signer_public_key=self.public_key if key is None else key,
            expected_exam_id="exam-20261031", expected_event_id=event,
            expected_objective_id="sentiment.binary",
            expected_scope_sha256=scope,
            expected_sealed_plan_sha256=(sha256(self.plan.public_bytes).hexdigest()
                                          if plan_digest is None else plan_digest),
            consumed_event_sha256s=consumed)

    def test_no_private_cases_labels_seeds_or_trainer_key_are_exported(self):
        result = self.verify()
        body = json.loads(self.precommit.public_bytes)
        self.assertEqual(result.exam_id, "exam-20261031")
        self.assertEqual(body["sealed_plan_sha256"], sha256(self.plan.public_bytes).hexdigest())
        self.assertFalse(result.graduation_authorized)
        for sensitive in ("SECRET_AFTER", "SECRET_BEFORE", "secret-case", "secret-semantic",
                          "seed", "gold", "label", "answer", "private_bank", "trainer_key",
                          "20261101", "situation", "prompt"):
            self.assertNotIn(sensitive, self.precommit.public_bytes.decode())
        self.assertEqual(body["external_anchor_verified"], False)
        self.assertEqual(body["physical_model_execution_verified"], False)
        self.assertEqual(body["graduation_authorized"], False)

    def test_wrong_external_pinned_key_fails_signature(self):
        other = Ed25519PrivateKey.generate().public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw)
        with self.assertRaisesRegex(FinalExamPrecommitError, "signature"):
            self.verify(key=other)

    def test_tampered_public_commitment_rejected(self):
        corrupted = replace(self.precommit, public_bytes=self.precommit.public_bytes + b" ")
        with self.assertRaisesRegex(FinalExamPrecommitError, "signature"):
            self.verify(precommit=corrupted)

    def test_plan_scope_and_event_mismatch_fails_closed(self):
        for kwargs in ({"event": "event-else"}, {"scope": "b" * 64},
                       {"plan_digest": "f" * 64}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                    FinalExamPrecommitError, "pins"):
                self.verify(**kwargs)

    def test_consumed_external_event_is_detected_but_roster_is_not_authoritative(self):
        event_digest = sha256((SCHEMA + "\0event-20261031").encode("ascii")).hexdigest()
        with self.assertRaisesRegex(FinalExamPrecommitError, "already consumed"):
            self.verify(consumed=frozenset({event_digest}))
        self.assertIsNotNone(self.verify(consumed=frozenset()))

    def test_reject_signed_duplicate_field_noncanonical_and_extra_private_field(self):
        for bad_bytes in (
            self.precommit.public_bytes.replace(
                b'"event_id":"event-20261031",',
                b'"event_id":"event-20261031","event_id":"event-20261031",'),
            self.precommit.public_bytes.replace(
                b'"schema":', b'"private_answer":"SECRET_GOLD","schema":'),
            self.precommit.public_bytes + b"\n",
        ):
            signed = SignedFinalExamPrecommit(
                bad_bytes, self.private_signer.sign(
                    b"labyrinth.final-exam-pretraining-precommit.v1\0" + bad_bytes).hex())
            with self.subTest(bad=bad_bytes[-60:]), self.assertRaises(FinalExamPrecommitError):
                self.verify(precommit=signed)

    def test_secret_embedded_in_signed_sealed_plan_must_not_be_projected(self):
        body = json.loads(self.plan.public_bytes)
        body["private_gold_answer"] = "SHOULD_NOT_EXPORT"
        raw = (json.dumps(body, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
        signed = SignedSealedPlan(raw, hmac.new(
            EVALUATOR, b"labyrinth.sealed-semantic-plan.v1\0" + raw,
            sha256).hexdigest())
        with self.assertRaisesRegex(FinalExamPrecommitError, "schema changed"):
            create_pretraining_precommit(
                signed, evaluator_key=EVALUATOR, signing_key=self.private_signer)

    def test_no_in_process_actor_can_self_assert_graduation_flag(self):
        forged = replace(self.precommit, graduation_authorized=True)
        with self.assertRaisesRegex(FinalExamPrecommitError, "invalid signed"):
            self.verify(precommit=forged)

    def test_reject_malformed_consumed_event_roster(self):
        for roster in (set(), frozenset({0}), frozenset({"a" * 63}),
                       frozenset({"A" * 64})):
            with self.subTest(roster=roster), self.assertRaisesRegex(
                    FinalExamPrecommitError, "spent-event roster"):
                self.verify(consumed=roster)


if __name__ == "__main__":
    unittest.main()
