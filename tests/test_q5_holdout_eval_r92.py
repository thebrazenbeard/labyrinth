"""Adversarial private Q5 scoring fixtures. No real model or final exam runs."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from labyrinth.five_arm_experiment_contract import ArmClaim, build_frozen_plan
from labyrinth.maze_conditioned_minilm_cpu_experiment import (
    PREREG_ARMS, default_scope, optimizer_learning_rate,
)
from labyrinth.minilm_cpu_experiment import HELDOUT
from labyrinth.q5_holdout_eval_r92 import (
    COMPLETION_SCHEMA, PREDICTION_SCHEMA, PRECOMMIT_SCHEMA, REPORT_SCHEMA,
    RETENTION_SCHEMA,
    PrivateCase, PrivateSeed, Q5HoldoutError, Q5TrustPins, RawArmPredictions,
    SignedBytes, _claims_digest, _digest, _public_key, bank_public_commitment,
    evaluate_private_q5, inference_bytes, precommit_bytes, prediction_commitment,
    retention_bytes, sign_source_bytes, training_completion_bytes, verify_public_aggregate,
)


SEEDS = (20261030, 20261031, 20261032, 20261033,
         20261034, 20261035, 20261037, 20261038)


def d(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def source_seeds() -> tuple[PrivateSeed, ...]:
    """Construct a demonstrably synthetic bank; all verdicts are protocol tests."""
    result = []
    for i, seed in enumerate(SEEDS):
        from dataclasses import replace as update
        scope = update(default_scope(), seed=seed)
        retention_ids = tuple(f"ret-{i}-{j:02d}" for j in range(20))
        plan = build_frozen_plan(
            scope, run_id=f"holdout-{i}", source_head="a" * 40,
            source_tree="b" * 40, base_model_sha256=d("base"),
            initial_trainable_sha256=d("initial"),
            initial_optimizer_state_sha256=d("initial-moments"),
            private_transfer_roster_commitment=d(f"transfer-{i}"),
            private_final_bank_commitment=d(f"sealed-final-{i}"),
            retention_case_ids=retention_ids, maximum_prior_correct_regressions=0,
            steps_per_cycle=4,
        )
        claims = []
        for arm in PREREG_ARMS:
            frozen = arm in ("zero_update", "zero_lr")
            start = plan.initial_trainable_sha256
            optimizer_start = plan.initial_optimizer_state_sha256
            cycle0 = start if frozen else d(f"after-cycle0-{i}-{arm}")
            final = start if frozen else d(f"after-cycle1-{i}-{arm}")
            moments0 = optimizer_start if arm == "zero_update" else d(f"moments0-{i}-{arm}")
            moments_final = (optimizer_start if arm == "zero_update"
                             else d(f"moments-final-{i}-{arm}"))
            claims.append(ArmClaim(
                arm=arm, plan_sha256=plan.source_contract_sha256,
                initial_trainable_sha256=start,
                initial_optimizer_state_sha256=optimizer_start,
                executed_visits=plan.train_visits,
                executed_labels=dict(plan.expected_labels)[arm],
                effective_learning_rate=optimizer_learning_rate(arm, plan.nominal_learning_rate),
                forward_calls=8, backward_calls=8,
                optimizer_step_calls=0 if arm == "zero_update" else 8,
                elapsed_wall_milliseconds=100,
                cycle_zero_trainable_sha256=cycle0,
                cycle_one_start_trainable_sha256=cycle0,
                cycle_zero_optimizer_state_sha256=moments0,
                cycle_one_start_optimizer_state_sha256=moments0,
                final_trainable_sha256=final,
                final_optimizer_state_sha256=moments_final,
            ))
        cases = []
        for split in ("development", "transfer", "retention"):
            for j in range(20):
                case_id = retention_ids[j] if split == "retention" else f"c-{i}-{split}-{j}"
                cases.append(PrivateCase(
                    run_id=plan.run_id, case_id=case_id, split=split,
                    prompt=f"Private generated situation {i} {split} {j} ephemeral nonce-{i}-{split}-{j}",
                    gold=(i + j) % 2, structure_family=f"private-{split}-{j % 4}",
                ))
        prior = tuple((c.case_id, str(c.gold)) for c in cases if c.split == "retention")
        # The raw predictions are filled/signed only after pretraining and
        # training-completion source envelopes have been assembled.
        result.append(PrivateSeed(plan, tuple(claims), tuple(cases), (), prior))
    return tuple(result)


class SyntheticFixture:
    def __init__(self, exam_id="synthetic-raw-exam", seeds=None, completion_seq=11,
                 correct_retention_regressed=False, retention_seq=9):
        self.seeds = source_seeds() if seeds is None else seeds
        self.exam_id = exam_id
        self.eval_key = Ed25519PrivateKey.generate()
        self.train_key = Ed25519PrivateKey.generate()
        self.infer_key = Ed25519PrivateKey.generate()
        self.retention_key = Ed25519PrivateKey.generate()
        self.gold_key = b"examiner-only-secret-gold-key-32B!"
        self.seeds = tuple(replace(seed, signed_retention=sign_source_bytes(
            retention_bytes(run_id=seed.plan.run_id,
                            plan_sha256=seed.plan.source_contract_sha256,
                            predictions_sha256=prediction_commitment(seed.retention_before),
                            retention_sequence=retention_seq),
            self.retention_key, RETENTION_SCHEMA))
            for seed in self.seeds)
        self.exam = sign_source_bytes(precommit_bytes(
            self.seeds, exam_id=exam_id, examiner_hmac_key=self.gold_key,
            pretraining_sequence=10), self.eval_key, PRECOMMIT_SCHEMA)
        exam_digest = _digest(self.exam.public_bytes)
        self.completion = sign_source_bytes(training_completion_bytes(
            exam_id=exam_id, exam_sha256=exam_digest,
            claims_sha256=_claims_digest(self.seeds),
            training_end_sequence=completion_seq), self.train_key, COMPLETION_SCHEMA)
        completion_digest = _digest(self.completion.public_bytes)
        receipts = []
        final_seeds = []
        for seed in self.seeds:
            outputs = []
            for arm, claim in zip(PREREG_ARMS, seed.arm_claims):
                responses = tuple((
                    c.case_id,
                    str(1 - c.gold if (arm != "correct" or
                                      (correct_retention_regressed and arm == "correct" and
                                       c.split == "retention" and
                                       c.case_id == seed.plan.retention_case_ids[0]))
                        else c.gold)) for c in seed.cases)
                process_id = f"proc-{seed.plan.run_id}-{arm}"
                source = inference_bytes(
                    exam_sha256=exam_digest, completion_sha256=completion_digest,
                    run_id=seed.plan.run_id, arm=arm, process_id=process_id,
                    final_trainable_sha256=claim.final_trainable_sha256,
                    predictions_sha256=prediction_commitment(responses),
                    inference_sequence=completion_seq + 1)
                signature = sign_source_bytes(source, self.infer_key, PREDICTION_SCHEMA)
                receipts.append((seed.plan.run_id, arm, _digest(source)))
                outputs.append(RawArmPredictions(
                    seed.plan.run_id, arm, process_id, claim.final_trainable_sha256,
                    responses, signature))
            final_seeds.append(replace(seed, predictions=tuple(outputs)))
        self.seeds = tuple(final_seeds)
        self.pins = Q5TrustPins(
            _public_key(self.eval_key), _public_key(self.train_key),
            _public_key(self.infer_key), exam_digest, completion_digest,
            _digest(_public_key(self.eval_key)), tuple(receipts),
            frozenset(), frozenset(), _public_key(self.retention_key),
            tuple((seed.plan.run_id, _digest(seed.signed_retention.public_bytes))
                  for seed in self.seeds))

    def evaluate(self, ledger: Path):
        return evaluate_private_q5(
            seeds=self.seeds, exam=self.exam, completion=self.completion,
            pins=self.pins, examiner_hmac_key=self.gold_key,
            evaluator_private_key=self.eval_key, custodian_db_path=ledger)


class PrivateQ5R92Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = source_seeds()

    def _ledger(self, tmp):
        return Path(tmp) / "examiner-controlled-one-shot.db"

    def test_synthetic_raw_predictions_score_with_no_gold_or_case_rows_in_report(self):
        fixture = SyntheticFixture(seeds=self.example)
        with tempfile.TemporaryDirectory() as tmp:
            report = fixture.evaluate(self._ledger(tmp))
        outcome = verify_public_aggregate(
            report, pinned_evaluator_public_key=fixture.pins.examiner_public_key,
            expected_report_sha256=_digest(report.public_bytes),
            expected_exam_sha256=fixture.pins.expected_exam_sha256)
        self.assertTrue(outcome["source_screen_pass"])  # synthetic unit-test ONLY
        self.assertEqual(len(outcome["comparisons"]), 8)
        self.assertFalse(outcome["graduation_authorized"])
        self.assertFalse(outcome["host_training_and_inference_authenticated"])
        self.assertFalse(outcome["pretraining_temporal_custody_authenticated"])
        for private in (b"gold", b"Private generated situation", b"c-0-development",
                        b"ret-0-", b"responses", b"executed_labels", b"training_visits"):
            self.assertNotIn(private, report.public_bytes)

    def test_public_fixture_reuse_with_new_case_identity_is_rejected(self):
        s = self.example[0]
        changed = replace(s.cases[0], case_id="new-blind-id", prompt=HELDOUT[0][0])
        poisoned = (replace(s, cases=(changed,) + s.cases[1:]),) + self.example[1:]
        with self.assertRaisesRegex(Q5HoldoutError, "public"):
            bank_public_commitment(poisoned)

    def test_invisible_unicode_suffix_does_not_make_public_fixture_private(self):
        for suffix in ("\u200b", "\u2060", "\u034f", "\ufe0f"):
            first = self.example[0]
            changed = replace(first.cases[0], case_id="novel-private-id",
                              prompt=HELDOUT[0][0] + suffix)
            poisoned = (replace(first, cases=(changed,) + first.cases[1:]),) + self.example[1:]
            with self.subTest(suffix=ascii(suffix)):
                with self.assertRaises(Q5HoldoutError):
                    bank_public_commitment(poisoned)

    def test_unsigned_predecessor_retention_cannot_flip_rejection_into_pass(self):
        fixture = SyntheticFixture(seeds=self.example, correct_retention_regressed=True)
        with tempfile.TemporaryDirectory() as tmp:
            # The first retained previously-correct case is now incorrect.
            with self.assertRaisesRegex(Q5HoldoutError, "eight-comparison rule"):
                fixture.evaluate(Path(tmp) / "first-exam.db")
            old_exam = fixture.exam
            old_completion = fixture.completion
            old_inference = tuple(p.signed_inference for s in fixture.seeds
                                  for p in s.predictions)
            mutated = []
            for seed in fixture.seeds:
                before = list(seed.retention_before)
                before[0] = (before[0][0], str(1 - int(before[0][1])))
                mutated.append(replace(seed, retention_before=tuple(before)))
            fixture.seeds = tuple(mutated)
            self.assertEqual(fixture.exam, old_exam)
            self.assertEqual(fixture.completion, old_completion)
            self.assertEqual(tuple(p.signed_inference for s in fixture.seeds
                                   for p in s.predictions), old_inference)
            with self.assertRaises(Q5HoldoutError):
                fixture.evaluate(Path(tmp) / "second-exam.db")

    def test_missing_predecessor_retention_signature_is_rejected(self):
        fixture = SyntheticFixture(seeds=self.example)
        first = fixture.seeds[0]
        fixture.seeds = (replace(first, signed_retention=None),) + fixture.seeds[1:]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Q5HoldoutError, "signed predecessor retention"):
                fixture.evaluate(self._ledger(tmp))

    def test_altered_predecessor_signature_burns_one_shot_ledger(self):
        fixture = SyntheticFixture(seeds=self.example)
        first = fixture.seeds[0]
        receipt = first.signed_retention
        altered = replace(receipt, signature_hex="0" * 128)
        fixture.seeds = (replace(first, signed_retention=altered),) + fixture.seeds[1:]
        with tempfile.TemporaryDirectory() as tmp:
            ledger = self._ledger(tmp)
            with self.assertRaisesRegex(Q5HoldoutError, "signature"):
                fixture.evaluate(ledger)
            with self.assertRaisesRegex(Q5HoldoutError, "consumed"):
                fixture.evaluate(ledger)

    def test_predecessor_source_requires_independent_key_and_exact_receipt_pin(self):
        first = SyntheticFixture(seeds=self.example)
        first.pins = replace(first.pins, retention_public_key=first.pins.inference_public_key)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Q5HoldoutError, "distinct externally pinned"):
                first.evaluate(self._ledger(tmp))

        second = SyntheticFixture(seeds=self.example)
        changed = list(second.pins.expected_retention_sha256s)
        changed[0] = (changed[0][0], "f" * 64)
        second.pins = replace(second.pins, expected_retention_sha256s=tuple(changed))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Q5HoldoutError, "digest pin mismatch"):
                second.evaluate(self._ledger(tmp))

    def test_signed_predecessor_source_sequence_must_claim_pretraining(self):
        # Validly signed, independently pinned receipts with a sequence equal
        # to the pretraining source sequence are still too late to qualify.
        fixture = SyntheticFixture(seeds=self.example, retention_seq=10)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Q5HoldoutError, "signed predecessor retention"):
                fixture.evaluate(self._ledger(tmp))

    def test_prompt_reuse_between_seeds_even_with_distinct_ids_is_rejected(self):
        second = self.example[1]
        duplicate_prompt = replace(second.cases[0], prompt=self.example[0].cases[0].prompt.upper())
        poisoned = (self.example[0], replace(second, cases=(duplicate_prompt,) + second.cases[1:])) + self.example[2:]
        with self.assertRaisesRegex(Q5HoldoutError, "duplicate case or prompt"):
            bank_public_commitment(poisoned)

    def test_training_prompt_leakage_detected_even_when_case_id_differs(self):
        s = self.example[0]
        # Public source training inventory is normalized; deliberately make
        # the private prompt match one exact source-generated TRAIN example.
        from labyrinth.maze_conditioned_minilm_cpu_experiment import build_training_cycles
        cycle = build_training_cycles(replace(default_scope(), seed=SEEDS[0]))[0]
        changed = replace(s.cases[0], prompt=cycle.examples[0].prompt)
        poisoned = (replace(s, cases=(changed,) + s.cases[1:]),) + self.example[1:]
        with self.assertRaisesRegex(Q5HoldoutError, "TRAIN"):
            bank_public_commitment(poisoned)

    def test_other_seeds_training_prompt_is_not_blind_either(self):
        first = self.example[0]
        from labyrinth.maze_conditioned_minilm_cpu_experiment import build_training_cycles
        second_train = build_training_cycles(replace(default_scope(), seed=SEEDS[1]))[0]
        changed = replace(first.cases[0], prompt=second_train.examples[0].prompt)
        poisoned = (replace(first, cases=(changed,) + first.cases[1:]),) + self.example[1:]
        with self.assertRaisesRegex(Q5HoldoutError, "TRAIN"):
            bank_public_commitment(poisoned)

    def test_report_metric_forgery_and_added_private_labels_rejected(self):
        fixture = SyntheticFixture(seeds=self.example)
        with tempfile.TemporaryDirectory() as tmp:
            report = fixture.evaluate(self._ledger(tmp))
        forged = json.loads(report.public_bytes)
        forged["source_screen_pass"] = False
        from labyrinth.q5_holdout_eval_r92 import _canonical
        payload = _canonical(forged)
        # External hash updated to the attacker's bytes, but signature stays
        # attached to the original examiner output.
        with self.assertRaisesRegex(Q5HoldoutError, "signature"):
            verify_public_aggregate(
                SignedBytes(payload, report.signature_hex),
                pinned_evaluator_public_key=fixture.pins.examiner_public_key,
                expected_report_sha256=_digest(payload),
                expected_exam_sha256=fixture.pins.expected_exam_sha256)
        forged["gold"] = {"case": "1"}
        attacker = Ed25519PrivateKey.generate()
        envelope = sign_source_bytes(_canonical(forged), attacker, REPORT_SCHEMA)
        with self.assertRaises(Q5HoldoutError):
            verify_public_aggregate(envelope,
                                    pinned_evaluator_public_key=fixture.pins.examiner_public_key,
                                    expected_report_sha256=_digest(envelope.public_bytes),
                                    expected_exam_sha256=fixture.pins.expected_exam_sha256)

        # Even correctly signed bytes cannot add hidden labels, impossible
        # statistical p-values, or set an unearned authority flag.
        for key, value in (("gold", {"private": 1}),
                           ("graduation_authorized", True),
                           ("comparisons", [{**forged["comparisons"][0],
                                             "unadjusted_p": 0.0}] + forged["comparisons"][1:])):
            modified = json.loads(report.public_bytes)
            modified[key] = value
            resigned = sign_source_bytes(_canonical(modified), fixture.eval_key, REPORT_SCHEMA)
            with self.subTest(key=key), self.assertRaises(Q5HoldoutError):
                verify_public_aggregate(
                    resigned, pinned_evaluator_public_key=fixture.pins.examiner_public_key,
                    expected_report_sha256=_digest(resigned.public_bytes),
                    expected_exam_sha256=fixture.pins.expected_exam_sha256)

    def test_repeated_private_bank_is_one_shot_even_under_new_exam_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = self._ledger(tmp)
            one = SyntheticFixture(exam_id="first", seeds=self.example)
            one.evaluate(ledger)
            two = SyntheticFixture(exam_id="second", seeds=self.example)
            with self.assertRaisesRegex(Q5HoldoutError, "consumed"):
                two.evaluate(ledger)
            with self.assertRaisesRegex(Q5HoldoutError, "consumed"):
                one.evaluate(ledger)

    def test_adaptive_retry_after_bad_inference_signature_burns_exam(self):
        fixture = SyntheticFixture(seeds=self.example)
        first = fixture.seeds[0]
        malformed = replace(first.predictions[0], signed_inference=SignedBytes(
            first.predictions[0].signed_inference.public_bytes, "0" * 128))
        fixture.seeds = (replace(first, predictions=(malformed,) + first.predictions[1:]),) + fixture.seeds[1:]
        with tempfile.TemporaryDirectory() as tmp:
            ledger = self._ledger(tmp)
            with self.assertRaisesRegex(Q5HoldoutError, "signature"):
                fixture.evaluate(ledger)
            with self.assertRaisesRegex(Q5HoldoutError, "consumed"):
                fixture.evaluate(ledger)

    def test_unmatched_training_budget_rejected_even_if_trainer_changes_claims(self):
        fixture = SyntheticFixture(seeds=self.example)
        first = fixture.seeds[0]
        fake = replace(first.arm_claims[0], forward_calls=9)
        fixture.seeds = (replace(first, arm_claims=(fake,) + first.arm_claims[1:]),) + fixture.seeds[1:]
        # Signed training-completion digest binds the original claims.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Q5HoldoutError, "training completion"):
                fixture.evaluate(self._ledger(tmp))

        # An attacker who can re-sign *all* source envelopes still cannot
        # satisfy the deterministic budget rule with nine forward calls.
        changed = (replace(self.example[0], arm_claims=(fake,) +
                           self.example[0].arm_claims[1:]),) + self.example[1:]
        fully_resigned = SyntheticFixture(seeds=changed)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Q5HoldoutError, "unmatched source TRAIN budget"):
                fully_resigned.evaluate(self._ledger(tmp))

    def test_post_training_required_and_synthetic_order_is_not_host_attestation(self):
        fixture = SyntheticFixture(seeds=self.example, completion_seq=10)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Q5HoldoutError, "must follow"):
                fixture.evaluate(self._ledger(tmp))

    def test_external_evaluator_and_inference_pins_cannot_be_self_asserted(self):
        fixture = SyntheticFixture(seeds=self.example)
        fake_key = _public_key(Ed25519PrivateKey.generate())
        fixture.pins = replace(fixture.pins, examiner_public_key=fake_key,
                               expected_evaluator_key_sha256=_digest(fake_key))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Q5HoldoutError, "evaluator private key"):
                fixture.evaluate(self._ledger(tmp))

    def test_externally_reserved_prior_prompt_and_case_exclusion(self):
        fixture = SyntheticFixture(seeds=self.example)
        from labyrinth.q5_holdout_eval_r92 import _prompt_hash
        prior_case = self.example[0].cases[0]
        fixture.pins = replace(
            fixture.pins,
            excluded_prompt_sha256s=frozenset({_prompt_hash(prior_case.prompt)}))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Q5HoldoutError, "externally excluded"):
                fixture.evaluate(self._ledger(tmp))

    def test_private_gold_mutation_after_precommit_is_rejected(self):
        fixture = SyntheticFixture(seeds=self.example)
        first = fixture.seeds[0]
        changed = replace(first.cases[0], gold=1 - first.cases[0].gold)
        fixture.seeds = (replace(first, cases=(changed,) + first.cases[1:]),) + fixture.seeds[1:]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Q5HoldoutError, "precommit"):
                fixture.evaluate(self._ledger(tmp))

    def test_validly_signed_noncanonical_duplicate_or_overflow_json_rejected(self):
        original = SyntheticFixture(seeds=self.example)
        contents = original.exam.public_bytes
        attacks = (
            contents.replace(b'"phase":"BEFORE_TRAINING"',
                             b'"phase":"BEFORE_TRAINING","phase":"BEFORE_TRAINING"'),
            contents.replace(b'"maximum_attempts":1', b'"maximum_attempts":1e9999'),
        )
        for index, forged_bytes in enumerate(attacks):
            self.assertNotEqual(forged_bytes, contents)
            fixture = SyntheticFixture(seeds=self.example)
            fixture.exam = sign_source_bytes(forged_bytes, fixture.eval_key, PRECOMMIT_SCHEMA)
            fixture.pins = replace(fixture.pins, expected_exam_sha256=_digest(forged_bytes))
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(Q5HoldoutError):
                    fixture.evaluate(self._ledger(tmp))


if __name__ == "__main__":
    unittest.main()
