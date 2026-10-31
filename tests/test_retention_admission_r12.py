"""Strict development-retention admission; synthetic signed evidence only."""

from dataclasses import replace
from hashlib import sha256
import unittest
from unittest.mock import patch

from labyrinth.journey_qualification import (
    JourneyQualificationPrecommit, PublicQualificationRequest,
)
from labyrinth.retention_entrypoint_guard import evaluate_retention_admission
from labyrinth.retention_family_registry import sign_family_taxonomy
from labyrinth.retention_qualification import (
    PROMPT_COMMITMENT_VERSION, RetentionObservation, RetentionPair,
    RetentionPolicy, _policy_sha256, normalized_prompt_sha256,
    retention_roster_sha256, sign_retention_development_evidence,
)
from labyrinth.retention_semantic_guard import require_distinct_arithmetic_retention
from labyrinth.retention_admission_r12 import evaluate_strict_retention_admission


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


EVAL_KEY = b"r12-independent-evaluator-key-abcdef123456"
TAX_KEY = b"r12-independent-taxonomy-key-abcdef123456"
NONCE = digest("r12-exam-taxonomy-nonce")
SIGNER = "r12-trusted-taxonomy-signer"


def pair(i, before, after):
    return RetentionPair(
        "arithmetic", f"situation-{i:04d}", f"family-{i:04d}",
        f"case-before-{i:04d}", f"case-after-{i:04d}", before, after,
        normalized_prompt_sha256(before), normalized_prompt_sha256(after),
    )


def diverse_roster():
    return tuple(pair(i, f"What is 0 plus {i}?", f"Compute 0 + {i}.")
                 for i in range(64))


def one_fact_roster():
    # All forms express the same 7+5 case. Deliberately many different IDs,
    # word aliases, operand order, stems and punctuation.
    forms = []
    for left, right in (("7", "5"), ("seven", "five"),
                        ("5", "7"), ("five", "seven"),
                        ("7", "five"), ("seven", "5"),
                        ("5", "seven"), ("five", "7")):
        for clause in (f"What is {left} plus {right}",
                       f"What is {left} + {right}",
                       f"Compute {left} plus {right}",
                       f"Compute {left} + {right}",
                       f"Add {left} to {right}",
                       f"{left} plus {right}", f"{left} + {right}"):
            forms.extend((clause, clause + "?", clause + "."))
    assert len(forms) >= 128 and len(set(forms[:128])) == 128
    return tuple(pair(i, forms[2 * i], forms[2 * i + 1])
                 for i in range(64))


def campaign(roster=None, *, evaluator_identity="evaluator-source", losses=0):
    roster = diverse_roster() if roster is None else roster
    scope, prior, candidate = (digest(value) for value in
                               ("r12-scope", "r12-prior", "r12-candidate"))
    training = "What is 99 plus 98?"
    policy = RetentionPolicy(
        "arithmetic", scope, prior, candidate, digest(evaluator_identity),
        digest("r12-evaluation-conditions"), "a" * 64, ("arithmetic",),
        retention_roster_sha256(roster), PROMPT_COMMITMENT_VERSION,
        (training,), (normalized_prompt_sha256(training),),
        ("r12-training-family",),
    )
    request = PublicQualificationRequest(
        "arithmetic", scope, candidate, digest("r12-entrant"),
        1, "development-1",
    )
    precommit = JourneyQualificationPrecommit(
        scope, "development-1", digest("r12-entrant"), 1,
    )
    observations = tuple(RetentionObservation(
        p.situation_id, True, i >= losses
    ) for i, p in enumerate(roster))
    signed = sign_retention_development_evidence(
        policy, roster, observations, independent_evaluator_key=EVAL_KEY,
    )
    taxonomy = sign_family_taxonomy(
        policy, roster, signer_id=SIGNER,
        precommitted_nonce_sha256=NONCE,
        independent_taxonomy_key=TAX_KEY,
        independent_evaluator_key=EVAL_KEY,
    )
    options = dict(
        trusted_journey_precommit=precommit,
        trusted_prior_checkpoint_sha256=prior,
        expected_policy_sha256=_policy_sha256(policy),
        expected_roster_sha256=retention_roster_sha256(roster),
        expected_semantic_inventory_sha256=(
            require_distinct_arithmetic_retention(policy, roster)
            .semantic_inventory_sha256 if roster == diverse_roster() else digest("one-fact-invalid")),
        expected_taxonomy_signer_id=SIGNER,
        expected_taxonomy_nonce_sha256=NONCE,
        independent_taxonomy_key=TAX_KEY,
        independent_evaluator_key=EVAL_KEY,
    )
    return (request, policy, roster, signed, taxonomy), options


class RetentionAdmissionR12Tests(unittest.TestCase):
    def test_diverse_signed_development_receipt_is_still_not_graduation(self):
        args, opts = campaign()
        # No development result may consume the separate stateful private
        # graduation campaign, regardless of its public development verdict.
        with patch("labyrinth.graduation_state.decide_campaign",
                   side_effect=AssertionError("private gate called")) as private_gate:
            result = evaluate_strict_retention_admission(*args, **opts)
        private_gate.assert_not_called()
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.retention.disposition, "PASS")
        self.assertEqual(result.strict_evidence.semantic_case_count, 64)
        self.assertFalse(result.graduation_authorized)
        self.assertFalse(result.private_exam_authorized)
        self.assertFalse(result.model_execution_authenticated)
        self.assertFalse(result.examiner_custody_verified)

    def test_old_entrypoint_passes_64_forms_of_7_plus_5_new_entrypoint_blocks(self):
        args, opts = campaign(one_fact_roster())
        legacy_opts = {key: value for key, value in opts.items()
                       if key not in {"expected_policy_sha256", "expected_roster_sha256",
                                      "expected_semantic_inventory_sha256"}}
        legacy = evaluate_retention_admission(*args, **legacy_opts)
        self.assertEqual(legacy.status, "PASS")
        result = evaluate_strict_retention_admission(*args, **opts)
        self.assertEqual(result.status, "BLOCKED")
        self.assertIsNone(result.retention)
        self.assertIsNone(result.strict_evidence)

    def test_operator_pinned_evaluator_commitment_cannot_be_rebound(self):
        args, opts = campaign()
        attacker_args, _ = campaign(evaluator_identity="attacker-replaced-evaluator")
        changed = evaluate_strict_retention_admission(*attacker_args, **opts)
        self.assertEqual(changed.status, "BLOCKED")
        self.assertIsNone(changed.retention)

    def test_missing_or_forged_taxonomy_signature_blocks(self):
        args, opts = campaign()
        self.assertEqual(evaluate_strict_retention_admission(
            *args, **(opts | {"independent_taxonomy_key": None})).status, "BLOCKED")
        modified = args[:4] + (replace(args[4], signature_hex="0" * 64),)
        self.assertEqual(evaluate_strict_retention_admission(
            *modified, **opts).status, "BLOCKED")

    def test_duplicate_case_identity_and_changed_scope_block(self):
        args, opts = campaign()
        damaged = (args[0], args[1],
                   args[2][:1] + (replace(args[2][1], before_case_id=args[2][0].before_case_id),)
                   + args[2][2:], args[3], args[4])
        self.assertEqual(evaluate_strict_retention_admission(
            *damaged, **opts).status, "BLOCKED")
        changed = (replace(args[0], scope_fingerprint=digest("changed-scope")),) + args[1:]
        self.assertEqual(evaluate_strict_retention_admission(
            *changed, **opts).status, "BLOCKED")

    def test_one_observed_loss_vetoes_even_on_distinct_facts(self):
        args, opts = campaign(losses=1)
        result = evaluate_strict_retention_admission(*args, **opts)
        self.assertEqual(result.status, "HOLD")
        self.assertEqual(result.retention.disposition, "HOLD")
        self.assertFalse(result.private_exam_authorized)

    def test_mismatched_semantic_pin_blocks_before_any_pass(self):
        args, opts = campaign()
        result = evaluate_strict_retention_admission(
            *args, **(opts | {"expected_semantic_inventory_sha256": digest("other-roster")}))
        self.assertEqual(result.status, "BLOCKED")
        self.assertIsNone(result.retention)


if __name__ == "__main__":
    unittest.main()
