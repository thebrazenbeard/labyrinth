"""Reject post-construction policy mutation at both retention trust boundaries."""

from dataclasses import asdict, replace
from hashlib import sha256
import hmac
import json
import unittest

from labyrinth.retention_qualification import (
    RetentionError,
    SignedRetentionEvidence,
    decide_retention,
    sign_retention_development_evidence,
)
import test_retention_qualification as retention_fixtures


KEY = b"independent-retention-evaluator-key-12345678"


def _campaign_with_one_loss():
    policy, roster, observed = retention_fixtures.RetentionQualificationTests().campaign()
    return policy, roster, (replace(observed[0], after_correct=False), *observed[1:])


def _canonical(data):
    return (json.dumps(data, sort_keys=True, separators=(",", ":"),
                       allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")


class RetentionPolicyMutationTests(unittest.TestCase):
    def test_signer_refuses_post_construction_relaxed_deterioration_margin(self):
        policy, roster, observed = _campaign_with_one_loss()
        object.__setattr__(policy, "max_deterioration_rate", 1.0)
        with self.assertRaisesRegex(RetentionError, "invalid retention policy"):
            sign_retention_development_evidence(
                policy, roster, observed, independent_evaluator_key=KEY,
            )

    def test_other_post_construction_policy_weakening_is_rejected(self):
        for field, invalid in (
            ("significance_alpha", 0.50),
            ("min_before_accuracy", 0.10),
            ("min_units_per_skill", 1),
        ):
            with self.subTest(field=field):
                policy, roster, observed = _campaign_with_one_loss()
                object.__setattr__(policy, field, invalid)
                with self.assertRaisesRegex(RetentionError, "invalid retention policy"):
                    sign_retention_development_evidence(
                        policy, roster, observed, independent_evaluator_key=KEY,
                    )

    def test_instance_shadowed_validator_cannot_hide_invalid_margin(self):
        policy, roster, observed = _campaign_with_one_loss()
        object.__setattr__(policy, "max_deterioration_rate", 1.0)
        object.__setattr__(policy, "__post_init__", lambda: None)
        with self.assertRaisesRegex(RetentionError, "invalid retention policy"):
            sign_retention_development_evidence(
                policy, roster, observed, independent_evaluator_key=KEY,
            )

    def test_decider_refuses_authentically_signed_invalid_policy(self):
        policy, roster, observed = _campaign_with_one_loss()
        valid = sign_retention_development_evidence(
            policy, roster, observed, independent_evaluator_key=KEY,
        )
        object.__setattr__(policy, "max_deterioration_rate", 1.0)
        body = json.loads(valid.receipt_bytes)
        body["policy_sha256"] = sha256(
            b"labyrinth.prior-skill-retention-policy.v2\0"
            + _canonical(asdict(policy))
        ).hexdigest()
        raw = _canonical(body)
        signed = SignedRetentionEvidence(
            raw,
            hmac.new(KEY, b"labyrinth.prior-skill-retention-development.v2\0" + raw,
                     sha256).hexdigest(),
        )
        with self.assertRaisesRegex(RetentionError, "invalid retention policy"):
            decide_retention(policy, roster, signed, independent_evaluator_key=KEY)

    def test_valid_precommitted_policy_still_holds_one_deterioration(self):
        policy, roster, observed = _campaign_with_one_loss()
        signed = sign_retention_development_evidence(
            policy, roster, observed, independent_evaluator_key=KEY,
        )
        result = decide_retention(policy, roster, signed, independent_evaluator_key=KEY)
        self.assertEqual(result.disposition, "HOLD")
        self.assertFalse(result.graduation_authorized)


if __name__ == "__main__":
    unittest.main()
