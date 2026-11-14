"""Hostile cross-invocation custody tests for per-invocation source validation."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from labyrinth import switchback_oracle as oracle


class InvocationContextValidationTests(unittest.TestCase):
    def test_same_seed_forged_case_not_whitelisted_by_prior_valid_call(self):
        valid = oracle.oracle_case(0)
        self.assertTrue(oracle._valid_case(valid))
        self.assertIn(oracle.adjudicate(valid, {"kind": "CLAIM_UNSAT"})["verdict"],
                      ("VERIFIED_FAILURE", "PROVEN_UNSAT_UNDER_DECLARED_RULES"))
        forged = replace(valid, target=(valid.target + 1) % 8)
        self.assertEqual(forged.seed, valid.seed)
        self.assertFalse(oracle._valid_case(forged))
        with self.assertRaises(ValueError):
            oracle.adjudicate(forged, {"kind": "CLAIM_UNSAT"})

    def test_changed_identity_requires_fresh_validation_not_just_prior_seed(self):
        original = oracle.oracle_case(1)
        self.assertTrue(oracle._valid_case(original))
        equal_copy = replace(original)
        self.assertIsNot(equal_copy, original)
        self.assertTrue(oracle._valid_case(equal_copy))
        self.assertIn(oracle.adjudicate(equal_copy, {"kind": "PLAN", "actions": ["COMMIT"]})
                      ["verdict"], ("VERIFIED_FAILURE", "UNKNOWN"))
        forged_copy = replace(equal_copy, switch_at=equal_copy.switch_at + 1)
        self.assertFalse(oracle._valid_case(forged_copy))
        with self.assertRaises(ValueError):
            oracle.adjudicate(forged_copy, {"kind": "PLAN", "actions": ["COMMIT"]})

    def test_prior_adjudication_cannot_cache_away_later_oracle_corruption(self):
        case = oracle.oracle_case(0)
        prior = oracle.adjudicate(case, {"kind": "CLAIM_UNSAT"})
        self.assertIn(prior["verdict"],
                      ("VERIFIED_FAILURE", "PROVEN_UNSAT_UNDER_DECLARED_RULES"))
        original_step = oracle.oracle_step

        def faulty_step(world, state, action):
            return state if state.ticks == 1 else original_step(world, state, action)

        with patch.object(oracle, "oracle_step", side_effect=faulty_step):
            corrupted = oracle.adjudicate(case, {"kind": "CLAIM_UNSAT"})
            self.assertEqual(corrupted["verdict"], "UNKNOWN")
            self.assertEqual(corrupted["review"]["reason"],
                             "INDEPENDENT_TRANSITION_DISAGREEMENT")
            self.assertFalse(oracle.verify_receipt_replay(prior))
        self.assertTrue(oracle.verify_receipt_replay(prior))

    def test_same_object_typed_seed_mutation_inside_invocation_is_rejected(self):
        for field_name in ("seed", "target", "switch_at"):
            with self.subTest(field_name=field_name):
                case = oracle.oracle_case(0)
                original_parity = oracle.independent_transition_parity

                def tamper_before_nested_validation(world):
                    self.assertIs(world, case)
                    original = getattr(world, field_name)
                    object.__setattr__(world, field_name, float(original))
                    try:
                        # False equals 0 and 2.0 equals 2, but different
                        # runtime types invalidate the canonical Case.
                        self.assertFalse(oracle._valid_case(world))
                        self.assertFalse(original_parity(world))
                    finally:
                        object.__setattr__(world, field_name, original)
                    return False

                with patch.object(oracle, "independent_transition_parity",
                                  side_effect=tamper_before_nested_validation):
                    result = oracle.adjudicate(case, {"kind": "CLAIM_UNSAT"})
                self.assertEqual(result["verdict"], "UNKNOWN")
                self.assertTrue(oracle._valid_case(case))

    def test_same_object_typed_nested_rule_mutation_inside_invocation_is_rejected(self):
        case = oracle.oracle_case(0)

        def tamper_before_nested_validation(world):
            original = world.before.family
            object.__setattr__(world.before, "family", float(original))
            try:
                self.assertFalse(oracle._valid_case(world))
            finally:
                object.__setattr__(world.before, "family", original)
            return False

        with patch.object(oracle, "independent_transition_parity",
                          side_effect=tamper_before_nested_validation):
            result = oracle.adjudicate(case, {"kind": "CLAIM_UNSAT"})
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertTrue(oracle._valid_case(case))


if __name__ == "__main__":
    unittest.main()
