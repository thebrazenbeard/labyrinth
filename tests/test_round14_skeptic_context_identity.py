"""Hostile scope tests for planned per-invocation case validation cache.

No process-wide or cross-invocation equivalence by identity may authorize a
changed forged Case. A correct optimization cannot mask a mid-call mutation.
"""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from threading import get_ident
import unittest
from unittest.mock import patch

from labyrinth import switchback_circuit as dut, switchback_oracle as oracle


class InvocationIdentityValidation(unittest.TestCase):
    @unittest.skipUnless(hasattr(oracle, "_PARITY_SCOPE"), "ContextVar optimization absent")
    def test_equal_value_wrong_type_cannot_reuse_scope(self):
        # In Python, False == 0 and 1.0 == 1. A tuple equality fingerprint
        # does not preserve the exact primitive-type invariant of _valid_case.
        for replacement in (False, 0.0):
            with self.subTest(replacement=repr(replacement)):
                case = dut.make_case(0)
                original_step = dut.step
                touched = False

                def replace_seed(world, state, action):
                    nonlocal touched
                    if not touched:
                        touched = True
                        object.__setattr__(case, "seed", replacement)
                    return original_step(world, state, action)

                try:
                    with patch.object(dut, "step", new=replace_seed):
                        parity = oracle.independent_transition_parity(case)
                    self.assertTrue(touched)
                    self.assertFalse(oracle._valid_case(case))
                    self.assertFalse(parity,
                                     "ContextVar fingerprint treated bool/float as exact int")
                finally:
                    object.__setattr__(case, "seed", 0)

    @unittest.skipUnless(hasattr(oracle, "_PARITY_SCOPE"), "ContextVar optimization absent")
    def test_exact_rule_subclass_swap_cannot_reuse_scope(self):
        # Typed leaf primitives still omit the enclosing Rule class itself.
        # A Rule subclass with identical field values must be noncanonical.
        case = dut.make_case(0)
        original_rule = case.before

        class FakeRule(dut.Rule):
            pass

        counts = []
        original_step = dut.step

        def count_step(world, state, action):
            counts.append(1)
            return original_step(world, state, action)

        with patch.object(dut, "step", new=count_step):
            self.assertTrue(oracle.independent_transition_parity(case))
        self.assertGreater(len(counts), 0)
        expected_final_call = len(counts)
        seen = 0

        def mutate_last_step(world, state, action):
            nonlocal seen
            result = original_step(world, state, action)
            seen += 1
            if seen == expected_final_call:
                object.__setattr__(case, "before", FakeRule(
                    original_rule.family, original_rule.mask,
                    original_rule.bias, original_rule.threshold))
            return result

        try:
            with patch.object(dut, "step", new=mutate_last_step):
                parity = oracle.independent_transition_parity(case)
            self.assertEqual(seen, expected_final_call)
            self.assertFalse(oracle._valid_case(case))
            self.assertFalse(parity,
                             "cache reused exact scalar leaves but ignored Rule subclass")
        finally:
            object.__setattr__(case, "before", original_rule)

    @unittest.skipUnless(hasattr(oracle, "_PARITY_SCOPE"), "ContextVar optimization absent")
    def test_same_identity_case_subclass_swap_cannot_reuse_scope(self):
        case = dut.make_case(0)
        original_step = dut.step

        class FakeCase(dut.Case):
            pass

        first_run_calls = []
        def count_step(world, state, action):
            first_run_calls.append(1)
            return original_step(world, state, action)

        with patch.object(dut, "step", new=count_step):
            self.assertTrue(oracle.independent_transition_parity(case))
        seen = 0

        def mutate_last_step(world, state, action):
            nonlocal seen
            result = original_step(world, state, action)
            seen += 1
            if seen == len(first_run_calls):
                object.__setattr__(case, "__class__", FakeCase)
            return result

        try:
            with patch.object(dut, "step", new=mutate_last_step):
                parity = oracle.independent_transition_parity(case)
            self.assertEqual(seen, len(first_run_calls))
            self.assertFalse(oracle._valid_case(case))
            self.assertFalse(parity,
                             "cached Case object identity ignored mutated class")
        finally:
            object.__setattr__(case, "__class__", dut.Case)

    @unittest.skipUnless(hasattr(oracle, "_PARITY_SCOPE"), "ContextVar optimization absent")
    def test_copied_context_cannot_reuse_attestation_after_parity(self):
        case = dut.make_case(0)
        copied = []
        original_step = dut.step

        def capture_step(world, state, action):
            if not copied:
                copied.append(copy_context())  # Scope is still active here.
            return original_step(world, state, action)

        with patch.object(dut, "step", new=capture_step):
            self.assertTrue(oracle.independent_transition_parity(case))
        self.assertEqual(len(copied), 1)
        with patch.object(oracle, "_valid_case", return_value=False):
            with self.assertRaises(ValueError):
                copied[0].run(oracle.oracle_step, case,
                              dut.State(case.initial), "WAIT")

    @unittest.skipUnless(hasattr(oracle, "_PARITY_SCOPE"), "ContextVar optimization absent")
    def test_copied_active_scope_cannot_cross_thread(self):
        case = dut.make_case(0)
        original_step = dut.step
        main_thread = get_ident()
        observations = []

        def observer_step(world, state, action):
            if not observations:
                copied = copy_context()
                original_valid = oracle._valid_case

                def worker():
                    def thread_gate(subject):
                        return (original_valid(subject) if get_ident() == main_thread
                                else False)
                    with patch.object(oracle, "_valid_case", new=thread_gate):
                        try:
                            copied.run(oracle.oracle_step, case,
                                       dut.State(case.initial), "WAIT")
                        except ValueError:
                            return "BLOCKED"
                        return "UNSAFE_CONTEXT_REUSE"

                with ThreadPoolExecutor(max_workers=1) as pool:
                    observations.append(pool.submit(worker).result(timeout=5))
            return original_step(world, state, action)

        with patch.object(dut, "step", new=observer_step):
            self.assertTrue(oracle.independent_transition_parity(case))
        self.assertEqual(observations, ["BLOCKED"])

    def test_successful_invocation_does_not_attest_future_mutated_case(self):
        case = dut.make_case(0)
        receipt = oracle.adjudicate(case, {"kind": "PLAN", "actions": ["COMMIT"]})
        self.assertFalse(receipt["verified_success"])
        original_signature = case.mechanic_signature
        try:
            object.__setattr__(case, "mechanic_signature", "f" * 64)
            self.assertFalse(oracle._valid_case(case))
            with self.assertRaises(ValueError):
                oracle.adjudicate(case, {"kind": "PLAN", "actions": ["COMMIT"]})
        finally:
            object.__setattr__(case, "mechanic_signature", original_signature)

    def test_nested_canonical_case_cannot_attest_forged_case(self):
        case = dut.make_case(0)
        forged = replace(case, target=(case.target + 2) % 8)
        original_kernel = dut.kernel
        triggered = False

        def nested_check(register, bit, rule):
            nonlocal triggered
            if not triggered:
                triggered = True
                with self.assertRaises(ValueError):
                    oracle.adjudicate(forged, {
                        "kind": "PLAN", "actions": ["COMMIT"]})
            return original_kernel(register, bit, rule)

        with patch.object(dut, "kernel", new=nested_check):
            receipt = oracle.adjudicate(case, {
                "kind": "PLAN", "actions": ["COMMIT"]})
        self.assertTrue(triggered)
        self.assertFalse(receipt["verified_success"])

    def test_distinct_case_same_seed_and_type_cannot_share_trust(self):
        valid = dut.make_case(0)
        forged = replace(valid, target=(valid.target + 1) % 8)
        self.assertTrue(oracle._valid_case(valid))
        self.assertFalse(oracle._valid_case(forged))
        self.assertTrue(oracle._valid_case(valid))

    def test_exception_does_not_keep_mutated_object_attested(self):
        case = dut.make_case(0)
        old_seed = case.seed
        with patch.object(oracle, "independent_transition_parity",
                          side_effect=RuntimeError("forced scoped abort")):
            with self.assertRaises(RuntimeError):
                oracle.adjudicate(case, {"kind": "PLAN", "actions": ["COMMIT"]})
        try:
            object.__setattr__(case, "seed", old_seed + 1)
            self.assertFalse(oracle._valid_case(case))
            with self.assertRaises(ValueError):
                oracle.adjudicate(case, {"kind": "PLAN", "actions": ["COMMIT"]})
        finally:
            object.__setattr__(case, "seed", old_seed)

    def test_thread_contexts_do_not_share_attestation(self):
        valid = dut.make_case(0)
        forged = replace(valid, mechanic_signature="0" * 64)
        def check(case):
            try:
                oracle.adjudicate(case, {"kind": "PLAN", "actions": ["COMMIT"]})
            except ValueError:
                return False
            return True
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(check, (valid, forged, valid, forged)))
        self.assertEqual(outcomes, [True, False, True, False])

    def test_mid_call_case_mutation_must_not_return_verified_success(self):
        case = dut.make_case(0)
        path = oracle.search(case)["witness"]
        self.assertIsNotNone(path)
        before_seed = case.seed
        original_kernel = dut.kernel
        touched = False

        def mutable_kernel(register, bit, rule):
            nonlocal touched
            if not touched:
                touched = True
                object.__setattr__(case, "seed", before_seed + 1)
            return original_kernel(register, bit, rule)

        try:
            with patch.object(dut, "kernel", new=mutable_kernel):
                try:
                    receipt = oracle.adjudicate(case, {
                        "kind": "PLAN", "actions": path})
                except (ValueError, RuntimeError):
                    receipt = {"verified_success": False}
            self.assertTrue(touched)
            self.assertFalse(receipt.get("verified_success"),
                             "identity-only cached Case admitted mutation during scoring")
        finally:
            object.__setattr__(case, "seed", before_seed)


if __name__ == "__main__":
    unittest.main()
