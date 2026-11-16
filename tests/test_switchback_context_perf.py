"""Hostile tests for narrowly scoped canonical-case fastpath (no global cache)."""

from contextvars import copy_context
from dataclasses import replace
from threading import Thread
import unittest
from unittest import mock

from labyrinth import switchback_circuit as dut
from labyrinth import switchback_oracle as oracle
from labyrinth.switchback_controls import MEMORIZED_SEED0_PATH
from labyrinth.switchback_perf_benchmark import measure


class ContextScopedValidationTests(unittest.TestCase):
    def test_profiled_plan_and_replay_remain_correct(self):
        receipt = measure()
        self.assertTrue(receipt["replay_valid"])
        self.assertEqual(receipt["verdict"], "VERIFIED_WIN")
        self.assertGreater(receipt["function_calls"]["oracle_step"], 0)

    def test_direct_oracle_step_still_validates_every_out_of_scope_call(self):
        case = dut.make_case(0)
        original = oracle._valid_case
        with mock.patch.object(oracle, "_valid_case", wraps=original) as validator:
            for _ in range(3):
                oracle.oracle_step(case, dut.State(case.initial), "WAIT")
            self.assertEqual(validator.call_count, 3)
        self.assertIsNone(oracle._PARITY_SCOPE.get())

    def test_copied_context_becomes_inactive_after_parity_returns(self):
        case = dut.make_case(0)
        original_step = dut.step
        copied = []

        def record_context(c, state, action):
            if not copied:
                copied.append(copy_context())
            return original_step(c, state, action)

        with mock.patch.object(dut, "step", side_effect=record_context):
            self.assertTrue(oracle.independent_transition_parity(case))
        self.assertEqual(len(copied), 1)
        inherited = copied[0].get(oracle._PARITY_SCOPE)
        self.assertIsNotNone(inherited)
        self.assertFalse(inherited.active)
        self.assertIsNone(oracle._PARITY_SCOPE.get())
        with mock.patch.object(oracle, "_valid_case", wraps=oracle._valid_case) as validator:
            copied[0].run(oracle.oracle_step, case, dut.State(case.initial), "WAIT")
            self.assertEqual(validator.call_count, 1)

    def test_wrong_case_with_same_seed_cannot_inherit_scope(self):
        case = dut.make_case(0)
        different = replace(case, target=(case.target + 1) % 8)
        original_step = dut.step
        checked = []

        def cross_case_attempt(c, state, action):
            if not checked:
                with self.assertRaises(ValueError):
                    oracle.oracle_step(different, state, action)
                checked.append(True)
            return original_step(c, state, action)

        with mock.patch.object(dut, "step", side_effect=cross_case_attempt):
            self.assertTrue(oracle.independent_transition_parity(case))
        self.assertEqual(checked, [True])

    def test_exact_same_case_in_place_mutation_invalidates_scope(self):
        case = dut.make_case(0)
        original_target = case.target
        original_step = dut.step
        altered = []

        def mutate_during_parity(c, state, action):
            if not altered:
                object.__setattr__(case, "target", (original_target + 1) % 8)
                altered.append(True)
            return original_step(c, state, action)

        try:
            with mock.patch.object(dut, "step", side_effect=mutate_during_parity):
                self.assertFalse(oracle.independent_transition_parity(case))
            self.assertEqual(altered, [True])
            self.assertIsNone(oracle._PARITY_SCOPE.get())
        finally:
            object.__setattr__(case, "target", original_target)

    def test_numeric_equality_must_not_launder_wrong_field_types(self):
        # Python compares False == 0 and 0.0 == 0. A raw-tuple fingerprint
        # would silently grant the corrupted Case canonical authority.
        for replacement in (False, 0.0):
            with self.subTest(replacement=repr(replacement)):
                case = dut.make_case(0)
                original_step = dut.step
                changed = []

                def corrupt_seed(c, state, action):
                    if not changed:
                        object.__setattr__(case, "seed", replacement)
                        changed.append(True)
                    return original_step(c, state, action)

                try:
                    with mock.patch.object(dut, "step", side_effect=corrupt_seed):
                        self.assertFalse(oracle.independent_transition_parity(case))
                    self.assertEqual(changed, [True])
                    self.assertIsNone(oracle._PARITY_SCOPE.get())
                finally:
                    object.__setattr__(case, "seed", 0)

    def test_rule_subclass_same_scalars_cannot_inherit_validated_scope(self):
        class DifferentRuleClass(dut.Rule):
            pass

        case = dut.make_case(0)
        original_before = case.before
        original_step = dut.step
        edge_count = []

        def swap_after_final_edge(c, state, action):
            evolved = original_step(c, state, action)
            edge_count.append(True)
            # Seed 0 has exactly 166 reachable states × four actions. Mutate
            # during the last edge, after initial truth-table validation.
            if len(edge_count) == 664:
                object.__setattr__(case, "before", DifferentRuleClass(
                    original_before.family, original_before.mask,
                    original_before.bias, original_before.threshold))
            return evolved

        try:
            with mock.patch.object(dut, "step", side_effect=swap_after_final_edge):
                self.assertFalse(oracle.independent_transition_parity(case))
            self.assertEqual(len(edge_count), 664)
            self.assertIsNone(oracle._PARITY_SCOPE.get())
        finally:
            object.__setattr__(case, "before", original_before)

    def test_kernel_prevalidation_side_effect_cannot_create_stale_scope(self):
        case = dut.make_case(0)
        original_seed = case.seed
        original_kernel = dut.kernel
        changed = []

        def mutate_during_truth_table(register, bit, rule):
            if not changed:
                object.__setattr__(case, "seed", original_seed + 1)
                changed.append(True)
            return original_kernel(register, bit, rule)

        try:
            with mock.patch.object(dut, "kernel", side_effect=mutate_during_truth_table):
                self.assertFalse(oracle.independent_transition_parity(case))
            self.assertEqual(changed, [True])
            self.assertIsNone(oracle._PARITY_SCOPE.get())
        finally:
            object.__setattr__(case, "seed", original_seed)

    def test_copied_context_cross_thread_does_not_inherit_trust(self):
        case = dut.make_case(0)
        original_step = dut.step
        observed = []

        def cross_thread(c, state, action):
            if not observed:
                context = copy_context()
                def worker():
                    with mock.patch.object(oracle, "_valid_case", wraps=oracle._valid_case) as check:
                        context.run(oracle.oracle_step, case, state, "WAIT")
                        observed.append(check.call_count)
                thread = Thread(target=worker)
                thread.start()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
            return original_step(c, state, action)

        with mock.patch.object(dut, "step", side_effect=cross_thread):
            self.assertTrue(oracle.independent_transition_parity(case))
        self.assertEqual(observed, [1])

    def test_exception_path_invalidates_and_resets_scope(self):
        case = dut.make_case(0)
        copied = []

        def crash(c, state, action):
            if not copied:
                copied.append(copy_context())
            raise RuntimeError("DUT transition crashed")

        with mock.patch.object(dut, "step", side_effect=crash):
            self.assertFalse(oracle.independent_transition_parity(case))
        self.assertFalse(copied[0].get(oracle._PARITY_SCOPE).active)
        self.assertIsNone(oracle._PARITY_SCOPE.get())

    def test_source_adjudication_and_replay_are_not_optimism_bias(self):
        case = dut.make_case(0)
        proposal = {"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)}
        receipt = dut.ProbeSession(case).submit(proposal)
        self.assertEqual(receipt["verdict"], "VERIFIED_WIN")
        self.assertTrue(oracle.verify_receipt_replay(receipt))
        self.assertIsNone(oracle._PARITY_SCOPE.get())


if __name__ == "__main__":
    unittest.main()
