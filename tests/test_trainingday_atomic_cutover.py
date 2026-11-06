"""Public synthetic R26 cutover recovery tests; never operate a deployed model."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import labyrinth.trainingday_atomic_cutover as cutover_module

from labyrinth.trainingday_atomic_cutover import (
    CUTOVER_STEPS, CutoverCandidate, CutoverInputError, ModelMinotaurBundle,
    bundle_commitment, rehearse_cutover,
)


PREVIOUS = ModelMinotaurBundle(
    "1" * 64, "2" * 64, "3" * 64, "4" * 64, 7,
)
NEXT = ModelMinotaurBundle(
    "5" * 64, "6" * 64, "7" * 64, "8" * 64, 8,
)
CANDIDATE = CutoverCandidate(
    transaction_id="rehearsal-8", expected_previous_commitment=bundle_commitment(PREVIOUS),
    proposed=NEXT, graduation_handoff_sha256="9" * 64,
)


class AtomicCutoverRehearsalTests(unittest.TestCase):
    def test_clean_synthetic_cutover_swaps_whole_bundle_only(self):
        result = rehearse_cutover(PREVIOUS, CANDIDATE)
        self.assertEqual(result.status, "SIMULATED_COMMIT_COMPLETE")
        self.assertEqual(result.active, NEXT)
        self.assertEqual(result.retained_previous, PREVIOUS)
        self.assertEqual(result.original_commitment, bundle_commitment(PREVIOUS))
        self.assertEqual(result.active_commitment, bundle_commitment(NEXT))
        self.assertEqual(result.steps, CUTOVER_STEPS)
        self.assertFalse(result.promotion_allowed)
        self.assertFalse(result.deployment_allowed)
        self.assertFalse(result.host_custody_verified)
        self.assertFalse(result.physical_rollback_verified)

    def test_each_injected_failure_rolls_back_all_three_surfaces(self):
        for step in CUTOVER_STEPS:
            with self.subTest(step=step):
                result = rehearse_cutover(PREVIOUS, CANDIDATE,
                                          injected_failure=step)
                self.assertEqual(result.status, "SIMULATED_ROLLBACK_COMPLETE")
                self.assertEqual(result.failure_step, step)
                self.assertEqual(result.active, PREVIOUS)
                self.assertEqual(result.retained_previous, PREVIOUS)
                self.assertEqual(result.staged_candidate, NEXT)
                self.assertEqual(result.active_commitment, bundle_commitment(PREVIOUS))
                self.assertTrue(result.synthetic_rollback_preserved)
                self.assertFalse(result.deployment_allowed)

    def test_post_swap_forward_recovery_keeps_all_components_coherent(self):
        for step in ("ATOMIC_SWAP", "POST_SWAP_VERIFY", "FINALIZE_JOURNAL"):
            with self.subTest(step=step):
                result = rehearse_cutover(PREVIOUS, CANDIDATE,
                                          injected_failure=step,
                                          recovery_policy="forward")
                self.assertEqual(result.status, "SIMULATED_FORWARD_RECOVERY")
                self.assertEqual(result.active, NEXT)
                self.assertEqual(result.retained_previous, PREVIOUS)
                self.assertEqual(result.active_commitment, bundle_commitment(NEXT))
                self.assertFalse(result.promotion_allowed)

    def test_pre_swap_failure_never_forwards_uncommitted_candidate(self):
        for step in ("VALIDATE_INPUT", "STAGE_MODEL", "STAGE_GATE",
                     "STAGE_EVIDENCE", "PRECOMMIT_VERIFY"):
            with self.subTest(step=step):
                result = rehearse_cutover(PREVIOUS, CANDIDATE,
                                          injected_failure=step,
                                          recovery_policy="forward")
                self.assertEqual(result.status, "SIMULATED_ROLLBACK_COMPLETE")
                self.assertEqual(result.active, PREVIOUS)

    def test_commitment_binds_model_adapter_gate_evidence_and_generation(self):
        for component, changed in (
            ("model", replace(PREVIOUS, model_sha256="a" * 64)),
            ("adapter", replace(PREVIOUS, adapter_sha256="b" * 64)),
            ("gate", replace(PREVIOUS, graduation_state_sha256="c" * 64)),
            ("evidence", replace(PREVIOUS, evidence_index_sha256="d" * 64)),
            ("generation", replace(PREVIOUS, generation=9)),
        ):
            with self.subTest(component=component):
                self.assertNotEqual(bundle_commitment(changed),
                                    bundle_commitment(PREVIOUS))
                with self.assertRaises(CutoverInputError):
                    rehearse_cutover(changed, CANDIDATE)

    def test_reject_stale_generation_noop_and_replay(self):
        for next_bundle in (
            replace(NEXT, generation=7),
            replace(NEXT, generation=9),
            replace(NEXT, adapter_sha256=PREVIOUS.adapter_sha256),
            replace(NEXT, graduation_state_sha256=PREVIOUS.graduation_state_sha256),
            replace(NEXT, evidence_index_sha256=PREVIOUS.evidence_index_sha256),
        ):
            with self.subTest(next_bundle=next_bundle):
                with self.assertRaises(CutoverInputError):
                    rehearse_cutover(PREVIOUS, replace(CANDIDATE, proposed=next_bundle))
        with self.assertRaises(CutoverInputError):
            rehearse_cutover(PREVIOUS, CANDIDATE,
                            consumed_transaction_ids=frozenset({"rehearsal-8"}))

    def test_malformed_inputs_fail_closed_and_are_never_promoted(self):
        for bad_previous in (
            replace(PREVIOUS, model_sha256="z" * 64),
            replace(PREVIOUS, generation=True),
            replace(PREVIOUS, generation=-1),
            replace(PREVIOUS, evidence_index_sha256=object()),
        ):
            with self.subTest(previous=bad_previous):
                with self.assertRaises(CutoverInputError):
                    rehearse_cutover(bad_previous, CANDIDATE)
        for bad_candidate in (
            replace(CANDIDATE, transaction_id="\u2603"),
            replace(CANDIDATE, expected_previous_commitment="f" * 64),
            replace(CANDIDATE, graduation_handoff_sha256="invalid"),
        ):
            with self.subTest(candidate=bad_candidate):
                with self.assertRaises(CutoverInputError):
                    rehearse_cutover(PREVIOUS, bad_candidate)
        for k in ("MISSING_STEP", "", "__dict__"):
            with self.assertRaises(CutoverInputError):
                rehearse_cutover(PREVIOUS, CANDIDATE, injected_failure=k)
        with self.assertRaises(CutoverInputError):
            rehearse_cutover(PREVIOUS, CANDIDATE, recovery_policy="deploy")

    def test_journal_commitment_changes_with_injected_failure_and_recovery(self):
        clean = rehearse_cutover(PREVIOUS, CANDIDATE)
        abort = rehearse_cutover(PREVIOUS, CANDIDATE,
                                injected_failure="POST_SWAP_VERIFY")
        forward = rehearse_cutover(PREVIOUS, CANDIDATE,
                                  injected_failure="POST_SWAP_VERIFY",
                                  recovery_policy="forward")
        self.assertEqual(len({clean.journal_sha256, abort.journal_sha256,
                              forward.journal_sha256}), 3)
        for result in (clean, abort, forward):
            self.assertEqual(len(result.journal_sha256), 64)
            self.assertEqual(result.original_commitment,
                             bundle_commitment(PREVIOUS))
            self.assertFalse(result.physical_rollback_verified)

    def test_subclass_mutable_digest_cannot_change_after_validation(self):
        class FickleModelBundle(ModelMinotaurBundle):
            """Override a frozen field getter to flip after the first commit check."""

            def __getattribute__(self, key):
                if key == "model_sha256":
                    reads = object.__getattribute__(self, "_model_reads") + 1
                    object.__setattr__(self, "_model_reads", reads)
                    return ("1" if reads <= 4 else "a") * 64
                return object.__getattribute__(self, key)

        fickle = FickleModelBundle(*PREVIOUS.__dict__.values())
        object.__setattr__(fickle, "_model_reads", 0)
        # Prior implementation returned SIMULATED_COMMIT_COMPLETE with a
        # receipt.original_commitment different from the accepted candidate
        # expected_previous_commitment. Reject caller-controlled subclasses.
        with self.assertRaises(CutoverInputError):
            rehearse_cutover(fickle, CANDIDATE)

        class DynamicCandidate(CutoverCandidate):
            def __getattribute__(self, key):
                if key == "expected_previous_commitment":
                    reads = object.__getattribute__(self, "_commitment_reads") + 1
                    object.__setattr__(self, "_commitment_reads", reads)
                    return (CANDIDATE.expected_previous_commitment
                            if reads <= 1 else "f" * 64)
                return object.__getattribute__(self, key)

        changing = DynamicCandidate(
            CANDIDATE.transaction_id, CANDIDATE.expected_previous_commitment,
            NEXT, CANDIDATE.graduation_handoff_sha256,
        )
        object.__setattr__(changing, "_commitment_reads", 0)
        with self.assertRaises(CutoverInputError):
            rehearse_cutover(PREVIOUS, changing)

        class CandidateSubclass(CutoverCandidate):
            pass

        with self.assertRaises(CutoverInputError):
            rehearse_cutover(PREVIOUS, CandidateSubclass(
                CANDIDATE.transaction_id, CANDIDATE.expected_previous_commitment,
                NEXT, CANDIDATE.graduation_handoff_sha256,
            ))

    def test_hostile_equality_objects_cannot_masquerade_as_phase_or_policy(self):
        class EqualAll:
            def __eq__(self, other):
                return True

        # This previously matched every name in tuple membership checks. A
        # forged phase yielded SIMULATED_ROLLBACK_COMPLETE; a forged policy
        # reached JSON journaling and raised uncaught TypeError.
        with self.assertRaises(CutoverInputError):
            rehearse_cutover(PREVIOUS, CANDIDATE,
                            injected_failure=EqualAll())
        with self.assertRaises(CutoverInputError):
            rehearse_cutover(PREVIOUS, CANDIDATE,
                            recovery_policy=EqualAll())
        with self.assertRaises(CutoverInputError):
            rehearse_cutover(PREVIOUS, CANDIDATE,
                            injected_failure=123)
        with self.assertRaises(CutoverInputError):
            rehearse_cutover(PREVIOUS, CANDIDATE,
                            recovery_policy=None)

    def test_post_validation_candidate_mutation_cannot_change_staged_receipt(self):
        candidate = replace(CANDIDATE, proposed=NEXT)
        clean = rehearse_cutover(PREVIOUS, candidate)
        mutated = replace(NEXT, generation=99,
                          model_sha256="a" * 64,
                          adapter_sha256="b" * 64)
        original_sha = cutover_module._canonical_sha
        calls = 0

        def mutate_during_journal(payload):
            nonlocal calls
            calls += 1
            if calls == 2:
                object.__setattr__(candidate, "proposed", mutated)
                object.__setattr__(candidate, "transaction_id", "invalid\ntransaction")
                object.__setattr__(candidate, "graduation_handoff_sha256", "invalid")
            return original_sha(payload)

        with patch.object(cutover_module, "_canonical_sha",
                          side_effect=mutate_during_journal):
            receipt = rehearse_cutover(PREVIOUS, candidate)

        self.assertEqual(receipt.active, receipt.staged_candidate)
        self.assertEqual(receipt.staged_candidate, NEXT)
        self.assertEqual(receipt.journal_sha256, clean.journal_sha256)
        self.assertEqual(receipt.active_commitment,
                         bundle_commitment(receipt.active))
        self.assertEqual(receipt.staged_candidate.generation, 8)

    def test_during_snapshot_bundle_mutation_fails_closed(self):
        proposed = replace(NEXT)
        candidate = replace(CANDIDATE, proposed=proposed)
        old_getattribute = ModelMinotaurBundle.__getattribute__
        changed = False

        def mutate_between_primitive_reads(bundle, name):
            nonlocal changed
            value = old_getattribute(bundle, name)
            if bundle is proposed and name == "model_sha256" and not changed:
                changed = True
                object.__setattr__(bundle, "model_sha256", "a" * 64)
                object.__setattr__(bundle, "adapter_sha256", "b" * 64)
                object.__setattr__(bundle, "graduation_state_sha256", "c" * 64)
                object.__setattr__(bundle, "evidence_index_sha256", "d" * 64)
            return value

        with patch.object(ModelMinotaurBundle, "__getattribute__",
                          mutate_between_primitive_reads):
            with self.assertRaisesRegex(CutoverInputError,
                                        "changed during synthetic snapshot"):
                rehearse_cutover(PREVIOUS, candidate)
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
