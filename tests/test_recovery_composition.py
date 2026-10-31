"""SOURCE-LEVEL composition tests; full neural checkpoint loading is MOCKED.

These tests exercise real MAC-sealed SQLite event/fence transitions, but do not
establish physical optimizer/RNG restoration, neural learning, or graduation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from labyrinth.hf_development_event_journal import (
    DevelopmentJournalError,
    complete_development_run,
    initialize_development_event_journal,
    inspect_development_event,
    reserve_development_event,
)
from labyrinth.neural_resume_checkpoint import FullCheckpointContext, FullCheckpointReceipt
from labyrinth.recovery_composition import (
    FencedCheckpointBinding,
    RecoveryCompositionError,
    commit_fenced_checkpoint,
    neural_checkpoint_loader,
    reservation_callback,
    restore_and_claim_same_run,
)


class MockTrustedFullLoader:
    """Receipt-only seam; deliberately does NOT load physical checkpoint bytes."""

    _VALID = object()

    def __init__(self):
        self.calls = []
        self.receipt_override = self._VALID
        self.fail = False

    def __call__(self, path, digest, context):
        self.calls.append((path, digest, context))
        if self.fail:
            raise ValueError("missing or corrupted checkpoint bytes")
        if self.receipt_override is not self._VALID:
            return self.receipt_override
        return FullCheckpointReceipt(digest, context, ("adapter.weight",))


class RecoveryCompositionSourceTests(TestCase):
    KEY = b"separate-host-journal-mac-key-only-32-bytes"
    INITIAL = "a" * 64
    COMMITMENT = "b" * 64
    BASE = "c" * 64
    NEXT = "d" * 64
    EXPOSURE_A = ("training-case-a", "e" * 64)
    EXPOSURE_B = ("training-case-b", "f" * 64)

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.journal = root / "host-events.sqlite"
        initialize_development_event_journal(self.journal, state_key=self.KEY)
        self.loader = MockTrustedFullLoader()
        context = FullCheckpointContext(
            run_id="neural-run-1", objective_id="math-reasoning",
            base_checkpoint_sha256=self.BASE,
            prior_checkpoint_sha256=self.BASE,
            fence=0, cursor=1,
            exposures=(self.EXPOSURE_A, self.EXPOSURE_B),
        )
        self.binding = FencedCheckpointBinding(
            event_id="signed-fail-1", run_id="neural-run-1",
            run_commitment_sha256=self.COMMITMENT,
            checkpoint_dir=root / "immutable-checkpoint-initial",
            checkpoint_sha256=self.INITIAL,
            checkpoint_context=context,
            journal_fence=0,
        )

    def state(self):
        return inspect_development_event(
            self.journal, self.binding.event_id, state_key=self.KEY,
        )

    def reserve(self):
        return reservation_callback(
            self.binding, journal_path=self.journal, state_key=self.KEY,
            checkpoint_loader=self.loader,
        )(self.binding.event_id)

    def resume(self, binding=None):
        return restore_and_claim_same_run(
            self.binding if binding is None else binding,
            journal_path=self.journal, state_key=self.KEY,
            checkpoint_loader=self.loader,
        )

    def next_context(self, current, *, cursor=2, exposures=None):
        return replace(
            current.checkpoint_context,
            prior_checkpoint_sha256=current.checkpoint_sha256,
            fence=current.journal_fence,
            cursor=cursor,
            exposures=(self.EXPOSURE_A, self.EXPOSURE_B) if exposures is None else exposures,
        )

    def commit(self, current, *, context=None, digest=None):
        return commit_fenced_checkpoint(
            current, next_checkpoint_dir=self.binding.checkpoint_dir.parent / "next-immutable",
            next_checkpoint_sha256=self.NEXT if digest is None else digest,
            next_context=self.next_context(current) if context is None else context,
            journal_path=self.journal, state_key=self.KEY,
            checkpoint_loader=self.loader,
        )

    def test_verified_event_callback_consumes_once_with_full_run_identity(self):
        self.assertTrue(self.reserve())
        state = self.state()
        self.assertEqual(state["run_id"], self.binding.run_id)
        self.assertEqual(state["run_commitment_sha256"], self.COMMITMENT)
        self.assertEqual(state["checkpoint_sha256"], self.INITIAL)
        self.assertEqual(state["fence"], 0)
        self.assertEqual(state["run_state"], "ACTIVE")
        self.assertEqual(self.loader.calls, [(
            self.binding.checkpoint_dir, self.INITIAL, self.binding.checkpoint_context,
        )])
        with self.assertRaisesRegex(DevelopmentJournalError, "already consumed"):
            self.reserve()
        self.assertFalse(reserve_development_event(
            self.journal, self.binding.event_id, state_key=self.KEY,
        ))

    def test_production_loader_factory_calls_strict_full_state_loader(self):
        model = object()
        optimizer = object()
        expected = FullCheckpointReceipt(
            self.INITIAL, self.binding.checkpoint_context, ("w",),
        )
        with patch("labyrinth.recovery_composition.load_full_checkpoint",
                   return_value=expected) as strict_loader:
            result = neural_checkpoint_loader(model=model, optimizer=optimizer)(
                self.binding.checkpoint_dir, self.INITIAL,
                self.binding.checkpoint_context,
            )
        self.assertIs(result, expected)
        strict_loader.assert_called_once_with(
            self.binding.checkpoint_dir, model=model, optimizer=optimizer,
            expected_sha256=self.INITIAL,
            expected_context=self.binding.checkpoint_context,
        )

    def test_mismatched_signed_event_never_loads_or_consumes(self):
        callback = reservation_callback(
            self.binding, journal_path=self.journal, state_key=self.KEY,
            checkpoint_loader=self.loader,
        )
        with self.assertRaisesRegex(RecoveryCompositionError, "signed event differs"):
            callback("another-signed-fail")
        self.assertFalse(self.state()["consumed"])
        self.assertFalse(self.loader.calls)

    def test_invalid_checkpoint_receipt_never_consumes(self):
        wrong_receipts = (
            FullCheckpointReceipt(self.NEXT, self.binding.checkpoint_context, ("w",)),
            FullCheckpointReceipt(self.INITIAL, replace(
                self.binding.checkpoint_context, cursor=0), ("w",)),
            FullCheckpointReceipt(self.INITIAL, self.binding.checkpoint_context, ()),
            FullCheckpointReceipt(self.INITIAL, self.binding.checkpoint_context,
                                  ("w",), graduation_authorized=True),
            None,
        )
        for wrong in wrong_receipts:
            with self.subTest(wrong=wrong):
                self.loader.receipt_override = wrong
                with self.assertRaisesRegex(RecoveryCompositionError, "loader did not attest"):
                    self.reserve()
                self.assertFalse(self.state()["consumed"])

    def test_loader_exception_does_not_consume(self):
        self.loader.fail = True
        with self.assertRaisesRegex(ValueError, "corrupted"):
            self.reserve()
        self.assertFalse(self.state()["consumed"])

    def test_context_run_mismatch_or_nonzero_initial_fence_is_denied(self):
        wrong_run = replace(
            self.binding, run_id="another-run",
        )
        with self.assertRaisesRegex(RecoveryCompositionError, "checkpoint run differs"):
            reservation_callback(
                wrong_run, journal_path=self.journal, state_key=self.KEY,
                checkpoint_loader=self.loader,
            )
        wrong_fence = replace(
            self.binding, checkpoint_context=replace(
                self.binding.checkpoint_context, fence=1), journal_fence=1,
        )
        with self.assertRaisesRegex(RecoveryCompositionError, "fence zero"):
            reservation_callback(
                wrong_fence, journal_path=self.journal, state_key=self.KEY,
                checkpoint_loader=self.loader,
            )
        self.assertFalse(self.state()["consumed"])

    def test_recovery_restores_exact_checkpoint_then_increments_fence(self):
        self.reserve()
        resumed = self.resume()
        self.assertEqual(resumed.journal_fence, 1)
        self.assertEqual(resumed.checkpoint_context.fence, 0)
        self.assertEqual(self.state()["fence"], 1)
        self.assertEqual(self.state()["checkpoint_sha256"], self.INITIAL)
        with self.assertRaisesRegex(RecoveryCompositionError, "mismatched"):
            self.resume()
        self.assertEqual(self.state()["fence"], 1)

    def test_resume_loader_failure_preserves_previous_fence(self):
        self.reserve()
        self.loader.fail = True
        with self.assertRaisesRegex(ValueError, "corrupted"):
            self.resume()
        self.assertEqual(self.state()["fence"], 0)
        self.assertTrue(self.state()["consumed"])

    def test_stale_or_other_run_binding_cannot_claim_or_commit(self):
        self.reserve()
        resumed = self.resume()
        wrong_commitment = replace(resumed, run_commitment_sha256=self.NEXT)
        with self.assertRaisesRegex(RecoveryCompositionError, "mismatched"):
            self.resume(wrong_commitment)
        with self.assertRaisesRegex(RecoveryCompositionError, "mismatched"):
            self.commit(self.binding)
        self.assertEqual(self.state()["checkpoint_sha256"], self.INITIAL)

    def test_new_checkpoint_advances_digest_without_releasing_consumed_event(self):
        self.reserve()
        resumed = self.resume()
        next_binding = self.commit(resumed)
        self.assertEqual(next_binding.checkpoint_sha256, self.NEXT)
        self.assertEqual(next_binding.checkpoint_context.fence, 1)
        self.assertEqual(next_binding.checkpoint_context.cursor, 2)
        self.assertEqual(self.state()["checkpoint_sha256"], self.NEXT)
        self.assertEqual(self.state()["fence"], 1)
        self.assertTrue(self.state()["consumed"])
        with self.assertRaisesRegex(DevelopmentJournalError, "already consumed"):
            self.reserve()
        resumed_again = self.resume(next_binding)
        self.assertEqual(resumed_again.journal_fence, 2)
        self.assertEqual(resumed_again.checkpoint_context.fence, 1)

    def test_regeneration_can_replace_pending_but_not_consumed_exposures(self):
        self.reserve()
        changed_pending = (self.EXPOSURE_A, ("new-pending-case", "1" * 64))
        next_binding = self.commit(
            self.binding, context=self.next_context(
                self.binding, cursor=2, exposures=changed_pending,
            ),
        )
        self.assertEqual(next_binding.checkpoint_context.exposures, changed_pending)
        self.assertEqual(self.state()["checkpoint_sha256"], self.NEXT)

    def test_forgotten_consumed_exposure_is_rejected_before_journal_mutation(self):
        self.reserve()
        changed_history = (("impostor-old-case", "1" * 64), self.EXPOSURE_B)
        with self.assertRaisesRegex(RecoveryCompositionError, "consumed exposures changed"):
            self.commit(self.binding, context=self.next_context(
                self.binding, exposures=changed_history,
            ))
        self.assertEqual(self.state()["checkpoint_sha256"], self.INITIAL)

    def test_cursor_rollback_and_prior_digest_change_are_denied(self):
        self.reserve()
        bad_contexts = (
            self.next_context(self.binding, cursor=0),
            replace(self.next_context(self.binding), prior_checkpoint_sha256=self.BASE),
            replace(self.next_context(self.binding), base_checkpoint_sha256=self.NEXT),
        )
        for context in bad_contexts:
            with self.subTest(context=context), self.assertRaises(RecoveryCompositionError):
                self.commit(self.binding, context=context)
        self.assertEqual(self.state()["checkpoint_sha256"], self.INITIAL)

    def test_commit_denied_if_full_loader_disagrees_with_new_digest(self):
        self.reserve()
        self.loader.receipt_override = FullCheckpointReceipt(
            self.INITIAL, self.next_context(self.binding), ("w",),
        )
        with self.assertRaisesRegex(RecoveryCompositionError, "loader did not attest"):
            self.commit(self.binding)
        self.assertEqual(self.state()["checkpoint_sha256"], self.INITIAL)

    def test_terminal_run_is_not_resumable_or_committable(self):
        self.reserve()
        complete_development_run(
            self.journal, self.binding.event_id, state_key=self.KEY,
            run_id=self.binding.run_id,
            run_commitment_sha256=self.binding.run_commitment_sha256,
            checkpoint_sha256=self.INITIAL, expected_fence=0,
            result_sha256="2" * 64,
        )
        with self.assertRaisesRegex(RecoveryCompositionError, "not ACTIVE"):
            self.resume()
        with self.assertRaisesRegex(RecoveryCompositionError, "not ACTIVE"):
            self.commit(self.binding)
        self.assertTrue(self.state()["consumed"])
        self.assertEqual(self.state()["run_state"], "COMPLETED")
        self.assertNotIn("graduation_authorized", self.state())
