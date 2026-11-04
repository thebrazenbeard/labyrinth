"""Source-only full-state successor ordering and custody boundary tests.

These intentionally mock the physical Torch/safetensors serialization boundary:
passing them does not establish that a pretrained model was resumed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.hf_development_event_journal import (
    FullStateRunBinding,
    claim_development_run_resume,
    initialize_development_event_journal,
    inspect_development_event,
)
from labyrinth.neural_resume_checkpoint import FullCheckpointError, FullCheckpointReceipt
from labyrinth.fullstate_successor import (
    DevelopmentStepReceipt,
    FullStateSuccessorError,
    hf_fullstate_training_step,
    run_fullstate_development_cycle,
)


class FullStateSuccessorContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "journal.sqlite3"
        self.key = b"journal-secret-for-tests-32-plus-bytes"
        initialize_development_event_journal(self.db, state_key=self.key)
        self.binding = FullStateRunBinding(
            run_id="run-fs-1", objective_id="program-reasoning",
            base_checkpoint_sha256="a" * 64,
            source_tree_sha256="b" * 64,
            maze_commitment_sha256="c" * 64,
            prior_checkpoint_sha256="d" * 64,
            initial_checkpoint_sha256="e" * 64,
            initial_cursor=1,
            initial_exposures=(("case-0", "f" * 64),
                               ("case-1", "1" * 64),
                               ("case-2", "2" * 64)),
        )
        self.event_id = "event-fs-1"
        self.initial = self.root / "initial"
        self.destination = self.root / "successor"

    def invoke(self, *, verify_event=None, attest_output=None, train=None,
               fingerprints=None):
        class CPUModel:
            def parameters(self):
                return (SimpleNamespace(device=SimpleNamespace(type="cpu")),)

            def named_buffers(self):
                return ()

        sequence = fingerprints if fingerprints is not None else [
            ("a" * 64, (0,), "f" * 64),
            ("b" * 64, (1,), "f" * 64),
            ("b" * 64, (1,), "f" * 64),
        ]
        with patch("labyrinth.fullstate_successor._neural_fingerprint",
                   side_effect=sequence, create=True):
            return run_fullstate_development_cycle(
                binding=self.binding,
                event_id=self.event_id,
                journal_db=self.db,
                state_key=self.key,
                initial_checkpoint_dir=self.initial,
                successor_checkpoint_dir=self.destination,
                model_factory=CPUModel,
                optimizer_factory=lambda _model: object(),
                training_step=train or (lambda _model, _optimizer, planned, cursor:
                    DevelopmentStepReceipt(tuple(planned), len(planned),
                                           cursor, cursor + len(planned))),
                step_count=1,
                verify_signed_event=verify_event,
                attest_successor_bytes=attest_output,
            )

    def test_missing_host_authorization_denies_before_model_and_journal(self):
        with self.assertRaisesRegex(FullStateSuccessorError, "signed.*host"):
            self.invoke(verify_event=None, attest_output=lambda *_: True)
        self.assertFalse(inspect_development_event(self.db, self.event_id,
                                                  state_key=self.key)["consumed"])

    def test_load_train_publish_readback_and_cas_in_order(self):
        order: list[str] = []

        def load(_path, *, model, optimizer, expected_sha256, expected_context):
            order.append("load_initial" if expected_sha256 == "e" * 64 else "readback")
            return FullCheckpointReceipt(expected_sha256, expected_context, ("weight",))

        def save(_path, *, model, optimizer, context):
            order.append("save_successor")
            self.assertEqual(context.cursor, 2)
            self.assertEqual(context.prior_checkpoint_sha256, "e" * 64)
            return FullCheckpointReceipt("9" * 64, context, ("weight",))

        def train(_model, _optimizer, planned, cursor):
            order.append("train")
            self.assertEqual(planned, (("case-1", "1" * 64),))
            self.assertEqual(cursor, 1)
            return DevelopmentStepReceipt(tuple(planned), 1, cursor, cursor + 1)

        def verify(_event, _binding):
            order.append("verify_host_event")
            return True

        def attest(_path, _receipt):
            order.append("attest_bytes")
            return True

        with (patch("labyrinth.fullstate_successor.load_full_checkpoint", side_effect=load),
              patch("labyrinth.fullstate_successor.save_full_checkpoint", side_effect=save)):
            receipt = self.invoke(verify_event=verify, attest_output=attest, train=train)
        self.assertEqual(order, ["verify_host_event", "load_initial", "train",
                                 "save_successor", "readback", "attest_bytes"])
        self.assertEqual(receipt.status, "PROVISIONAL_SOURCE_ONLY")
        self.assertFalse(receipt.graduation_authorized)
        self.assertEqual(receipt.next_cursor, 2)
        record = inspect_development_event(self.db, self.event_id, state_key=self.key)
        self.assertEqual(record["checkpoint_sha256"], "9" * 64)

    def test_forged_training_exposures_never_publish_or_advance(self):
        def load(_path, *, expected_sha256, expected_context, **_):
            return FullCheckpointReceipt(expected_sha256, expected_context, ("weight",))

        def wrong_train(_model, _optimizer, _planned, cursor):
            return DevelopmentStepReceipt((("alien", "8" * 64),), 1,
                                          cursor, cursor + 1)

        with (patch("labyrinth.fullstate_successor.load_full_checkpoint", side_effect=load),
              patch("labyrinth.fullstate_successor.save_full_checkpoint") as save):
            with self.assertRaisesRegex(FullStateSuccessorError, "exposure"):
                self.invoke(verify_event=lambda *_: True,
                            attest_output=lambda *_: True, train=wrong_train)
            save.assert_not_called()
        record = inspect_development_event(self.db, self.event_id, state_key=self.key)
        self.assertEqual(record["checkpoint_sha256"], "e" * 64)

    def test_missing_independent_output_attestation_keeps_orphan_uncommitted(self):
        def load(_path, *, expected_sha256, expected_context, **_):
            return FullCheckpointReceipt(expected_sha256, expected_context, ("weight",))

        def save(_path, *, context, **_):
            return FullCheckpointReceipt("9" * 64, context, ("weight",))

        with (patch("labyrinth.fullstate_successor.load_full_checkpoint", side_effect=load),
              patch("labyrinth.fullstate_successor.save_full_checkpoint", side_effect=save)):
            with self.assertRaisesRegex(FullStateSuccessorError, "attestation"):
                self.invoke(verify_event=lambda *_: True, attest_output=lambda *_: False)
        record = inspect_development_event(self.db, self.event_id, state_key=self.key)
        self.assertEqual(record["checkpoint_sha256"], "e" * 64)

    def test_partial_initial_restore_aborts_without_consuming_event(self):
        with patch("labyrinth.fullstate_successor.load_full_checkpoint",
                   side_effect=FullCheckpointError("partial restored state")):
            with self.assertRaisesRegex(FullStateSuccessorError, "discard this process"):
                self.invoke(verify_event=lambda *_: True,
                            attest_output=lambda *_: True)
        self.assertFalse(inspect_development_event(self.db, self.event_id,
                                                  state_key=self.key)["consumed"])

    def test_stale_worker_fence_never_commits_orphan(self):
        def load(_path, *, expected_sha256, expected_context, **_):
            return FullCheckpointReceipt(expected_sha256, expected_context, ("weight",))

        def save(_path, *, context, **_):
            return FullCheckpointReceipt("9" * 64, context, ("weight",))

        def invalidate_fence(_path, _receipt):
            fresh = claim_development_run_resume(
                self.db, self.event_id, state_key=self.key,
                run_id=self.binding.run_id,
                run_commitment_sha256=self.binding.commitment_sha256(),
                checkpoint_sha256=self.binding.initial_checkpoint_sha256,
                expected_fence=0,
            )
            self.assertEqual(fresh, 1)
            return True

        with (patch("labyrinth.fullstate_successor.load_full_checkpoint", side_effect=load),
              patch("labyrinth.fullstate_successor.save_full_checkpoint", side_effect=save)):
            with self.assertRaisesRegex(FullStateSuccessorError, "fence mismatch"):
                self.invoke(verify_event=lambda *_: True, attest_output=invalidate_fence)
        record = inspect_development_event(self.db, self.event_id, state_key=self.key)
        self.assertEqual(record["checkpoint_sha256"], "e" * 64)
        self.assertEqual(record["fence"], 1)

    def test_model_with_uncheckpointed_buffer_is_rejected_before_reservation(self):
        class ModelWithBuffer:
            def parameters(self):
                return (SimpleNamespace(device=SimpleNamespace(type="cpu")),)

            def named_buffers(self):
                return (("running_mean", object()),)

        with patch("labyrinth.fullstate_successor.load_full_checkpoint") as load:
            with self.assertRaisesRegex(FullStateSuccessorError, "uncheckpointed model buffers"):
                run_fullstate_development_cycle(
                    binding=self.binding, event_id=self.event_id,
                    journal_db=self.db, state_key=self.key,
                    initial_checkpoint_dir=self.initial,
                    successor_checkpoint_dir=self.destination,
                    model_factory=ModelWithBuffer,
                    optimizer_factory=lambda _model: object(),
                    training_step=lambda *_: DevelopmentStepReceipt((), 0, 0, 0),
                    step_count=1,
                    verify_signed_event=lambda *_: True,
                    attest_successor_bytes=lambda *_: True,
                )
            load.assert_not_called()
        self.assertFalse(inspect_development_event(self.db, self.event_id,
                                                  state_key=self.key)["consumed"])

    def test_context_only_checkpoint_change_cannot_launder_noop_training(self):
        def load(_path, *, expected_sha256, expected_context, **_):
            return FullCheckpointReceipt(expected_sha256, expected_context, ("weight",))

        def save(_path, *, context, **_):
            # A new cursor/parent SHA alone changes the manifest digest.
            return FullCheckpointReceipt("9" * 64, context, ("weight",))

        with (patch("labyrinth.fullstate_successor.load_full_checkpoint", side_effect=load),
              patch("labyrinth.fullstate_successor.save_full_checkpoint", side_effect=save)):
            with self.assertRaisesRegex(FullStateSuccessorError, "neural|optimizer"):
                self.invoke(verify_event=lambda *_: True,
                            attest_output=lambda *_: True,
                            fingerprints=[("a" * 64, (0,), "f" * 64),
                                          ("a" * 64, (0,), "f" * 64)])
        record = inspect_development_event(self.db, self.event_id, state_key=self.key)
        self.assertEqual(record["checkpoint_sha256"], "e" * 64)

    def test_trainer_cannot_reset_or_skip_custodied_exposure_cursor(self):
        def load(_path, *, expected_sha256, expected_context, **_):
            return FullCheckpointReceipt(expected_sha256, expected_context, ("weight",))

        def reset_cursor(_model, _optimizer, planned, _cursor):
            return DevelopmentStepReceipt(tuple(planned), 1, 0, 1)

        with (patch("labyrinth.fullstate_successor.load_full_checkpoint", side_effect=load),
              patch("labyrinth.fullstate_successor.save_full_checkpoint") as save):
            with self.assertRaisesRegex(FullStateSuccessorError, "cursor"):
                self.invoke(verify_event=lambda *_: True,
                            attest_output=lambda *_: True, train=reset_cursor)
            save.assert_not_called()

    def test_mutated_frozen_base_cannot_commit_as_full_state(self):
        def load(_path, *, expected_sha256, expected_context, **_):
            return FullCheckpointReceipt(expected_sha256, expected_context, ("weight",))

        with (patch("labyrinth.fullstate_successor.load_full_checkpoint", side_effect=load),
              patch("labyrinth.fullstate_successor.save_full_checkpoint") as save):
            with self.assertRaisesRegex(FullStateSuccessorError, "frozen"):
                self.invoke(verify_event=lambda *_: True,
                            attest_output=lambda *_: True,
                            fingerprints=[("a" * 64, (0,), "f" * 64),
                                          ("b" * 64, (1,), "0" * 64)])
            save.assert_not_called()

    def test_hf_training_step_glues_owned_optimizer_and_global_cursor(self):
        from types import SimpleNamespace

        train = [SimpleNamespace(id="case-1", prompt="P", answer="A")]
        config = SimpleNamespace(steps=1, device="cpu")
        planned = (("case-1", "1" * 64),)

        def fake_hf(_model, _tokenizer, _torch, cases, labels, config_arg, **kwargs):
            self.assertIs(cases, train)
            self.assertEqual(labels, ["A"])
            self.assertIs(config_arg, config)
            self.assertEqual(kwargs["rng_policy"], "preserve")
            self.assertEqual(kwargs["visit_cursor"], 1)
            self.assertEqual(kwargs["expected_visit_exposures"], planned)
            self.assertEqual(kwargs["scheduler"], None)
            self.assertIs(kwargs["optimizer"], optimizer)
            return {"steps": 1, "visit_cursor_before": 1,
                    "visit_cursor_after": 2, "visited_exposures": list(planned)}

        optimizer = object()
        adapter = hf_fullstate_training_step(
            tokenizer=object(), torch=object(), train=train,
            labels=["A"], config=config,
        )
        with patch("labyrinth.hf_pilot._train_steps", side_effect=fake_hf):
            receipt = adapter(object(), optimizer, planned, 1)
        self.assertEqual(receipt, DevelopmentStepReceipt(planned, 1, 1, 2))

    def test_hf_training_step_rejects_cuda_before_training(self):
        adapter = hf_fullstate_training_step(
            tokenizer=object(), torch=object(), train=[], labels=[],
            config=SimpleNamespace(steps=1, device="cuda"),
        )
        with patch("labyrinth.hf_pilot._train_steps") as train:
            with self.assertRaisesRegex(FullStateSuccessorError, "CPU"):
                adapter(object(), object(), (("case-0", "a" * 64),), 0)
            train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
