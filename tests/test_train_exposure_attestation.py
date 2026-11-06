"""Hostile source-only TRAIN inventory completeness checks, not physical proof."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import unittest

from labyrinth.train_exposure_attestation import (
    SignedTrainCapture, SignedTrainSchedule, TrainExposureEvent,
    TrainingExposureAttestationError, TrainingWitnessContext,
    capture_training_visits, precommit_training_schedule,
    verify_complete_train_inventory,
)
from labyrinth.sealed_semantic_exam_protocol import training_inventory_sha256


SCHEDULE_KEY = b"independent-training-schedule-plan-key-v1"
RECORDER_KEY = b"independent-training-event-recorder-key-v1"


def fixture():
    context = TrainingWitnessContext(
        logical_run_id="neural-seed-301", objective_id="sentiment.binary_polarity",
        scope_sha256="a" * 64, base_checkpoint_sha256="b" * 64,
        source_sha256="c" * 64, fence=21, cycles=2,
        steps_per_cycle=2, visits_per_step=2,
    )
    cases = (
        ("case-arith", "arith-five-seven", "arithmetic-train",
         "Compute seven plus five."),
        ("case-pos", "sentiment-positive", "sentiment-train",
         "This person was helpful."),
        ("case-neg", "sentiment-negative", "sentiment-train",
         "The instructions were confusing."),
        ("case-kind", "sentiment-kind", "sentiment-train",
         "They acted thoughtfully."),
    )
    events = tuple(TrainExposureEvent(
        ordinal=i, cycle=i // 4, step=(i // 2) % 2, visit=i % 2,
        event_id=f"train-event-{i}",
        case_id=cases[i % 4][0], semantic_key=cases[i % 4][1],
        family_id=cases[i % 4][2], prompt=cases[i % 4][3],
    ) for i in range(8))
    return context, events


def signed_fixture(context=None, events=None):
    original_context, original_events = fixture()
    context = original_context if context is None else context
    events = original_events if events is None else events
    schedule = precommit_training_schedule(
        context, events, schedule_key=SCHEDULE_KEY)
    pinned = sha256(schedule.receipt_bytes).hexdigest()
    capture = capture_training_visits(
        context, events, schedule_sha256=pinned,
        checkpoint_sha256="d" * 64, recorder_key=RECORDER_KEY)
    return context, events, schedule, pinned, capture


class TrainingExposureAttestationTests(unittest.TestCase):
    def verify(self, context=None, events=None, schedule=None, pinned=None,
               capture=None, **kwargs):
        c, e, s, p, v = signed_fixture()
        return verify_complete_train_inventory(
            context=c if context is None else context,
            scheduled=e if events is None else events,
            observed=e if "observed" not in kwargs else kwargs.pop("observed"),
            signed_schedule=s if schedule is None else schedule,
            signed_capture=v if capture is None else capture,
            externally_pinned_schedule_sha256=p if pinned is None else pinned,
            expected_checkpoint_sha256="d" * 64,
            schedule_key=SCHEDULE_KEY, recorder_key=RECORDER_KEY,
            **kwargs,
        )

    def test_complete_two_cycle_stream_yields_exact_sealed_exam_inventory(self):
        verified = self.verify()
        inventory = verified.inventory
        self.assertEqual(inventory.family_ids,
                         ("arithmetic-train", "sentiment-train"))
        self.assertEqual(len(inventory.prompt_alias_sha256s), 4)
        self.assertEqual(len(inventory.semantic_keys), 4)
        self.assertEqual(inventory.arithmetic_semantic_keys, ("add:5:7",))
        self.assertEqual(verified.inventory_sha256,
                         training_inventory_sha256(inventory))
        self.assertEqual(verified.checkpoint_sha256, "d" * 64)
        self.assertEqual(len(verified.schedule_sha256), 64)
        self.assertEqual(len(verified.capture_sha256), 64)
        self.assertFalse(verified.physical_execution_authenticated)
        self.assertFalse(verified.independent_host_verified)
        self.assertFalse(verified.graduation_authorized)

    def test_receipts_do_not_expose_training_prompts_or_case_ids(self):
        _, _, schedule, _, capture = signed_fixture()
        schedule_body = json.loads(schedule.receipt_bytes)
        capture_body = json.loads(capture.receipt_bytes)
        self.assertEqual(schedule_body["ordered_family_roster_sha256"],
                         capture_body["ordered_family_roster_sha256"])
        self.assertEqual(len(schedule_body["ordered_family_roster_sha256"]), 64)
        for private in ("Compute seven plus five", "case-arith",
                        "This person was helpful", "train-event-7"):
            self.assertNotIn(private, schedule.receipt_bytes.decode())
            self.assertNotIn(private, capture.receipt_bytes.decode())
        self.assertFalse(schedule.independently_hosted)
        self.assertFalse(capture.physically_authenticated)

    def test_any_omission_duplicate_reorder_or_wrong_step_is_refused(self):
        context, events, _, _, _ = signed_fixture()
        cases = (
            events[:-1],
            events[:3] + (events[2],) + events[4:],
            events[:2] + events[3:4] + events[2:3] + events[4:],
            events[:4] + (replace(events[4], step=1),) + events[5:],
            events[:5] + (replace(events[5], cycle=0),) + events[6:],
            events[:3] + (replace(events[3], visit=True),) + events[4:],
            events[:7] + (replace(events[7], event_id=events[0].event_id),),
        )
        for damaged in cases:
            with self.subTest(damaged=damaged), self.assertRaises(
                    TrainingExposureAttestationError):
                precommit_training_schedule(
                    context, damaged, schedule_key=SCHEDULE_KEY)

    def test_forged_or_changed_observation_refused_against_pinned_schedule(self):
        context, events, schedule, pinned, _ = signed_fixture()
        for observed in (
            events[:4] + (replace(events[4], prompt="A completely new prompt."),)
            + events[5:],
            events[:4] + (replace(events[4], event_id="fresh-id"),) + events[5:],
            events[:6] + (replace(events[6], semantic_key="other-semantic"),)
            + events[7:],
            events[:3] + (replace(events[3], family_id="foreign-family"),)
            + events[4:],
        ):
            # The malicious recorder can re-sign its changed stream; the
            # previously signed/pinned schedule still catches the mutation.
            try:
                capture = capture_training_visits(
                    context, observed, schedule_sha256=pinned,
                    checkpoint_sha256="d" * 64, recorder_key=RECORDER_KEY)
            except TrainingExposureAttestationError:
                # A repeated case changing its meaning may fail closed even
                # before a compromised recorder can sign its report.
                continue
            with self.subTest(observed=observed), self.assertRaisesRegex(
                    TrainingExposureAttestationError,
                    "recorder evidence"):
                self.verify(capture=capture, observed=observed,
                            schedule=schedule, pinned=pinned)

    def test_validly_resigned_shorter_schedule_cannot_replace_prior_external_pin(self):
        context, events, _, pinned, _ = signed_fixture()
        shorter_context = replace(context, visits_per_step=1)
        shorter = tuple(replace(events[i * 2], ordinal=i,
                                cycle=i // 2, step=i % 2, visit=0)
                        for i in range(4))
        new_schedule = precommit_training_schedule(
            shorter_context, shorter, schedule_key=SCHEDULE_KEY)
        new_pin = sha256(new_schedule.receipt_bytes).hexdigest()
        new_capture = capture_training_visits(
            shorter_context, shorter, schedule_sha256=new_pin,
            checkpoint_sha256="d" * 64, recorder_key=RECORDER_KEY)
        with self.assertRaisesRegex(TrainingExposureAttestationError,
                                    "prior external pin"):
            verify_complete_train_inventory(
                context=shorter_context, scheduled=shorter, observed=shorter,
                signed_schedule=new_schedule, signed_capture=new_capture,
                externally_pinned_schedule_sha256=pinned,
                expected_checkpoint_sha256="d" * 64,
                schedule_key=SCHEDULE_KEY, recorder_key=RECORDER_KEY)

    def test_run_scope_base_model_source_and_fence_are_pinned(self):
        _, _, schedule, pinned, _ = signed_fixture()
        context, events = fixture()
        failures = (
            replace(context, logical_run_id="foreign-run"),
            replace(context, objective_id="other.objective"),
            replace(context, scope_sha256="f" * 64),
            replace(context, base_checkpoint_sha256="f" * 64),
            replace(context, source_sha256="f" * 64),
            replace(context, fence=context.fence + 1),
        )
        for changed in failures:
            with self.subTest(changed=changed), self.assertRaises(
                    TrainingExposureAttestationError):
                new_capture = capture_training_visits(
                    changed, events, schedule_sha256=pinned,
                    checkpoint_sha256="d" * 64, recorder_key=RECORDER_KEY)
                verify_complete_train_inventory(
                    context=changed, scheduled=events, observed=events,
                    signed_schedule=schedule, signed_capture=new_capture,
                    externally_pinned_schedule_sha256=pinned,
                    expected_checkpoint_sha256="d" * 64,
                    schedule_key=SCHEDULE_KEY, recorder_key=RECORDER_KEY)

    def test_checkpoint_and_signature_tampering_rejected(self):
        context, events, schedule, pin, capture = signed_fixture()
        wrong_checkpoint = capture_training_visits(
            context, events, schedule_sha256=pin,
            checkpoint_sha256="e" * 64, recorder_key=RECORDER_KEY)
        for changed in (
            wrong_checkpoint,
            replace(capture, signature_hex="0" * 64),
            replace(capture, receipt_bytes=capture.receipt_bytes + b" "),
        ):
            with self.subTest(changed=changed), self.assertRaises(
                    TrainingExposureAttestationError):
                self.verify(capture=changed)
        with self.assertRaisesRegex(TrainingExposureAttestationError,
                                    "signature"):
            self.verify(schedule=replace(schedule, signature_hex="0" * 64))

    def test_same_signer_keys_and_forged_source_qualification_flags_refused(self):
        context, events, schedule, pin, capture = signed_fixture()
        for kwargs in (
            {"schedule_key": RECORDER_KEY, "recorder_key": RECORDER_KEY},
            {"schedule_key": b"short", "recorder_key": RECORDER_KEY},
        ):
            with self.assertRaises(TrainingExposureAttestationError):
                verify_complete_train_inventory(
                    context=context, scheduled=events, observed=events,
                    signed_schedule=schedule, signed_capture=capture,
                    externally_pinned_schedule_sha256=pin,
                    expected_checkpoint_sha256="d" * 64, **kwargs)
        for signed_schedule, signed_capture in (
            (replace(schedule, independently_hosted=True), capture),
            (schedule, replace(capture, physically_authenticated=True)),
        ):
            with self.assertRaisesRegex(TrainingExposureAttestationError,
                                        "unqualified source receipt flags"):
                self.verify(schedule=signed_schedule, capture=signed_capture)

    def test_training_case_cannot_change_between_visits_or_alias_under_new_id(self):
        context, events = fixture()
        cases = (
            events[:4] + (replace(events[4], prompt="A changed meaning."),) + events[5:],
            events[:4] + (replace(events[4], case_id="new-case"),) + events[5:],
            events[:4] + (replace(events[4], case_id="new-case",
                                  prompt="Compute 7 + 5."),) + events[5:],
        )
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(
                    TrainingExposureAttestationError):
                precommit_training_schedule(context, changed,
                                            schedule_key=SCHEDULE_KEY)

    def test_invalid_dimensions_or_source_metadata_refused(self):
        context, events = fixture()
        invalids = (
            replace(context, fence=0),
            replace(context, fence=True),
            replace(context, cycles=False),
            replace(context, steps_per_cycle=100_001),
            replace(context, scope_sha256="nothex"),
            replace(context, objective_id="../foreign"),
        )
        for invalid in invalids:
            with self.subTest(invalid=invalid), self.assertRaises(
                    TrainingExposureAttestationError):
                precommit_training_schedule(invalid, events,
                                            schedule_key=SCHEDULE_KEY)


if __name__ == "__main__":
    unittest.main()
