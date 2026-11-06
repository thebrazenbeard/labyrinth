"""Real CLI wiring for independently pinned source-only measured TRAIN schedules."""

from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import qualify_maze_conditioned_minilm_cpu as runner
from tests import test_measured_neural_batch_binding as fixtures


class MeasuredMiniLMCliTests(unittest.TestCase):
    def setUp(self):
        signed = fixtures.BoundNeuralMeasuredBatchesTests(
            "test_actual_four_key_signed_development_produces_bound_five_arm_schedule")
        signed.setUp()
        self.addCleanup(signed.doCleanups)
        self.fixture = signed
        self.schedule = signed.pinned(signed.proposal())
        self.expected = {
            "expected_schedule_sha256": self.schedule.schedule_sha256,
            "expected_run_id": self.schedule.run_id,
            "expected_candidate_artifact_sha256": self.schedule.candidate_artifact_sha256,
            "expected_predecessor_checkpoint_sha256": self.schedule.predecessor_checkpoint_sha256,
            "expected_fence": self.schedule.fence,
        }

    def export(self, schedule=None):
        return runner.export_measured_cli_schedule_json(schedule or self.schedule)

    def load(self, raw=None, **changed):
        expected = dict(self.expected)
        expected.update(changed)
        return runner.load_measured_cli_schedule(
            raw if raw is not None else self.export(),
            scope=self.fixture.signed.scope, steps_per_cycle=16,
            **expected,
        )

    def test_signed_fixture_serializes_and_reconstructs_typed_source_schedule(self):
        request, cycles = self.load()
        self.assertIsInstance(request, runner.MeasuredNeuralExecutionRequest)
        self.assertEqual(request.schedule.schedule_sha256, self.schedule.schedule_sha256)
        self.assertEqual(cycles, self.fixture.signed.cycles)
        self.assertEqual(request.schedule.batches, self.schedule.batches)
        self.assertFalse(request.schedule.host_signature_attested)
        self.assertFalse(request.schedule.graduation_authorized)

    def test_foreign_or_missing_independent_pins_fail_closed(self):
        variants = (
            {"expected_schedule_sha256": "f" * 64},
            {"expected_run_id": "other-run"},
            {"expected_candidate_artifact_sha256": "f" * 64},
            {"expected_predecessor_checkpoint_sha256": "f" * 64},
            {"expected_fence": self.schedule.fence + 1},
            {"expected_fence": True},
            {"expected_schedule_sha256": None},
        )
        for change in variants:
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                self.load(**change)

    def test_reordered_train_visits_and_forged_weakness_rejected(self):
        data = json.loads(self.export())
        variants = (
            {"ordered_exposures": list(reversed(data["ordered_exposures"]))},
            {"weak_skill": "positive_polarity"},
            {"source_event_id": "forged-source-event"},
            {"fence": self.schedule.fence + 1},
            {"arms": list(runner.ARMS)},
            {"schedule_sha256": "f" * 64},
        )
        for change in variants:
            amended = dict(data)
            amended.update(change)
            raw = (json.dumps(amended, sort_keys=True,
                              separators=(",", ":")) + "\n").encode()
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                self.load(raw)

    def test_extra_privileged_fields_duplicate_keys_and_bad_json_rejected(self):
        data = json.loads(self.export())
        enriched = dict(data, graduation_authorized=True)
        cases = (
            (json.dumps(enriched, sort_keys=True, separators=(",", ":")) + "\n").encode(),
            (self.export().rstrip(b"\n")[:-1] + b',"fence":7}\n'),
            self.export() + b" ",
            b"{" + b"x" * 65_537 + b"}",
            b"not-json",
        )
        for raw in cases:
            with self.subTest(raw_prefix=raw[:40]), self.assertRaises(ValueError):
                self.load(raw)

    def _args(self, temp: str, *, measured: bool = True) -> list[str]:
        args = ["--output-parent", temp, "--run-name", "r53-measured",
                "--steps-per-cycle", "16", "--preregistered-controls",
                "--full-optimizer-cycle-continuity",
                "--optimizer-run-id", self.schedule.run_id,
                "--optimizer-prior-sha256", self.schedule.predecessor_checkpoint_sha256,
                "--optimizer-fence", str(self.schedule.fence)]
        if measured:
            path = Path(temp) / "schedule.json"
            path.write_bytes(self.export())
            args += ["--measured-schedule-file", str(path),
                     "--measured-schedule-sha256", self.schedule.schedule_sha256,
                     "--measured-run-id", self.schedule.run_id,
                     "--measured-candidate-sha256", self.schedule.candidate_artifact_sha256,
                     "--measured-predecessor-sha256", self.schedule.predecessor_checkpoint_sha256,
                     "--measured-fence", str(self.schedule.fence)]
        return args

    def test_real_main_preflight_labels_measured_source_but_no_host_authority(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()) as out:
            result = runner.main(self._args(temp))
        self.assertEqual(result, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["measured_schedule_sha256"], self.schedule.schedule_sha256)
        self.assertEqual(payload["weakness_feedback_provenance"],
                         "EXTERNALLY_PINNED_SOURCE_SCHEDULE_HOST_UNATTESTED")
        self.assertFalse(payload["graduation_authorized"])

    def test_real_main_execute_passes_typed_request_before_neural_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = {}
            def fake_execute(*args, **kwargs):
                capture.update(kwargs)
                return {"public_development_comparison": {},
                        "public_transfer_comparison": {}}
            with patch.object(runner, "execute", side_effect=fake_execute):
                with redirect_stdout(StringIO()):
                    result = runner.main(self._args(temp) + ["--execute"])
        self.assertEqual(result, 0)
        self.assertEqual(capture["measured_batches"].schedule.schedule_sha256,
                         self.schedule.schedule_sha256)
        self.assertEqual(capture["optimizer_continuity"].logical_run_id,
                         self.schedule.run_id)
        self.assertEqual(capture["cycles"], self.fixture.signed.cycles)

    def test_main_rejects_partial_pins_conflicts_without_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            full = self._args(temp)
            cases = (
                full + ["--weak-cycle-one", "negative_polarity"],
                full[:full.index("--measured-candidate-sha256")]
                + full[full.index("--measured-predecessor-sha256"):],
                [item for item in full if item != "--preregistered-controls"],
                [item for item in full if item != "--full-optimizer-cycle-continuity"],
                self._args(temp, measured=False) + ["--measured-fence", "7"],
            )
            for argv in cases:
                with self.subTest(argv=argv[-4:]), redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        runner.main(argv)
                    self.assertEqual(caught.exception.code, 2)

    def test_execute_flag_never_downgrades_bad_measured_pin_to_operator_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            arguments = self._args(temp) + ["--execute"]
            index = arguments.index("--measured-schedule-sha256")
            arguments[index + 1] = "f" * 64
            with patch.object(runner, "execute") as neural_execute:
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
                    runner.main(arguments)
                self.assertEqual(caught.exception.code, 2)
                neural_execute.assert_not_called()
            self.assertFalse((Path(temp) / "r53-measured").exists())

    def test_existing_operator_cli_stays_distinct_and_unattested(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()) as out:
            result = runner.main(self._args(temp, measured=False)
                                 + ["--weak-cycle-one", "negative_polarity"])
        self.assertEqual(result, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["weakness_feedback_provenance"],
                         "OPERATOR_DECLARED_NOT_MEASURED")
        self.assertNotIn("measured_schedule_sha256", payload)

    def test_real_python_module_entrypoint_imports_one_measured_schedule_type(self):
        # Unlike direct main(), -m creates a __main__ module first. The
        # signed development binder imports its canonical module identity.
        with tempfile.TemporaryDirectory() as temp:
            completed = subprocess.run(
                [sys.executable, "-m", "scripts.qualify_maze_conditioned_minilm_cpu",
                 *self._args(temp)],
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(completed.returncode, 0,
                         completed.stdout + completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["measured_schedule_sha256"],
                         self.schedule.schedule_sha256)
        self.assertFalse(report["host_pin_custody_authenticated"])


if __name__ == "__main__":
    unittest.main()
