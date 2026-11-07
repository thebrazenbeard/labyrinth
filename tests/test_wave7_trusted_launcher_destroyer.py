"""Independent hostile black-box checks for the new trusted launcher.

Run in a checkout that contains labyrinth/minotaur_trusted_launcher.py.
These tests do not contact any Ollama runtime or start GPU work.
"""

from __future__ import annotations

import math
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from labyrinth.minotaur_cross_exam import ExamSpec
from labyrinth.minotaur_trusted_launcher import (
    ProcessCandidateConfig, candidate_public_summary, run_trusted_cross_exam,
)


class UntrustedBoundaryAttack(unittest.TestCase):
    def test_hostile_callable_is_never_invoked_as_a_candidate_factory(self):
        class Trap:
            invoked = False

            def __call__(self, _public):
                self.invoked = True
                raise AssertionError("UNTRUSTED_HOST_STACK_EXECUTION")

        trap = Trap()
        with self.assertRaises(TypeError):
            run_trusted_cross_exam((ExamSpec("oracle", 739),), trap)
        self.assertFalse(trap.invoked)

    def test_primitive_config_mutation_fuzz_is_rejected_before_child_launch(self):
        class StrSubclass(str):
            pass

        class IntSubclass(int):
            pass

        invalid = (
            ("mode", None), ("mode", StrSubclass("probe")), ("mode", "breach"),
            ("actions", ["wait"]), ("actions", (StrSubclass("wait"),)),
            ("actions", ("wait",)), ("actions", ("w" * 65,)),
            ("actions", ("w",) * 33),
            ("generation_seed", True), ("generation_seed", IntSubclass(7)),
            ("generation_seed", -1), ("generation_seed", 2**31),
            ("generation_seed", "0"),
            ("timeout_seconds", math.nan), ("timeout_seconds", math.inf),
            ("timeout_seconds", 0), ("timeout_seconds", 999999),
            ("timeout_seconds", True),
            ("max_output_bytes", True), ("max_output_bytes", 127),
            ("max_output_bytes", 65537), ("max_output_bytes", .5),
            ("model", "unapproved"), ("model", StrSubclass("unapproved")),
            ("base_url", StrSubclass("http://127.0.0.1:11434")),
            ("authorize_local_model", True), ("authorize_local_model", 1),
        )
        with patch("labyrinth.minotaur_process_adapter.subprocess.Popen",
                   side_effect=AssertionError("UNEXPECTED_CHILD_SPAWN")) as spawn:
            for key, value in invalid:
                with self.subTest(field=key, value=value):
                    config = ProcessCandidateConfig(mode="probe")
                    object.__setattr__(config, key, value)
                    with self.assertRaises((TypeError, ValueError)):
                        run_trusted_cross_exam((ExamSpec("oracle", 739),), config)
        spawn.assert_not_called()

    def test_distinct_host_rng_seeds_are_explicitly_unmatched_in_receipt(self):
        config = ProcessCandidateConfig(mode="probe", generation_seed=None)
        # The source modules share the SAME imported `secrets` module object;
        # patching its function twice would silently contaminate the control.
        # Patch independent source references instead.
        with (patch("labyrinth.minotaur_cross_exam.secrets",
                    SimpleNamespace(randbelow=lambda bound: 4242)),
              patch("labyrinth.minotaur_trusted_launcher.secrets",
                    SimpleNamespace(randbelow=lambda bound: 2026))):
            result = run_trusted_cross_exam((ExamSpec("oracle", 739, sealed=True),), config)[0]
        self.assertEqual(result.cross_case.candidate.seed, 4242)
        self.assertEqual(result.cross_case.baseline.seed, 4242)
        self.assertEqual(result.process_calls[0].generation_seed, 2026)
        self.assertEqual(result.model_seed_source, "HOST_CSPRNG_REQUEST_UNATTESTED")
        self.assertFalse(result.stochastic_comparison_matched)

    def test_private_case_seed_is_present_in_host_receipt_do_not_reveal_to_worker(self):
        result = run_trusted_cross_exam(
            (ExamSpec("oracle", 739, sealed=True),),
            ProcessCandidateConfig(mode="probe", generation_seed=17))[0]
        self.assertEqual(result.cross_case.seed, 739)
        self.assertEqual(result.private_custody, "UNQUALIFIED_SAME_OS_USER")
        self.assertEqual(result.host_factory, "TRUSTED_CONFIG_ONLY_NO_CANDIDATE_CALLBACK")

    def test_after_exam_public_projection_never_serializes_host_private_seed(self):
        host_only = run_trusted_cross_exam(
            (ExamSpec("oracle", 739, sealed=True),),
            ProcessCandidateConfig(mode="probe", generation_seed=17))
        view = candidate_public_summary(host_only)
        self.assertEqual(type(view), tuple)
        self.assertEqual(len(view), 1)
        self.assertEqual(set(view[0]), {
            "family", "candidate_verdict", "candidate_escape_verified",
            "baseline_escape_verified", "candidate_kind",
        })
        self.assertEqual(view[0]["family"], "oracle")
        encoded = json.dumps(view)
        for forbidden in ("seed", "transcript", "process_calls", "case_id",
                          "receipt_sha256", "generation_seed"):
            self.assertNotIn(forbidden, encoded)


if __name__ == "__main__":
    unittest.main()
