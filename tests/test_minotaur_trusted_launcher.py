"""A trusted config-only launch must not execute a candidate Python factory."""

from __future__ import annotations

import os
import json
import unittest

from labyrinth.minotaur_cross_exam import ExamSpec
from labyrinth.minotaur_trusted_launcher import (
    ProcessCandidateConfig, candidate_public_summary, run_trusted_cross_exam,
)


WINNING_SEED_ZERO = ("forward", "take", "forward", "take", "arm", "forward", "escape")


class TrustedLauncherTests(unittest.TestCase):
    def test_public_config_only_child_probe_no_parent_python_stack(self):
        results = run_trusted_cross_exam(
            (ExamSpec("timegate", 0, sealed=True),), ProcessCandidateConfig(mode="probe"))
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.candidate_kind, "DETERMINISTIC_ZERO_UPDATE_CONTROL")
        self.assertEqual(result.model_authentication, "UNATTESTED")
        self.assertEqual(result.private_custody, "UNQUALIFIED_SAME_OS_USER")
        self.assertFalse(result.stochastic_comparison_matched)
        self.assertEqual(len(result.process_calls), 1)
        self.assertNotEqual(result.process_calls[0].child_pid, os.getpid())
        self.assertFalse(result.process_calls[0].parent_session_frame_visible)
        self.assertEqual(result.process_calls[0].generation_seed_independence, "UNATTESTED")
        self.assertEqual(result.receipt_visibility, "HOST_PRIVATE_NOT_FOR_CANDIDATE")
        self.assertEqual(result.model_seed_source, "HOST_CSPRNG_REQUEST_UNATTESTED")

    def test_public_summary_strips_secret_case_seed_and_host_receipts(self):
        result = run_trusted_cross_exam((ExamSpec("timegate", 739),),
                                        ProcessCandidateConfig(mode="probe"))
        self.assertEqual(result[0].cross_case.seed, 739)  # private host receipt
        summaries = candidate_public_summary(result)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(set(summaries[0]), {
            "family", "candidate_verdict", "candidate_escape_verified",
            "baseline_escape_verified", "candidate_kind",
        })
        raw = json.dumps(summaries)
        for private in ('"seed"', '"case_id"', '"receipt_sha256"',
                        '"generation_seed"', '"public_start_json"'):
            self.assertNotIn(private, raw)

    def test_fixed_zero_update_script_replays_without_learning_claim(self):
        results = run_trusted_cross_exam(
            (ExamSpec("timegate", 0),),
            ProcessCandidateConfig(mode="fixed", actions=WINNING_SEED_ZERO,
                                   generation_seed=17))
        item = results[0]
        self.assertTrue(item.cross_case.candidate.verified_escape)
        self.assertEqual(item.model_seed_source, "FIXED_CONFIG_UNATTESTED")
        self.assertEqual(len(item.process_calls), len(WINNING_SEED_ZERO))
        self.assertTrue(all(call.generation_seed == 17 for call in item.process_calls))
        self.assertFalse(item.neural_training_performed)

    def test_no_user_supplied_candidate_factory_or_callable_is_accepted(self):
        class HostileFactory:
            invoked = False

            def __call__(self, public):
                self.invoked = True
                raise AssertionError("hostile factory reached examiner frame")

        evil = HostileFactory()
        specs = (ExamSpec("timegate", 0),)
        with self.assertRaises(TypeError):
            run_trusted_cross_exam(specs, evil)
        with self.assertRaises(TypeError):
            run_trusted_cross_exam(specs, ProcessCandidateConfig(), candidate_factory=evil)
        self.assertFalse(evil.invoked)

    def test_hostile_subclasses_and_custom_field_instances_rejected(self):
        class BogusConfig(ProcessCandidateConfig):
            pass

        class BogusSpec(ExamSpec):
            pass

        with self.assertRaises(TypeError):
            run_trusted_cross_exam((ExamSpec("timegate", 0),), BogusConfig())
        with self.assertRaises(TypeError):
            run_trusted_cross_exam((BogusSpec("timegate", 0),), ProcessCandidateConfig())
        config = ProcessCandidateConfig(mode="probe")
        object.__setattr__(config, "generation_seed", "secretly-not-an-int")
        with self.assertRaises(ValueError):
            run_trusted_cross_exam((ExamSpec("timegate", 0),), config)

    def test_configuration_is_frozen_and_invalid_modes_refused(self):
        config = ProcessCandidateConfig()
        with self.assertRaises((AttributeError, TypeError)):
            config.mode = "ollama"
        with self.assertRaises(ValueError):
            ProcessCandidateConfig(mode="breach")
        with self.assertRaises(ValueError):
            ProcessCandidateConfig(mode="ollama", model="qwen")
        with self.assertRaises(ValueError):
            ProcessCandidateConfig(mode="fixed", actions=(object(),))


if __name__ == "__main__":
    unittest.main()
