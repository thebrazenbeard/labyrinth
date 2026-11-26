"""Source-level and loopback-HTTP trials, not hidden or model-origin attestation."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from labyrinth.minotaur_local_trial import run_local_trial, write_trial_new
from tests.test_minotaur_ollama_child import fake_local_ollama


class LocalTrialTests(unittest.TestCase):
    def test_differing_private_variants_share_public_input_but_not_score_authority(self):
        with fake_local_ollama() as (url, calls):
            report = run_local_trial(
                "synthetic:local", endpoint=url, seeds=(7,),
                sealed_modes=(False, True), max_calls=2,
                timeout_seconds=6.0)
        self.assertEqual(report["schema"], "MINOTAUR_LOCAL_DEV_TRIAL_V1")
        self.assertFalse(report["comparison_qualified"])
        self.assertFalse(report["learning_qualified"])
        self.assertFalse(report["model_origin_attested"])
        self.assertEqual(len(report["cases"]), 2)
        self.assertEqual(len(calls), 4)
        left, right = report["cases"]
        self.assertEqual(left["public_sha256"], right["public_sha256"])
        self.assertFalse(left["sealed_for_development_only"])
        self.assertTrue(right["sealed_for_development_only"])
        self.assertEqual(left["source_development_oracle"]["verdict"], "VERIFIED_WIN")
        self.assertEqual(left["source_development_oracle"]["known_witness_actions"], 10)
        self.assertFalse(left["source_development_oracle"]["call_budget_can_fit_known_witness"])
        self.assertEqual(right["source_development_oracle"]["verdict"],
                         "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertIsNone(right["source_development_oracle"]["known_witness_actions"])
        for case in report["cases"]:
            self.assertEqual(case["candidate"]["dispositions"], ["ACTION", "ACTION"])
            self.assertEqual(case["control"]["dispositions"], ["ACTION", "ACTION"])
            self.assertTrue(case["candidate"]["audit_replay"])
            self.assertTrue(case["control"]["audit_replay"])
            self.assertFalse(case["candidate"]["verified_escape"])
            self.assertFalse(case["control"]["verified_escape"])
            self.assertEqual(case["candidate"]["model_id_unattested"],
                             "ollama:synthetic:local:unattested")
            self.assertEqual(case["control"]["model_id_unattested"],
                             "fixed:forward:zero-update")
        for _path, request in calls:
            self.assertIs(request["think"], False)
            self.assertEqual(request["options"]["seed"], 7)
            self.assertIn("public_start", request["prompt"])
            self.assertNotIn("case_id", request["prompt"])

    def test_source_oracle_budget_does_not_grant_model_win(self):
        abstain = '{"kind":"CLAIM","claim":{"kind":"ABSTAIN","assessment":"UNKNOWN"}}'
        with fake_local_ollama(response=abstain) as (url, calls):
            report = run_local_trial("synthetic:local", endpoint=url,
                                     seeds=(7,), sealed_modes=(False,),
                                     max_calls=16, timeout_seconds=6.0)
        self.assertEqual(len(calls), 1)
        case = report["cases"][0]
        self.assertTrue(case["source_development_oracle"]["call_budget_can_fit_known_witness"])
        self.assertEqual(case["source_development_oracle"]["known_witness_actions"], 10)
        self.assertEqual(case["candidate"]["dispositions"], ["CLAIM"])
        self.assertFalse(case["candidate"]["verified_escape"])
        self.assertFalse(report["learning_qualified"])

    def test_failed_model_call_cannot_be_reported_as_replay_or_win(self):
        with fake_local_ollama(done=False) as (url, calls):
            report = run_local_trial("synthetic:local", endpoint=url,
                                     seeds=(7,), sealed_modes=(False,),
                                     max_calls=1, timeout_seconds=6.0)
        self.assertEqual(len(calls), 1)
        candidate = report["cases"][0]["candidate"]
        self.assertEqual(candidate["final_verdict"], "MODEL_CALL_ERROR")
        self.assertFalse(candidate["audit_replay"])
        self.assertFalse(candidate["verified_escape"])
        self.assertFalse(report["comparison_qualified"])

    def test_rejects_invalid_trial_before_any_external_call(self):
        changes = [
            dict(model=""), dict(model="x", seeds=(True,)),
            dict(model="x", seeds=(7, 7)),
            dict(model="x", sealed_modes=(False, False)),
            dict(model="x", sealed_modes=(1,)),
            dict(model="x", seeds=()),
            dict(model="x", max_calls=0),
            dict(model="x", max_calls=17),
            dict(model="x", timeout_seconds=0),
            dict(model="x", timeout_seconds=61),
            dict(model="x", endpoint="http://192.0.2.1:11434"),
        ]
        for arguments in changes:
            with self.subTest(arguments=arguments), self.assertRaises(
                    (ValueError, TypeError, PermissionError)):
                run_local_trial(**arguments)

    def test_exclusive_evidence_writer_never_overwrites_or_follows_symlinks(self):
        report = {
            "schema": "MINOTAUR_LOCAL_DEV_TRIAL_V1",
            "comparison_qualified": False,
            "learning_qualified": False,
            "model_origin_attested": False,
            "cases": [],
        }
        with TemporaryDirectory() as folder:
            path = Path(folder) / "trial.json"
            write_trial_new(path, report)
            self.assertEqual(json.loads(path.read_text("utf-8")), report)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_trial_new(path, report)
            self.assertEqual(path.read_bytes(), before)
            linked = Path(folder) / "linked.json"
            try:
                linked.symlink_to(path)
            except (OSError, NotImplementedError):
                pass
            else:
                with self.assertRaises(FileExistsError):
                    write_trial_new(linked, report)
                self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
