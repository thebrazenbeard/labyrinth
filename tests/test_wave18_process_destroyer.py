"""Hostile regression: deterministic fixture responses cannot be rewritten as wins.

Independent red baseline: Labyrinth #165 (3e29c87), recorded by Destroyer.
"""

import base64
from copy import deepcopy
import json
import unittest

from labyrinth import switchback_novel_transfer as novel
from scripts import novel_router_model_trial as trial


class Wave18ProcessDestroyer(unittest.TestCase):
    @staticmethod
    def _rewrite_child_as_plan(report: dict, turn_index: int,
                               plan: dict, source: dict) -> dict:
        forged = deepcopy(report)
        raw = json.dumps(plan, separators=(",", ":")).encode("utf-8")
        child = forged["turns"][turn_index]["child"]
        child["raw_stdout_base64"] = base64.b64encode(raw).decode("ascii")
        child["response_sha256"] = trial._sha(raw)
        child["response_bytes"] = len(raw)
        forged["turns"][turn_index]["decision"] = plan
        forged["source_receipt"] = source
        forged["source_verdict"] = "VERIFIED_WIN"
        forged["verified_success"] = True
        forged["receipt_sha256"] = trial._sha({
            key: value for key, value in forged.items() if key != "receipt_sha256"})
        return forged

    def test_declared_abstain_fixture_cannot_be_selfhashed_into_verified_win(self):
        case = novel.make_case(0)
        candidate_plan = {"kind": "PLAN", "actions": novel.search(case)["witness"]}
        config = trial.TrialConfig(case_seed=0, generation_seed=73,
                                  max_calls=1, call_timeout_seconds=2,
                                  total_timeout_seconds=2,
                                  max_output_bytes=4096)
        adapter = trial.AdapterSpec(mode="fixture", fixture_responses=(
            '{"kind":"ABSTAIN"}',))
        original = trial.run_trial(config, adapter)
        self.assertTrue(trial.verify_trial_receipt(original))
        self.assertFalse(original["verified_success"])

        forged = self._rewrite_child_as_plan(
            original, 0, candidate_plan,
            novel.ProbeSession(case).submit(candidate_plan))
        self.assertEqual(forged["source_receipt"]["verdict"], "VERIFIED_WIN")
        self.assertEqual(tuple(forged["adapter"]["fixture_responses"]),
                         ('{"kind":"ABSTAIN"}',))
        self.assertFalse(trial.verify_trial_receipt(forged))

    def test_exhausted_fixture_sequence_defaults_abstain_not_forged_win(self):
        case = novel.make_case(0)
        candidate_plan = {"kind": "PLAN", "actions": novel.search(case)["witness"]}
        config = trial.TrialConfig(case_seed=0, generation_seed=73,
                                  max_calls=2, call_timeout_seconds=2,
                                  total_timeout_seconds=4,
                                  max_output_bytes=4096)
        adapter = trial.AdapterSpec(mode="fixture", fixture_responses=(
            '{"kind":"ACTION","action":"TICK"}',))
        original = trial.run_trial(config, adapter)
        self.assertTrue(trial.verify_trial_receipt(original))
        self.assertEqual(original["turns"][1]["decision"], {"kind": "ABSTAIN"})

        source_session = novel.ProbeSession(case)
        source_session.probe("TICK")
        forged = self._rewrite_child_as_plan(
            original, 1, candidate_plan, source_session.submit(candidate_plan))
        self.assertFalse(trial.verify_trial_receipt(forged))


if __name__ == "__main__":
    unittest.main()
