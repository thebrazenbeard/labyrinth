"""Scoped pilot receipt auditing: source-level consistency, not model evidence."""

from __future__ import annotations

import hashlib
import unittest

from labyrinth.evaluation import score_response
from labyrinth.hf_pilot import (
    PilotConfig, SCHEMA, _prompt_commitment, _scoped_training_cases,
    _training_visit_indices, derange_answers,
)
from labyrinth.hf_report_audit import audit_hf_report
from labyrinth.training_scope import TrainingScopeSpec
from tests.test_hf_report_audit import pilot_fixture


def scoped_pilot_fixture(*, cycle_index: int = 0):
    report = pilot_fixture()
    scope = TrainingScopeSpec(
        objective_id="audit-scoped-arithmetic", domain="reasoning",
        skills=("arithmetic",), specificity=1, scale=3,
        max_cycles=3, seed=41,
    )
    config = PilotConfig(
        model_path="C:/fixture/base", output_dir="C:/fixture/run",
        seed=13, train_count=2, eval_count=2, tiers=(1,), steps=2,
    )
    train, heldout, plan = _scoped_training_cases(config, scope, cycle_index, ())
    report["schema"] = SCHEMA
    report["config"]["answer_only"] = True
    report["initial_persistent_buffers_sha256"] = "9" * 64
    report["training_case_ids"] = [case.id for case in train]
    report["heldout_case_ids"] = [case.id for case in heldout]
    report["heldout_public_prompt_sha256"] = _prompt_commitment(heldout)
    report["control_data"].update({
        "corrupted_label_collisions": sum(
            case.answer == label for case, label in zip(train, derange_answers(train))
        ),
        "answer_scaffold_supplied": True,
        "sham_optimizer_steps": 2, "sham_train_case_visits": 2,
        "sham_evaluation_before_updates": False,
        "sham_is_compute_matched": True,
    })
    visits = [train[i].id for i in _training_visit_indices(train, seed=config.seed, steps=2)]
    for arm in ("sham", "candidate", "corrupted"):
        report["training"][arm] = {
            "steps": 2, "example_visits": 2, "visited_case_ids": visits,
            "input_tokens_seen": 200, "losses": [2.5, 2.0],
        }
    trials = []
    for challenge in heldout:
        response = "ANSWER: 0"
        score = score_response(challenge, response)
        trials.append({
            "id": challenge.id, "correct": score.correct,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "response_preview": response, "score_reason": score.reason,
        })
    count = sum(int(trial["correct"]) for trial in trials)
    summary = {
        "correct": count, "total": len(trials), "accuracy": count / len(trials),
        "format_valid": len(trials), "format_rate": 1.0,
    }
    report["evaluations"] = {
        arm: {**summary, "trials": [dict(trial) for trial in trials]}
        for arm in ("baseline", "sham", "candidate", "corrupted")
    }
    all_cases = train + heldout
    report["resumed_journey"] = {
        "scope": scope.to_dict(), "scope_fingerprint": scope.fingerprint,
        "cycle_index": cycle_index, "weak_skills": [],
        "training_room_ids": [room.room_id for room in plan.rooms],
        "historical_case_ids": sorted(case.id for case in all_cases),
        "historical_prompt_sha256": sorted(
            hashlib.sha256(case.prompt.encode()).hexdigest() for case in all_cases
        ),
        "same_initial_lora_for_all_controls": True,
    }
    return report


class HfScopedReportAuditTests(unittest.TestCase):
    def test_honest_scoped_v3_report_does_not_fail_unscoped_case_replay(self):
        report = scoped_pilot_fixture()
        result = audit_hf_report(report)
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["checked_trials"], 8)
        self.assertFalse(result["authenticated_model_execution"])

    def test_scoped_successor_reports_incomplete_without_predecessor_custody(self):
        report = scoped_pilot_fixture(cycle_index=1)
        result = audit_hf_report(report)
        self.assertEqual(result["status"], "INCOMPLETE", result)
        self.assertTrue(any("predecessor custody" in item for item in result["unchecked"]))

    def test_scoped_scope_room_and_case_forgery_fail(self):
        fixture = scoped_pilot_fixture()
        mutations = (
            lambda report: report["resumed_journey"].__setitem__("scope_fingerprint", "f" * 64),
            lambda report: report["resumed_journey"]["training_room_ids"].__setitem__(0, "a" * 32),
            lambda report: report["training_case_ids"].__setitem__(0, "train-forged"),
            lambda report: report["heldout_case_ids"].__setitem__(0, "eval-forged"),
            lambda report: report["resumed_journey"]["historical_case_ids"].clear(),
            lambda report: report["resumed_journey"].__setitem__("same_initial_lora_for_all_controls", False),
        )
        import copy
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                report = copy.deepcopy(fixture)
                mutate(report)
                self.assertEqual(audit_hf_report(report)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
