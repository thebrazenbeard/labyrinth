"""Source-only tests for isolated multi-objective adaptive TRAIN scheduling."""

from dataclasses import replace
from hashlib import sha256
from tempfile import TemporaryDirectory
import re
import unittest

from labyrinth.adaptive_minotaur_r93 import ObjectiveLane, schedule_adaptive_training
from labyrinth.evaluation import score_response
from labyrinth.journey import (
    TRAINING, TrainingCompletionReceipt, begin_journey, complete_maze, face_minotaur,
)
from labyrinth.maze import generate_maze
from labyrinth.minotaur_exam import ExamCaseResult, MinotaurExamReceipt
from labyrinth.objective_challenge_provider_r92 import (
    AUDIT_FOCUS, AUDIT_OBSERVABLE, AUDIT_RUBRIC_ID,
    prepare_objective_audit_room, score_objective_audit_response,
)
from labyrinth.objective_contract import ObjectiveContract, SkillTarget
from labyrinth.objective_learning_loop_r92 import (
    arithmetic_objective, run_objective_learning_loop,
)
from labyrinth.training_scope import TrainingScopeSpec


def _audit():
    scope = TrainingScopeSpec(
        objective_id="zzz-audit-r93", domain="evidence_reasoning",
        skills=("source_priority",), specificity=4, scale=2,
        max_cycles=2, seed=92031,
    )
    return ObjectiveContract(
        scope=scope,
        intention="Exclude invalid reports before choosing reliable source priority.",
        targets=(SkillTarget(
            skill="source_priority", focus_tags=(AUDIT_FOCUS,),
            observable_behaviors=(AUDIT_OBSERVABLE,), rubric_id=AUDIT_RUBRIC_ID,
            minimum_score_percent=75, minimum_gain_percent=10,
        ),),
        excluded_scopes=("Independent final assessment.",),
        forbidden_claims=("Public TRAIN data is independent examination.",),
    )


def _program():
    scope = TrainingScopeSpec(
        objective_id="aaa-program-r93", domain="reasoning",
        skills=("program_reasoning",), specificity=3, scale=2,
        max_cycles=2, seed=81042,
    )
    return ObjectiveContract(
        scope=scope,
        intention="Trace a bounded four-step register dependency graph.",
        targets=(SkillTarget(
            skill="program_reasoning",
            focus_tags=("four_step_cross_register_dependency_path",),
            observable_behaviors=("Compute register A after four instructions.",),
            rubric_id="program_structure_v1", minimum_score_percent=75,
            minimum_gain_percent=10,
        ),),
        excluded_scopes=("Independent final assessment.",),
        forbidden_claims=("Public TRAIN data is independent examination.",),
    )


def _failed_audit(contract):
    scope = contract.scope
    start = begin_journey(scope, checkpoint_sha256="a" * 64)
    prepared = tuple(prepare_objective_audit_room(contract, start.maze, room,
                                                  journey=start)
                     for room in start.maze.rooms)
    trained = complete_maze(start, TrainingCompletionReceipt(
        objective_id=scope.objective_id, cycle_index=0,
        checkpoint_before_sha256="a" * 64,
        checkpoint_after_sha256="b" * 64,
        room_case_ids=tuple((item.room_id, item.challenge.id,
                             item.trainee_view.prompt_sha256) for item in prepared),
    ))
    cases = []
    for split in ("heldout", "transfer"):
        for case in generate_maze(seed=1701 + (split == "transfer"),
                                  tiers=(3,), count=4, split=split):
            score = score_response(case, "ANSWER: 0")
            cases.append(ExamCaseResult(
                case_id=case.id, skill="source_priority", split=split,
                score=float(score.points), baseline_score=0.0,
                prompt_sha256=sha256(case.prompt.encode("utf-8")).hexdigest(),
            ))
    failed = face_minotaur(trained, MinotaurExamReceipt(
        exam_id="public-dev-r93", objective_id=scope.objective_id,
        cycle_index=0, checkpoint_sha256="b" * 64,
        baseline_sha256="a" * 64, evaluator_id="public-development-toy",
        cases=tuple(cases),
    ))
    assert failed.phase == TRAINING and failed.attempts[-1].exam_status == "FAIL"
    return failed


def _source_rank_heuristic(prompt):
    priority = re.search(r"^Priority: ([a-z]+(?: > [a-z]+){3})$", prompt, re.M)
    assert priority
    names = priority.group(1).split(" > ")
    rows = re.findall(
        r"^- source=([a-z]+); audit_status=(VALID|INVALID); measured_value=([0-9]+)",
        prompt, re.M,
    )
    values = {name: (valid, value) for name, valid, value in rows}
    # Deliberately weak: never examine priority beyond second place.
    first, second = names[:2]
    return values[first][1] if values[first][0] == "VALID" else values[second][1]


class AdaptiveMinotaurR93Tests(unittest.TestCase):
    def setUp(self):
        self.audit = _audit()
        self.program = _program()
        self.fresh_audit = begin_journey(self.audit.scope, checkpoint_sha256="a" * 64)
        self.fresh_program = begin_journey(self.program.scope, checkpoint_sha256="d" * 64)

    def test_two_distinct_approved_providers_produce_disjoint_prompt_only_schedule(self):
        schedule = schedule_adaptive_training((
            ObjectiveLane(self.audit, self.fresh_audit),
            ObjectiveLane(self.program, self.fresh_program),
        ))
        self.assertEqual(schedule.evidence_class, "SOURCE_ONLY_UNATTESTED")
        self.assertEqual(schedule.objective_order, (
            "aaa-program-r93", "zzz-audit-r93"))
        expected = self.audit.scope.total_chambers + self.program.scope.total_chambers
        self.assertEqual(len(schedule.ordered_rooms), expected)
        self.assertEqual(len(set(item.prepared.challenge.id for item in schedule.ordered_rooms)), expected)
        self.assertEqual(len(set(item.trainee_view.prompt_sha256 for item in schedule.ordered_rooms)), expected)
        self.assertEqual({item.prepared.challenge.kind for item in schedule.ordered_rooms},
                         {"register_program", "source_priority"})
        for objective_id, view in schedule.trainee_sequence():
            self.assertIn(objective_id, schedule.objective_order)
            self.assertNotIn("answer", vars(view))
            self.assertEqual(view.prompt_sha256, sha256(view.prompt.encode()).hexdigest())
        self.assertFalse(hasattr(schedule, "graduation_authorized"))
        with self.assertRaises(AttributeError):
            object.__setattr__(schedule, "evidence_class", "Q9_GRADUATED")

    def test_real_scored_development_failure_prioritizes_changed_objective_train(self):
        failed = _failed_audit(self.audit)
        schedule = schedule_adaptive_training((
            ObjectiveLane(self.program, self.fresh_program),
            ObjectiveLane(self.audit, failed),
        ))
        self.assertEqual(schedule.objective_order, (
            "zzz-audit-r93", "aaa-program-r93"))
        first = schedule.ordered_rooms[0]
        self.assertTrue(first.failed_development_predecessor)
        self.assertEqual(first.cycle_index, 1)
        self.assertIn("Remediation focus", first.trainee_view.prompt)
        self.assertEqual(
            len([r for r in schedule.ordered_rooms if r.objective_id == "zzz-audit-r93"]),
            len(failed.maze.rooms),
        )
        old = schedule_adaptive_training((
            ObjectiveLane(self.audit, self.fresh_audit),
            ObjectiveLane(self.program, self.fresh_program),
        ))
        self.assertTrue({x.prepared.challenge.id for x in schedule.ordered_rooms
                         if x.objective_id == "zzz-audit-r93"}.isdisjoint(
                             x.prepared.challenge.id for x in old.ordered_rooms
                             if x.objective_id == "zzz-audit-r93"))

    def test_admitted_trainee_and_trainer_gold_cannot_change_after_scheduling(self):
        # A frozen dataclass is not a deep immutable snapshot; Python callers
        # can mutate nested objects despite a frozen schedule wrapper.
        for field, replacement in (
            ("prompt", "Calculate 1 + 2.\nRespond with exactly one line: ANSWER: <integer>."),
            ("answer", "FORGED GOLD"),
            ("metadata", (("hostile", "injected"),)),
            ("id", "train-forged-case"),
        ):
            with self.subTest(field=field):
                schedule = schedule_adaptive_training((
                    ObjectiveLane(self.audit, self.fresh_audit),
                    ObjectiveLane(self.program, self.fresh_program),
                ))
                first = schedule.ordered_rooms[0]
                object.__setattr__(first.prepared.challenge, field, replacement)
                with self.assertRaisesRegex(ValueError, "admitted TRAIN material changed"):
                    schedule.trainee_sequence()
                with self.assertRaisesRegex(ValueError, "admitted TRAIN material changed"):
                    _ = first.trainee_view

    def test_forged_admitted_trainee_snapshot_cannot_override_source(self):
        schedule = schedule_adaptive_training((
            ObjectiveLane(self.audit, self.fresh_audit),
            ObjectiveLane(self.program, self.fresh_program),
        ))
        first = schedule.ordered_rooms[0]
        forged_prompt = "This forged TRAIN prompt was never admitted."
        assert first.trainee_view.prompt != forged_prompt
        fields = first.admitted_trainee_fields
        with self.assertRaises(AttributeError):
            object.__setattr__(
                first, "admitted_trainee_fields",
                fields[:3] + (forged_prompt, sha256(forged_prompt.encode()).hexdigest()),
            )
        self.assertEqual(first.trainee_view.prompt, fields[3])

    def test_route_objective_identity_cannot_mutate_after_admission(self):
        schedule = schedule_adaptive_training((
            ObjectiveLane(self.audit, self.fresh_audit),
            ObjectiveLane(self.program, self.fresh_program),
        ))
        first = schedule.ordered_rooms[0]
        with self.assertRaises(AttributeError):
            object.__setattr__(first, "objective_id", "foreign-objective-no-custody")
        self.assertIn(schedule.trainee_sequence()[0][0], schedule.objective_order)

    def test_forged_unsealed_failure_is_not_accepted_as_development_lineage(self):
        failed = _failed_audit(self.audit)
        stale = replace(failed, sealed_exam_ids=(), exam_event_statuses=())
        with self.assertRaises(ValueError):
            schedule_adaptive_training((
                ObjectiveLane(self.audit, stale),
                ObjectiveLane(self.program, self.fresh_program),
            ))
        forged_pass = replace(failed.attempts[-1], exam_status="PASS")
        with self.assertRaises(ValueError):
            schedule_adaptive_training((
                ObjectiveLane(self.audit, replace(failed, attempts=(forged_pass,))),
                ObjectiveLane(self.program, self.fresh_program),
            ))

    def test_unsupported_focus_duplicate_objective_or_wrong_journey_fails(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            schedule_adaptive_training((
                ObjectiveLane(self.audit, self.fresh_audit),
                ObjectiveLane(self.audit, self.fresh_audit),
            ))
        mutated = replace(self.audit, targets=(replace(
            self.audit.targets[0], rubric_id="other_unapproved_rubric"),))
        with self.assertRaisesRegex(ValueError, "approved"):
            schedule_adaptive_training((
                ObjectiveLane(mutated, self.fresh_audit),
                ObjectiveLane(self.program, self.fresh_program),
            ))
        with self.assertRaisesRegex(ValueError, "own active TRAIN journey"):
            schedule_adaptive_training((
                ObjectiveLane(self.audit, self.fresh_program),
                ObjectiveLane(self.program, self.fresh_program),
            ))
        unsupported_scope = TrainingScopeSpec(
            objective_id="unsupported-empathy-r93", domain="empathy",
            skills=("cognitive_empathy",), specificity=3, scale=1,
            max_cycles=2, seed=123,
        )
        unsupported_contract = ObjectiveContract(
            scope=unsupported_scope,
            intention="Understand uncertain emotional perspective cues.",
            targets=(SkillTarget(
                skill="cognitive_empathy", focus_tags=("perspective",),
                observable_behaviors=("Offer a tentative perspective explanation.",),
                rubric_id="empathy_behavior_v1", minimum_score_percent=75,
                minimum_gain_percent=10,
            ),),
            excluded_scopes=("Independent clinical diagnoses.",),
            forbidden_claims=("Source scheduling proves subjective empathy.",),
        )
        with self.assertRaisesRegex(ValueError, "no approved executable provider"):
            schedule_adaptive_training((
                ObjectiveLane(unsupported_contract, begin_journey(
                    unsupported_scope, checkpoint_sha256="e" * 64)),
                ObjectiveLane(self.program, self.fresh_program),
            ))

    def test_public_audit_positional_and_format_shortcuts_do_not_pass_threshold(self):
        for seed in (1, 2, 7, 17, 31, 101, 420, 92031):
            with self.subTest(seed=seed):
                audit_contract = replace(self.audit, scope=replace(
                    self.audit.scope, seed=seed))
                initial = begin_journey(audit_contract.scope,
                                        checkpoint_sha256="a" * 64)
                schedule = schedule_adaptive_training((
                    ObjectiveLane(audit_contract, initial),
                    ObjectiveLane(self.program, self.fresh_program),
                ))
                audit = [x.prepared.challenge for x in schedule.ordered_rooms
                         if x.objective_id == "zzz-audit-r93"]
                shortcuts = sum(score_objective_audit_response(
                    case, "ANSWER: " + _source_rank_heuristic(case.prompt)
                ).correct for case in audit)
                self.assertLess(shortcuts / len(audit), 0.75)
                for case in audit:
                    self.assertFalse(score_objective_audit_response(
                        case, "ANSWER: " + case.answer + "\n").correct)

    def test_actual_arithmetic_toy_sgd_and_matched_sham_remain_separate(self):
        # This is a *distinct* approved arithmetic objective. Its four toy
        # parameters are not shared with audit/program objective lanes.
        with TemporaryDirectory() as tmp:
            report = run_objective_learning_loop(
                arithmetic_objective(seed=7), checkpoint_dir=tmp,
                learner_seed=0, epochs_by_cycle=(1, 80), exam_count_per_slice=8,
            )
        self.assertGreaterEqual(len(report["cycles"]), 1)
        for cycle in report["cycles"]:
            self.assertGreater(cycle["optimizer"]["parameter_updates"], 0)
            self.assertEqual(cycle["sham_parameter_updates"], 0)
            self.assertNotEqual(cycle["checkpoint_before_sha256"],
                                cycle["checkpoint_after_sha256"])
            self.assertEqual(cycle["sham_correct"], cycle["frozen_baseline_correct"])
        self.assertFalse(report["graduation_authorized"])
        self.assertFalse(report["pretrained_improvement_established"])


if __name__ == "__main__":
    unittest.main()
