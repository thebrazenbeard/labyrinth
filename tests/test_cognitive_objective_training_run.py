"""Cognitive empathy TRAIN routing exercises, never neural-training claims."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from labyrinth.approved_cognitive_empathy_train_r90 import (
    PROVIDER_ID, prepare_cognitive_empathy_train_rooms,
)
from labyrinth.journey import (
    AWAITING_EXAM, TRAINING, begin_journey, complete_maze, face_minotaur,
)
from labyrinth.minotaur_exam import ExamCaseResult, MinotaurExamReceipt
from labyrinth.objective_contract import ObjectiveContract, SkillTarget
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.training_scope import TrainingScopeSpec


def cognitive_contract(*, seed=103, focus="context", specificity=5, scale=5,
                       cycles=2, domain="empathy", skill="cognitive_empathy",
                       rubric="perspective_choice_train_v1"):
    scope = TrainingScopeSpec(
        objective_id=f"cognitive-runtime-{seed}", domain=domain,
        skills=(skill,), specificity=specificity, scale=scale,
        max_cycles=cycles, seed=seed,
    )
    return ObjectiveContract(
        scope=scope, intention="Practice synthetic perspective-taking choices.",
        targets=(SkillTarget(
            skill=skill, focus_tags=(focus,),
            observable_behaviors=("Select a perspective without assuming private motives.",),
            rubric_id=rubric, minimum_score_percent=80, minimum_gain_percent=10,
        ),), excluded_scopes=("Reactive or emotional empathy claims",),
        forbidden_claims=("No human empathy mastery or graduation can be inferred.",),
    )


def development_fail_for_reroll(waiting):
    """Synthetic *test-only* DEV diagnostic, not an evaluator attestation."""
    cases = tuple(
        ExamCaseResult(
            case_id=f"dev-cognitive-{waiting.cycle_index}-{kind}-{i}",
            skill="cognitive_empathy", split=kind,
            score=0.0, baseline_score=0.0,
            prompt_sha256=f"{1 + i + (4 if kind == 'transfer' else 0):064x}",
        ) for kind in ("heldout", "transfer") for i in range(4)
    )
    receipt = MinotaurExamReceipt(
        exam_id=f"dev-diagnostic-cycle-{waiting.cycle_index}",
        objective_id=waiting.scope.objective_id, cycle_index=waiting.cycle_index,
        checkpoint_sha256=waiting.checkpoint_sha256,
        baseline_sha256=waiting.baseline_checkpoint_sha256,
        evaluator_id="test-only-synthetic-development", cases=cases,
    )
    return face_minotaur(waiting, receipt)


class CognitiveObjectiveTrainingRunTests(unittest.TestCase):
    def test_fifteen_rooms_public_choice_scoring_and_completion_handoff(self):
        from labyrinth.cognitive_objective_training_run import (
            CognitiveObjectiveTrainingRun, open_approved_objective_training_run,
        )
        contract = cognitive_contract()
        journey = begin_journey(contract.scope, checkpoint_sha256="1" * 64)
        run = open_approved_objective_training_run(contract, journey, provider_id=PROVIDER_ID)
        self.assertIsInstance(run, CognitiveObjectiveTrainingRun)
        gold = prepare_cognitive_empathy_train_rooms(contract, journey.maze)
        observed = []
        for private in gold:
            def learner(view, private=private):
                self.assertEqual(view, private.trainee_view)
                self.assertEqual(view.skill, "cognitive_empathy")
                self.assertNotIn("answer", vars(view))
                self.assertIn("Respond with only A, B, or C.", view.prompt)
                self.assertNotIn("ANSWER: <integer>", view.prompt)
                return private.prepared.challenge.answer
            result = run.run_training_chamber(learner)
            observed.append(result)
            self.assertEqual((result.points, result.correct), (100, True))
            self.assertFalse(result.independent_exam_evidence)
            self.assertFalse(result.graduation_authorized)
        self.assertTrue(run.is_complete)
        self.assertEqual(run.opened_count, 15)
        self.assertEqual(len({item.prompt_sha256 for item in observed}), 15)
        self.assertFalse(run.authorizes_host_execution)
        self.assertFalse(run.authorizes_training)
        self.assertFalse(run.authorizes_exam)
        self.assertFalse(run.authorizes_graduation)
        receipt = run.self_reported_completion_receipt(checkpoint_after_sha256="2" * 64)
        self.assertEqual(receipt.room_case_ids, tuple(
            (item.room_id, item.case_id, item.prompt_sha256) for item in observed
        ))
        self.assertEqual(complete_maze(journey, receipt).phase, AWAITING_EXAM)

    def test_wrong_choice_is_zero_not_arithmetic_and_pending_callback_is_replay_safe(self):
        from labyrinth.cognitive_objective_training_run import CognitiveObjectiveTrainingRun
        contract = cognitive_contract(seed=202)
        journey = begin_journey(contract.scope, checkpoint_sha256="3" * 64)
        run = CognitiveObjectiveTrainingRun(contract, journey)
        with self.assertRaisesRegex(RuntimeError, "disconnect"):
            run.run_training_chamber(lambda view: (_ for _ in ()).throw(RuntimeError("disconnect")))
        self.assertEqual(run.opened_count, 1)
        with self.assertRaisesRegex(ValueError, "already exposed"):
            run.next_training_prompt()
        for invalid in ("ANSWER: 0", "1", "D", "A or B"):
            # Wrong bounded answers count as incorrect training, not a parse error.
            observed = run.submit_training_response(invalid)
            self.assertEqual((observed.points, observed.correct), (0, False))
            break
        for _ in range(1, 15):
            scored = run.run_training_chamber(lambda view: "ANSWER: 1")
            self.assertEqual(scored.points, 0)
        self.assertTrue(run.is_complete)
        with self.assertRaises(ValueError):
            run.next_training_prompt()

    def test_two_fifteen_room_cycles_canonical_prior_exposure_and_exhaustion(self):
        from labyrinth.cognitive_objective_training_run import CognitiveObjectiveTrainingRun
        contract = cognitive_contract(seed=304)
        first = begin_journey(contract.scope, checkpoint_sha256="4" * 64)
        run0 = CognitiveObjectiveTrainingRun(contract, first)
        original = prepare_cognitive_empathy_train_rooms(contract, first.maze)
        for _ in range(15):
            run0.run_training_chamber(lambda view: "A")
        waiting = complete_maze(first, run0.self_reported_completion_receipt(
            checkpoint_after_sha256="5" * 64,
        ))
        second = development_fail_for_reroll(waiting)
        self.assertEqual((second.phase, second.cycle_index), (TRAINING, 1))
        run1 = CognitiveObjectiveTrainingRun(contract, second)
        fresh = prepare_cognitive_empathy_train_rooms(contract, second.maze)
        self.assertEqual(len(original + fresh), 30)
        self.assertEqual(len({r.semantic_sha256 for r in original + fresh}), 30)
        for _ in range(15):
            run1.run_training_chamber(lambda view: "B")
        waiting2 = complete_maze(second, run1.self_reported_completion_receipt(
            checkpoint_after_sha256="6" * 64,
        ))
        self.assertEqual(waiting2.phase, AWAITING_EXAM)
        self.assertEqual(len(set(waiting2.used_training_case_ids)), 30)
        self.assertEqual(len(set(waiting2.used_training_prompt_sha256)), 30)
        contract3 = cognitive_contract(seed=304, cycles=3)
        with self.assertRaisesRegex(ValueError, "exhausted"):
            prepare_cognitive_empathy_train_rooms(
                contract3, generate_maze_plan(contract3.scope, 2),
            )

    def test_operator_prior_semantic_exclusion_fails_before_exposure(self):
        from labyrinth.cognitive_objective_training_run import CognitiveObjectiveTrainingRun
        contract = cognitive_contract(seed=401)
        journey = begin_journey(contract.scope, checkpoint_sha256="7" * 64)
        prior = prepare_cognitive_empathy_train_rooms(contract, journey.maze)
        with self.assertRaisesRegex(ValueError, "duplicate or operator-reserved"):
            CognitiveObjectiveTrainingRun(
                contract, journey, forbidden_semantic_sha256=(prior[0].semantic_sha256,),
            )
        with self.assertRaises(ValueError):
            CognitiveObjectiveTrainingRun(
                contract, journey, forbidden_semantic_sha256=("invalid",),
            )
        with self.assertRaises(ValueError):
            CognitiveObjectiveTrainingRun(
                contract, journey, forbidden_semantic_sha256=[prior[0].semantic_sha256],
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            CognitiveObjectiveTrainingRun(
                contract, journey,
                forbidden_semantic_sha256=(prior[0].semantic_sha256,
                                           prior[0].semantic_sha256),
            )

    def test_rejects_wrong_provider_scope_forged_journey_and_nontraining_phase(self):
        from labyrinth.cognitive_objective_training_run import (
            CognitiveObjectiveTrainingRun, open_approved_objective_training_run,
        )
        contract = cognitive_contract()
        journey = begin_journey(contract.scope, checkpoint_sha256="8" * 64)
        for bad in ("semantic_program_train_v1", "unknown", 1):
            with self.subTest(provider=bad), self.assertRaises(ValueError):
                open_approved_objective_training_run(contract, journey, provider_id=bad)
        for invalid in (
            cognitive_contract(domain="reasoning"),
            cognitive_contract(skill="reactive_empathy"),
            cognitive_contract(rubric="fake_rubric"),
        ):
            with self.assertRaises(ValueError):
                CognitiveObjectiveTrainingRun(
                    invalid, begin_journey(invalid.scope, checkpoint_sha256="9" * 64),
                )
        with self.assertRaises(ValueError):
            CognitiveObjectiveTrainingRun(contract, replace(journey, phase=AWAITING_EXAM))
        with self.assertRaises(ValueError):
            CognitiveObjectiveTrainingRun(contract, replace(journey, checkpoint_sha256="f"*64))
        forged = replace(journey)
        object.__setattr__(forged.maze.rooms[0], "room_id", "forged")
        with self.assertRaises(ValueError):
            CognitiveObjectiveTrainingRun(contract, forged)

    def test_forged_view_observation_receipt_and_provider_substitution_fail_closed(self):
        from labyrinth.cognitive_objective_training_run import CognitiveObjectiveTrainingRun
        contract = cognitive_contract(seed=502)
        journey = begin_journey(contract.scope, checkpoint_sha256="a" * 64)
        run = CognitiveObjectiveTrainingRun(contract, journey)
        first_view = run.next_training_prompt()
        original = (first_view.challenge_id, first_view.prompt_sha256)
        object.__setattr__(first_view, "challenge_id", "fake-gold-case")
        object.__setattr__(first_view, "prompt_sha256", "0" * 64)
        first_obs = run.submit_training_response("C")
        self.assertEqual((first_obs.case_id, first_obs.prompt_sha256), original)
        object.__setattr__(first_obs, "correct", True)
        object.__setattr__(first_obs, "case_id", "fake")
        self.assertEqual(run.observations[0].case_id, original[0])
        for _ in range(1, 15):
            run.run_training_chamber(lambda view: "A")
        receipt = run.self_reported_completion_receipt(checkpoint_after_sha256="b" * 64)
        object.__setattr__(receipt, "checkpoint_after_sha256", "c" * 64)
        object.__setattr__(receipt, "room_case_ids", (("forged", "forged", "0" * 64),))
        repeated = run.self_reported_completion_receipt(checkpoint_after_sha256="b" * 64)
        self.assertEqual(repeated.room_case_ids[0][1:], original)
        self.assertEqual(repeated.checkpoint_after_sha256, "b" * 64)
        with self.assertRaisesRegex(ValueError, "already pinned"):
            run.self_reported_completion_receipt(checkpoint_after_sha256="c" * 64)
        self.assertEqual(complete_maze(journey, repeated).phase, AWAITING_EXAM)

    def test_synthetic_source_change_cannot_forge_opened_case_on_score_or_receipt(self):
        from labyrinth.cognitive_objective_training_run import CognitiveObjectiveTrainingRun
        contract = cognitive_contract(seed=604)
        journey = begin_journey(contract.scope, checkpoint_sha256="d" * 64)
        run = CognitiveObjectiveTrainingRun(contract, journey)
        run.next_training_prompt()
        # A compromised/replaced source regeneration cannot silently substitute
        # a different source case for an already opened public prompt.
        with patch("labyrinth.cognitive_objective_training_run.prepare_cognitive_empathy_train_rooms",
                   return_value=()):
            with self.assertRaises(ValueError):
                run.submit_training_response("A")
        self.assertEqual(run.opened_count, 1)
        self.assertEqual(run.observations, ())
        run.submit_training_response("A")


if __name__ == "__main__":
    unittest.main()
