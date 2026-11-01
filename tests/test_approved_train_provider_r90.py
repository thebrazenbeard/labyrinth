"""An objective declaration cannot install a provider or confer exam authority."""

from dataclasses import FrozenInstanceError, replace
import unittest

from labyrinth.approved_train_provider_r90 import (
    approved_train_provider_capabilities,
    rate_objective_train_response,
    route_objective_train_room,
)
from labyrinth.objective_contract import ObjectiveContract, SkillTarget
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.semantic_chambers import semantic_focus_profile
from labyrinth.training_scope import TrainingScopeSpec


def _objective(*, skill="arithmetic", domain="reasoning", specificity=2,
               focus=None, rubric=None):
    if focus is None:
        focus = ("integer_add_subtract" if skill == "arithmetic"
                 else semantic_focus_profile(specificity)
                 if skill == "program_reasoning" else "context")
    if rubric is None:
        rubric = ("arithmetic_exact_answer_train_v1" if skill == "arithmetic"
                  else "program_structure_v1" if skill == "program_reasoning"
                  else "invented_rubric_v1")
    scope = TrainingScopeSpec(
        objective_id=f"r90-{skill}-{specificity}", domain=domain,
        skills=(skill,), specificity=specificity, scale=1, max_cycles=2, seed=90,
    )
    return ObjectiveContract(
        scope=scope,
        intention="Practice narrowly specified exercises in a TRAIN environment.",
        targets=(SkillTarget(
            skill=skill, focus_tags=(focus,),
            observable_behaviors=("Produces the expected answer format.",),
            rubric_id=rubric, minimum_score_percent=75,
            minimum_gain_percent=5,
        ),),
        excluded_scopes=("Private examination tasks.",),
        forbidden_claims=("Local TRAIN scoring proves mastery.",),
    )


class ApprovedTrainProviderR90Tests(unittest.TestCase):
    def test_manifest_is_explicit_with_generator_and_rater_capabilities(self):
        manifest = approved_train_provider_capabilities()
        self.assertEqual(tuple(x.provider_id for x in manifest),
                         ("arithmetic_static_train_v1", "semantic_program_train_v1"))
        self.assertEqual(tuple(x.skill for x in manifest),
                         ("arithmetic", "program_reasoning"))
        self.assertTrue(all(x.has_generator and x.has_local_rater for x in manifest))
        self.assertTrue(all(x.split == "train" for x in manifest))
        self.assertFalse(any(x.can_access_private_exam for x in manifest))
        self.assertFalse(any(x.can_authorize_graduation for x in manifest))
        with self.assertRaises(FrozenInstanceError):
            manifest[0].can_authorize_graduation = True

    def test_arithmetic_routes_and_rates_train_only_with_actual_content(self):
        contract = _objective()
        plan = generate_maze_plan(contract.scope, 0)
        routed = route_objective_train_room(contract, plan, plan.rooms[0])
        self.assertEqual(routed.provider_id, "arithmetic_static_train_v1")
        self.assertEqual(routed.prepared.challenge.split, "train")
        self.assertEqual(routed.prepared.challenge.kind, "arithmetic")
        self.assertIn("Calculate ", routed.trainee_view.prompt)
        self.assertNotIn("answer", vars(routed.trainee_view))
        self.assertEqual(dict(routed.prepared.challenge.metadata)["objective_focus_tag"],
                         "integer_add_subtract")
        self.assertEqual(dict(routed.prepared.challenge.metadata)["objective_contract_sha256"],
                         contract.fingerprint)
        correct = rate_objective_train_response(
            contract, plan, plan.rooms[0],
            "ANSWER: " + routed.prepared.challenge.answer,
        )
        self.assertTrue(correct.correct)
        self.assertEqual(correct.points, 1)
        self.assertEqual(correct.evidence_split, "train")
        self.assertFalse(correct.independent_exam_evidence)
        self.assertFalse(correct.graduation_authorized)
        self.assertEqual(correct.prompt_sha256, routed.trainee_view.prompt_sha256)
        incorrect = rate_objective_train_response(
            contract, plan, plan.rooms[0], "relabel train as private final",
        )
        self.assertFalse(incorrect.correct)
        self.assertEqual(incorrect.points, 0)

    def test_existing_semantic_program_provider_is_compatible(self):
        contract = _objective(skill="program_reasoning", specificity=3)
        plan = generate_maze_plan(contract.scope, 0)
        routed = route_objective_train_room(contract, plan, plan.rooms[0])
        self.assertEqual(routed.provider_id, "semantic_program_train_v1")
        self.assertEqual(dict(routed.prepared.challenge.metadata)["semantic_focus"],
                         semantic_focus_profile(3))
        result = rate_objective_train_response(
            contract, plan, plan.rooms[0],
            f"ANSWER: {routed.prepared.challenge.answer}",
        )
        self.assertTrue(result.correct)
        self.assertFalse(result.independent_exam_evidence)

    def test_nonexecutable_objectives_fail_closed(self):
        for contract in (
            _objective(skill="cognitive_empathy", domain="empathy"),
            _objective(skill="custom_skill", domain="reasoning"),
            _objective(skill="arithmetic", focus="harmful_arbitrary_tag"),
            _objective(skill="arithmetic", rubric="invented_rubric_v1"),
            _objective(skill="arithmetic", domain="arbitrary_domain"),
            _objective(skill="program_reasoning", focus="integer_add_subtract"),
        ):
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                plan = generate_maze_plan(contract.scope, 0)
                route_objective_train_room(contract, plan, plan.rooms[0])

    def test_provider_id_cannot_install_unauthorized_plugin(self):
        contract = _objective()
        plan = generate_maze_plan(contract.scope, 0)
        for provider_id in ("cognitive_empathy", "untrusted", "semantic_program_train_v1", 9):
            with self.subTest(provider_id=provider_id), self.assertRaises(ValueError):
                route_objective_train_room(contract, plan, plan.rooms[0],
                                           provider_id=provider_id)
        with self.assertRaises(TypeError):
            route_objective_train_room(contract, plan, plan.rooms[0],
                                       generator=lambda *args: None)

    def test_forged_maze_and_exam_routes_fail_closed(self):
        contract = _objective()
        plan = generate_maze_plan(contract.scope, 0)
        different = generate_maze_plan(replace(contract.scope, seed=91), 0)
        # Frozen dataclasses are not security boundaries; mutate a separate
        # duplicate in the same way a hostile Python caller could.
        forged_exam_room = replace(plan.rooms[0])
        object.__setattr__(forged_exam_room, "purpose", "exam")
        for bad_plan, bad_room in (
            (different, different.rooms[0]),
            (plan, forged_exam_room),
            (plan, replace(plan.rooms[0], case_seed=5)),
            (replace(plan, objective_id="other"), plan.rooms[0]),
        ):
            with self.subTest(bad_plan=bad_plan, bad_room=bad_room):
                with self.assertRaises(ValueError):
                    route_objective_train_room(contract, bad_plan, bad_room)
        regenerated = generate_maze_plan(contract.scope, 1)
        with self.assertRaisesRegex(ValueError, "JourneyState"):
            route_objective_train_room(contract, regenerated, regenerated.rooms[0])

    def test_forged_response_does_not_become_signed_development_evidence(self):
        contract = _objective()
        plan = generate_maze_plan(contract.scope, 0)
        score = rate_objective_train_response(
            contract, plan, plan.rooms[0], "ANSWER: 0",
        )
        for forged_field in ("graduation_authorized", "independent_exam_evidence"):
            with self.subTest(field=forged_field), self.assertRaises((TypeError, ValueError)):
                replace(score, **{forged_field: True})


if __name__ == "__main__":
    unittest.main()
