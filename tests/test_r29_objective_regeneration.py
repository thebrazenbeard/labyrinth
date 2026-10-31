"""R29 deterministic PUBLIC TOY Minotaur regeneration falsifiers."""

from dataclasses import replace
import unittest

from labyrinth.r28_reward_hacking import ChallengeObservation, build_public_toy_challenge
from labyrinth.r29_objective_regeneration import (
    PublicTemplateFamily,
    freeze_public_regeneration_boundary,
    regenerate_from_public_shortcut,
    verify_public_regeneration_receipt,
)
from labyrinth import r29_objective_regeneration as regen_module
from labyrinth.trainingday_objective_metamorphic import (
    FormalObjectiveContract, FormalSkillCriterion,
)


CONTRACT = FormalObjectiveContract(
    "r29-toy-objective", "skill_mastery",
    (FormalSkillCriterion("arithmetic", "exact", 2, 2),
     FormalSkillCriterion("source_priority", "rank", 2, 1)),
    ("private_ood",), "a" * 64, "b" * 64, "c" * 64,
)
TEMPLATES = (
    PublicTemplateFamily("arithmetic", "1" * 64, "2" * 64),
    PublicTemplateFamily("source_priority", "3" * 64, "4" * 64),
)
TEMPLATES_N2 = (
    PublicTemplateFamily("arithmetic", "5" * 64, "6" * 64),
    PublicTemplateFamily("source_priority", "7" * 64, "8" * 64),
)
TEMPLATES_N3 = (
    PublicTemplateFamily("arithmetic", "9" * 64, "a" * 64),
    PublicTemplateFamily("source_priority", "b" * 64, "c" * 64),
)


def fixture(contract=CONTRACT, seed="source-1"):
    source = build_public_toy_challenge(contract, seed, 2)
    observations = tuple(
        ChallengeObservation(
            case_id=case.case_id, objective_id=source.objective_id,
            skill_id=case.skill_id, prompt_sha256=case.prompt_sha256,
            correct=True, format_valid=True, evidence_source="PUBLIC_TOY_ORACLE",
            hint_embedded=(i == 0),
        ) for i, case in enumerate(source.cases)
    )
    return source, observations, tuple(case.case_id for case in source.cases)


def regen(*, contract=CONTRACT, source=None, observations=None, selected=None,
          boundary=None, generation=1, seed="cycle-1", per_skill=2,
          templates=TEMPLATES, predecessor=None, predecessor_ancestors=()):
    fresh_source, fresh_observations, fresh_selected = fixture(contract)
    if boundary is None:
        boundary = freeze_public_regeneration_boundary(contract, "d" * 64, "e" * 64)
    return regenerate_from_public_shortcut(
        contract,
        fresh_source if source is None else source,
        fresh_observations if observations is None else observations,
        fresh_selected if selected is None else selected,
        boundary=boundary,
        generation=generation,
        seed=seed,
        per_skill=per_skill,
        template_families=templates,
        predecessor=predecessor,
        predecessor_ancestors=predecessor_ancestors,
    )


class ObjectiveRegenerationTests(unittest.TestCase):
    def test_seeded_regeneration_is_repeatable_isolated_and_non_authorizing(self):
        first = regen()
        self.assertEqual(first, regen())
        self.assertEqual(first.attack_findings, ("PROMPT_HINT_LEAKAGE",))
        self.assertEqual(first.generation, 1)
        self.assertFalse(first.graduation_authorized)
        self.assertFalse(first.promotion_allowed)
        self.assertFalse(first.independent_exam_verified)
        self.assertTrue(first.boundary.objective_fingerprint)
        self.assertEqual(first.boundary.rights_sha256, "d" * 64)
        self.assertEqual(first.boundary.reference_sha256, "e" * 64)
        development = first.development_plan.cases
        challenge = first.untouched_challenge_plan.cases
        self.assertTrue({c.case_id for c in development}.isdisjoint(c.case_id for c in challenge))
        self.assertTrue({c.prompt_sha256 for c in development}.isdisjoint(
            c.prompt_sha256 for c in challenge))
        self.assertEqual(len(development), 4)
        self.assertEqual(len(challenge), 4)
        self.assertEqual(first.untouched_challenge_plan.training_prompt_sha256,
                         tuple(sorted(c.prompt_sha256 for c in development)))
        self.assertEqual(verify_public_regeneration_receipt(CONTRACT, first).status,
                         "PUBLIC_TOY_RECEIPT_VALID")

    def test_distinct_seed_changes_all_synthetic_case_ids(self):
        a, b = regen(seed="cycle-1"), regen(seed="cycle-2")
        self.assertNotEqual(a.commitment_sha256, b.commitment_sha256)
        all_a = {c.case_id for c in a.development_plan.cases + a.untouched_challenge_plan.cases}
        all_b = {c.case_id for c in b.development_plan.cases + b.untouched_challenge_plan.cases}
        self.assertTrue(all_a.isdisjoint(all_b))

    def test_weakness_and_template_condition_case_id_namespace(self):
        source, observations, selected = fixture()
        hint = regen(source=source, observations=observations, selected=selected)
        imitation = regen(source=source, observations=(replace(
            observations[0], hint_embedded=False, grading_template_imitation=True),)
            + observations[1:], selected=selected)
        self.assertEqual(imitation.attack_findings, ("GRADING_TEMPLATE_IMITATION",))
        self.assertNotEqual(
            {c.case_id for c in hint.development_plan.cases},
            {c.case_id for c in imitation.development_plan.cases},
        )
        templates_changed = regen(templates=TEMPLATES_N2)
        self.assertNotEqual(
            {c.case_id for c in hint.untouched_challenge_plan.cases},
            {c.case_id for c in templates_changed.untouched_challenge_plan.cases},
        )

    def test_no_demonstrated_shortcut_is_not_regeneration_authority(self):
        source, observations, selected = fixture()
        clean = tuple(replace(x, hint_embedded=False) for x in observations)
        with self.assertRaises(ValueError):
            regen(source=source, observations=clean, selected=selected)
        with self.assertRaises(ValueError):
            regen(observations=(replace(observations[0], case_id="stale-id"),)
                 + observations[1:])

    def test_same_template_across_splits_or_skills_is_rejected(self):
        with self.assertRaises(ValueError):
            regen(templates=(replace(TEMPLATES[0], challenge_sha256="1" * 64),
                             TEMPLATES[1]))
        with self.assertRaises(ValueError):
            regen(templates=(TEMPLATES[0],
                             replace(TEMPLATES[1], challenge_sha256="1" * 64)))
        with self.assertRaises(ValueError):
            regen(templates=(TEMPLATES[0], TEMPLATES[0]))

    def test_immutable_objective_rights_reference_and_receipt_tampering_fail(self):
        receipt = regen()
        changed = replace(CONTRACT, evaluator_sha256="f" * 64)
        with self.assertRaises(ValueError):
            regen(contract=changed, boundary=receipt.boundary)
        self.assertEqual(verify_public_regeneration_receipt(changed, receipt).status,
                         "PUBLIC_TOY_RECEIPT_INVALID")
        self.assertEqual(verify_public_regeneration_receipt(
            CONTRACT, replace(receipt, boundary=replace(receipt.boundary,
                                                       rights_sha256="f" * 64))).status,
            "PUBLIC_TOY_RECEIPT_INVALID")
        self.assertEqual(verify_public_regeneration_receipt(
            CONTRACT, replace(receipt, boundary=replace(receipt.boundary,
                                                       reference_sha256="f" * 64))).status,
            "PUBLIC_TOY_RECEIPT_INVALID")
        self.assertEqual(verify_public_regeneration_receipt(
            CONTRACT, replace(receipt, attack_findings=("FAKE_EVAL_PASS",))).status,
            "PUBLIC_TOY_RECEIPT_INVALID")

    def test_impossible_min_cases_and_curriculum_downgrade_rejected(self):
        with self.assertRaises(ValueError):
            regen(per_skill=1)
        source = build_public_toy_challenge(CONTRACT, "source-difficulty-3", 3)
        observations = tuple(ChallengeObservation(
            c.case_id, source.objective_id, c.skill_id, c.prompt_sha256,
            True, True, "PUBLIC_TOY_ORACLE", hint_embedded=(i == 0),
        ) for i, c in enumerate(source.cases))
        with self.assertRaises(ValueError):
            regen(source=source, observations=observations,
                  selected=tuple(c.case_id for c in source.cases), per_skill=2)
        impossible = replace(CONTRACT, skills=(
            FormalSkillCriterion("giant", "exact", 300, 300),
        ))
        # The underlying R28 generator itself caps public per-skill case
        # count at 256. The contradictory min_cases=300 must stay invalid.
        with self.assertRaises(ValueError):
            build_public_toy_challenge(impossible, "giant-1", 300)
        with self.assertRaises(ValueError):
            regenerate_from_public_shortcut(
                impossible, *fixture()[:2], (),
                boundary=freeze_public_regeneration_boundary(
                    impossible, "d" * 64, "e" * 64),
                generation=1, seed="giant-2", per_skill=1,
                template_families=(PublicTemplateFamily("giant", "1" * 64, "2" * 64),),
            )

    def test_stale_replay_requires_next_generation_new_seed_and_correct_source(self):
        first = regen()
        with self.assertRaises(ValueError):
            regen(predecessor=first, generation=1, seed="cycle-1")
        with self.assertRaises(ValueError):
            regen(predecessor=first, generation=2, seed="cycle-1")
        with self.assertRaises(ValueError):
            regen(predecessor=first, generation=2, seed="cycle-2")
        fresh_source = first.untouched_challenge_plan
        fresh_obs = tuple(ChallengeObservation(
            c.case_id, fresh_source.objective_id, c.skill_id, c.prompt_sha256,
            True, True, "PUBLIC_TOY_ORACLE", grading_template_imitation=(i == 0),
        ) for i, c in enumerate(fresh_source.cases))
        with self.assertRaises(ValueError):
            regen(source=fresh_source, observations=fresh_obs,
                  selected=tuple(c.case_id for c in fresh_source.cases),
                  boundary=first.boundary, predecessor=first,
                  generation=2, seed="cycle-2", templates=TEMPLATES)
        second = regen(source=fresh_source, observations=fresh_obs,
                       selected=tuple(c.case_id for c in fresh_source.cases),
                       boundary=first.boundary, predecessor=first,
                       generation=2, seed="cycle-2", templates=TEMPLATES_N2)
        self.assertEqual(second.attack_findings, ("GRADING_TEMPLATE_IMITATION",))
        self.assertEqual(second.predecessor_commitment_sha256, first.commitment_sha256)
        self.assertNotEqual(second.commitment_sha256, first.commitment_sha256)
        self.assertEqual(verify_public_regeneration_receipt(
            CONTRACT, second, predecessor=first).status,
                         "PUBLIC_TOY_RECEIPT_VALID")
        self.assertEqual(verify_public_regeneration_receipt(CONTRACT, replace(
            second, predecessor_commitment_sha256="0" * 64),
            predecessor=first).status,
            "PUBLIC_TOY_RECEIPT_INVALID")

    def test_three_generation_lineage_requires_entire_custody_chain(self):
        first = regen()

        def shortcut_on(plan):
            return tuple(ChallengeObservation(
                c.case_id, plan.objective_id, c.skill_id, c.prompt_sha256,
                True, True, "PUBLIC_TOY_ORACLE", training_example_reused=(i == 0),
            ) for i, c in enumerate(plan.cases))

        second = regen(source=first.untouched_challenge_plan,
                       observations=shortcut_on(first.untouched_challenge_plan),
                       selected=tuple(c.case_id for c in first.untouched_challenge_plan.cases),
                       boundary=first.boundary, predecessor=first,
                       templates=TEMPLATES_N2, generation=2, seed="cycle-2")
        third = regen(source=second.untouched_challenge_plan,
                      observations=shortcut_on(second.untouched_challenge_plan),
                      selected=tuple(c.case_id for c in second.untouched_challenge_plan.cases),
                      boundary=first.boundary, predecessor=second,
                      predecessor_ancestors=(first,),
                      templates=TEMPLATES_N3, generation=3, seed="cycle-3")
        self.assertEqual(verify_public_regeneration_receipt(
            CONTRACT, third, predecessor=second,
            predecessor_ancestors=(first,)).status, "PUBLIC_TOY_RECEIPT_VALID")
        self.assertEqual(verify_public_regeneration_receipt(
            CONTRACT, third, predecessor=second).status, "PUBLIC_TOY_RECEIPT_INVALID")

    def test_rehashed_g2_is_not_verifiable_without_exact_predecessor(self):
        first = regen()
        source = first.untouched_challenge_plan
        observations = tuple(ChallengeObservation(
            c.case_id, source.objective_id, c.skill_id, c.prompt_sha256,
            True, True, "PUBLIC_TOY_ORACLE", hint_embedded=(i == 0),
        ) for i, c in enumerate(source.cases))
        second = regen(source=source, observations=observations,
                       selected=tuple(c.case_id for c in source.cases),
                       boundary=first.boundary, predecessor=first,
                       generation=2, seed="cycle-2", templates=TEMPLATES_N2)
        self.assertEqual(verify_public_regeneration_receipt(CONTRACT, second).status,
                         "PUBLIC_TOY_RECEIPT_INVALID")
        self.assertEqual(verify_public_regeneration_receipt(
            CONTRACT, second, predecessor=first).status,
            "PUBLIC_TOY_RECEIPT_VALID")
        wrong = regen(seed="unrelated")
        self.assertEqual(verify_public_regeneration_receipt(
            CONTRACT, second, predecessor=wrong).status,
            "PUBLIC_TOY_RECEIPT_INVALID")

        # The attacker can recompute EVERY public hash/plan; that still cannot
        # make a reused predecessor challenge template valid at the verifier.
        draft = replace(second, template_families=TEMPLATES)
        dev = build_public_toy_challenge(CONTRACT, regen_module._seed_for(draft, "dev"),
                                         draft.per_skill, training_prompt_sha256=(
                                             draft.source_training_prompt_sha256))
        challenge = build_public_toy_challenge(
            CONTRACT, regen_module._seed_for(draft, "holdout"), draft.per_skill,
            training_prompt_sha256=tuple(sorted(c.prompt_sha256 for c in dev.cases)))
        draft = replace(draft, development_plan=dev, untouched_challenge_plan=challenge)
        replay = replace(draft, commitment_sha256=regen_module._commit(
            b"labyrinth.r29.public-toy-receipt.v1",
            regen_module._receipt_payload(draft),
        ))
        self.assertEqual(verify_public_regeneration_receipt(CONTRACT, replay).status,
                         "PUBLIC_TOY_RECEIPT_INVALID")
        self.assertEqual(verify_public_regeneration_receipt(
            CONTRACT, replay, predecessor=first).status,
            "PUBLIC_TOY_RECEIPT_INVALID")

    def test_training_eval_overlap_injected_into_receipt_fails_closed(self):
        receipt = regen()
        tampered = replace(receipt, untouched_challenge_plan=replace(
            receipt.untouched_challenge_plan,
            cases=(receipt.development_plan.cases[0],)
            + receipt.untouched_challenge_plan.cases[1:],
        ))
        verdict = verify_public_regeneration_receipt(CONTRACT, tampered)
        self.assertEqual(verdict.status, "PUBLIC_TOY_RECEIPT_INVALID")
        self.assertFalse(verdict.graduation_authorized)
        self.assertFalse(verdict.promotion_allowed)
        tampered_templates = replace(receipt, template_families=(
            replace(receipt.template_families[0], challenge_sha256="1" * 64),
            receipt.template_families[1],
        ))
        self.assertEqual(verify_public_regeneration_receipt(
            CONTRACT, tampered_templates).status, "PUBLIC_TOY_RECEIPT_INVALID")


if __name__ == "__main__":
    unittest.main()
