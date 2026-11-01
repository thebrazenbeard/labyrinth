"""Required source-only TRAIN provider boundary and novelty regressions."""

import dataclasses
import hashlib
import re
import unittest

from labyrinth.approved_cognitive_empathy_train_r90 import (
    APPROVED_TRAIN_PROVIDER_REGISTRY, MAX_COGNITIVE_ROOMS, PROVIDER_ID,
    CognitiveTrainingRoom, prepare_cognitive_empathy_train_rooms,
    semantic_scenario_sha256, semantic_sha256_from_prompt,
)
from labyrinth.objective_contract import ObjectiveContract, SkillTarget
from labyrinth.regenerative_maze import MazePlan, MazeRoom, generate_maze_plan
from labyrinth.training_scope import TrainingScopeSpec


def contract(seed=3, scale=3, specificity=5, focus="context", domain="empathy",
             skill="cognitive_empathy", rubric="perspective_choice_train_v1", cycles=2):
    scope = TrainingScopeSpec(
        objective_id=f"cognitive-perspective-{seed}", domain=domain,
        skills=(skill,), specificity=specificity, scale=scale, max_cycles=cycles, seed=seed,
    )
    return ObjectiveContract(
        scope=scope, intention="Practice synthetic perspective-taking choices.",
        targets=(SkillTarget(
            skill=skill, focus_tags=(focus,),
            observable_behaviors=("Select a perspective without assuming private motives.",),
            rubric_id=rubric, minimum_score_percent=80, minimum_gain_percent=10,
        ),),
        excluded_scopes=("Reactive or emotional empathy claims",),
        forbidden_claims=("No human empathy mastery or graduation can be inferred.",),
    )


def prepared(seed=3, scale=3, **kwargs):
    objective = contract(seed=seed, scale=scale)
    plan = generate_maze_plan(objective.scope, 0)
    return prepare_cognitive_empathy_train_rooms(objective, plan, **kwargs)


class CognitivePerspectiveTrainTests(unittest.TestCase):
    def test_registry_is_static_and_unmodifiable(self):
        self.assertEqual(tuple(APPROVED_TRAIN_PROVIDER_REGISTRY), (PROVIDER_ID,))
        with self.assertRaises(TypeError):
            APPROVED_TRAIN_PROVIDER_REGISTRY["unauthorized"] = ("x", "y", "z")

    def test_seed_three_novelty_regression_nine_rooms(self):
        rooms = prepared(3)
        self.assertEqual(len(rooms), 9)
        self.assertEqual(len({room.semantic_sha256 for room in rooms}), 9)

    def test_ninety_nine_seed_nine_room_fuzz_has_no_semantic_duplicates(self):
        for seed in range(99):
            rooms = prepared(seed)
            digests = [r.semantic_sha256 for r in rooms]
            self.assertEqual(len(set(digests)), 9, f"seed {seed} duplicate semantic scenario")

    def test_fifteen_room_maximum_stress(self):
        for seed in (0, 3, 17, 71, 98):
            rooms = prepared(seed, scale=5)
            self.assertEqual(len(rooms), MAX_COGNITIVE_ROOMS)
            self.assertEqual(len(set(r.semantic_sha256 for r in rooms)), 15)

    def test_cannot_materialize_above_fifteen_rooms(self):
        objective = contract(scale=5)
        plan = generate_maze_plan(objective.scope, 0)
        extra = MazeRoom(room_id="f" * 32, skill="cognitive_empathy", case_seed=2**31 - 1)
        expanded = MazePlan(
            objective_id=plan.objective_id, scope_fingerprint=plan.scope_fingerprint,
            cycle_index=plan.cycle_index, rooms=plan.rooms + (extra,),
        )
        with self.assertRaises(ValueError):
            prepare_cognitive_empathy_train_rooms(objective, expanded)

    def test_same_scope_seed_reproduces_ids_prompts_golds(self):
        self.assertEqual(prepared(3), prepared(3))

    def test_semantic_digest_is_case_id_and_option_order_invariant(self):
        room = prepared()[0]
        prompt = room.prepared.challenge.prompt
        altered = prompt.replace(room.prepared.challenge.id, "train-cognitive-different-case")
        lines = altered.splitlines()
        lines[4], lines[6] = "C. " + lines[4][3:], "A. " + lines[6][3:]
        lines[5] = "B. " + lines[5][3:]
        self.assertEqual(semantic_sha256_from_prompt(prompt),
                         semantic_sha256_from_prompt("\n".join(lines)))

    def test_nfkc_case_and_spacing_normalization(self):
        first = semantic_scenario_sha256("A  FOCUS   case", ("Full width Ａ", "B option", "C option"))
        second = semantic_scenario_sha256(" a focus case ", ("c OPTION", "full width a", "b OPTION"))
        self.assertEqual(first, second)

    def test_semantic_parser_rejects_duplicate_or_missing_choices(self):
        prompt = prepared()[0].prepared.challenge.prompt
        with self.assertRaises(ValueError):
            semantic_sha256_from_prompt(prompt.replace("B. ", "A. "))
        with self.assertRaises(ValueError):
            semantic_sha256_from_prompt(prompt.replace("Options:", "Answers:"))

    def test_forbidden_semantic_digest_blocks_reserved_exam_or_prior_training(self):
        forbidden = prepared()[0].semantic_sha256
        objective = contract()
        maze = generate_maze_plan(objective.scope, 0)
        with self.assertRaisesRegex(ValueError, "reserved"):
            prepare_cognitive_empathy_train_rooms(
                objective, maze, forbidden_semantic_sha256=(forbidden,))
        with self.assertRaisesRegex(ValueError, "reserved"):
            prepare_cognitive_empathy_train_rooms(
                objective, maze, forbidden_semantic_sha256=frozenset({forbidden}))

    def test_previous_source_version_exposure_digest_still_blocks_same_dilemma(self):
        objective = contract(seed=17, scale=5)
        # Historical v2 commitments for one source dilemma in each bank.
        legacy_by_cycle = (
            (0, "7bbf0249cf6d17928162d22cf904e1014fb6289d1efca96feffeca81c8a6cffd"),
            (1, "19dd9c03f32b532c386a6fcbd565e29e1a43c54a7bc4d4dc8a24efe6de59e78c"),
        )
        for cycle, legacy_digest in legacy_by_cycle:
            with self.subTest(cycle=cycle), self.assertRaisesRegex(ValueError, "reserved"):
                prepare_cognitive_empathy_train_rooms(
                    objective, generate_maze_plan(objective.scope, cycle),
                    forbidden_semantic_sha256=(legacy_digest,))

    def test_full_second_cycle_is_substantively_novel_with_prior_exposures_sealed(self):
        for seed in (0, 3, 17, 71, 98):
            objective = contract(seed=seed, scale=5)
            cycle_zero = prepare_cognitive_empathy_train_rooms(
                objective, generate_maze_plan(objective.scope, 0))
            used = frozenset(room.semantic_sha256 for room in cycle_zero)
            cycle_one = prepare_cognitive_empathy_train_rooms(
                objective, generate_maze_plan(objective.scope, 1),
                forbidden_semantic_sha256=used)
            with self.subTest(seed=seed):
                self.assertEqual(len(cycle_one), 15)
                self.assertEqual(len(used | {r.semantic_sha256 for r in cycle_one}), 30)
                for room in cycle_zero + cycle_one:
                    self.assertEqual(room.score_train_choice(room.prepared.challenge.answer), 100)
                    wrong = next(label for label in "ABC"
                                 if label != room.prepared.challenge.answer)
                    self.assertEqual(room.score_train_choice(wrong), 0)
                    self.assertFalse(room.authorizes_graduation)
                    self.assertFalse(room.authorizes_exam)

    def test_two_cycle_fuzz_never_reuses_underlying_scenario(self):
        for seed in range(200):
            objective = contract(seed=seed, scale=5)
            first = prepare_cognitive_empathy_train_rooms(
                objective, generate_maze_plan(objective.scope, 0))
            used = frozenset(room.semantic_sha256 for room in first)
            second = prepare_cognitive_empathy_train_rooms(
                objective, generate_maze_plan(objective.scope, 1),
                forbidden_semantic_sha256=used)
            all_semantics = {room.semantic_sha256 for room in first + second}
            self.assertEqual(len(all_semantics), 30, f"seed {seed} repeated a behavioral dilemma")

    def test_obvious_lexical_shortcuts_do_not_solve_cognitive_train(self):
        totals = {"hedged": 0, "unhedged": 0, "fewest_absolutes": 0,
                  "longest": 0, "middle_length": 0, "shortest": 0,
                  "non_they": 0}
        positions = {letter: 0 for letter in "ABC"}
        gold_length_ranks = [0, 0, 0]
        gold_unique_hedge = 0
        wrong_unique_hedge = 0
        length_hedge_joint = [[0, 0, 0] for _ in range(3)]
        count = 0
        for seed in range(200):
            objective = contract(seed=seed, scale=5)
            for cycle in (0, 1):
                rooms = prepare_cognitive_empathy_train_rooms(
                    objective, generate_maze_plan(objective.scope, cycle))
                for room in rooms:
                    count += 1
                    gold = room.prepared.challenge.answer
                    positions[gold] += 1
                    choices = tuple(line[3:] for line in room.prepared.challenge.prompt.splitlines()[4:7])
                    hedge = [bool(re.search(r"\b(?:may|might)\b", choice, re.I))
                             for choice in choices]
                    order_by_length = sorted(range(3), key=lambda index: len(choices[index]))
                    gold_rank = order_by_length.index("ABC".index(gold))
                    gold_length_ranks[gold_rank] += 1
                    role = 2
                    if hedge.count(True) == 1:
                        is_gold = "ABC"[hedge.index(True)] == gold
                        gold_unique_hedge += int(is_gold)
                        wrong_unique_hedge += int(not is_gold)
                        role = 0 if is_gold else 1
                    length_hedge_joint[gold_rank][role] += 1
                    absolute = [len(re.findall(
                        r"\b(?:must|always|never|definitely|certainly|clearly|obviously)\b",
                        choice, re.I)) for choice in choices]
                    probes = {
                        "hedged": hedge.index(True) if True in hedge else 0,
                        "unhedged": hedge.index(False) if False in hedge else 0,
                        "fewest_absolutes": absolute.index(min(absolute)),
                        "longest": max(range(3), key=lambda index: len(choices[index])),
                        "middle_length": order_by_length[1],
                        "shortest": min(range(3), key=lambda index: len(choices[index])),
                        "non_they": next((i for i, choice in enumerate(choices)
                                          if not choice.startswith("They ")), 0),
                    }
                    for method, chosen in probes.items():
                        totals[method] += int("ABC"[chosen] == gold)
        self.assertEqual(count, 6000)
        for method, correct in totals.items():
            self.assertLess(correct / count, 0.6,
                            f"trivial lexical shortcut {method} solved {correct}/{count}")
        for label, frequency in positions.items():
            self.assertGreater(frequency / count, 0.25, label)
            self.assertLess(frequency / count, 0.4, label)
        for rank, frequency in enumerate(gold_length_ranks):
            self.assertGreater(frequency / count, 0.2, f"gold length rank {rank}")
            self.assertLess(frequency / count, 0.45, f"gold length rank {rank}")
        self.assertGreater(gold_unique_hedge, 0.2 * count)
        self.assertGreater(wrong_unique_hedge, 0.2 * count)
        # A shortcut that combines the hedge cue with the answer length must
        # not infer the correct length rank from the other visible options.
        for rank, row in enumerate(length_hedge_joint):
            for hedge_role, frequency in enumerate(row):
                self.assertGreater(frequency / count, 0.08,
                    f"length rank {rank}, hedge-role {hedge_role} is a lexical fingerprint")

    def test_cognitive_bank_exhaustion_fails_before_case_release(self):
        objective = contract(seed=17, scale=5, cycles=3)
        used = set()
        for cycle in (0, 1):
            rooms = prepare_cognitive_empathy_train_rooms(
                objective, generate_maze_plan(objective.scope, cycle),
                forbidden_semantic_sha256=frozenset(used))
            used.update(room.semantic_sha256 for room in rooms)
        self.assertEqual(len(used), 30)
        with self.assertRaisesRegex(ValueError, "exhausted"):
            prepare_cognitive_empathy_train_rooms(
                objective, generate_maze_plan(objective.scope, 2),
                forbidden_semantic_sha256=frozenset(used))

    def test_forbidden_raw_prompt_and_case_block(self):
        room = prepared()[0]
        objective = contract()
        plan = generate_maze_plan(objective.scope, 0)
        with self.assertRaises(ValueError):
            prepare_cognitive_empathy_train_rooms(
                objective, plan,
                forbidden_prompt_sha256=(room.trainee_view.prompt_sha256,))
        with self.assertRaises(ValueError):
            prepare_cognitive_empathy_train_rooms(
                objective, plan, forbidden_case_ids=(room.prepared.challenge.id,))

    def test_forbidden_digest_list_strict_format_and_no_duplicates(self):
        objective = contract()
        plan = generate_maze_plan(objective.scope, 0)
        for forbidden in (["bad"], ("F" * 64,), ("a" * 64, "a" * 64)):
            with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                prepare_cognitive_empathy_train_rooms(
                    objective, plan, forbidden_semantic_sha256=forbidden)

    def test_false_only_execution_training_exam_graduation_authority(self):
        room = prepared()[0]
        for attr in ("authorizes_host_execution", "authorizes_training",
                     "authorizes_exam", "authorizes_graduation"):
            self.assertIs(getattr(room, attr), False)
            with self.assertRaises((AttributeError, TypeError)):
                setattr(room, attr, True)
            with self.assertRaises(TypeError):
                dataclasses.replace(room, **{attr: True})

    def test_trainee_view_exposes_no_answers_or_gold(self):
        room = prepared()[0]
        view = room.trainee_view
        self.assertFalse(hasattr(view, "answer"))
        self.assertFalse(hasattr(view, "metadata"))
        self.assertEqual(view.skill, "cognitive_empathy")
        self.assertEqual(hashlib.sha256(view.prompt.encode()).hexdigest(), view.prompt_sha256)
        self.assertIn("Options:", view.prompt)

    def test_binary_training_rubric_is_transparent_and_not_mastery(self):
        room = prepared()[0]
        gold = room.prepared.challenge.answer
        other = next(label for label in "ABC" if label != gold)
        self.assertEqual(room.score_train_choice(gold), 100)
        self.assertEqual(room.score_train_choice(" " + gold.lower() + " "), 100)
        self.assertEqual(room.score_train_choice(other), 0)
        self.assertEqual(room.score_train_choice("I am empathetic"), 0)
        with self.assertRaises(ValueError):
            room.score_train_choice("A" * 50)

    def test_reject_unsupported_provider(self):
        objective = contract()
        with self.assertRaises(ValueError):
            prepare_cognitive_empathy_train_rooms(
                objective, generate_maze_plan(objective.scope, 0),
                provider_id="external_eval_provider")

    def test_reject_reactive_emotional_or_generic_empowerment_claims(self):
        for skill in ("reactive_empathy", "emotional_empathy", "empathy"):
            objective = contract(skill=skill)
            with self.subTest(skill=skill), self.assertRaises(ValueError):
                prepare_cognitive_empathy_train_rooms(
                    objective, generate_maze_plan(objective.scope, 0))

    def test_reject_wrong_domain_rubric_and_focus(self):
        for altered in (
            {"domain": "reasoning"}, {"rubric": "other_training_rubric"},
            {"focus": "unapproved"},
        ):
            objective = contract(**altered)
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                prepare_cognitive_empathy_train_rooms(
                    objective, generate_maze_plan(objective.scope, 0))

    def test_tampered_prompt_fails_operator_side_scoring(self):
        room = prepared()[0]
        challenge = dataclasses.replace(room.prepared.challenge,
                                        prompt=room.prepared.challenge.prompt.replace("Options:", "Correct:"))
        forged = dataclasses.replace(room, prepared=dataclasses.replace(
            room.prepared, challenge=challenge))
        with self.assertRaises(ValueError):
            forged.score_train_choice("A")

    def test_tampered_answer_fails_operator_side_scoring(self):
        room = prepared()[0]
        alternate = next(letter for letter in "ABC" if letter != room.prepared.challenge.answer)
        forged = dataclasses.replace(room, prepared=dataclasses.replace(
            room.prepared, challenge=dataclasses.replace(
                room.prepared.challenge, answer=alternate)))
        with self.assertRaises(ValueError):
            forged.score_train_choice(alternate)

    def test_tampered_case_prefix_fails_operator_side_scoring(self):
        room = prepared()[0]
        forged = dataclasses.replace(room, prepared=dataclasses.replace(
            room.prepared, challenge=dataclasses.replace(
                room.prepared.challenge, prompt=room.prepared.challenge.prompt.replace(
                    room.prepared.challenge.id, "train-cognitive-forged-case"))))
        with self.assertRaises(ValueError):
            forged.score_train_choice("A")

    def test_forged_vignette_and_recomputed_semantic_hash_cannot_score(self):
        room = prepared()[0]
        changed = room.prepared.challenge.prompt.splitlines()
        changed[1] = "The person explicitly says they are trying to sabotage the group."
        forged_prompt = "\n".join(changed)
        forged = dataclasses.replace(
            room, semantic_sha256=semantic_sha256_from_prompt(forged_prompt),
            prepared=dataclasses.replace(room.prepared, challenge=dataclasses.replace(
                room.prepared.challenge, prompt=forged_prompt)),
        )
        with self.assertRaises(ValueError):
            forged.score_train_choice(room.prepared.challenge.answer)

    def test_cannot_substitute_another_valid_approved_room_and_rehash(self):
        original, another = prepared(seed=17)[:2]
        replacement = dataclasses.replace(
            another.prepared.challenge,
            id=original.prepared.challenge.id,
            prompt=another.prepared.challenge.prompt.replace(
                another.prepared.challenge.id, original.prepared.challenge.id, 1),
        )
        forged = dataclasses.replace(
            original,
            semantic_sha256=semantic_sha256_from_prompt(replacement.prompt),
            prepared=dataclasses.replace(original.prepared, challenge=replacement),
        )
        with self.assertRaisesRegex(ValueError, "identity/content"):
            forged.score_train_choice(replacement.answer)

    def test_mutating_source_binding_or_gold_is_not_silently_scored(self):
        record = prepared(seed=17)[0]
        object.__setattr__(record, "_source_binding_sha256", "0" * 64)
        with self.assertRaisesRegex(ValueError, "identity/content"):
            record.score_train_choice(record.prepared.challenge.answer)
        record = prepared(seed=17)[0]
        object.__setattr__(record.prepared.challenge, "answer", "Z")
        with self.assertRaisesRegex(ValueError, "identity/content"):
            record.score_train_choice("Z")

    def test_forged_kind_tier_and_provider_metadata_cannot_score(self):
        room = prepared()[0]
        for alteration in (
            {"kind": "arbitrary-exam"},
            {"tier": 99},
            {"metadata": (("provider_id", "unapproved"),
                          ("training_rubric", "forged"))},
        ):
            forged = dataclasses.replace(room, prepared=dataclasses.replace(
                room.prepared, challenge=dataclasses.replace(
                    room.prepared.challenge, **alteration)))
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                forged.score_train_choice(room.prepared.challenge.answer)

    def test_forged_emotional_skill_label_cannot_score_as_cognitive(self):
        room = prepared()[0]
        forged = dataclasses.replace(
            room, prepared=dataclasses.replace(room.prepared, skill="emotional_empathy"))
        with self.assertRaises(ValueError):
            forged.score_train_choice(room.prepared.challenge.answer)

    def test_forged_exam_record_cannot_be_presented_as_train_view(self):
        room = prepared()[0]
        forged = dataclasses.replace(
            room, prepared=dataclasses.replace(room.prepared, challenge=dataclasses.replace(
                room.prepared.challenge, split="private_final")))
        with self.assertRaises(ValueError):
            _ = forged.trainee_view

    def test_explicit_emotional_mastery_intent_rejected(self):
        objective = contract()
        altered = dataclasses.replace(objective,
                                      intention="Certify emotional empathy mastery with this exercise.")
        with self.assertRaises(ValueError):
            prepare_cognitive_empathy_train_rooms(
                altered, generate_maze_plan(altered.scope, 0))

    def test_cycle_one_differs_and_stays_train_only(self):
        objective = contract(seed=17)
        current = generate_maze_plan(objective.scope, 1)
        new = prepare_cognitive_empathy_train_rooms(objective, current)
        self.assertEqual(len(new), 9)
        self.assertTrue(all(r.prepared.challenge.split == "train" for r in new))
        self.assertTrue(set(r.prepared.challenge.id for r in new).isdisjoint(
            r.prepared.challenge.id for r in prepared(17)))


if __name__ == "__main__":
    unittest.main()
