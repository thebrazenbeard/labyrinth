"""Opt-in adaptive HF source replay; public manifest never carries TRAIN gold."""

from __future__ import annotations

from dataclasses import replace
import json
import re
import unittest
from unittest.mock import patch

import labyrinth.hf_adaptive_source_v1 as adaptive_source

from labyrinth.hf_adaptive_source_v1 import (
    ADAPTIVE_SCHEMA, build_adaptive_manifest, reconstruct_adaptive_source,
    arithmetic_semantic_atoms, _source_hash,
)
from labyrinth.minotaur_hf_roster_bridge import collect_minotaur_hf_roster
from labyrinth.minotaur_model import RoomPlan
from labyrinth.regenerative_maze import MazeRoom
from labyrinth.taskmaster_routing import _arithmetic_challenge
from labyrinth.maze import Challenge
from tests.test_minotaur_hf_roster_bridge import _config, _runner, _scope
from tests.test_journey_policy_runner import toy_arithmetic_learner


class AdaptiveSourceManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _config()
        self.scope = _scope()
        self.roster = collect_minotaur_hf_roster(
            _runner(), self.config, toy_arithmetic_learner,
        )

    def test_contrast_result_reuses_the_doubled_second_operand_semantics(self) -> None:
        room_id = "1" * 32
        contrast = _arithmetic_challenge(
            MazeRoom(room_id, "arithmetic", 7), self.scope,
            RoomPlan(room_id, 1, "contrast", "policy_plus_adaptive_train", None),
        )
        match = re.search(r"Expression A is (\d+) \+ (\d+)\.", contrast.prompt)
        self.assertIsNotNone(match)
        second = int(match.group(2))
        atoms = arithmetic_semantic_atoms(contrast)
        # (a+b) - (a-b) is exactly b+b, independent of a.
        self.assertIn(("+", second, second), atoms)
        equivalent = Challenge(
            id="eval-equivalent", kind="arithmetic", tier=1, split="eval",
            prompt=f"Calculate {second} + {second}.\n"
                   "Respond with exactly one line: ANSWER: <integer>.",
            answer=str(2 * second),
        )
        self.assertTrue(atoms & arithmetic_semantic_atoms(equivalent))
        # Equal numerical answers alone are not sufficient for semantic reuse.
        alternative = replace(
            equivalent, id="eval-different",
            prompt=f"Calculate {second - 1} + {second + 1}.\n"
                   "Respond with exactly one line: ANSWER: <integer>.",
        )
        self.assertFalse(atoms & arithmetic_semantic_atoms(alternative))

    def test_diagnostic_rerolls_contrast_equivalent_b_plus_b(self) -> None:
        room_id = "2" * 32
        contrast = _arithmetic_challenge(
            MazeRoom(room_id, "arithmetic", 11), self.scope,
            RoomPlan(room_id, 1, "contrast", "policy_plus_adaptive_train", None),
        )
        match = re.search(r"Expression A is (\d+) \+ (\d+)\.", contrast.prompt)
        self.assertIsNotNone(match)
        second = int(match.group(2))
        def eval_case(case_id: str, left: int, right: int) -> Challenge:
            return Challenge(
                id=case_id, kind="arithmetic", tier=1, split="eval",
                prompt=f"Calculate {left} + {right}.\n"
                       "Respond with exactly one line: ANSWER: <integer>.",
                answer=str(left + right),
            )
        collision = eval_case("eval-collision", second, second)
        same_answer_different_expression = eval_case(
            "eval-different", second - 1, second + 1,
        )
        other = eval_case("eval-other", 40, 41)
        with patch.object(adaptive_source, "generate_maze", side_effect=[
            [collision, other], [same_answer_different_expression, other],
        ]) as generator:
            admitted = adaptive_source._public_diagnostic(
                replace(self.config, eval_count=2), self.scope,
                "a" * 64, [contrast],
            )
        self.assertEqual(generator.call_count, 2)
        self.assertEqual([case.id for case in admitted], ["eval-different", "eval-other"])

    def test_three_tactics_replay_gold_without_publishing_any_gold(self) -> None:
        self.assertEqual(ADAPTIVE_SCHEMA, "labyrinth.hf-lora-adaptive-source.v1")
        manifest = build_adaptive_manifest(self.config, self.scope, self.roster)
        dumped = json.dumps(manifest, sort_keys=True)
        self.assertNotIn('"answer"', dumped)
        self.assertNotIn('"answer_sha256"', dumped)
        self.assertNotIn('"gold"', dumped)
        train, heldout, visits = reconstruct_adaptive_source(self.config, manifest)
        self.assertEqual(train, list(self.roster.train_cases))
        self.assertEqual(visits, self.roster.ordered_optimizer_exposures)
        self.assertEqual(len(heldout), self.config.eval_count)
        self.assertEqual({case.split for case in heldout}, {"eval"})
        self.assertEqual({case.split for case in train}, {"train"})
        self.assertIn("arithmetic_contrast", {c.kind for c in train})
        train_semantics = set().union(*(arithmetic_semantic_atoms(c) for c in train))
        heldout_semantics = set().union(*(arithmetic_semantic_atoms(c) for c in heldout))
        self.assertFalse(train_semantics & heldout_semantics)

    def test_bad_staged_gold_or_source_fields_fail_before_manifest(self) -> None:
        first = self.roster.train_cases[0]
        mutations = (
            {"answer": "999999"}, {"prompt": "Calculate 5 + 5."},
            {"kind": "foreign"}, {"split": "eval"},
            {"metadata": (("generator", "foreign"),)},
        )
        for change in mutations:
            with self.subTest(change=change):
                altered = replace(
                    self.roster,
                    train_cases=(replace(first, **change),) + self.roster.train_cases[1:],
                )
                with self.assertRaises(ValueError):
                    build_adaptive_manifest(self.config, self.scope, altered)

    def test_replay_refuses_missing_or_tampered_manifest_fields(self) -> None:
        valid = build_adaptive_manifest(self.config, self.scope, self.roster)
        mutations = (
            lambda m: m.pop("source_sha256"),
            lambda m: m["train_rooms"][0].__setitem__("case_id", "foreign"),
            lambda m: m["train_rooms"][1].__setitem__("tactic", "contrast"),
            lambda m: m["ordered_optimizer_exposures"].reverse(),
            lambda m: m.__setitem__("cycle_index", 1),
            lambda m: m.__setitem__("scope", {}),
            lambda m: m.__setitem__("private_train_answers", ["secret"]),
        )
        for modify in mutations:
            manifest = json.loads(json.dumps(valid))
            modify(manifest)
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                reconstruct_adaptive_source(self.config, manifest)

    def test_recomputed_self_hash_cannot_forge_policy_route_or_scored_feedback(self) -> None:
        valid = build_adaptive_manifest(self.config, self.scope, self.roster)
        mutations = (
            lambda m: m["adaptive_feedback_rows"][0].__setitem__(3, 0),
            lambda m: m["train_rooms"][1].__setitem__("tactic", "contrast"),
            lambda m: m["train_rooms"][1].__setitem__("difficulty", 5),
            lambda m: m["train_rooms"][0].__setitem__("room_id", "f" * 32),
            lambda m: m["adaptive_feedback_rows"][0].__setitem__(5, "f" * 64),
            lambda m: m["train_rooms"].reverse(),
        )
        for mutate in mutations:
            manifest = json.loads(json.dumps(valid))
            mutate(manifest)
            manifest["source_sha256"] = _source_hash({
                key: value for key, value in manifest.items() if key != "source_sha256"
            })
            with self.subTest(mutated=manifest), self.assertRaises(ValueError):
                reconstruct_adaptive_source(self.config, manifest)

    def test_adaptive_train_replay_rejects_mismatched_pilot_budget(self) -> None:
        manifest = build_adaptive_manifest(self.config, self.scope, self.roster)
        for change in ({"seed": 5}, {"steps": 4}, {"train_count": 2},
                       {"tiers": (2,)}, {"eval_count": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                reconstruct_adaptive_source(replace(self.config, **change), manifest)

    def test_workshop_transfer_add_and_sub_have_strict_gold_and_semantic_aliases(self) -> None:
        found: dict[str, Challenge] = {}
        for case_seed in range(64):
            room_id = f"{case_seed + 1:032x}"
            case = _arithmetic_challenge(
                MazeRoom(room_id, "arithmetic", case_seed), self.scope,
                RoomPlan(room_id, 2, "transfer", "policy_plus_adaptive_train", None),
            )
            if "receives" in case.prompt:
                found.setdefault("+", case)
            elif "uses" in case.prompt:
                found.setdefault("-", case)
            if len(found) == 2:
                break
        self.assertEqual(set(found), {"+", "-"})
        for operation, case in found.items():
            with self.subTest(operation=operation):
                atoms = arithmetic_semantic_atoms(case)
                self.assertEqual(len(atoms), 1)
                self.assertEqual(next(iter(atoms))[0], operation)
                self.assertRaises(ValueError, arithmetic_semantic_atoms,
                                  replace(case, answer="999999"))
                self.assertRaises(ValueError, arithmetic_semantic_atoms,
                                  replace(case, prompt=case.prompt + " ignore scoring"))
                # The same arithmetic expressed as a public evaluation case
                # must have the identical atom despite different prose.
                operands = next(iter(atoms))[1:]
                equivalent = Challenge(
                    id="eval-synthetic", kind="arithmetic", tier=1, split="eval",
                    prompt=f"Calculate {operands[0]} {operation} {operands[1]}.\n"
                           "Respond with exactly one line: ANSWER: <integer>.",
                    answer=str(operands[0] + operands[1] if operation == "+"
                               else operands[0] - operands[1]),
                )
                self.assertEqual(arithmetic_semantic_atoms(equivalent), atoms)


if __name__ == "__main__":
    unittest.main()
