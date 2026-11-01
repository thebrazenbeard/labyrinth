"""Cross-contract arithmetic-atom novelty for staged Minotaur TRAIN only.

These checks do not authenticate physical learner feedback or examination.
"""

from __future__ import annotations

import unittest

from labyrinth.hf_adaptive_source_v1 import arithmetic_semantic_atoms
from labyrinth.hf_pilot import PilotConfig, _training_visit_exposures
from labyrinth.journey_policy_runner import TrainingCorridorRunner
from labyrinth.maze import Challenge
from labyrinth.minotaur_hf_roster_bridge import (
    MinotaurHfTrainRoster, _source_arithmetic, collect_minotaur_hf_roster,
    replay_staged_roster, snapshot_staged_roster,
)
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.training_scope import TrainingScopeSpec


class TrainingAtomNoveltyTests(unittest.TestCase):
    def _scope(self, seed: int) -> TrainingScopeSpec:
        return TrainingScopeSpec(
            "r93-weak-seed-sweep", "reasoning", ("arithmetic",),
            5, 5, 2, seed,
        )

    @staticmethod
    def _config(count: int = 15) -> PilotConfig:
        return PilotConfig(
            "unused-model", "unused-output", seed=31,
            train_count=count, eval_count=5, steps=count, tiers=(1,),
        )

    def _collect_mixed(self, seed: int) -> MinotaurHfTrainRoster:
        scope = self._scope(seed)
        runner = TrainingCorridorRunner.begin(
            scope, generate_maze_plan(scope, 0), checkpoint_sha256="0" * 64,
        )
        counter = 0

        def learner(view):
            nonlocal counter
            index = counter
            counter += 1
            if (index // 3) % 2:
                return "ANSWER: invalid"
            kind = ("arithmetic_contrast" if view.prompt.startswith("Expression A")
                    else "arithmetic")
            gold, _ = _source_arithmetic(Challenge(
                view.challenge_id, kind, 1, "train", view.prompt, "",
            ))
            return "ANSWER: " + gold

        roster = collect_minotaur_hf_roster(runner, self._config(), learner)
        self.assertTrue(runner.is_complete)
        self.assertEqual(len(roster.train_cases), 15)
        return roster

    @staticmethod
    def _assert_atoms_distinct(cases: tuple[Challenge, ...]) -> None:
        seen: set[tuple[str, int, int]] = set()
        for case in cases:
            atoms = arithmetic_semantic_atoms(case)
            if seen & atoms:
                raise AssertionError(f"TRAIN reused semantic atom at {case.id}")
            seen.update(atoms)

    def test_mixed_feedback_seed_46_redraws_internal_contrast_alias(self) -> None:
        # Previously two contrast prompts A=9+4/B=9-4 and A=4+9/B=4-9
        # were admitted because their b+b keys differ, even though both
        # contain the exact source computation 4+9.
        roster = self._collect_mixed(46)
        self._assert_atoms_distinct(roster.train_cases)
        snapshot_staged_roster(roster, self._config(), self._scope(46))

    def test_source_only_staged_replay_rejects_forged_shared_internal_atom(self) -> None:
        scope = TrainingScopeSpec(
            "r110-atom-forgery", "reasoning", ("arithmetic",), 1, 2, 2, 42,
        )
        cases = (
            Challenge(
                "train-taskmaster-" + "1" * 32 + "-" + "a" * 12,
                "arithmetic_contrast", 1, "train",
                "Expression A is 9 + 4. Expression B is 9 - 4. "
                "Calculate A minus B.\nRespond with exactly one line: ANSWER: <integer>.",
                "8",
            ),
            Challenge(
                "train-taskmaster-" + "2" * 32 + "-" + "b" * 12,
                "arithmetic_contrast", 1, "train",
                "Expression A is 4 + 9. Expression B is 4 - 9. "
                "Calculate A minus B.\nRespond with exactly one line: ANSWER: <integer>.",
                "18",
            ),
        )
        self.assertTrue(arithmetic_semantic_atoms(cases[0])
                        & arithmetic_semantic_atoms(cases[1]))
        config = self._config(2)
        exposures = _training_visit_exposures(list(cases), seed=config.seed, steps=config.steps)
        roster = MinotaurHfTrainRoster(scope.fingerprint, 0, cases, exposures)
        with self.assertRaisesRegex(ValueError, "semantic overlap"):
            snapshot_staged_roster(roster, config, scope)

        # The public report contains no answers; a forged self-consistent
        # two-entry roster still must be refused during answer regeneration.
        import hashlib
        public = {
            "schema": "labyrinth.minotaur-hf-staged-train.v1",
            "scope": scope.to_dict(), "scope_sha256": scope.fingerprint,
            "cycle_index": 0,
            "source_authority": "SELF_REPORTED_TRAIN_ONLY_NOT_HOST_AUTHENTICATED",
            "train": [
                {"id": case.id, "kind": case.kind, "prompt": case.prompt,
                 "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest()}
                for case in cases
            ],
            "optimizer_exposures": [list(pair) for pair in exposures],
        }
        with self.assertRaisesRegex(ValueError, "semantic overlap"):
            replay_staged_roster(public, config)

    def test_two_hundred_weak_and_mixed_corridors_remain_stagable(self) -> None:
        # A stronger novelty check must retain the bounded redraw's source
        # availability for ordinary weak and varied-feedback corridors.
        for seed in range(200):
            with self.subTest(seed=seed, learner="mixed"):
                roster = self._collect_mixed(seed)
                self._assert_atoms_distinct(roster.train_cases)
            with self.subTest(seed=seed, learner="weak"):
                scope = self._scope(seed)
                runner = TrainingCorridorRunner.begin(
                    scope, generate_maze_plan(scope, 0), checkpoint_sha256="0" * 64,
                )
                weak = collect_minotaur_hf_roster(
                    runner, self._config(), lambda _view: "ANSWER: invalid",
                )
                self._assert_atoms_distinct(weak.train_cases)


if __name__ == "__main__":
    unittest.main()
