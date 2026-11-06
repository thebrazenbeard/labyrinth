"""Regression for arithmetic equivalence across regenerated HF journeys.

These are public development cases, not private graduation examination cases.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import unittest

from labyrinth.hf_pilot import (
    PilotConfig, _arithmetic_equivalence_key, _arithmetic_semantic_commitments,
    _arithmetic_history_witnesses,
    _arithmetic_case_binding_records, _arithmetic_case_binding_root,
    _resume_training_cases,
    _scoped_training_cases,
)
from labyrinth.training_scope import TrainingScopeSpec


class CrossCycleSemanticHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scope = TrainingScopeSpec(
            "arith-test", "reasoning", ("arithmetic",), 3, 3, 2, 15708,
        )
        self.config = PilotConfig(
            model_path="/unused/model", output_dir="/unused/output",
            seed=42, train_count=4, eval_count=8, tiers=(1,), steps=4,
        )

    def test_reversed_operands_from_prior_development_cannot_enter_train(self):
        previous_train, previous_eval, _ = _scoped_training_cases(
            self.config, self.scope, 0, (),
        )
        next_train, _, _ = _scoped_training_cases(self.config, self.scope, 1, ())
        previous_keys = {_arithmetic_equivalence_key(case) for case in previous_eval}
        collisions = [case for case in next_train
                      if _arithmetic_equivalence_key(case) in previous_keys]
        self.assertEqual([case.prompt.splitlines()[0] for case in collisions],
                         ["Calculate 502 + 459."])
        self.assertIn("Calculate 459 + 502.",
                      [case.prompt.splitlines()[0] for case in previous_eval])

        all_previous = previous_train + previous_eval
        prior = {
            "config": asdict(self.config),
            "training_case_ids": [case.id for case in previous_train],
            "heldout_case_ids": [case.id for case in previous_eval],
            "resumed_journey": {
                "scope_fingerprint": self.scope.fingerprint,
                "cycle_index": 0,
                "weak_skills": [],
                "historical_case_ids": sorted(case.id for case in all_previous),
                "historical_prompt_sha256": sorted(
                    hashlib.sha256(case.prompt.encode()).hexdigest()
                    for case in all_previous
                ),
            },
        }
        with self.assertRaisesRegex(ValueError, "semantic arithmetic"):
            _resume_training_cases(self.config, self.scope, 1, (), prior)

    def test_three_cycle_history_persists_and_missing_or_truncated_seal_fails_closed(self):
        scope = TrainingScopeSpec(
            "arith-history", "reasoning", ("arithmetic",), 3, 3, 3, 1,
        )
        def make_report(cycle: int, history: list):
            train, evaluation, _ = _scoped_training_cases(
                self.config, scope, cycle, (),
            )
            all_cases = history + train + evaluation
            witnesses = _arithmetic_history_witnesses(all_cases)
            bindings = _arithmetic_case_binding_records(witnesses)
            return {
                "config": asdict(self.config),
                "training_case_ids": [case.id for case in train],
                "heldout_case_ids": [case.id for case in evaluation],
                "resumed_journey": {
                    "scope_fingerprint": scope.fingerprint,
                    "cycle_index": cycle, "weak_skills": [],
                    "historical_case_ids": sorted(case.id for case in all_cases),
                    "historical_prompt_sha256": sorted(
                        hashlib.sha256(case.prompt.encode()).hexdigest()
                        for case in all_cases
                    ),
                    "historical_arithmetic_sha256": sorted(
                        _arithmetic_semantic_commitments(all_cases)
                    ),
                    "historical_arithmetic_witnesses_v1": witnesses,
                    "historical_arithmetic_case_bindings_v1": bindings,
                    "historical_arithmetic_binding_root_v1": _arithmetic_case_binding_root(bindings),
                },
            }, all_cases

        first, first_cases = make_report(0, [])
        train1, eval1, _ = _resume_training_cases(
            self.config, scope, 1, (), first,
        )
        second, second_cases = make_report(1, first_cases)
        self.assertEqual({case.id for case in train1 + eval1},
                         {case.id for case in second_cases} -
                         {case.id for case in first_cases})
        train2, eval2, _ = _resume_training_cases(
            self.config, scope, 2, (), second,
        )
        self.assertTrue(train2 and eval2)
        self.assertEqual(len(second["resumed_journey"]["historical_arithmetic_sha256"]),
                         len(second_cases))

        missing = {**second, "resumed_journey": dict(second["resumed_journey"])}
        del missing["resumed_journey"]["historical_arithmetic_sha256"]
        with self.assertRaisesRegex(ValueError, "semantic arithmetic history"):
            _resume_training_cases(self.config, scope, 2, (), missing)

        truncated = {**second, "resumed_journey": dict(second["resumed_journey"])}
        truncated["resumed_journey"]["historical_arithmetic_sha256"] = (
            second["resumed_journey"]["historical_arithmetic_sha256"][:-1]
        )
        with self.assertRaisesRegex(ValueError, "semantic arithmetic history"):
            _resume_training_cases(self.config, scope, 2, (), truncated)

        # The first cycle is reconstructible from the immediate predecessor,
        # so an old cycle-zero v3 report may cross this one bounded boundary.
        old_first = {**first, "resumed_journey": dict(first["resumed_journey"])}
        del old_first["resumed_journey"]["historical_arithmetic_sha256"]
        del old_first["resumed_journey"]["historical_arithmetic_witnesses_v1"]
        del old_first["resumed_journey"]["historical_arithmetic_case_bindings_v1"]
        del old_first["resumed_journey"]["historical_arithmetic_binding_root_v1"]
        self.assertTrue(_resume_training_cases(self.config, scope, 1, (), old_first)[0])

    def test_older_public_case_witnesses_cannot_be_replaced_by_forged_semantic_digest(self):
        # The old digest-only history accepted both of these substitutions:
        # EVAL0->TRAIN2 for seed 5970 and TRAIN0->EVAL2 for seed 4504.
        # They are different public-development contamination directions.
        config = PilotConfig(
            model_path="/unused/model", output_dir="/unused/output",
            seed=42, train_count=4, eval_count=32, tiers=(1,), steps=4,
        )
        for seed, old_split, new_split in (
            (5970, "eval", "train"),
            (4504, "train", "eval"),
        ):
            with self.subTest(seed=seed):
                scope = TrainingScopeSpec(
                    "history-audit", "reasoning", ("arithmetic",), 3, 3, 3, seed,
                )
                cycles = [_scoped_training_cases(config, scope, cycle, ())
                          for cycle in range(3)]
                historic = cycles[0][0] + cycles[0][1] + cycles[1][0] + cycles[1][1]
                witnesses = _arithmetic_history_witnesses(historic)
                bindings = _arithmetic_case_binding_records(witnesses)
                old = cycles[0][0] if old_split == "train" else cycles[0][1]
                new = cycles[2][0] if new_split == "train" else cycles[2][1]
                matches = [(a, b) for a in old for b in new
                           if _arithmetic_equivalence_key(a) == _arithmetic_equivalence_key(b)]
                self.assertTrue(matches, "seed must exercise the intended cross-cycle collision")
                collision_digest = next(iter(_arithmetic_semantic_commitments([matches[0][0]])))
                prior = {
                    "config": asdict(config),
                    "training_case_ids": [case.id for case in cycles[1][0]],
                    "heldout_case_ids": [case.id for case in cycles[1][1]],
                    "resumed_journey": {
                        "scope_fingerprint": scope.fingerprint,
                        "cycle_index": 1,
                        "weak_skills": [],
                        "historical_case_ids": sorted(case.id for case in historic),
                        "historical_prompt_sha256": sorted(
                            hashlib.sha256(case.prompt.encode()).hexdigest() for case in historic
                        ),
                        "historical_arithmetic_sha256": sorted(
                            _arithmetic_semantic_commitments(historic)
                        ),
                        "historical_arithmetic_witnesses_v1": witnesses,
                        "historical_arithmetic_case_bindings_v1": bindings,
                        "historical_arithmetic_binding_root_v1": _arithmetic_case_binding_root(bindings),
                    },
                }
                with self.assertRaisesRegex(ValueError, "overlaps prior semantic arithmetic"):
                    _resume_training_cases(config, scope, 2, (), prior)

                forged = {**prior, "resumed_journey": dict(prior["resumed_journey"])}
                digests = list(forged["resumed_journey"]["historical_arithmetic_sha256"])
                digests[digests.index(collision_digest)] = "f" * 64
                forged["resumed_journey"]["historical_arithmetic_sha256"] = sorted(digests)
                with self.assertRaisesRegex(ValueError, "disagrees with prompt witnesses"):
                    _resume_training_cases(config, scope, 2, (), forged)

                missing = {**prior, "resumed_journey": dict(prior["resumed_journey"])}
                missing["resumed_journey"].pop("historical_arithmetic_witnesses_v1")
                with self.assertRaisesRegex(ValueError, "replayable arithmetic history witnesses"):
                    _resume_training_cases(config, scope, 2, (), missing)

    def test_older_case_id_prompt_binding_cannot_be_swapped_without_detection(self):
        # Source report sets alone contain the same IDs, prompt hashes and
        # semantic keys after swapping two old witnesses. The ID-to-prompt
        # relationship must remain committed across later maze cycles.
        scope = TrainingScopeSpec(
            "arith-history", "reasoning", ("arithmetic",), 3, 3, 3, 1,
        )
        cycles = [_scoped_training_cases(self.config, scope, cycle, ())
                  for cycle in range(3)]
        old = cycles[0][0] + cycles[0][1]
        immediate = cycles[1][0] + cycles[1][1]
        historic = old + immediate
        witnesses = _arithmetic_history_witnesses(historic)
        bindings = _arithmetic_case_binding_records(witnesses)
        old_ids = {case.id for case in old}
        indices = [i for i, record in enumerate(witnesses)
                   if record["case_id"] in old_ids]
        self.assertGreaterEqual(len(indices), 2)
        prior = {
            "config": asdict(self.config),
            "training_case_ids": [case.id for case in cycles[1][0]],
            "heldout_case_ids": [case.id for case in cycles[1][1]],
            "resumed_journey": {
                "scope_fingerprint": scope.fingerprint,
                "cycle_index": 1,
                "weak_skills": [],
                "historical_case_ids": sorted(case.id for case in historic),
                "historical_prompt_sha256": sorted(
                    hashlib.sha256(case.prompt.encode()).hexdigest() for case in historic
                ),
                "historical_arithmetic_sha256": sorted(
                    _arithmetic_semantic_commitments(historic)
                ),
                "historical_arithmetic_witnesses_v1": witnesses,
                "historical_arithmetic_case_bindings_v1": bindings,
                "historical_arithmetic_binding_root_v1": _arithmetic_case_binding_root(bindings),
            },
        }
        self.assertTrue(_resume_training_cases(self.config, scope, 2, (), prior)[0])
        swapped = {**prior, "resumed_journey": dict(prior["resumed_journey"])}
        rows = [dict(record) for record in witnesses]
        first, second = indices[:2]
        rows[first]["prompt"], rows[second]["prompt"] = rows[second]["prompt"], rows[first]["prompt"]
        swapped["resumed_journey"]["historical_arithmetic_witnesses_v1"] = rows
        with self.assertRaisesRegex(ValueError, "case-to-prompt binding"):
            _resume_training_cases(self.config, scope, 2, (), swapped)

        swapped_bindings = {**swapped, "resumed_journey": dict(swapped["resumed_journey"])}
        # Recompute each per-case binding from the swapped witness rows but
        # leave the original aggregate root pinned to the earlier records.
        swapped_bindings["resumed_journey"]["historical_arithmetic_case_bindings_v1"] = (
            _arithmetic_case_binding_records(rows)
        )
        with self.assertRaisesRegex(ValueError, "case-to-prompt binding"):
            _resume_training_cases(self.config, scope, 2, (), swapped_bindings)

        missing_binding = {**prior, "resumed_journey": dict(prior["resumed_journey"])}
        missing_binding["resumed_journey"].pop("historical_arithmetic_case_bindings_v1")
        with self.assertRaisesRegex(ValueError, "case-to-prompt binding is missing"):
            _resume_training_cases(self.config, scope, 2, (), missing_binding)


if __name__ == "__main__":
    unittest.main()
