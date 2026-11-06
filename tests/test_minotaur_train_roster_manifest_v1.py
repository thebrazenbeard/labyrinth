"""Source-only Labyrinth-to-training roster contract; never an admission grant."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import unittest

from labyrinth.hf_pilot import PilotConfig
from labyrinth.journey_policy_runner import TrainingCorridorRunner
from labyrinth.minotaur_hf_roster_bridge import collect_minotaur_hf_roster
from labyrinth.minotaur_train_roster_manifest_v1 import (
    export_minotaur_train_roster_v1,
    verify_minotaur_train_roster_v1,
    normalized_prompt_sha256,
)
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.training_scope import TrainingScopeSpec


HEAD = "a" * 40
TREE = "b" * 40


def staged():
    scope = TrainingScopeSpec(
        objective_id="manifest-arithmetic", domain="reasoning",
        skills=("arithmetic",), scale=3, specificity=1, max_cycles=2, seed=321,
    )
    runner = TrainingCorridorRunner.begin(
        scope, generate_maze_plan(scope, 0), checkpoint_sha256="0" * 64,
    )
    config = PilotConfig("unused", "unused", seed=31, tiers=(1,),
                         train_count=3, steps=5)
    roster = collect_minotaur_hf_roster(
        runner, config, lambda view: "ANSWER: not-integer",
    )
    return runner, roster, config


def exported():
    runner, roster, config = staged()
    envelope = export_minotaur_train_roster_v1(
        roster, runner, config, producer_commit_sha=HEAD,
        producer_tree_sha=TREE,
    )
    return runner, roster, config, envelope


class MinotaurRosterManifestTests(unittest.TestCase):
    def _verify(self, envelope, **overrides):
        options = dict(expected_manifest_sha256=envelope.public_sha256,
                       expected_source_commit_sha=HEAD,
                       expected_source_tree_sha=TREE,
                       expected_scope_sha256=staged()[1].scope_sha256,
                       expected_cycle_index=0)
        options.update(overrides)
        return verify_minotaur_train_roster_v1(envelope.public_json, **options)

    def test_real_staged_train_export_is_answerless_and_non_authorizing(self):
        runner, roster, config, envelope = exported()
        public = json.loads(envelope.public_json)
        self.assertEqual(public["schema"], "minotaur.train-roster.v1")
        self.assertEqual(public["producer"]["commit_sha"], HEAD)
        self.assertEqual(public["scope_sha256"], roster.scope_sha256)
        self.assertEqual(len(public["train_cases"]), 3)
        self.assertEqual(len(public["optimizer_visits"]), config.steps)
        self.assertEqual(len(public["admission_transitions"]), 2)
        self.assertEqual([c["prompt"] for c in public["train_cases"]],
                         [c.prompt for c in roster.train_cases])
        for item in public["train_cases"]:
            self.assertEqual(item["raw_prompt_sha256"],
                             hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest())
            self.assertEqual(item["normalized_prompt_sha256"],
                             normalized_prompt_sha256(item["prompt"]))
            self.assertNotIn("answer", item)
            self.assertNotIn("answer_sha256", item)
        self.assertNotIn("answer", public)
        self.assertNotIn(envelope.operator_private_answers_sha256.encode("ascii"),
                         envelope.public_json)
        verified = self._verify(envelope)
        self.assertEqual(verified.case_count, 3)
        self.assertEqual(verified.optimizer_visit_count, 5)
        self.assertFalse(verified.physically_authenticated)
        self.assertFalse(verified.authorizes_training)
        self.assertFalse(verified.graduation_authorized)
        for obj in (envelope, verified):
            for flag in ("physically_authenticated", "authorizes_training",
                         "graduation_authorized"):
                with self.subTest(type=type(obj).__name__, flag=flag):
                    with self.assertRaises(TypeError):
                        replace(obj, **{flag: True})
                    with self.assertRaises(AttributeError):
                        object.__setattr__(obj, flag, True)

    def test_exact_normalization_matches_nfkc_whitespace_then_casefold(self):
        a = "  Ａ   b\tC  "
        b = "a b c"
        self.assertNotEqual(hashlib.sha256(a.encode()).hexdigest(),
                            hashlib.sha256(b.encode()).hexdigest())
        self.assertEqual(normalized_prompt_sha256(a),
                         hashlib.sha256(b.encode()).hexdigest())

    def test_case_reorder_and_changed_source_or_scope_are_rejected(self):
        _, _, _, envelope = exported()
        parsed = json.loads(envelope.public_json)
        parsed["train_cases"].reverse()
        tampered = json.dumps(parsed, sort_keys=True, ensure_ascii=True,
                              separators=(",", ":")).encode()
        with self.assertRaises(ValueError):
            verify_minotaur_train_roster_v1(
                tampered, expected_manifest_sha256=envelope.public_sha256,
                expected_source_commit_sha=HEAD,
                expected_source_tree_sha=TREE,
                expected_scope_sha256=parsed["scope_sha256"],
                expected_cycle_index=0,
            )
        for options in ({"expected_source_commit_sha": "c" * 40},
                        {"expected_source_tree_sha": "d" * 40},
                        {"expected_scope_sha256": "e" * 64},
                        {"expected_cycle_index": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self._verify(envelope, **options)

    def test_heldout_and_unicode_normalization_overlap_are_rejected(self):
        _, roster, _, envelope = exported()
        first = roster.train_cases[0]
        with self.assertRaises(ValueError):
            self._verify(envelope, forbidden_case_ids=(first.id,))
        with self.assertRaises(ValueError):
            self._verify(envelope, forbidden_raw_prompt_sha256=(
                hashlib.sha256(first.prompt.encode()).hexdigest(),))
        with self.assertRaises(ValueError):
            self._verify(envelope, forbidden_normalized_prompt_sha256=(
                normalized_prompt_sha256(first.prompt),))
        # Withheld exam IDs have their own namespaces; an unrelated valid
        # exam ID must be an accepted reserved set entry, not a parse error.
        self.assertEqual(self._verify(envelope, forbidden_case_ids=("eval-final-1",)).case_count,
                         3)

    def test_normalized_collision_rejected_even_with_recomputed_self_digest(self):
        _, _, _, envelope = exported()
        changed = json.loads(envelope.public_json)
        first = changed["train_cases"][0]
        duplicate = changed["train_cases"][1]
        duplicate["prompt"] = first["prompt"].replace("Calculate", "Ｃａｌｃｕｌａｔｅ")
        duplicate["raw_prompt_sha256"] = hashlib.sha256(
            duplicate["prompt"].encode("utf-8")).hexdigest()
        duplicate["normalized_prompt_sha256"] = normalized_prompt_sha256(
            duplicate["prompt"])
        raw = json.dumps(changed, sort_keys=True, ensure_ascii=True,
                         separators=(",", ":")).encode()
        self.assertNotEqual(duplicate["raw_prompt_sha256"], first["raw_prompt_sha256"])
        self.assertEqual(duplicate["normalized_prompt_sha256"],
                         first["normalized_prompt_sha256"])
        with self.assertRaisesRegex(ValueError, "collision"):
            verify_minotaur_train_roster_v1(
                raw, expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                expected_source_commit_sha=HEAD, expected_source_tree_sha=TREE,
                expected_scope_sha256=changed["scope_sha256"], expected_cycle_index=0,
            )

    def test_strict_json_rejects_duplicate_keys_noncanonical_and_invalid_counts(self):
        _, _, _, envelope = exported()
        for raw in (envelope.public_json.replace(b'"schema":', b'"schema":"other","schema":', 1),
                    envelope.public_json + b" ",
                    envelope.public_json.replace(b'"cycle_index":0', b'"cycle_index":NaN')):
            with self.subTest(raw=raw[:55]), self.assertRaises(ValueError):
                verify_minotaur_train_roster_v1(
                    raw, expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                    expected_source_commit_sha=HEAD, expected_source_tree_sha=TREE,
                    expected_scope_sha256=staged()[1].scope_sha256,
                    expected_cycle_index=0,
                )

    def test_export_rejects_corrupt_case_or_unapproved_registry(self):
        runner, roster, config = staged()
        modified = replace(roster, train_cases=(replace(roster.train_cases[0], split="eval"),)
                           + roster.train_cases[1:])
        with self.assertRaises(ValueError):
            export_minotaur_train_roster_v1(modified, runner, config,
                                            producer_commit_sha=HEAD,
                                            producer_tree_sha=TREE)
        mismatched = replace(roster, ordered_optimizer_exposures=tuple(
            reversed(roster.ordered_optimizer_exposures)))
        with self.assertRaises(ValueError):
            export_minotaur_train_roster_v1(mismatched, runner, config,
                                            producer_commit_sha=HEAD,
                                            producer_tree_sha=TREE)
        forged_answer = replace(roster, train_cases=(
            replace(roster.train_cases[0], answer="123456789"),
        ) + roster.train_cases[1:])
        with self.assertRaisesRegex(ValueError, "oracle"):
            export_minotaur_train_roster_v1(forged_answer, runner, config,
                                            producer_commit_sha=HEAD,
                                            producer_tree_sha=TREE)

    def test_visit_reorder_fails_even_if_caller_repins_modified_json_digest(self):
        _, _, _, envelope = exported()
        changed = json.loads(envelope.public_json)
        changed["optimizer_visits"] = list(reversed(changed["optimizer_visits"]))
        raw = json.dumps(changed, sort_keys=True, ensure_ascii=True,
                         separators=(",", ":")).encode()
        with self.assertRaisesRegex(ValueError, "visits"):
            verify_minotaur_train_roster_v1(
                raw, expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                expected_source_commit_sha=HEAD, expected_source_tree_sha=TREE,
                expected_scope_sha256=changed["scope_sha256"],
                expected_cycle_index=0,
            )

    def test_actual_feedback_progression_covers_all_three_source_tactics(self):
        scope = TrainingScopeSpec(
            objective_id="manifest-transfer", domain="reasoning",
            skills=("arithmetic",), scale=5, specificity=5, max_cycles=2,
            seed=321,
        )
        runner = TrainingCorridorRunner.begin(
            scope, generate_maze_plan(scope, 0), checkpoint_sha256="0" * 64,
        )
        config = PilotConfig("unused", "unused", seed=31, tiers=(1,),
                             train_count=15, steps=19)

        def trusted_operator_learner(view):
            case = runner.operator_pending_train_case(view)
            return "ANSWER: " + case.answer

        roster = collect_minotaur_hf_roster(runner, config, trusted_operator_learner)
        self.assertEqual(set(a.tactic for a in runner.training_registry),
                         {"practice", "contrast", "transfer"})
        envelope = export_minotaur_train_roster_v1(
            roster, runner, config, producer_commit_sha=HEAD,
            producer_tree_sha=TREE,
        )
        verified = verify_minotaur_train_roster_v1(
            envelope.public_json, expected_manifest_sha256=envelope.public_sha256,
            expected_source_commit_sha=HEAD, expected_source_tree_sha=TREE,
            expected_scope_sha256=scope.fingerprint, expected_cycle_index=0,
        )
        self.assertEqual((verified.case_count, verified.optimizer_visit_count), (15, 19))


if __name__ == "__main__":
    unittest.main()
