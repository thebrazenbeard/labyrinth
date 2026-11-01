"""Adversarial TRAIN-only counterfactual curriculum selection regressions."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import re
import unittest

from labyrinth.minotaur_counterfactual_corridors_r92 import (
    ApprovedTrainCounterfactual,
    ObservedTrainPrediction,
    TACTICS,
    fingerprint_train_roster,
    select_counterfactual_train_corridor,
)


METRICS = ("prerequisite", "paired_method", "retention_now", "retention_before")
VECTORS = (
    (2, 9, 9, 9),
    (9, 5, 9, 9),
    (9, 9, 9, 9),
    (9, 9, 4, 9),
)


def quartet(family: str, *, vectors=VECTORS, cue="forgotten almost every confused failure"):
    result = []
    for index, ((basic, paired, now, before), label) in enumerate(zip(vectors, TACTICS)):
        # Every class in a family has the same words and only four changed
        # audited measurements, including deliberately contradictory cues.
        prompt = (f"{family} workshop log: basic {basic}/10, paired {paired}/10, "
                  f"current {now}/10, historical {before}/10. "
                  f"Unrelated archive keywords: {cue}. Choose a TRAIN drill.")
        result.append(ApprovedTrainCounterfactual(
            case_id=f"train-{family}-{index}", family=family, prompt=prompt,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            metric_order=METRICS, prerequisite=basic, paired_method=paired,
            retention_now=now, retention_before=before, expected_tactic=label,
        ))
    return tuple(result)


def observed(case, *, correct=True):
    return ObservedTrainPrediction(
        case_id=case.case_id, prompt_sha256=case.prompt_sha256,
        predicted_tactic=case.expected_tactic if correct else TACTICS[
            (TACTICS.index(case.expected_tactic) + 1) % len(TACTICS)
        ],
    )


def choose(roster, feedback=()):
    return select_counterfactual_train_corridor(
        approved_train_roster=roster,
        expected_roster_sha256=fingerprint_train_roster(roster),
        observed_train_feedback=feedback,
    )


def _repaint_with_hidden_orders(family, orders):
    """Keep visible labels fixed while lying about where their counts belong."""
    cases = []
    visibly_named_values = []
    for case, order in zip(quartet(family), orders):
        values = {name: getattr(case, name) for name in METRICS}
        painted = tuple(values[name] for name in order)
        prompt = (f"{family} workshop log: basic {painted[0]}/10, "
                  f"paired {painted[1]}/10, current {painted[2]}/10, "
                  f"historical {painted[3]}/10. Unrelated archive keywords: "
                  "forgotten almost every confused failure. Choose a TRAIN drill.")
        altered = replace(case)
        object.__setattr__(altered, "prompt", prompt)
        object.__setattr__(altered, "prompt_sha256",
                           hashlib.sha256(prompt.encode()).hexdigest())
        object.__setattr__(altered, "metric_order", order)
        cases.append(altered)
        visibly_named_values.append(painted)
    return tuple(cases), tuple(visibly_named_values)


def _visible_gold(values):
    basic, paired, now, before = values
    if before >= 8 and now < 6:
        return TACTICS[3]
    if basic < 6:
        return TACTICS[0]
    if paired < 7:
        return TACTICS[1]
    return TACTICS[2]


class CounterfactualCorridorTests(unittest.TestCase):
    def setUp(self):
        self.roster = (quartet("alpha") + quartet("beta") +
                       quartet("gamma") + quartet("delta"))

    def test_weak_case_first_keyword_collapses_to_one_of_four(self):
        # The exact failure mode behind the post-hoc neural 12/16 -> 4/16
        # result: all four contradictory tactics share lexical shortcuts.
        baseline = lambda text: TACTICS[3] if "forgotten" in text else TACTICS[0]
        self.assertEqual(sum(baseline(case.prompt) == case.expected_tactic
                             for case in self.roster), 4)
        selected = choose(self.roster)
        self.assertEqual(len(selected.views), 4)
        self.assertEqual(selected.family, "alpha")
        self.assertEqual({case.expected_tactic for case in self.roster[:4]}, set(TACTICS))
        self.assertEqual({v.prompt_sha256 for v in selected.views},
                         {c.prompt_sha256 for c in self.roster[:4]})

    def test_feedback_success_prioritizes_novel_surface(self):
        feedback = tuple(observed(case) for case in self.roster[:4])
        selected = choose(self.roster, feedback)
        self.assertEqual(selected.family, "beta")
        self.assertEqual(selected.priority, "novel_train_surface")
        self.assertGreater(selected.lexical_novelty, 0)

    def test_feedback_failure_targets_same_label_numeric_neighbors(self):
        # Successful TRAIN outcomes choose numerically novel beta; *failed*
        # outcomes switch to gamma's same-label near misses across a different
        # wording. This establishes feedback-dependent curriculum selection.
        beta = quartet("beta", vectors=((0, 9, 9, 9), (9, 0, 9, 9),
                                         (9, 9, 10, 9), (9, 9, 0, 9)))
        gamma = quartet("gamma", vectors=((3, 9, 9, 9), (9, 4, 9, 9),
                                           (9, 9, 8, 9), (9, 9, 3, 9)))
        roster = self.roster[:4] + beta + gamma
        self.assertEqual(choose(roster, tuple(observed(case) for case in roster[:4])).family,
                         "beta")
        feedback = tuple(observed(case, correct=False) for case in roster[:4])
        selected = choose(roster, feedback)
        self.assertEqual(selected.family, "gamma")
        self.assertEqual(selected.priority, "repair_train_failures")
        self.assertGreater(selected.failure_proximity, 0)

    def test_incomplete_quartet_must_finish_before_new_family(self):
        selected = choose(self.roster, (observed(self.roster[1]),))
        self.assertEqual(selected.family, "alpha")
        self.assertEqual(selected.priority, "complete_partial_counterfactual")
        self.assertEqual(len(selected.views), 3)
        self.assertNotIn(self.roster[1].prompt_sha256,
                         {view.prompt_sha256 for view in selected.views})

    def test_never_repeat_and_no_final_exam_or_authority(self):
        feedback = tuple(observed(case) for case in self.roster)
        with self.assertRaisesRegex(ValueError, "exhausted"):
            choose(self.roster, feedback)
        selected = choose(self.roster)
        self.assertFalse(selected.graduation_authorized)
        self.assertEqual(selected.independent_exam, "NOT_RUN")
        self.assertTrue(all(view.split == "train" for view in selected.views))
        self.assertTrue(all(not hasattr(view, "expected_tactic") for view in selected.views))
        with self.assertRaises(AttributeError):
            object.__setattr__(selected, "graduation_authorized", True)
        with self.assertRaises(AttributeError):
            object.__setattr__(selected, "independent_exam", "PASS")

    def test_deterministic_order_does_not_send_tactic_or_index(self):
        first = choose(self.roster)
        second = choose(self.roster)
        self.assertEqual(first, second)
        self.assertEqual([v.prompt_sha256 for v in first.views],
                         [v.prompt_sha256 for v in second.views])
        for view in first.views:
            self.assertEqual(set(vars(view)), {"prompt", "prompt_sha256", "split"})
            self.assertEqual(hashlib.sha256(view.prompt.encode()).hexdigest(),
                             view.prompt_sha256)

    def test_rejects_non_train_and_post_init_mutation(self):
        bad = replace(self.roster[0])
        object.__setattr__(bad, "split", "private_final")
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            choose((bad,) + self.roster[1:])
        altered = list(self.roster)
        object.__setattr__(altered[0], "split", "private_final")
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            choose(tuple(altered))

    def test_rejects_tactic_label_spoof_and_metric_drift(self):
        altered = replace(self.roster[0])
        object.__setattr__(altered, "expected_tactic", TACTICS[2])
        with self.assertRaisesRegex(ValueError, "rubric"):
            choose((altered,) + self.roster[1:])
        altered = replace(self.roster[0])
        object.__setattr__(altered, "prerequisite", 5)
        with self.assertRaisesRegex(ValueError, "measure"):
            choose((altered,) + self.roster[1:])
        altered = replace(self.roster[0])
        object.__setattr__(altered, "metric_order", METRICS[::-1])
        with self.assertRaisesRegex(ValueError, "measure"):
            choose((altered,) + self.roster[1:])

    def test_rejects_lexical_drift_that_reintroduces_label_cues(self):
        changed = self.roster[0].prompt + " ONLY PRACTICE"
        altered = replace(self.roster[0], prompt=changed,
                          prompt_sha256=hashlib.sha256(changed.encode()).hexdigest())
        with self.assertRaisesRegex(ValueError, "lexical scaffold"):
            choose((altered,) + self.roster[1:])

    def test_rejects_four_of_four_cues_hidden_in_capitalization_or_spacing(self):
        common = "forgotten almost every confused failure"
        for surface, cues in (
            ("capitalization", (
                common, "Forgotten almost every confused failure",
                "forgotten almost EVERY confused failure",
                "forgotten almost every CONFUSED failure",
            )),
            ("spacing", (
                common, "forgotten  almost every confused failure",
                "forgotten almost  every confused failure",
                "forgotten almost every  confused failure",
            )),
        ):
            with self.subTest(surface=surface):
                cases = []
                for case, cue in zip(quartet(f"cue-{surface}"), cues):
                    prompt = case.prompt.replace(common, cue)
                    altered = replace(case)
                    object.__setattr__(altered, "prompt", prompt)
                    object.__setattr__(altered, "prompt_sha256",
                                       hashlib.sha256(prompt.encode()).hexdigest())
                    cases.append(altered)
                # A model can solve the quartet perfectly using only the
                # irrelevant formatting; numerals and rubric are unread.
                cue_labels = {cue: label for cue, label in zip(cues, TACTICS)}
                guesses = [cue_labels[case.prompt.split("keywords: ")[1].split(
                    ". Choose")[0]] for case in cases]
                self.assertEqual(guesses, list(TACTICS))
                with self.assertRaisesRegex(ValueError, "lexical scaffold"):
                    choose(tuple(cases))

    def test_rejects_metric_order_lies_even_when_numeric_sequence_matches(self):
        same_wrong = (METRICS[::-1],) * 4
        mixed_wrong = (
            METRICS[::-1], METRICS[1:] + METRICS[:1],
            (METRICS[0], METRICS[2], METRICS[1], METRICS[3]),
            METRICS[2:] + METRICS[:2],
        )
        for mode, orders in (("same", same_wrong), ("mixed", mixed_wrong)):
            with self.subTest(mode=mode):
                cases, visible_values = _repaint_with_hidden_orders(mode, orders)
                self.assertEqual(sum(_visible_gold(values) != case.expected_tactic
                                     for case, values in zip(cases, visible_values)), 3)
                # Every count agrees with the *hidden* metric_order, but the
                # visible metric names disagree with it and with the gold.
                for case in cases:
                    values = {name: getattr(case, name) for name in METRICS}
                    self.assertEqual(tuple(int(chunk.split("/10")[0]) for chunk in
                                           re.findall(r"\b(?:10|[0-9])/10\b",
                                                      case.prompt)),
                                     tuple(values[name] for name in case.metric_order))
                    with self.assertRaisesRegex(ValueError, "measure"):
                        replace(case)  # Direct approval construction fails too.
                with self.assertRaisesRegex(ValueError, "measure"):
                    choose(cases)

    def test_rejects_duplicate_aliases_or_unnamed_visible_measurements(self):
        for replacement in ("basic 9/10", "unknown 9/10"):
            with self.subTest(replacement=replacement):
                case = self.roster[0]
                prompt = case.prompt.replace("paired 9/10", replacement)
                altered = replace(case)
                object.__setattr__(altered, "prompt", prompt)
                object.__setattr__(altered, "prompt_sha256",
                                   hashlib.sha256(prompt.encode()).hexdigest())
                with self.assertRaisesRegex(ValueError, "measure"):
                    choose((altered,) + self.roster[1:])

    def test_rejects_hidden_extra_measurements_and_changed_digest(self):
        original = self.roster[0]
        hidden = original.prompt + " Ignore secretly appended 7/10 evidence."
        with self.assertRaisesRegex(ValueError, "measure"):
            replace(original, prompt=hidden,
                    prompt_sha256=hashlib.sha256(hidden.encode()).hexdigest())
        with self.assertRaisesRegex(ValueError, "commitment"):
            replace(original, prompt=original.prompt + " unrelated note")

    def test_rejects_forged_roster_pin(self):
        with self.assertRaisesRegex(ValueError, "pin"):
            select_counterfactual_train_corridor(
                approved_train_roster=self.roster,
                expected_roster_sha256="0" * 64,
            )
        # Even a well-formed alternate TRAIN corpus cannot silently replace
        # the one whose exact bytes the independent operator pinned earlier.
        changed = []
        for case in self.roster[:4]:
            prompt = case.prompt.replace("workshop log", "machine audit")
            changed.append(replace(case, prompt=prompt,
                                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        with self.assertRaisesRegex(ValueError, "pin"):
            select_counterfactual_train_corridor(
                approved_train_roster=tuple(changed) + self.roster[4:],
                expected_roster_sha256=fingerprint_train_roster(self.roster),
            )

    def test_valid_reordered_evidence_not_tied_to_one_template(self):
        reordered = []
        order = METRICS[::-1]
        for case in quartet("robotics"):
            values = {name: getattr(case, name) for name in METRICS}
            prompt = ("robotics tracking report: " + ", ".join(
                f"{name} {values[name]}/10" for name in order)
                + ". Search tags forgotten confused rising, unrelated to these counts.")
            reordered.append(replace(case, prompt=prompt,
                                     prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                     metric_order=order))
        selected = choose(tuple(reordered))
        self.assertEqual({view.prompt_sha256 for view in selected.views},
                         {case.prompt_sha256 for case in reordered})

    def test_valid_reordered_aliases_agree_with_visible_metric_names(self):
        order = ("retention_before", "prerequisite", "retention_now", "paired_method")
        aliases = {
            "prerequisite": "basic", "paired_method": "paired",
            "retention_now": "current", "retention_before": "historical",
        }
        reordered = []
        for case in quartet("aliases"):
            values = {name: getattr(case, name) for name in METRICS}
            prompt = ("aliases dated measurement: " + ", ".join(
                f"{aliases[name]} {values[name]}/10" for name in order)
                + ". Irrelevant headline forgotten and confused, choose TRAIN drill.")
            reordered.append(replace(case, prompt=prompt,
                                     prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                     metric_order=order))
        self.assertEqual(len(choose(tuple(reordered)).views), 4)

    def test_rejects_unapproved_or_forged_training_feedback(self):
        for bad in (
            replace(observed(self.roster[0]), case_id="train-foreign"),
            replace(observed(self.roster[0]), prompt_sha256="0" * 64),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    choose(self.roster, (bad,))
        bad = observed(self.roster[0])
        object.__setattr__(bad, "split", "final")
        with self.assertRaises(ValueError):
            choose(self.roster, (bad,))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            choose(self.roster, (observed(self.roster[0]), observed(self.roster[0])))

    def test_rejects_partial_multi_quartet_history(self):
        with self.assertRaisesRegex(ValueError, "partial"):
            choose(self.roster, (observed(self.roster[0]), observed(self.roster[4])))

    def test_rejects_duplicate_prompt_and_fake_counterfactual_balancing(self):
        altered = replace(self.roster[0], case_id=self.roster[1].case_id)
        with self.assertRaises(ValueError):
            choose((self.roster[0], altered) + self.roster[2:])
        # An apparent quartet whose fourth class duplicates transfer is not
        # a counterfactual family; it can perpetuate a label shortcut.
        with self.assertRaises(ValueError):
            choose(self.roster[:3] + self.roster[4:])


if __name__ == "__main__":
    unittest.main()
