"""Independent randomized mathematical and custody reattack of graduation v2.

Synthetic HMAC fixtures are NOT measured model-training evidence. Random seeds
are pinned for exact reproducibility. No production code or secrets are used.
"""

from dataclasses import replace
from fractions import Fraction
import hashlib
import hmac
import math
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest

from labyrinth.graduation_gate import GateState, decide_graduation
from labyrinth.graduation_state import decide_campaign, initialize_campaign
from tests.test_graduation_gate import (
    KEY, POLICY, decision, evaluated_receipt, independently_signed,
)


STATE_KEY = b"mathematician-wave7-state-key-is-independent-of-evaluator"
ARMS = ("candidate", "baseline", "zero_update", "corrupted")


def _independent_tail(wins, losses):
    n = wins + losses
    return Fraction(sum(math.comb(n, k) for k in range(wins, n + 1)), 1 << n)


def _independent_mastery(successes, n, probability=Fraction(3, 4)):
    return sum((math.comb(n, k) * probability ** k *
                (1 - probability) ** (n - k) for k in range(successes, n + 1)),
               Fraction())


def _with_family_clustering(policy, receipt, cluster_size):
    assignments = {}
    for name, trials in receipt["slices"].items():
        for index, trial in enumerate(trials):
            family_id = hashlib.sha256(
                f"round7:frozen:{name}:{index // cluster_size}".encode()
            ).hexdigest()
            trial["semantic_family_id"] = family_id
            assignments[trial["case_id"]] = family_id
    roster = tuple(
        replace(entry, semantic_family_id=assignments[entry.case_id])
        if entry.attempt == 1 and entry.run_id == receipt["run_id"]
        and entry.slice_name != "train"
        else entry
        for entry in policy.independent_semantic_registry
    )
    return replace(policy, independent_semantic_registry=roster)


def _reference_graduation(policy, receipt):
    for slice_name, trials in receipt["slices"].items():
        by_family = {}
        for trial in trials:
            by_family.setdefault(trial["semantic_family_id"], []).append(trial)
        if len(by_family) < policy.min_semantic_families_per_slice:
            return "BLOCKED"
        family_score = [{arm: all(t[f"{arm}_semantic_correct"] for t in members)
                         for arm in ARMS} |
                        {"format": all(t["candidate_format"] for t in members)}
                        for members in by_family.values()]
        n = len(family_score)
        cand = sum(x["candidate"] for x in family_score)
        strongest = max(sum(x[arm] for x in family_score) for arm in ARMS[1:])
        candidate_format = sum(x["format"] for x in family_score)
        if (Fraction(cand, n) < Fraction(str(policy.min_accuracy))
                or Fraction(candidate_format, n) < Fraction(str(policy.min_format_rate))):
            return "STAY"
        if policy.objective_kind == "skill_mastery":
            if (_independent_mastery(cand, n, Fraction(str(policy.min_accuracy))) *
                    len(policy.required_slices) * policy.max_attempts >=
                    Fraction(str(policy.significance_alpha))):
                return "STAY"
        else:
            if Fraction(cand - strongest, n) < Fraction(str(policy.min_gain_over_controls)):
                return "STAY"
            for control in ARMS[1:]:
                wins = sum(s["candidate"] and not s[control] for s in family_score)
                losses = sum(s[control] and not s["candidate"] for s in family_score)
                if (_independent_tail(wins, losses) *
                        3 * len(policy.required_slices) * policy.max_attempts >=
                        Fraction(str(policy.significance_alpha))):
                    return "STAY"
    return "GRADUATED"


class GraduationWave7MathTests(unittest.TestCase):
    def test_randomized_clustered_score_differential_against_fraction_oracle(self):
        rng = random.Random(0xBEEF2026)
        evaluated = 0
        results = {"GRADUATED": 0, "STAY": 0, "BLOCKED": 0}
        for objective_kind in ("training_effect", "skill_mastery"):
            for cluster_size in (1, 2):
                for sample in range(180):
                    receipt = evaluated_receipt()
                    policy = _with_family_clustering(
                        replace(POLICY, objective_kind=objective_kind),
                        receipt, cluster_size,
                    )
                    for trials in receipt["slices"].values():
                        for trial in trials:
                            for arm in ARMS:
                                # Include genuine graduation opportunities,
                                # near-threshold inputs and ordinary failures.
                                strong = sample % 4 == 0
                                candidate_probability = 0.995 if strong else 0.72
                                control_probability = 0.01 if strong else 0.48
                                semantic = rng.random() < (
                                    candidate_probability if arm == "candidate"
                                    else control_probability
                                )
                                valid_format = rng.random() < 0.96
                                trial[f"{arm}_semantic_correct"] = semantic
                                trial[f"{arm}_format"] = valid_format
                                trial[f"{arm}_correct"] = semantic and valid_format
                    expected = _reference_graduation(policy, receipt)
                    result = decision(receipt, policy=policy)
                    with self.subTest(objective=objective_kind,
                                      cluster_size=cluster_size, trial=evaluated):
                        self.assertEqual(result.disposition, expected, result.reasons)
                    results[result.disposition] += 1
                    evaluated += 1
        self.assertEqual(evaluated, 720)
        self.assertGreater(results["GRADUATED"], 10)
        self.assertGreater(results["STAY"], 10)

    def test_authorized_signer_cannot_bypass_roster_with_mutated_tuple_fields(self):
        rng = random.Random(721701)
        for sample in range(160):
            receipt = evaluated_receipt()
            group = rng.choice(list(receipt["slices"]))
            trial = rng.choice(receipt["slices"][group])
            field = rng.choice(("case_id", "prompt_commitment", "semantic_family_id"))
            if field == "case_id":
                trial[field] = f"unexpected-{sample}"
            else:
                trial[field] = hashlib.sha256(f"altered:{sample}:{field}".encode()).hexdigest()
            result = decision(receipt)
            with self.subTest(sample=sample, field=field):
                self.assertEqual(result.disposition, "BLOCKED")

    def test_receipt_domain_and_unsigned_mismatch_rejected(self):
        raw, sig = independently_signed(evaluated_receipt())
        self.assertEqual(decision(evaluated_receipt(), signature=sig).disposition,
                         "GRADUATED")
        domain_mismatch = hmac.new(
            KEY, b"labyrinth.minotaur.atomic-custody.v2\x00" + raw,
            hashlib.sha256,
        ).hexdigest()
        self.assertEqual(
            decide_graduation(POLICY, GateState(), raw, domain_mismatch,
                              independent_evaluator_key=KEY).disposition,
            "BLOCKED",
        )
        tampered = bytearray(raw)
        tampered[-2] = (tampered[-2] + 1) % 256
        self.assertEqual(
            decide_graduation(POLICY, GateState(), bytes(tampered), sig,
                              independent_evaluator_key=KEY).disposition,
            "BLOCKED",
        )

    def test_authenticated_sqlite_snapshot_rollback_is_external_unsolved_risk(self):
        """A full old *authentic* snapshot is undetectable without external witness."""
        with tempfile.TemporaryDirectory() as temporary:
            custody = Path(temporary) / "custody.sqlite3"
            snapshot = Path(temporary) / "baseline.sqlite3"
            initialize_campaign(custody, POLICY, state_key=STATE_KEY)
            with sqlite3.connect(custody) as source, sqlite3.connect(snapshot) as backup:
                source.backup(backup)

            def act(candidate_correct):
                raw, signature = independently_signed(
                    evaluated_receipt(candidate_correct=candidate_correct))
                return decide_campaign(
                    custody, POLICY, raw, signature,
                    independent_evaluator_key=KEY, state_key=STATE_KEY,
                )

            self.assertEqual(act(False).disposition, "STAY")
            self.assertEqual(act(True).disposition, "BLOCKED")
            # Host-level replacement of full authenticated old snapshot:
            # adversary has no MAC key but controls mutable DB files.
            with sqlite3.connect(snapshot) as source, sqlite3.connect(custody) as target:
                source.backup(target)
            self.assertEqual(act(True).disposition, "GRADUATED")


if __name__ == "__main__":
    unittest.main()
