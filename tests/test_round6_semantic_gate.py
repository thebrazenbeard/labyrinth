"""Red-first hostile admission tests for semantic-family independence.

Only test-key HMAC fixtures are used. No real private evaluator, model, or
production graduation is invoked.
"""

from dataclasses import fields, replace
from hashlib import sha256
import hmac
import inspect
import json
import threading
import unittest
from unittest.mock import patch

from labyrinth import graduation_gate as gate
from labyrinth.graduation_gate import (
    GateState, GraduationPolicy, SemanticCaseCommitment,
    _policy_fingerprint, decide_graduation,
)
from tests.test_graduation_gate import KEY, POLICY, decision, evaluated_receipt


def _id(label):
    return sha256(f"hostile-semantic-family:{label}".encode()).hexdigest()


def _bound_relabel(receipt, relabels, *, max_attempts=None):
    """Make the TEST-ONLY policy registry and signed rows agree on labels.

    This deliberately exercises the statistical gate *after* custody checks,
    not a test of trusted registry authorship or real-world semantics.
    """
    new_registry = tuple(replace(entry,
                                 semantic_family_id=relabels.get(entry.case_id,
                                                                 entry.semantic_family_id))
                         for entry in POLICY.independent_semantic_registry)
    policy = replace(POLICY, independent_semantic_registry=new_registry,
                     **({"max_attempts": max_attempts} if max_attempts else {}))
    for group in receipt["slices"].values():
        for trial in group:
            if trial["case_id"] in relabels:
                trial["semantic_family_id"] = relabels[trial["case_id"]]
    return policy


class SemanticFamilyGateTests(unittest.TestCase):
    def test_concurrent_mutation_after_admission_cannot_switch_objective(self):
        """The signed objective must be snapshotted for the full decision."""

        class ChameleonObjective(str):
            def __eq__(self, other):
                return other in ("training_effect", "skill_mastery")

        policy = replace(POLICY)
        actual_compare = gate.hmac.compare_digest
        altered = []

        def tamper_during_digest(first, second):
            if not altered:
                worker = threading.Thread(target=lambda: object.__setattr__(
                    policy, "objective_kind", ChameleonObjective("training_effect")))
                worker.start()
                worker.join()
                altered.append(True)
            return actual_compare(first, second)

        with patch.object(gate.hmac, "compare_digest", side_effect=tamper_during_digest):
            verdict = decision(evaluated_receipt(controls_correct=True), policy=policy)
        self.assertTrue(altered)
        self.assertEqual(_policy_fingerprint(policy), _policy_fingerprint(POLICY))
        self.assertEqual(verdict.disposition, "STAY")

    def test_policy_subclass_cannot_hide_different_objective_from_fingerprint(self):
        """A policy's admission semantics must match its serialized identity."""

        class TrojanPolicy(GraduationPolicy):
            def __getattribute__(self, name):
                if (name == "objective_kind"
                        and inspect.currentframe().f_back.f_code.co_name == "decide_graduation"):
                    return "skill_mastery"
                return super().__getattribute__(name)

        adversarial = TrojanPolicy(**{f.name: getattr(POLICY, f.name)
                                      for f in fields(POLICY)})
        self.assertEqual(_policy_fingerprint(POLICY), _policy_fingerprint(adversarial))
        self.assertEqual(decision(evaluated_receipt(controls_correct=True),
                                  policy=POLICY).disposition, "STAY")
        with self.assertRaises(TypeError):
            decision(evaluated_receipt(controls_correct=True), policy=adversarial)

    def test_trusted_state_subclass_not_admitted(self):
        class MockTrustedState(GateState):
            pass

        with self.assertRaises(TypeError):
            decision(evaluated_receipt(), MockTrustedState())

    def test_frozen_policy_mutation_cannot_reinterpret_signed_objective(self):
        class ChameleonObjective(str):
            def __eq__(self, other):
                return other in ("training_effect", "skill_mastery")

        tampered = replace(POLICY)
        object.__setattr__(tampered, "objective_kind", ChameleonObjective("training_effect"))
        self.assertIs(type(tampered), GraduationPolicy)
        self.assertEqual(_policy_fingerprint(tampered), _policy_fingerprint(POLICY))
        self.assertEqual(decision(evaluated_receipt(controls_correct=True),
                                  policy=POLICY).disposition, "STAY")
        with self.assertRaises(ValueError):
            decision(evaluated_receipt(controls_correct=True), policy=tampered)

    def test_deceptive_string_subclass_cannot_switch_graduation_objective(self):
        """Do not let Python rich comparisons bypass signed policy semantics."""

        class ChameleonObjective(str):
            def __eq__(self, other):
                return other in ("training_effect", "skill_mastery")

        ordinary = decision(evaluated_receipt(controls_correct=True), policy=POLICY)
        self.assertEqual(ordinary.disposition, "STAY")
        with self.assertRaises(ValueError):
            replace(POLICY, objective_kind=ChameleonObjective("training_effect"))

    def test_invalid_attempt_type_fails_cleanly_before_registry_comparison(self):
        for bad in ("3", True, None):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                replace(POLICY, max_attempts=bad)

    def test_legacy_signed_schema_must_not_graduate_without_family_evidence(self):
        receipt = evaluated_receipt()
        receipt["schema"] = "labyrinth.minotaur-evaluator-attestation.v1"
        result = decision(receipt)
        self.assertEqual(result.disposition, "BLOCKED")
        self.assertFalse(result.state.graduated)

    def test_historical_v1_signature_domain_cannot_authenticate_v2_receipt(self):
        receipt = evaluated_receipt()
        raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        old_signature = hmac.new(KEY, b"labyrinth.minotaur.graduation.v1\x00" + raw,
                                 "sha256").hexdigest()
        result = decide_graduation(POLICY, GateState(), raw, old_signature,
                                   independent_evaluator_key=KEY)
        self.assertEqual(result.disposition, "BLOCKED")
        self.assertFalse(result.state.graduated)

    def test_signed_fabricated_family_ids_cannot_override_custodian_registry(self):
        receipt = evaluated_receipt()
        for group in receipt["slices"].values():
            for index, trial in enumerate(group):
                trial["semantic_family_id"] = _id(f"invented:{trial['case_id']}:{index}")
        result = decision(receipt)
        self.assertEqual(result.disposition, "BLOCKED")
        self.assertIn("semantic-family", " ".join(result.reasons))

    def test_seven_distinct_families_are_not_eight_independent_units(self):
        receipt = evaluated_receipt()
        relabels = {trial["case_id"]: _id(f"seven:{name}:{index % 7}")
                    for name, rows in receipt["slices"].items()
                    for index, trial in enumerate(rows)}
        policy = _bound_relabel(receipt, relabels)
        result = decision(receipt, policy=policy)
        self.assertEqual(result.disposition, "BLOCKED")
        self.assertIn("insufficient independent", " ".join(result.reasons))

    def test_eight_two_case_families_do_not_get_sixteen_paired_wins(self):
        receipt = evaluated_receipt()
        relabels = {trial["case_id"]: _id(f"eight:{name}:{index // 2}")
                    for name, rows in receipt["slices"].items()
                    for index, trial in enumerate(rows)}
        policy = _bound_relabel(receipt, relabels, max_attempts=3)
        result = decision(receipt, policy=policy)
        # Eight discordant family wins => p=1/256; Bonferroni over 3 arms,
        # 2 slices, 3 possible attempts gives 18/256 > 0.05, hence STAY.
        self.assertEqual(result.disposition, "STAY")
        self.assertFalse(result.state.graduated)

    def test_precommitted_family_seen_in_training_blocks_transfer(self):
        receipt = evaluated_receipt()
        case_id = receipt["slices"]["transfer"][0]["case_id"]
        relabels = {case_id: receipt["training_semantic_family_ids"][0]}
        policy = _bound_relabel(receipt, relabels)
        result = decision(receipt, policy=policy)
        self.assertEqual(result.disposition, "BLOCKED")
        self.assertFalse(result.state.graduated)

    def test_precommitted_family_reused_across_slices_blocks_graduation(self):
        receipt = evaluated_receipt()
        prior = receipt["slices"]["heldout"][0]["semantic_family_id"]
        case_id = receipt["slices"]["transfer"][0]["case_id"]
        policy = _bound_relabel(receipt, {case_id: prior})
        result = decision(receipt, policy=policy)
        self.assertEqual(result.disposition, "BLOCKED")

    def test_semantic_family_replay_fails_across_signed_attempts(self):
        first_receipt = evaluated_receipt(candidate_correct=False)
        second_receipt = evaluated_receipt(attempt=2, run_id="fresh-run-002")
        prior = first_receipt["slices"]["heldout"][0]["semantic_family_id"]
        case_id = second_receipt["slices"]["heldout"][0]["case_id"]
        policy = _bound_relabel(second_receipt, {case_id: prior})
        first = decision(first_receipt, policy=policy)
        self.assertEqual(first.disposition, "STAY")
        second = decision(second_receipt, first.state, policy=policy)
        self.assertEqual(second.disposition, "BLOCKED")

    def test_signer_cannot_cherry_pick_only_successful_precommitted_cases(self):
        receipt = evaluated_receipt()
        # This case was registered BEFORE evaluation. A signer attempts to
        # omit its unfavorable result and submit the 16 easy cases instead.
        extra = SemanticCaseCommitment(
            1, "fresh-run-001", "heldout", "predeclared-hard-case-017",
            _id("additional-hard-case-prompt"), _id("independent-hard-mechanic"))
        policy = replace(POLICY, independent_semantic_registry=(
            *POLICY.independent_semantic_registry, extra))
        result = decision(receipt, policy=policy)
        self.assertEqual(result.disposition, "BLOCKED")
        self.assertIn("roster", " ".join(result.reasons))


if __name__ == "__main__":
    unittest.main()
