"""Dual-contract source binding; synthetic signer is NOT independent host custody."""

from dataclasses import replace
from hashlib import sha256
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from labyrinth.formal_objective_bridge_r16 import (
    DualObjectivePins, FormalObjectiveBridgeError,
    approved_dual_objective_payload, verify_dual_objective_binding,
)
from labyrinth.objective_contract import ObjectiveContract, SkillTarget
from labyrinth.training_scope import TrainingScopeSpec
from labyrinth.trainingday_objective_metamorphic import (
    FormalObjectiveContract, FormalSkillCriterion,
)


class FormalObjectiveBridgeR16Tests(unittest.TestCase):
    def setUp(self):
        self.scope = TrainingScopeSpec(
            objective_id="prog-v1", domain="reasoning", skills=("program_reasoning",),
            specificity=3, scale=1, max_cycles=2, seed=19,
        )
        self.operational = ObjectiveContract(
            scope=self.scope, intention="Learn structurally varied register programs.",
            targets=(SkillTarget(
                skill="program_reasoning", focus_tags=("causal_structure",),
                observable_behaviors=("Compute the resulting register value.",),
                rubric_id="program_rubric", minimum_score_percent=70,
                minimum_gain_percent=10,
            ),),
            excluded_scopes=(), forbidden_claims=("TRAIN scores are not graduation.",),
        )
        self.formal = FormalObjectiveContract(
            objective_id="prog-v1", objective_kind="skill_mastery",
            skills=(FormalSkillCriterion("program_reasoning", "program_rubric", 10, 8),),
            excluded_skill_ids=(), evaluator_sha256="a" * 64,
            conditions_sha256="b" * 64, public_context_sha256="c" * 64,
            presentation_label="This wording is nonnormative.",
        )
        self.signer = Ed25519PrivateKey.generate()
        self.pins = DualObjectivePins(
            authority_id="host-operator", candidate_id="learner-one",
            formal_sha256="", operational_sha256=self.operational.fingerprint,
            scope_sha256=self.scope.fingerprint, host_binding_sha256="d" * 64,
            policy_sha256="e" * 64, source_tree_sha256="f" * 64,
            public_key=self.signer.public_key().public_bytes_raw(),
        )
        from labyrinth.trainingday_objective_metamorphic import _formal_fingerprint
        self.pins = replace(self.pins, formal_sha256=_formal_fingerprint(self.formal))

    def verify(self, **override):
        formal = override.get("formal", self.formal)
        operational = override.get("operational", self.operational)
        pins = override.get("pins", self.pins)
        payload = override.get("payload", approved_dual_objective_payload(
            self.formal, self.operational, self.pins))
        signature = override.get("signature", self.signer.sign(payload))
        return verify_dual_objective_binding(
            formal, operational, pins, payload, signature)

    def test_valid_exact_signed_pair_is_only_source_binding(self):
        receipt = self.verify()
        self.assertEqual(receipt.status, "DUAL_OBJECTIVE_SOURCE_BINDING_ONLY")
        self.assertFalse(receipt.graduation_authorized)
        self.assertFalse(receipt.semantic_equivalence_proven)
        self.assertFalse(receipt.neural_execution_authenticated)
        self.assertFalse(receipt.training_authorized)

    def test_changed_formal_threshold_or_evaluator_is_not_equivalent(self):
        for formal in (
            replace(self.formal, skills=(replace(self.formal.skills[0], min_correct=9),)),
            replace(self.formal, evaluator_sha256="9" * 64),
            replace(self.formal, conditions_sha256="8" * 64),
        ):
            with self.subTest(formal=formal), self.assertRaises(FormalObjectiveBridgeError):
                self.verify(formal=formal)

    def test_operational_focus_and_rubric_cannot_be_silently_translated(self):
        for operational in (
            replace(self.operational, targets=(replace(
                self.operational.targets[0], focus_tags=("other",)),)),
            replace(self.operational, targets=(replace(
                self.operational.targets[0], rubric_id="alternate"),)),
            replace(self.operational, targets=(replace(
                self.operational.targets[0], minimum_gain_percent=100),)),
        ):
            with self.subTest(operational=operational), self.assertRaises(FormalObjectiveBridgeError):
                self.verify(operational=operational)

    def test_authority_policy_source_and_signer_rebinding_rejected(self):
        for pins in (
            replace(self.pins, host_binding_sha256="0" * 64),
            replace(self.pins, policy_sha256="1" * 64),
            replace(self.pins, source_tree_sha256="2" * 64),
            replace(self.pins, public_key=Ed25519PrivateKey.generate().public_key().public_bytes_raw()),
        ):
            with self.subTest(pins=pins), self.assertRaises(FormalObjectiveBridgeError):
                self.verify(pins=pins)

    def test_noncanonical_manifest_tampering_and_wrong_signature_fail(self):
        payload = approved_dual_objective_payload(self.formal, self.operational, self.pins)
        with self.assertRaises(FormalObjectiveBridgeError):
            self.verify(payload=payload + b" ")
        with self.assertRaises(FormalObjectiveBridgeError):
            self.verify(signature=Ed25519PrivateKey.generate().sign(payload))

    def test_excluded_training_scope_and_undeclared_formal_skill_refused(self):
        with self.assertRaises(FormalObjectiveBridgeError):
            self.verify(formal=replace(self.formal, excluded_skill_ids=("other",)))
        with self.assertRaises(FormalObjectiveBridgeError):
            self.verify(formal=replace(self.formal, skills=(
                FormalSkillCriterion("other_skill", "new_metric", 5, 4),)))


if __name__ == "__main__":
    unittest.main()
