"""PASS-as-EXPLOIT: signer-controlled semantic registry can fabricate diversity.

This does NOT violate the declared API contract. It demonstrates why graduation
is unqualified without an independently audited, actually diverse challenge
registry and truthful external evaluator. Exact base: 140840be58b9a8c.
"""

from dataclasses import replace
import hashlib
import unittest

from labyrinth.graduation_gate import GateState, decide_graduation
from test_graduation_gate import (
    KEY, POLICY, evaluated_receipt, independently_signed,
)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


class ForgedSemanticIndependence(unittest.TestCase):
    def test_cosmetic_prompt_nonce_claimed_as_32_independent_mechanics_graduates(self):
        # All 16 prompts in each split are the same arithmetic problem. The
        # unique header is merely a nonce; no transfer mechanics differ.
        altered = {}
        registry = []
        for entry in POLICY.independent_semantic_registry:
            if entry.slice_name == "train":
                registry.append(entry)
                continue
            prompt = f"Question {entry.case_id}: What is 2 + 2?"
            # Every case gets a unique hash and fake semantic-family digest.
            family = sha("pretend-independent-family:" + entry.case_id)
            commitment = sha(prompt)
            altered[entry.case_id] = (family, commitment)
            registry.append(replace(entry, semantic_family_id=family,
                                    prompt_commitment=commitment))
        policy = replace(POLICY, independent_semantic_registry=tuple(registry))
        receipt = evaluated_receipt()
        for records in receipt["slices"].values():
            for trial in records:
                family, commitment = altered[trial["case_id"]]
                trial["semantic_family_id"] = family
                trial["prompt_commitment"] = commitment
        raw, mac = independently_signed(receipt)
        decision = decide_graduation(policy, GateState(), raw, mac,
                                     independent_evaluator_key=KEY)
        self.assertEqual(decision.disposition, "GRADUATED")
        # This is successful SOURCE acceptance of a dishonest semantic claim,
        # NOT a real qualification, independently verified task, or model run.


if __name__ == "__main__":
    unittest.main()
