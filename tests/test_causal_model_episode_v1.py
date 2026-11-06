"""Evidence boundary tests for the opt-in, open-ended causal-model episode."""

import hashlib
import json
import unittest

from labyrinth.causal_model_episode_v1 import run_causal_episode
from labyrinth.causal_reality_v1 import CausalBlueprint
from labyrinth.local_model_action_adapter_v1 import (
    LocalModelActionConfig, ModelActionBoundaryError,
)


class _FakeTransport:
    def __init__(self, content, *, finish_reason="stop"):
        self.content = content
        self.finish_reason = finish_reason
        self.calls = []

    def complete(self, config, payload):
        self.calls.append((config, payload))
        return {"choices": [{"message": {"content": self.content},
                             "finish_reason": self.finish_reason}]}


class CausalModelEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.config = LocalModelActionConfig(
            port=1234, model="local-test-model", retries=2,
            json_mode=False, max_tokens=256)

    def assert_public_payload_protected(self, payload, seed, challenge_class):
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertNotIn("response_format", payload)
        scene = json.loads(payload["messages"][1]["content"])
        self.assertEqual(set(scene), {
            "scenario", "visible_objects", "appearances", "public_clues",
            "terminal_temperature_c", "remaining_manipulations",
        })
        self.assertEqual(scene["remaining_manipulations"], 1)
        self.assertEqual(len(scene["visible_objects"]), 4)
        whole_payload = json.dumps(payload).lower()
        for forbidden in (
            "allowed_actions", "allowed_action_json", "solution",
            "challenge_class", "private_verdict", "possible_safe_manipulations",
            "causal_links", "blueprint_sha256", "failure_causes",
            str(seed), challenge_class,
        ):
            # Prevent hidden class/seed/feasibility and answer-token leakage.
            self.assertNotIn(forbidden, whole_payload)

    def test_freeform_utterance_executes_once_and_keeps_private_protected(self):
        # Fake model is explicitly supplied a successful utterance for the
        # unit test; the real learner receives no hidden graph/class/solution.
        safe_target = CausalBlueprint.from_seed(39261517, "solvable").actuators[0]
        verb = {"valve": "rotate", "lever": "pull", "cable": "cut", "brace": "push"}[safe_target.form]
        fake = _FakeTransport(
            f"I {verb} the {safe_target.label.replace('_', ' ')} once, "
            "because the public gauges suggest a cooling route.")
        run = run_causal_episode(self.config, fake, seed=39261517,
                                 challenge_class="solvable")
        self.assertEqual(len(fake.calls), 1, "retries must not become feedback loops")
        self.assert_public_payload_protected(fake.calls[0][1], 39261517,
                                             "solvable")
        public = run.public_receipt
        self.assertEqual(public["intervention_status"], "EXECUTED")
        self.assertEqual(public["event_evidence"], "SOURCE_TOY_WORLD_EVENT")
        self.assertEqual(public["transport_evidence"], "TEST_FAKE_OR_CUSTOM_TRANSPORT_UNVERIFIED")
        self.assertEqual(public["resolution"]["interpreted_action"],
                         {"verb": {"rotate": "turn", "pull": "pull", "cut": "cut",
                                   "push": "push"}[verb], "target": safe_target.label})
        self.assertEqual(public["source_event"]["observed_outcome"], "STATION_STABILIZED")
        self.assertFalse(public["model_endpoint_identity_attested"])
        self.assertFalse(public["training_performed"])
        self.assertFalse(public["weights_changed"])
        self.assertNotIn("challenge_class", run.public_json())
        self.assertNotIn("blueprint_sha256", run.public_json())
        self.assertNotIn("blueprint_commitment_sha256", run.public_json())
        self.assertNotIn("possible_safe_manipulations", run.public_json())
        self.assertEqual(public["rationale_quality"], "NOT_EVALUATED")
        self.assertIn("because", public["proposal_and_rationale_as_submitted"])
        self.assertIn("private_verdict", run.protected_evaluation)
        self.assertEqual(run.protected_evaluation["blueprint"]["challenge_class"],
                         "solvable")
        self.assertEqual(run.protected_evaluation["private_verdict"]["outcome"],
                         "STATION_STABILIZED")
        # Public digest is reproducible and excludes private grading details.
        without_hash = dict(public)
        without_hash.pop("receipt_sha256")
        expected = hashlib.sha256(json.dumps(
            without_hash, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
        self.assertEqual(run.receipt_sha256, expected)

    def test_unresolved_freeform_action_has_no_fabricated_event_or_verdict(self):
        fake = _FakeTransport("I cannot justify a specific intervention from these observations.")
        run = run_causal_episode(self.config, fake, seed=39261517,
                                 challenge_class="impossible")
        self.assertEqual(len(fake.calls), 1)
        self.assert_public_payload_protected(fake.calls[0][1], 39261517,
                                             "impossible")
        self.assertEqual(run.public_receipt["intervention_status"], "UNRESOLVED")
        self.assertIsNone(run.public_receipt["source_event"])
        self.assertEqual(run.public_receipt["event_evidence"], "NO_SOURCE_EVENT")
        self.assertEqual(run.public_receipt["resolution"]["status"], "UNKNOWN")
        self.assertIsNone(run.public_receipt["resolution"]["interpreted_action"])
        self.assertIsNone(run.protected_evaluation["private_verdict"])
        self.assertFalse(run.protected_evaluation["world_state"]["manipulated"])

    def test_malformed_completion_is_explicit_unresolved_not_catastrophe(self):
        fake = _FakeTransport("I turn the amber valve.", finish_reason="length")
        run = run_causal_episode(self.config, fake, seed=55,
                                 challenge_class="solvable")
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(run.public_receipt["intervention_status"], "UNRESOLVED")
        self.assertEqual(run.public_receipt["resolution"]["reason"],
                         "INCOMPLETE_CHAT_COMPLETION")
        self.assertIsNone(run.public_receipt["source_event"])
        self.assertIsNone(run.protected_evaluation["private_verdict"])

    def test_explicit_json_mode_rejected_before_model_or_world_execution(self):
        fake = _FakeTransport("I turn the amber valve.")
        cfg = LocalModelActionConfig(port=1234, model="local-test-model",
                                     json_mode=True)
        with self.assertRaises(ModelActionBoundaryError):
            run_causal_episode(cfg, fake, seed=5, challenge_class="solvable")
        self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()
