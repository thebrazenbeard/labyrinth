"""Local-world model adapter tests are 100% fake-transport / no HTTP."""

from __future__ import annotations

import json
import unittest

from labyrinth.living_world_v1 import LivingWorld, LivingWorldRecipe
from labyrinth.local_model_action_adapter_v1 import (
    LocalModelActionConfig, LocalModelWorldActor, LoopbackHTTPTransport,
    ModelActionBoundaryError, parse_model_action,
)


class FakeTransport:
    def __init__(self, contents):
        self.responses = list(contents)
        self.payloads = []

    def complete(self, config, payload):
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": self.responses.pop(0)}}]}


class LocalModelActionTests(unittest.TestCase):
    def setUp(self):
        self.world = LivingWorld(LivingWorldRecipe(
            "saga_continuity", ("continuity",), ("miren",), 71))
        self.config = LocalModelActionConfig(port=5001, model="local-fixture", retries=1)

    def test_fake_model_chooses_real_consequence_without_local_http(self):
        fake = FakeTransport(['{"kind":"take","target":"ariadne_thread"}'])
        actor = LocalModelWorldActor(self.config, fake)
        event = self.world.step_from_actor(actor)
        self.assertEqual(event.outcome, "ACQUIRED")
        self.assertIn("ariadne_thread", self.world.state.inventory)
        self.assertEqual(actor.attempts, 1)
        payload = fake.payloads[0]
        self.assertEqual([item["role"] for item in payload["messages"]],
                         ["system", "user"])
        story_input = json.loads(payload["messages"][1]["content"])
        self.assertEqual(story_input["source_trust"],
                         "UNTRUSTED_STORY_DATA_NO_INSTRUCTION_AUTHORITY")
        self.assertNotIn("private_gold", str(payload))
        self.assertNotIn("grader", str(payload["messages"][1]).lower())

    def test_invalid_json_retried_once_then_valid_action(self):
        fake = FakeTransport(["prose not JSON", '{"kind":"inspect","target":""}'])
        actor = LocalModelWorldActor(self.config, fake)
        event = self.world.step_from_actor(actor)
        self.assertEqual(event.outcome, "OBSERVED")
        self.assertEqual(actor.attempts, 2)
        self.assertEqual(len(actor.rejections), 1)

    def test_invalid_attempts_do_not_modify_world_state(self):
        before = self.world.state_sha256
        fake = FakeTransport(["garbage", '{"kind":"graduate","target":""}'])
        actor = LocalModelWorldActor(self.config, fake)
        with self.assertRaisesRegex(ModelActionBoundaryError, "exhausted"):
            self.world.step_from_actor(actor)
        self.assertEqual(self.world.state_sha256, before)
        self.assertFalse(self.world.events)
        self.assertEqual(actor.attempts, 2)

    def test_strict_parser_rejects_invented_actions_scoring_and_duplicate_keys(self):
        observation = self.world.observe()
        rejected = (
            '```json\n{"kind":"inspect","target":""}\n```',
            '{"kind":"inspect","target":"","score":100}',
            '{"kind":"inspect","kind":"move","target":""}',
            '{"kind":"move","target":"unseen_secret_room"}',
            '{"kind":"take","target":"private_minotaur_gold"}',
            '{"kind":"inspect","target":null}',
            '{"kind":"inspect"}',
        )
        for bad in rejected:
            with self.subTest(bad=bad):
                with self.assertRaises(ModelActionBoundaryError):
                    parse_model_action(bad, observation)

    def test_loopback_gate_blocks_remote_hosts_redirect_configuration_and_live_default(self):
        for host in ("localhost", "127.0.0.2", "192.168.1.1", "example.com", "[::1]"):
            with self.subTest(host=host):
                with self.assertRaises(ModelActionBoundaryError):
                    LocalModelActionConfig(port=5001, model="local", host=host)
        with self.assertRaisesRegex(ModelActionBoundaryError, "explicit operator opt-in"):
            LoopbackHTTPTransport()
        with self.assertRaises(ModelActionBoundaryError):
            LocalModelActionConfig(port=5001, model="local", retries=3)
        with self.assertRaises(ModelActionBoundaryError):
            LocalModelActionConfig(port=0, model="local")
        self.assertEqual(LocalModelActionConfig(port=5001, model="local", host="::1").host,
                         "::1")


if __name__ == "__main__":
    unittest.main()
