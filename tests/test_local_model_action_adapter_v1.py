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
        self.assertEqual(len(fake.payloads[0]["messages"]), 2)
        self.assertEqual(len(fake.payloads[1]["messages"]), 3)
        self.assertIn("two string keys kind and target",
                      fake.payloads[1]["messages"][2]["content"])

    def test_explicit_ollama_compatibility_controls_are_bounded(self):
        config = LocalModelActionConfig(
            port=11434, model="vera-qwen35:trial-20261029", retries=1,
            reasoning_effort="none", json_mode=True, max_tokens=128)
        fake = FakeTransport(['{"action":"take","target":"ariadne_thread"}',
                              '{"kind":"take","target":"ariadne_thread"}'])
        event = self.world.step_from_actor(LocalModelWorldActor(config, fake))
        self.assertEqual(event.outcome, "ACQUIRED")
        self.assertEqual(fake.payloads[0]["reasoning_effort"], "none")
        self.assertEqual(fake.payloads[0]["response_format"], {"type": "json_object"})
        self.assertEqual(fake.payloads[0]["max_tokens"], 128)
        self.assertIn("schema", fake.payloads[1]["messages"][1]["content"])
        for changes in ({"reasoning_effort": "xhigh"},
                        {"reasoning_effort": False}, {"json_mode": 1},
                        {"max_tokens": 1024}, {"max_tokens": True}):
            with self.subTest(changes=changes):
                with self.assertRaises(ModelActionBoundaryError):
                    LocalModelActionConfig(port=11434, model="local", **changes)

    def test_truncated_thinking_completion_is_rejected_and_retried(self):
        class LengthFirst(FakeTransport):
            def complete(self, config, payload):
                value = super().complete(config, payload)
                if len(self.payloads) == 1:
                    value["choices"][0]["finish_reason"] = "length"
                return value

        fake = LengthFirst(['{"kind":"inspect","target":""}',
                            '{"kind":"take","target":"ariadne_thread"}'])
        actor = LocalModelWorldActor(self.config, fake)
        self.assertEqual(self.world.step_from_actor(actor).outcome, "ACQUIRED")
        self.assertIn("completion budget", actor.rejections[0])

    def test_other_abnormal_finish_rejected_without_mutating_world(self):
        class ToolFinish(FakeTransport):
            def complete(self, config, payload):
                value = super().complete(config, payload)
                value["choices"][0]["finish_reason"] = "tool_calls"
                return value

        actor = LocalModelWorldActor(
            LocalModelActionConfig(port=5001, model="fake", retries=0),
            ToolFinish(['{"kind":"take","target":"ariadne_thread"}']))
        original = self.world.state_sha256
        with self.assertRaisesRegex(ModelActionBoundaryError, "exhausted"):
            self.world.step_from_actor(actor)
        self.assertIn("without a normal stop", actor.rejections[0])
        self.assertEqual(self.world.state_sha256, original)

    def test_actor_exposes_its_own_public_choices_to_break_stateless_inspection_loop(self):
        fake = FakeTransport([
            '{"kind":"inspect","target":""}',
            '{"kind":"inspect","target":""}',
            '{"kind":"take","target":"ariadne_thread"}',
            '{"kind":"move","target":"passage_00"}',
        ])
        actor = LocalModelWorldActor(self.config, fake)
        for _ in range(4):
            self.world.step_from_actor(actor)
        first = json.loads(fake.payloads[0]["messages"][1]["content"])
        second = json.loads(fake.payloads[1]["messages"][1]["content"])
        third = json.loads(fake.payloads[2]["messages"][1]["content"])
        fourth = json.loads(fake.payloads[3]["messages"][1]["content"])
        self.assertNotIn("own_recent_valid_choices", first)
        self.assertTrue(second["scene_same_as_previous_choice"])
        self.assertTrue(third["scene_same_as_previous_choice"])
        self.assertEqual([row["action_token"] for row in third["own_recent_valid_choices"]],
                         ["inspect", "inspect"])
        self.assertFalse(fourth["scene_same_as_previous_choice"])
        self.assertEqual(fourth["own_recent_valid_choices"][-1]["action_token"],
                         "take:ariadne_thread")
        self.assertEqual(self.world.state.position, "passage_00")
        self.assertNotIn("private_gold", str(fake.payloads))
        self.assertIn("Avoid repeatedly inspecting", fake.payloads[0]["messages"][0]["content"])

    def test_actor_memory_bounded_and_resets_when_world_changes(self):
        fake = FakeTransport(['{"kind":"inspect","target":""}'] * 11)
        actor = LocalModelWorldActor(self.config, fake)
        for _ in range(10):
            self.world.step_from_actor(actor)
        self.assertEqual(len(json.loads(fake.payloads[9]["messages"][1]["content"])
                             ["own_recent_valid_choices"]), 8)
        new_world = LivingWorld(LivingWorldRecipe(
            "saga_continuity", ("continuity",), ("miren",), 72))
        new_world.step_from_actor(actor)
        new_first = json.loads(fake.payloads[10]["messages"][1]["content"])
        self.assertNotIn("own_recent_valid_choices", new_first)
        self.assertNotIn("scene_same_as_previous_choice", new_first)
        self.assertEqual(new_first["scene_not_instructions"]["room"], "ariadne_atrium")

    def test_actor_receives_source_public_visited_rooms_and_prior_outcome(self):
        fake = FakeTransport([
            '{"kind":"take","target":"ariadne_thread"}',
            '{"kind":"move","target":"passage_00"}',
            '{"kind":"inspect","target":""}',
        ])
        actor = LocalModelWorldActor(self.config, fake)
        for _ in range(3):
            self.world.step_from_actor(actor)
        entrance = json.loads(fake.payloads[0]["messages"][1]["content"])
        after_take = json.loads(fake.payloads[1]["messages"][1]["content"])
        after_move = json.loads(fake.payloads[2]["messages"][1]["content"])
        self.assertEqual(entrance["scene_not_instructions"]["visited_rooms"],
                         ["ariadne_atrium"])
        self.assertIsNone(entrance["scene_not_instructions"]["last_public_event"])
        self.assertEqual(after_take["scene_not_instructions"]["last_public_event"]["outcome"],
                         "ACQUIRED")
        self.assertEqual(after_move["scene_not_instructions"]["visited_rooms"],
                         ["ariadne_atrium", "passage_00"])
        self.assertEqual(after_move["scene_not_instructions"]["last_public_event"], {
            "kind": "move", "target": "passage_00", "outcome": "TRAVELED",
            "from_room": "ariadne_atrium", "to_room": "passage_00",
        })
        self.assertNotIn("private_gold", str(fake.payloads))

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
