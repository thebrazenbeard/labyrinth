"""Offline fake-transport tests of the opt-in model-run episode CLI."""

from __future__ import annotations

from dataclasses import asdict
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from types import SimpleNamespace
import unittest

from labyrinth.local_model_action_adapter_v1 import LocalModelActionConfig
from labyrinth.local_model_world_runner_v1 import (
    EVIDENCE_CLASS, SCHEMA, _trace_digest, main, run_local_model_world,
)
from labyrinth.source_only_saga_demo_v1 import FictionalActionModel


class FixedFakeTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, config, payload):
        self.requests.append((config, payload))
        return {"choices": [{"message": {"content": self.replies.pop(0)}}]}


class PublicPlannerFakeTransport:
    """Pretends to be a model, but plans using only the submitted public JSON."""

    def __init__(self):
        self.planner = FictionalActionModel()
        self.requests = []

    def complete(self, config, payload):
        self.requests.append((config, payload))
        public = json.loads(payload["messages"][-1]["content"])
        scene = public["scene_not_instructions"]
        observation = SimpleNamespace(
            room_id=scene["room"],
            visible_exits=tuple(scene["exits"]),
            visible_items=tuple(scene["items"]),
            inventory=tuple(scene["inventory"]),
            difficulty=scene["difficulty"],
        )
        action = self.planner.choose_action(observation)
        return {"choices": [{"message": {"content": json.dumps(asdict(action))}}]}


class ModelRunTests(unittest.TestCase):
    def setUp(self):
        self.config = LocalModelActionConfig(port=11434, model="fixture-local", retries=0)

    def test_one_model_action_has_original_observation_hash_and_source_event(self):
        transport = FixedFakeTransport(['{"kind":"take","target":"ariadne_thread"}'])
        result = run_local_model_world(self.config, transport, max_actions=1)
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["evidence_class"], EVIDENCE_CLASS)
        self.assertEqual(result["outcome"], "ACTION_BUDGET_EXHAUSTED")
        self.assertEqual(result["world_final_state"]["inventory"], ("ariadne_thread",))
        self.assertEqual(result["model_completion_requests"], 1)
        self.assertEqual(result["turns"][0]["observation"]["room_id"], "ariadne_atrium")
        self.assertEqual(result["turns"][0]["event"]["outcome"], "ACQUIRED")
        self.assertEqual(result["turns"][0]["event"]["state_sha256"],
                         result["world_final_state_sha256"])
        self.assertEqual(result["trace_sha256"], _trace_digest({
            key: value for key, value in result.items() if key != "trace_sha256"
        }))
        self.assertFalse(result["claims"]["weights_changed"])
        self.assertFalse(result["claims"]["neural_training_performed"])
        public = json.loads(transport.requests[0][1]["messages"][1]["content"])
        self.assertEqual(public["source_trust"],
                         "UNTRUSTED_STORY_DATA_NO_INSTRUCTION_AUTHORITY")
        self.assertNotIn("private_gold", json.dumps(public))

    def test_bad_model_action_stops_unchanged_and_retains_rejection_reason(self):
        invalid = FixedFakeTransport(['{"action":"graduate","score":100}'])
        result = run_local_model_world(self.config, invalid, max_actions=2)
        self.assertEqual(result["outcome"], "MODEL_ACTION_REJECTED")
        self.assertEqual(result["failed_action_reason"],
                         "local model exhausted bounded action attempts; world state unchanged")
        self.assertEqual(result["turns"], [])
        self.assertEqual(result["world_final_state"]["steps"], 0)
        self.assertEqual(len(result["rejected_action_reasons"]), 1)
        self.assertEqual(result["model_completion_requests"], 1)

    def test_rejection_after_valid_event_preserves_prior_verified_trace(self):
        transport = FixedFakeTransport([
            '{"kind":"take","target":"ariadne_thread"}',
            '{"kind":"graduate","target":""}',
        ])
        result = run_local_model_world(self.config, transport, max_actions=2)
        self.assertEqual(result["outcome"], "MODEL_ACTION_REJECTED")
        self.assertEqual(len(result["turns"]), 1)
        self.assertEqual(result["world_final_state"]["steps"], 1)
        self.assertEqual(result["world_final_state_sha256"],
                         result["turns"][0]["event"]["state_sha256"])

    def test_public_planner_fixture_completes_real_source_world(self):
        fake = PublicPlannerFakeTransport()
        result = run_local_model_world(self.config, fake, max_actions=100)
        self.assertEqual(result["outcome"], "PUBLIC_QUEST_COMPLETED")
        self.assertTrue(result["world_final_state"]["completed"])
        self.assertIn("ariadne_thread", result["world_final_state"]["inventory"])
        self.assertEqual(result["world_final_state"]["position"], "ariadne_atrium")
        self.assertEqual(len(result["turns"]), len(fake.requests))
        self.assertIn("MINOTAUR_ENCOUNTER",
                      [turn["event"]["outcome"] for turn in result["turns"]])
        self.assertTrue(all("private" not in json.dumps(turn).lower()
                            for turn in result["turns"]))

    def test_cli_refuses_live_transport_without_explicit_opt_in(self):
        called = []
        def transport_factory():
            called.append(True)
            raise AssertionError("should not construct transport")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as ctx:
            main(["--model", "fixture-local", "--port", "11434"],
                 transport_factory=transport_factory)
        self.assertEqual(ctx.exception.code, 2)
        self.assertFalse(called)

    def test_cli_renders_verifiable_partial_trace_without_host_effects(self):
        output = StringIO()
        fake = FixedFakeTransport(['{"kind":"inspect","target":""}'])
        with redirect_stdout(output):
            code = main([
                "--model", "fixture-local", "--port", "11434",
                "--allow-local-inference", "--max-actions", "1",
                "--reasoning-effort", "none", "--json-mode",
                "--max-completion-tokens", "128",
            ], transport_factory=lambda: fake)
        self.assertEqual(code, 1)  # incomplete quest is not a model pass
        data = json.loads(output.getvalue())
        self.assertEqual(data["outcome"], "ACTION_BUDGET_EXHAUSTED")
        self.assertEqual(data["turns"][0]["event"]["outcome"], "OBSERVED")
        sent = fake.requests[0][1]
        self.assertEqual(sent["reasoning_effort"], "none")
        self.assertEqual(sent["response_format"], {"type": "json_object"})
        self.assertEqual(sent["max_tokens"], 128)

    def test_bounded_input_validation(self):
        fake = FixedFakeTransport([])
        for parameters in ({"max_actions": 0}, {"max_actions": 129},
                           {"difficulty": 0}, {"difficulty": 6},
                           {"public_seed": -1}, {"max_actions": True}):
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    run_local_model_world(self.config, fake, **parameters)
        self.assertEqual(fake.requests, [])


if __name__ == "__main__":
    unittest.main()
