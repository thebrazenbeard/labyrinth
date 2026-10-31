"""Source-only Rezon collective snapshot compatibility and denial fixtures."""

import copy
from hashlib import sha256
import json
import unittest

from experiments.labyrinth_rezon_bridge.projection import (
    RezonProjectionError, project_current_collective_synthesis,
)


HEAD = "7dcb8b09c61c98c255e57ac0217250d821dfdd04"


def fixture():
    agents = ["alice", "bob"]
    answers = [
        {"id": "answer:alice", "agent_id": "alice", "kind": "answer",
         "content": "Hypothesis A", "source_refs": [], "target_ids": []},
        {"id": "answer:bob", "agent_id": "bob", "kind": "answer",
         "content": "Hypothesis B", "source_refs": [], "target_ids": []},
    ]
    reviews = [
        {"id": "review:alice", "agent_id": "alice", "kind": "review",
         "content": "Challenge B", "source_refs": [], "target_ids": ["answer:bob"]},
        {"id": "review:bob", "agent_id": "bob", "kind": "review",
         "content": "Challenge A", "source_refs": [], "target_ids": ["answer:alice"]},
    ]
    contributions = [
        {"contribution_id": item["id"], "agent_id": item["agent_id"],
         "kind": item["kind"], "content": item["content"],
         "source_refs": item["source_refs"], "target_ids": item["target_ids"]}
        for item in answers + reviews
    ]
    basis = {"episode_id": "episode-001", "prompt": "Public dev case",
             "agents": agents, "contributions": contributions}
    digest = sha256(json.dumps(basis, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"episode_id": basis["episode_id"], "prompt": basis["prompt"],
            "agents": agents, "answers": answers, "reviews": reviews,
            "syntheses": [{"synthesis_id": f"synthesis:{digest}:integrator",
                           "integrator_id": "integrator", "input_digest": digest,
                           "content": "Unverified suggestion: seek contrasting cases.",
                           "status": "unverified_model_synthesis", "is_current": True}],
            "missing_agents": [], "complete": True, "input_digest": digest}


def project(record, **overrides):
    params = {"rezon_head": HEAD, "independently_expected_head": HEAD,
              "independently_expected_episode": "episode-001",
              "independently_expected_input_digest": fixture()["input_digest"]}
    params.update(overrides)
    return project_current_collective_synthesis(record, **params)


class RezonProjectionTests(unittest.TestCase):
    def test_current_complete_collective_projects_only_manual_advisory(self):
        p = project(fixture())
        self.assertEqual(p.rezon_head, HEAD)
        self.assertEqual(p.evidence_class, "SOURCE_ONLY_UNVERIFIED_TEXT")
        self.assertEqual(len(p.advisory_text), 1)
        self.assertFalse(p.neural_weight_update_verified)
        self.assertFalse(p.distributed_tensor_execution_verified)
        self.assertFalse(p.can_authorize_training_or_graduation)

    def test_rejects_cross_source_and_episode_rebinding(self):
        for kw in ({"independently_expected_head": "a" * 40},
                   {"independently_expected_episode": "other"},
                   {"independently_expected_input_digest": "f" * 64}):
            with self.subTest(kw=kw), self.assertRaises(RezonProjectionError):
                project(fixture(), **kw)

    def test_rejects_rewritten_answer_even_with_old_synthesis(self):
        modified = fixture()
        modified["answers"][0]["content"] = "Changed answer"
        with self.assertRaisesRegex(RezonProjectionError, "input digest"):
            project(modified)

    def test_rejects_stale_or_promoted_synthesis(self):
        for field, value in (("is_current", False), ("status", "QUALIFIED"),
                             ("input_digest", "0" * 64),
                             ("synthesis_id", "synthesis:old:integrator")):
            altered = fixture()
            altered["syntheses"][0][field] = value
            with self.subTest(field=field), self.assertRaises(RezonProjectionError):
                project(altered)

    def test_retains_older_source_history_but_projects_only_current_synthesis(self):
        snapshot = fixture()
        old = {"synthesis_id": f"synthesis:{'0' * 64}:integrator",
               "integrator_id": "integrator", "input_digest": "0" * 64,
               "content": "Old advice must never reenter current review",
               "status": "unverified_model_synthesis", "is_current": False}
        snapshot["syntheses"].insert(0, old)
        result = project(snapshot)
        self.assertEqual(result.advisory_text,
                         ("Unverified suggestion: seek contrasting cases.",))

    def test_rejects_incomplete_or_missing_actors(self):
        for field, value in (("complete", False), ("missing_agents", ["bob"]),
                             ("agents", ["alice"]), ("answers", [])):
            altered = fixture()
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(RezonProjectionError):
                project(altered)

    def test_rejects_claim_of_peer_review_on_self(self):
        altered = fixture()
        altered["reviews"][0]["target_ids"] = ["answer:alice"]
        with self.assertRaisesRegex(RezonProjectionError, "peer review"):
            project(altered)

    def test_rejects_malformed_nested_and_unknown_fields(self):
        variants = []
        a = fixture()
        a["model_weight_delta"] = "forged"
        variants.append(a)
        b = fixture()
        b["answers"][0]["source_refs"] = [[]]
        variants.append(b)
        c = fixture()
        c["answers"][0]["source_refs"] = [1]
        variants.append(c)
        d = fixture()
        d["syntheses"][0]["learning_verified"] = True
        variants.append(d)
        for idx, bad in enumerate(variants):
            with self.subTest(idx=idx), self.assertRaises(RezonProjectionError):
                project(bad)

    def test_string_subclass_cannot_override_kind_or_complete(self):
        class Forged(str):
            def __eq__(self, other):
                return True
            def __ne__(self, other):
                return False
        x = fixture()
        x["answers"][0]["kind"] = Forged("gradient_operation")
        with self.assertRaises(RezonProjectionError):
            project(x)
        y = fixture()
        y["syntheses"][0]["status"] = Forged("QUALIFIED")
        with self.assertRaises(RezonProjectionError):
            project(y)

    def test_copying_all_source_data_and_pins_does_not_promote_to_execution(self):
        copied = copy.deepcopy(fixture())
        projected = project(copied)
        self.assertFalse(projected.can_authorize_training_or_graduation)
        self.assertFalse(projected.distributed_tensor_execution_verified)
        self.assertFalse(projected.neural_weight_update_verified)


if __name__ == "__main__":
    unittest.main()
