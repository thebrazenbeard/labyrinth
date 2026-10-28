"""Pretrained experiment protocol: balanced visits and matched negative control."""
import unittest

from labyrinth.hf_pilot import PilotConfig, _training_visit_indices, training_cases
from tests.test_hf_report_audit import pilot_fixture
from labyrinth.hf_report_audit import audit_hf_report


class MatchedControlsTests(unittest.TestCase):
    def test_tier_interleaving_short_budget(self):
        cfg = PilotConfig(model_path="unused", output_dir="unused", seed=217,
                          train_count=5, eval_count=5, tiers=(1, 2), steps=4)
        train, _ = training_cases(cfg)
        ids = _training_visit_indices(train, seed=cfg.seed, steps=cfg.steps)
        self.assertEqual(len(ids), 4)
        self.assertEqual([train[i].tier for i in ids], [1, 2, 1, 2])
        self.assertEqual(len(set(ids)), 4)
        self.assertEqual(ids, _training_visit_indices(train, seed=cfg.seed, steps=cfg.steps))

    def test_complete_epoch_uses_every_training_id(self):
        cfg = PilotConfig(model_path="unused", output_dir="unused", seed=318,
                          train_count=5, eval_count=5, tiers=(1, 2), steps=10)
        train, _ = training_cases(cfg)
        ids = _training_visit_indices(train, seed=cfg.seed, steps=cfg.steps)
        self.assertEqual(set(ids), set(range(10)))
        self.assertEqual(len(ids), len(set(ids)))

    def test_insufficient_tier_coverage_fails(self):
        cfg = PilotConfig(model_path="unused", output_dir="unused", seed=218,
                          train_count=5, eval_count=5, tiers=(1, 2), steps=1)
        train, _ = training_cases(cfg)
        with self.assertRaisesRegex(ValueError, "tier"):
            _training_visit_indices(train, seed=cfg.seed, steps=cfg.steps)

    def test_v3_audit_requires_matched_sham_and_original_v2_stays_valid(self):
        old = pilot_fixture()
        old["schema"] = "labyrinth.hf-lora-pilot.v2"
        self.assertEqual(audit_hf_report(old)["status"], "PASS")
        import labyrinth.hf_pilot as pilot
        newer = pilot_fixture()
        newer["schema"] = pilot.SCHEMA
        newer["config"]["answer_only"] = True
        newer["control_data"].update({
            "answer_scaffold_supplied": True,
            "sham_optimizer_steps": 2, "sham_train_case_visits": 2,
            "sham_evaluation_before_updates": False, "sham_is_compute_matched": True,
        })
        newer["training"]["sham"] = dict(newer["training"]["candidate"])
        from labyrinth.hf_pilot import _training_visit_indices
        train, _ = training_cases(PilotConfig(
            model_path="unused", output_dir="unused", seed=13,
            train_count=2, eval_count=2, tiers=(1,), steps=2
        ))
        expected = [train[i].id for i in _training_visit_indices(train, seed=13, steps=2)]
        for arm in ("sham", "candidate", "corrupted"):
            newer["training"][arm]["visited_case_ids"] = expected
        self.assertEqual(audit_hf_report(newer)["status"], "PASS")
        newer["control_data"]["sham_is_compute_matched"] = False
        self.assertEqual(audit_hf_report(newer)["status"], "FAIL")
        newer["control_data"]["sham_is_compute_matched"] = True
        newer["training"]["sham"]["visited_case_ids"] = list(reversed(expected))
        self.assertEqual(audit_hf_report(newer)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
