"""Independent Wave16 adversarial scoring and receipt-trust controls.

Source baseline: PR157 integration 16f84839df869d2a89243d55baa212cf92527944.
No model, daemon, or host service is installed by these tests; fixed child
subprocesses are the only actual candidate execution.
"""

from copy import deepcopy
from hashlib import sha256
from unittest.mock import patch
import json
import unittest

from labyrinth.minotaur_process_adapter import ProcessModelAdapter, _ABSTAIN
from labyrinth.switchback_circuit import ProbeSession, make_case
from labyrinth.switchback_oracle import search
from scripts import switchback_model_trial as trial


class SkepticModelEvidence(unittest.TestCase):
    @staticmethod
    def _fixed():
        return ProcessModelAdapter(mode="fixed", actions=(), generation_seed=0,
                                   timeout_seconds=1.0, max_output_bytes=1024)

    @staticmethod
    def _config(seed=0):
        return trial.TrialConfig(seed, 0, "fixed", 1, 1.0, 3.0, 1024)

    def test_single_shared_json_decoder_fault_cannot_invent_model_win(self):
        path = search(make_case(0))["witness"]
        self.assertIsNotNone(path)
        original = trial._read_json

        def corrupt_decoder(raw):
            if raw == _ABSTAIN.encode("utf-8"):
                return {"kind": "PLAN", "actions": path}
            return original(raw)

        with patch.object(trial, "_read_json", new=corrupt_decoder):
            report = trial.run_trial(self._config(0), self._fixed())

        # Actual fixed-process stdout was ABSTAIN, NOT the forged PLAN.
        self.assertEqual(report["turns"][0]["response_sha256"],
                         sha256(_ABSTAIN.encode("utf-8")).hexdigest())
        self.assertFalse(report["verified_success"],
                         "both supposed independent wire decoders shared one corrupted function")

    def test_single_stdlib_json_decoder_fault_cannot_self_certify_false_win(self):
        """Independent wrappers may still share one mutable library function."""
        path = search(make_case(0))["witness"]
        original_loads = json.loads

        def broken_stdlib_decoder(raw, *args, **kwargs):
            if raw in (_ABSTAIN, _ABSTAIN.encode("utf-8")):
                return {"kind": "PLAN", "actions": path}
            return original_loads(raw, *args, **kwargs)

        with patch.object(json, "loads", new=broken_stdlib_decoder):
            report = trial.run_trial(self._config(0), self._fixed())
        self.assertEqual(report["turns"][0]["response_sha256"],
                         sha256(_ABSTAIN.encode("utf-8")).hexdigest())
        self.assertFalse(report["verified_success"],
                         "both wire parsers delegated to a shared patched json.loads")

    def test_unkeyed_outer_checksum_cannot_authenticate_model_claims(self):
        """Negative control: a selfhash is reconstructible without a signer."""
        report = trial.run_trial(self._config(0), self._fixed())
        self.assertFalse(report["verified_success"])
        forged = deepcopy(report)
        forged["model_neural_identity_verified"] = True
        forged["model_weight_update_verified"] = True
        forged["generalization_qualified"] = True
        forged["source_verdict"] = "VERIFIED_WIN"
        forged["verified_success"] = True
        forged["receipt_sha256"] = sha256(trial._canonical({
            key: value for key, value in forged.items()
            if key != "receipt_sha256"})).hexdigest()
        self.assertEqual(forged["receipt_sha256"], sha256(trial._canonical({
            key: value for key, value in forged.items()
            if key != "receipt_sha256"})).hexdigest())
        self.assertNotEqual(forged["source_receipt"]["verdict"], "VERIFIED_WIN")
        self.assertFalse(report["model_neural_identity_verified"])

    def test_public_child_view_hides_seed_but_host_report_exposes_it(self):
        """Negative control: host-private report must never become candidate input."""
        case = make_case(0)
        public = trial._public_request(ProbeSession(case).view(), 0)
        self.assertNotIn('"seed":', public)
        self.assertNotIn('"case_id":', public)
        report = trial.run_trial(self._config(0), self._fixed())
        self.assertEqual(report["case_seed_host_only"], 0)
        self.assertEqual(report["configuration"]["case_seed"], 0)
        self.assertFalse(report["model_neural_identity_verified"])


if __name__ == "__main__":
    unittest.main()
