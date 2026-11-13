"""An unkeyed precommit SHA cannot impersonate an externally held examiner key.

These are synthetic local keys. Valid Ed25519 verifies signer-key possession
only; key custody, independence and live model outputs remain unverified.
"""
from __future__ import annotations

import base64
from contextlib import redirect_stdout
from io import StringIO
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from labyrinth.live_model_probe import commit_cases, compare_reports, main
from test_live_model_probe_receipt_integrity import _observation, _receipt


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


class ExaminerSignedPrecommitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cases = self.root / "cases.json"
        self.commit = self.root / "precommit.json"
        self.before = self.root / "before.json"
        self.after = self.root / "after.json"
        self.out = self.root / "paired.json"
        self.envelope = self.root / "attestation.json"
        self.pub = self.root / "examiner.pub"
        self.exam_id = "blind-exam-W37"
        self.prompt = "Compute 2 + 2; output only ANSWER: followed by integer."
        self.cases.write_text(json.dumps({
            "schema": "labyrinth.live-probe-cases.v1",
            "cases": [{"case_id": "trial-1", "prompt": self.prompt, "answer": "4"}],
        }), encoding="utf-8")
        self.plan = commit_cases(self.cases, self.commit)
        report = _observation()
        report["commitment_sha256"] = self.plan["receipt_sha256"]
        report["cases_file_sha256"] = self.plan["cases_file_sha256"]
        report["trials"][0]["prompt_sha256"] = sha256(self.prompt.encode()).hexdigest()
        self.before.write_text(json.dumps(_receipt(report)), encoding="utf-8")
        self.after.write_text(json.dumps(_receipt(report)), encoding="utf-8")
        self.key = Ed25519PrivateKey.generate()
        self.key_pin = self._write_public_key(self.key)
        self._sign(self.key)

    def _write_public_key(self, signer):
        raw = signer.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.pub.write_bytes(raw)
        return sha256(raw).hexdigest()

    def _sign(self, signer, *, payload_changes=None, fields=None):
        payload = {
            "schema": "labyrinth.examiner-signed-precommit-payload.v1",
            "exam_id": self.exam_id,
            "precommit_sha256": self.plan["receipt_sha256"],
            "examiner_key_sha256": sha256(signer.public_key().public_bytes(
                Encoding.Raw, PublicFormat.Raw)).hexdigest(),
            "scope": "EXAMINER_AUTHORSHIP_OF_PRECOMMIT_ONLY",
        }
        payload.update(payload_changes or {})
        envelope = {
            "schema": "labyrinth.examiner-signed-precommit.v1",
            "payload": payload,
            "signature_ed25519_b64": base64.b64encode(
                signer.sign(_canonical(payload))).decode("ascii"),
        }
        envelope.update(fields or {})
        self.envelope.write_text(json.dumps(envelope), encoding="utf-8")

    def _compare(self, **options):
        args = {
            "cases_path": self.cases,
            "commitment_path": self.commit,
            "expected_commitment_sha256": self.plan["receipt_sha256"],
            "examiner_attestation_path": self.envelope,
            "examiner_public_key_path": self.pub,
            "expected_examiner_key_sha256": self.key_pin,
            "examiner_exam_id": self.exam_id,
        }
        args.update(options)
        return compare_reports(self.before, self.after, self.out, **args)

    def test_valid_signature_binds_precommit_but_not_independent_custody(self):
        result = self._compare()
        self.assertEqual(result["examiner_signature_evidence"],
                         "ED25519_VALID_CALLER_PINNED_KEY")
        self.assertEqual(result["examiner_key_sha256"], self.key_pin)
        self.assertEqual(result["examiner_exam_id"], self.exam_id)
        self.assertFalse(result["independent_custody_verified"])
        self.assertFalse(result["neural_learning_verified"])

    def test_rehashed_claim_with_changed_precommit_does_not_forge_signature(self):
        data = json.loads(self.envelope.read_text(encoding="utf-8"))
        data["payload"]["precommit_sha256"] = "e" * 64
        self.envelope.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "signature|precommit|attestation"):
            self._compare()
        self.assertFalse(self.out.exists())

    def test_new_attacker_key_and_valid_new_signature_do_not_replace_trusted_key(self):
        attacker = Ed25519PrivateKey.generate()
        self._write_public_key(attacker)
        self._sign(attacker)
        with self.assertRaisesRegex(ValueError, "examiner|key|anchor"):
            self._compare()
        self.assertFalse(self.out.exists())

    def test_signer_cannot_set_independent_custody_in_payload(self):
        self._sign(self.key, payload_changes={"independent_custody_verified": True})
        with self.assertRaisesRegex(ValueError, "attestation|scope|schema|field"):
            self._compare()
        self.assertFalse(self.out.exists())

    def test_attestation_for_different_exam_id_is_rejected(self):
        self._sign(self.key, payload_changes={"exam_id": "another-exam"})
        with self.assertRaisesRegex(ValueError, "exam|attestation"):
            self._compare()
        self.assertFalse(self.out.exists())

    def test_partial_signature_arguments_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "attestation|examiner|signature"):
            self._compare(examiner_public_key_path=None)
        self.assertFalse(self.out.exists())

    def test_malformed_signature_cannot_be_accepted(self):
        data = json.loads(self.envelope.read_text(encoding="utf-8"))
        data["signature_ed25519_b64"] = "!!!"
        self.envelope.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "signature|attestation"):
            self._compare()
        self.assertFalse(self.out.exists())


    def test_mutated_signature_bytes_rejected_after_all_fields_match(self):
        data = json.loads(self.envelope.read_text(encoding="utf-8"))
        raw = bytearray(base64.b64decode(data["signature_ed25519_b64"]))
        raw[22] ^= 1
        data["signature_ed25519_b64"] = base64.b64encode(raw).decode("ascii")
        self.envelope.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "signature"):
            self._compare()
        self.assertFalse(self.out.exists())

    def test_compare_cli_verifies_exact_signer_and_records_ceiling(self):
        output = StringIO()
        with redirect_stdout(output):
            main(["compare", "--before", str(self.before), "--after", str(self.after),
                  "--cases", str(self.cases), "--commitment", str(self.commit),
                  "--expected-precommit-sha256", self.plan["receipt_sha256"],
                  "--examiner-attestation", str(self.envelope),
                  "--examiner-public-key", str(self.pub),
                  "--expected-examiner-key-sha256", self.key_pin,
                  "--examiner-exam-id", self.exam_id, "--out", str(self.out)])
        on_disk = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["examiner_signature_evidence"],
                         "ED25519_VALID_CALLER_PINNED_KEY")
        self.assertFalse(on_disk["independent_custody_verified"])
        self.assertIn(on_disk["receipt_sha256"], output.getvalue())

    def test_unpinned_signature_not_counted_as_authenticated_signer(self):
        result = compare_reports(
            self.before, self.after, self.out,
            cases_path=self.cases, commitment_path=self.commit,
            expected_commitment_sha256=self.plan["receipt_sha256"])
        self.assertEqual(result["examiner_signature_evidence"], "NOT_SUPPLIED")
        self.assertFalse(result["independent_custody_verified"])


if __name__ == "__main__":
    unittest.main()
