"""An untrusted host cannot certify independent examiner custody from hashes.

Synthetic ACL evidence exercises conservative denial; on Windows a separate
integration test reads the current OS ACL rather than trusting metadata.
"""
from __future__ import annotations

import json
from unittest.mock import patch
from subprocess import CompletedProcess
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from labyrinth.examiner_custody_acl import (
    classify_acl_snapshot, audit_local_examiner_custody)


def entry(sid, rights, access="Allow", inherited=True):
    return {"sid": sid, "mask": rights, "access": access, "inherited": inherited}


def fixture(*, file_aces=(), parent_aces=(), key_aces=None):
    targets = [{
        "role": "private_cases",
        "layers": [
            {"is_target": True, "aces": list(file_aces)},
            {"is_target": False, "aces": list(parent_aces)}],
    }]
    if key_aces is not None:
        targets.append({
            "role": "signer_private_key",
            "layers": [{"is_target": True, "aces": list(key_aces)}],
        })
    return {"platform": "windows", "identity_sid": "S-1-5-21-99-1001",
            "targets": targets}


class CustodyAclHostileTests(unittest.TestCase):
    def test_authenticated_users_modify_private_file_is_explicitly_insecure(self):
        case = fixture(file_aces=[entry("S-1-5-11", 0x001301BF)])
        result = classify_acl_snapshot(case)
        self.assertEqual(result["status"], "INSECURE_BROAD_ALLOW")
        self.assertIn("BROAD_READ_PRIVATE_CASES", result["findings"])
        self.assertIn("BROAD_WRITE_PRIVATE_CASES", result["findings"])
        self.assertFalse(result["independent_custody_verified"])
        self.assertFalse(result["acl_isolation_verified"])

    def test_generic_read_flag_on_target_is_not_ignored(self):
        # Windows ACCESS_MASK GENERIC_READ can survive in negative signed
        # FileSystemRights; no FILE_READ_DATA low bit is necessary.
        for mask in (0x80000000, -2147483648):
            with self.subTest(mask=mask):
                report = classify_acl_snapshot(fixture(
                    file_aces=[entry("S-1-5-11", mask)]))
                self.assertEqual(report["status"], "INSECURE_BROAD_ALLOW")
                self.assertIn("BROAD_READ_PRIVATE_CASES", report["findings"])
                self.assertFalse(report["independent_custody_verified"])

    def test_generic_write_on_parent_is_not_ignored(self):
        report = classify_acl_snapshot(fixture(
            file_aces=[entry("S-1-5-21-99-1001", 0x1F01FF)],
            parent_aces=[entry("S-1-5-32-545", 0x40000000)]))
        self.assertEqual(report["status"], "INSECURE_BROAD_ALLOW")
        self.assertIn("BROAD_WRITE_ANCESTOR_OF_PRIVATE_CASES", report["findings"])

    def test_generic_all_on_signer_secret_is_read_and_write_exposure(self):
        report = classify_acl_snapshot(fixture(
            file_aces=[entry("S-1-5-21-99-1001", 0x1F01FF)],
            key_aces=[entry("S-1-1-0", 0x10000000)]))
        self.assertIn("BROAD_READ_SIGNER_PRIVATE_KEY", report["findings"])
        self.assertIn("BROAD_WRITE_SIGNER_PRIVATE_KEY", report["findings"])
        self.assertEqual(report["status"], "INSECURE_BROAD_ALLOW")

    def test_generic_execute_alone_is_not_file_read_or_write(self):
        report = classify_acl_snapshot(fixture(
            file_aces=[entry("S-1-5-11", 0x20000000)]))
        self.assertEqual(report["status"], "NO_BROAD_ALLOW_DETECTED_NOT_ATTESTED")

    def test_broad_writable_parent_blocks_candidate_exclusion_claim(self):
        case = fixture(file_aces=[entry("S-1-5-21-99-1001", 0x001F01FF)],
                       parent_aces=[entry("S-1-5-32-545", 0x001301BF)])
        result = classify_acl_snapshot(case)
        self.assertEqual(result["status"], "INSECURE_BROAD_ALLOW")
        self.assertIn("BROAD_WRITE_ANCESTOR_OF_PRIVATE_CASES", result["findings"])
        self.assertFalse(result["independent_custody_verified"])

    def test_signing_key_with_broad_read_is_insecure_even_when_cases_appear_private(self):
        case = fixture(file_aces=[entry("S-1-5-21-99-1001", 0x1F01FF)],
                       key_aces=[entry("S-1-1-0", 0x00120089)])
        result = classify_acl_snapshot(case)
        self.assertIn("BROAD_READ_SIGNER_PRIVATE_KEY", result["findings"])
        self.assertEqual(result["status"], "INSECURE_BROAD_ALLOW")

    def test_clean_allowlist_still_cannot_certify_independence(self):
        case = fixture(file_aces=[entry("S-1-5-21-99-1001", 0x1F01FF)],
                       parent_aces=[entry("S-1-5-18", 0x1F01FF)])
        result = classify_acl_snapshot(case)
        self.assertEqual(result["status"], "NO_BROAD_ALLOW_DETECTED_NOT_ATTESTED")
        self.assertEqual(result["findings"], [])
        self.assertFalse(result["acl_isolation_verified"])
        self.assertFalse(result["independent_custody_verified"])

    def test_incomplete_acl_or_missing_sid_fails_closed_as_unknown(self):
        case = fixture(file_aces=[{"sid": None, "mask": 0x1F01FF,
                                     "access": "Allow", "inherited": True}])
        result = classify_acl_snapshot(case)
        self.assertEqual(result["status"], "ACL_INSPECTION_INCOMPLETE")
        self.assertFalse(result["acl_isolation_verified"])
        self.assertFalse(result["independent_custody_verified"])

    def test_malformed_layer_is_not_a_success_or_uncaught_exception(self):
        case = fixture(file_aces=[entry("S-1-5-11", 0x001301BF)])
        case["targets"][0]["layers"][0] = None
        result = classify_acl_snapshot(case)
        self.assertEqual(result["status"], "ACL_INSPECTION_INCOMPLETE")
        self.assertFalse(result["independent_custody_verified"])

    @unittest.skipUnless(os.name == "nt", "requires actual Windows pathname")
    def test_truncated_getacl_ancestry_cannot_claim_complete_local_inspection(self):
        # A target-file Get-Acl alone can look private while its parent is
        # writable by Authenticated Users. The policy must NOT mark it complete.
        with TemporaryDirectory() as root:
            case_dir = Path(root) / "private"
            case_dir.mkdir()
            private = case_dir / "answer.json"
            private.write_text("test only", encoding="utf-8")
            out = Path(root) / "receipt.json"
            snapshot = {
                "platform": "windows",
                "identity_sid": "S-1-5-21-99-1001",
                "targets": [{
                    "role": "private_cases",
                    "layers": [{
                        "is_target": True,
                        "aces": [entry("S-1-5-21-99-1001", 0x1F01FF)],
                    }],
                }],
            }
            response = CompletedProcess(
                args=["Get-Acl"], returncode=0, stdout=json.dumps(snapshot), stderr="")
            with patch("labyrinth.examiner_custody_acl.subprocess.run",
                       return_value=response):
                result = audit_local_examiner_custody(private, out)
            self.assertEqual(result["status"], "ACL_INSPECTION_INCOMPLETE")
            self.assertFalse(result["acl_lineage_complete"])
            self.assertFalse(result["acl_inspection_complete"])
            self.assertFalse(result["independent_custody_verified"])

    @unittest.skipUnless(os.name == "nt", "requires actual Windows pathname")
    def test_forged_intermediate_ancestor_cannot_claim_complete_inspection(self):
        with TemporaryDirectory() as root:
            case_dir = Path(root) / "private"
            case_dir.mkdir()
            private = case_dir / "answers.json"
            private.write_text("test only", encoding="utf-8")
            out = Path(root) / "receipt.json"
            paths = [private, *private.parents]
            # A same-count fake walk with no verifiable path identities
            # is not evidence that the actual intermediate ACLs were read.
            fake_layers = [{
                "is_target": i == 0,
                "aces": [],
            } for i, p in enumerate(paths)]
            snapshot = {
                "platform": "windows",
                "identity_sid": "S-1-5-21-99-1001",
                "targets": [{
                    "role": "private_cases",
                    "layers": fake_layers,
                }],
            }
            response = CompletedProcess(
                args=["Get-Acl"], returncode=0, stdout=json.dumps(snapshot), stderr="")
            with patch("labyrinth.examiner_custody_acl.subprocess.run",
                       return_value=response):
                result = audit_local_examiner_custody(private, out)
            self.assertEqual(result["status"], "ACL_INSPECTION_INCOMPLETE")
            self.assertFalse(result["acl_lineage_complete"])
            self.assertFalse(result["independent_custody_verified"])

    @unittest.skipUnless(os.name == "nt", "requires actual Windows pathname")
    def test_valid_target_path_but_missing_rootward_layers_is_still_incomplete(self):
        with TemporaryDirectory() as root:
            private = Path(root) / "case.json"
            private.write_text("test only", encoding="utf-8")
            output = Path(root) / "receipt.json"
            snapshot = {
                "platform": "windows",
                "identity_sid": "S-1-5-21-99-1001",
                "targets": [{"role": "private_cases", "layers": [{
                    "path": str(private),
                    "is_target": True,
                    "aces": [],
                }]}],
            }
            process = CompletedProcess(
                args=["Get-Acl"], returncode=0, stdout=json.dumps(snapshot), stderr="")
            with patch("labyrinth.examiner_custody_acl.subprocess.run",
                       return_value=process):
                report = audit_local_examiner_custody(private, output)
            self.assertEqual(report["status"], "ACL_INSPECTION_INCOMPLETE")
            self.assertFalse(report["acl_lineage_complete"])
            self.assertFalse(report["acl_inspection_complete"])

    @unittest.skipUnless(os.name == "nt", "requires actual Windows pathname")
    def test_all_layers_present_but_one_parent_path_wrong_is_incomplete(self):
        with TemporaryDirectory() as root:
            private = Path(root) / "case.json"
            private.write_text("test only", encoding="utf-8")
            output = Path(root) / "receipt.json"
            paths = [private, *private.parents]
            layers = [{"path": str(p), "is_target": i == 0, "aces": []}
                      for i, p in enumerate(paths)]
            layers[1]["path"] = str(Path(root) / "fake-parent")
            snapshot = {
                "platform": "windows",
                "identity_sid": "S-1-5-21-99-1001",
                "targets": [{"role": "private_cases", "layers": layers}],
            }
            process = CompletedProcess(
                args=["Get-Acl"], returncode=0, stdout=json.dumps(snapshot), stderr="")
            with patch("labyrinth.examiner_custody_acl.subprocess.run",
                       return_value=process):
                report = audit_local_examiner_custody(private, output)
            self.assertEqual(report["status"], "ACL_INSPECTION_INCOMPLETE")
            self.assertFalse(report["acl_lineage_complete"])
            self.assertFalse(report["independent_custody_verified"])

    @unittest.skipUnless(os.name == "nt", "requires actual Windows pathname")
    def test_bound_ancestry_mismatch_keeps_broad_allow_denial_and_reports_incomplete(self):
        with TemporaryDirectory() as root:
            private = Path(root) / "case.json"
            private.write_text("test only", encoding="utf-8")
            out = Path(root) / "result.json"
            paths = [private, *private.parents]
            layers = [{
                "path": str(path),
                "is_target": i == 0,
                "aces": [entry("S-1-5-11", 0x001301BF)] if i == 0 else [],
            } for i, path in enumerate(paths)]
            layers[1]["path"] = str(Path(root) / "not-the-real-parent")
            snapshot = {
                "platform": "windows",
                "identity_sid": "S-1-5-21-99-1001",
                "targets": [{"role": "private_cases", "layers": layers}],
            }
            response = CompletedProcess(
                args=["Get-Acl"], returncode=0, stdout=json.dumps(snapshot), stderr="")
            with patch("labyrinth.examiner_custody_acl.subprocess.run",
                       return_value=response):
                result = audit_local_examiner_custody(private, out)
            self.assertEqual(result["status"], "INSECURE_BROAD_ALLOW")
            self.assertIn("BROAD_READ_PRIVATE_CASES", result["findings"])
            self.assertFalse(result["acl_lineage_complete"])
            self.assertFalse(result["acl_inspection_complete"])
            self.assertFalse(result["independent_custody_verified"])
            self.assertNotIn(str(private), out.read_text(encoding="utf-8"))

    def test_explicit_deny_ace_does_not_get_mistaken_for_allow(self):
        case = fixture(file_aces=[entry("S-1-5-11", 0x1F01FF, "Deny")])
        result = classify_acl_snapshot(case)
        self.assertEqual(result["status"], "NO_BROAD_ALLOW_DETECTED_NOT_ATTESTED")
        self.assertFalse(result["independent_custody_verified"])

    @unittest.skipUnless(os.name == "nt", "requires actual Windows NTFS ACL readback")
    def test_actual_host_readback_returns_nonpromoted_receipt(self):
        with TemporaryDirectory() as root:
            base = Path(root)
            private = base / "private-cases.json"
            private.write_text('{"fake":"test-only"}', encoding="utf-8")
            report = base / "acl-audit.json"
            actual = audit_local_examiner_custody(private, report)
            self.assertTrue(report.is_file())
            self.assertEqual(actual["receipt_sha256"],
                             json.loads(report.read_text())["receipt_sha256"])
            self.assertFalse(actual["independent_custody_verified"])
            self.assertFalse(actual["acl_isolation_verified"])
            self.assertTrue(actual["acl_lineage_complete"])
            self.assertTrue(actual["acl_inspection_complete"])
            self.assertIn(actual["status"], {
                "INSECURE_BROAD_ALLOW", "ACL_INSPECTION_INCOMPLETE",
                "NO_BROAD_ALLOW_DETECTED_NOT_ATTESTED"})
            with self.assertRaises(FileExistsError):
                audit_local_examiner_custody(private, report)


if __name__ == "__main__":
    unittest.main()
