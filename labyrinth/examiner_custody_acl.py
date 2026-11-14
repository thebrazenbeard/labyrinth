"""Read-only, fail-closed local examiner filesystem ACL assessment.

The initial test scaffold intentionally does not implement ACL inspection.
"""
from __future__ import annotations


def classify_acl_snapshot(snapshot):
    return {"status": "UNKNOWN", "findings": [], "independent_custody_verified": False}


def audit_local_examiner_custody(private_cases_path, out_path, *,
                                 signing_private_key_path=None):
    raise NotImplementedError("examiner local ACL readback not yet implemented")
