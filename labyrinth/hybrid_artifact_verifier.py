"""Read-only verification of V2 public-rule hybrid artifact and held source pins."""
from __future__ import annotations


def verify_hybrid_artifacts(public_path, candidate_path,
                            selected_path, report_path, *,
                            expected_public_sha256, expected_report_sha256,
                            observation_path=None,
                            expected_observation_sha256=None):
    raise NotImplementedError("V2 bound source-only verification not implemented")
