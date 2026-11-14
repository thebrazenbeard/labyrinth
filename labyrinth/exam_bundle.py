"""Optional source-only examiner bundle publication (implementation pending)."""
from __future__ import annotations


def prepare_exam_bundle(destination, *, count=8, rng=None, worlds=None):
    raise NotImplementedError("source-only staged exam bundle not yet implemented")


def verify_exam_bundle(bundle, *, expected_manifest_sha256):
    raise NotImplementedError("source-only staged exam verifier not yet implemented")


def main(argv=None):
    raise NotImplementedError("source-only staged exam CLI not yet implemented")
