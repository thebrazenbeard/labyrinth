"""Matched-case negative controls for a blind structural exam.

Only source-derived paired measurements; no generalization qualification.
"""
def compare_with_controls(private_path, public_path, precommit_path,
                          responses_path, out_path, *, expected_precommit_sha256,
                          observation_path=None, expected_observation_sha256=None):
    raise NotImplementedError("paired trivial-control falsification not implemented")
