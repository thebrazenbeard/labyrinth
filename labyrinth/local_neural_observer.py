"""Local Ollama HTTP observation of public blind structural questions only.

No private examiner input, model-weight attestation, signature custody or
training assertion. This module is being implemented test-first.
"""
def observe_public_exam(public_path, responses_out, observation_out, endpoint, model, *,
                        max_tokens=96, timeout=30):
    raise NotImplementedError("public-model invocation evidence not implemented")


def verify_local_observation(observation_path, expected_observation_sha256,
                             public_raw, responses_raw):
    raise NotImplementedError("bound observation validation not implemented")
