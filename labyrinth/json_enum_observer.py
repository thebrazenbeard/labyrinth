"""Bounded public-only JSON-enumerated model output experiment.

This protocol is distinct from V2 free-text observation. It must never
be mistaken for weight attestation or a neural learning certificate.
"""
from __future__ import annotations

OBSERVATION_SCHEMA = "labyrinth.blind-json-enum-local-http.v1"
EVIDENCE = "LOCAL_HTTP_JSON_ENUM_SELF_REPORTED_NOT_ATTESTED"


def observe_json_enum_exam(public_path, responses_out, observation_out,
                           endpoint, model, *, max_tokens=128, timeout=60):
    raise NotImplementedError("JSON-enumerated actual public model invocation not yet implemented")


def verify_json_enum_observation(observation_path, expected_observation_sha256,
                                 public_raw, responses_raw):
    raise NotImplementedError("JSON-enumerated raw evidence replay not yet implemented")
