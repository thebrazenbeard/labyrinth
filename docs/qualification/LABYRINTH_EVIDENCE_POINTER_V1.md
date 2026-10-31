# Labyrinth source evidence pointer v1 — non-authorizing cross-repository contract

**Status:** Source-only interface; no model-execution or release qualification.

`labyrinth.evidence_pointer` publishes an inert, canonical, bounded pointer to a
particular source declaration for the **five-arm regenerative MiniLM**
experiment. A downstream reader can compare the pointer to its *own*
independently acquired expectations without importing the entire Labyrinth
training runtime. A successful comparison is **not** a signed attestation.

The only allowed schema is `labyrinth.source-evidence-pointer.v1`; its experiment
identity is fixed to `labyrinth.minilm-five-arm-independent-evidence-source.v1`
and status fixed to `SOURCE_ONLY_UNATTESTED`. This is not the separate Qwen
four-control experiment. The pointer cannot select or change experiment type.

## Contract

- `run_id`: bounded, explicit source-run label, not a host lease identifier.
- `source_head`, `source_tree`: exact lowercase 40-digit Git commit/tree IDs.
- `plan_sha256`, `train_exposure_sha256`, `control_matrix_sha256`,
  `development_event_sha256`, `checkpoint_manifest_sha256`, `scope_sha256`:
  exact lowercase 64-digit source commitments. A digest here is a claim, not
  evidence that the bytes were read by an independent host.
- `rezon_source_head` and `rezon_receipt_bundle_sha256`: optional **paired**,
  opaque references to a Rezon *source* commit and a source receipt bundle.
  `null` for both means no Rezon reference. Neither field grants control-plane
  authority, distributed-training custody, signer identity, consent, or a right
  to replay a receipt.
- `schema`, `experiment`, `qualification_state`: fixed literals above.

Only canonical ASCII JSON with sorted keys, compact separators, exactly one
terminal newline, and at most 4096 bytes is accepted. Duplicate keys, omitted
keys, extra properties (including any `identity`, `consent`, `permissions`,
`can_authorize_*` or signing claims), wrong digest lengths, noncanonical
encoding, unknown experiment, and stale or mismatched independent expectations
all fail closed.

## Producer/consumer boundary

```python
from dataclasses import asdict
from labyrinth.evidence_pointer import (
    SCHEMA, EXPERIMENT, QUALIFICATION,
    ExpectedSourcePinsV1, SourceEvidencePointerV1,
    encode_source_evidence_pointer, verify_source_evidence_pointer,
)

# The consumer must fetch/pin these values through its OWN governed source.
# Do not derive them from the submitted pointer or a caller's self-assertion.
expected = ExpectedSourcePinsV1(
    run_id="minilm-run-001",
    source_head="a" * 40,
    source_tree="b" * 40,
    plan_sha256="1" * 64,
    train_exposure_sha256="2" * 64,
    control_matrix_sha256="3" * 64,
    development_event_sha256="4" * 64,
    checkpoint_manifest_sha256="5" * 64,
    scope_sha256="6" * 64,
)

# A source producer can encode a pointer but cannot authenticate those claims.
pointer = SourceEvidencePointerV1(
    schema=SCHEMA, experiment=EXPERIMENT,
    qualification_state=QUALIFICATION, **asdict(expected),
)
wire = encode_source_evidence_pointer(pointer)

# The consumer performs exact independent pin matching with no side effects.
match = verify_source_evidence_pointer(wire, expected=expected)
assert match.qualification_state == "SOURCE_ONLY_UNATTESTED"
assert not match.can_authorize_training
assert not match.can_authorize_graduation
assert not match.can_authorize_deployment
```

**Threat boundaries:** Matching source strings cannot prove real optimizer
updates, base-model identity, checkpoint byte readback, cross-process private
evaluation, host signer/counter custody, nonrollback/replay exclusion, source
precommit chronology, user/model identity, authorization, or consent. An
in-process Python caller may forge its own objects or expectations. Therefore
no consumer should use this success value as an execution switch or trust it
as Q5–Q9 acceptance evidence.

**Owners:** Labyrinth publishes frozen source semantics; Rezon separately
accepts producer compatibility; governed model consumers separately implement
read-only ingestion; independent host authority controls runtime rights.
Only owner-approved exact-head CI, independent evidence review, and host
qualification can advance release gates. This interface does not merge or
deploy anything.
