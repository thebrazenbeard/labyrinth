# Q8 external acceptance source validator (R68)

**Qualification status: Q8 HOLD. Source-only test ledger verifier; never a
physical host attestation, protected reservation receipt, or release gate.**

This iteration adds
`labyrinth/q8_external_acceptance_source_r68.py` and its adversarial tests.
It intentionally does **not** modify the existing authority service, Unix
transport, actual signing key, host provisioning, CAS store, or deployment
workflow. See `Q8_HOST_AUTHORITY_R57.md` for the controlling physical criteria.

## Caller-owned independent pins

The caller must obtain `Q8ExpectedWitnessPins` **out-of-band** from the
separately administered host authority operator, not from the bundle under
test. The expected values bind provider/run, host and service digests,
source commit/tree, event roster and policy digests, protected signer and
witness generations, and **two different** 32-byte Ed25519 verification keys.
The JSON bundle cannot provide replacement verification keys.

This pin struct is only a testable software interface. Source code cannot
prove those expected values were independently obtained, that an operator
is a different legal or administrative party, that keys are nonexportable,
or that the witness store is outside the trainer's rollback domain. In
particular, supplying both the fake "operator" and fake "witness" keys
yourself still yields only a source parse result.

## Proposed source ledger wire shape

The validator accepts up to 128 KiB of UTF-8 JSON with exactly these
statement fields:

`schema`, `provider_id`, `run_id`, `host_sha256`, `service_sha256`,
`source_commit_sha`, `source_tree_sha`, `roster_sha256`, `policy_sha256`,
`signer_generation`, `witness_generation` and `cases`.

Schema is `labyrinth.q8-external-acceptance-source.v1`; the signed statement
must match every independently supplied pin. The `cases` list must contain
**exactly and in order** Q8-A1 through Q8-A10. Every row contains:

`test_id`, `operation_nonce`, `before_index`, `before_head_sha256`,
`after_index`, `after_head_sha256`, `refusal_code`, `audit_sha256`,
`trace_sha256`, `operator_signature_hex`, `witness_signature_hex`.

Both signatures cover a domain-separated canonical JSON of the complete
statement and the individual unsigned row. The statement and row are pinned
against JSON duplicate keys, unexpected fields, wrong casing/length of
digests, boolean-as-integer coercion and bad signature material. Within the
bundle the nonces must be unique, and a digest used for either an audit or
a trace cannot appear in **any other audit or trace field**.

The `refusal_code` values are a **proposed source-ledger vocabulary**, not
proof that a host provider returns those codes. An independent examiner
must check each against the real service's original audit and replay traces:

| Row | Source outcome label | Declared protected-head transition |
| --- | --- | --- |
| A1 | `RESTORED_TRAINER_REFUSED` | Unchanged |
| A2 | `RESTORED_SERVICE_REFUSED` | Unchanged |
| A3 | `CLONE_REFUSED` | Unchanged |
| A4 | `REPLAY_REFUSED` | Unchanged |
| A5 | `RECONCILED_COMMIT` / `REMAINS_UNKNOWN` | Exactly +1 new head / unchanged |
| A6 | `BINDING_MISMATCH` | Unchanged |
| A7 | `PEER_UNAUTHORIZED` | Unchanged |
| A8 | `ROTATION_STALE` | Unchanged |
| A9 | `SINGLE_CAS_WINNER` | Exactly +1 new head |
| A10 | `FAIL_CLOSED_UNAVAILABLE` | Unchanged |

This is a format for *records supplied after real tests*, **not a harness
that performs those tests**. The source validator cannot observe whether
the before/after protected heads actually existed, whether the head was
backed by one atomic CAS store, or whether A5 was read-only. Even the
separate witness signature is only proof of possession of a particular
private key; it does not establish that its private key is outside a
cloned service image. A replay of a whole previously valid source bundle
is not detectable without a separately maintained replay register; callers
must never use this parser as that register.

## Acceptance, tests, and deployment holds

`validate_q8_acceptance_bundle(bundle_bytes, independently_pinned_values)`
returns an instance-immutable, tuple-backed `Q8SourceObservation`, with the
bundle SHA-256, ten checked row IDs, and source-defined read-only properties:

- `status == SOURCE_ONLY_UNATTESTED`
- `q8_physically_qualified == False`
- `can_graduate == can_merge == can_deploy == False`

**Python in-process trust caveat:** immutable tuple instances do not defend
against code with authority to monkeypatch their class, replace imported
functions, or tamper with the interpreter. For example,
`Q8SourceObservation.can_merge = property(lambda self: True)` can replace
the class-level property in a compromised process. Consequently consumers
must never treat these fields as a security trust boundary. The *actual*
permission decision must be enforced by an independently administered host
authority with its own protected state and identity binding; this parser
provides no such boundary.

Malformed, incomplete, forged, unpinned or ambiguous source data raises
`Q8AcceptanceSourceError`. This result **must not** be accepted as a host
reservation, Q8 PASS, Q5/Q6 learning, Q7 final, or Q9 release authorization.

Local adversarial command:

```shell
python3 -m unittest tests.test_q8_external_acceptance_source_r68 -q
```

Hard holds to clear outside this repository, with evidence independently
examined: a real protected external signer; nonrollback monotonic CAS and
consumed-event register; provisioned transport principal and immutable
operator roster; actual independently witnessed execution of all ten
physical A1..A10 attacks; binding of each recorded before/after head to
the protected store; key rotation under a separate operator; a real
outage/reconciliation trace; and independent host owner qualification.
No source-level green test, including these tests, clears those holds.
