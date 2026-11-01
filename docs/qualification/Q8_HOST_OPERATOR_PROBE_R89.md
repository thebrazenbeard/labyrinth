# Q8 signed operator/witness evidence intake (R89)

**Status: SOURCE-ONLY; Q8 HOLD.** `labyrinth.q8_host_authority_probe`
validates a signed *statement about* the ten Q8-A1–A10 host challenges
defined in `Q8_HOST_AUTHORITY_R57.md`. It does **not** perform those
challenges, implement a signer, provision a host, consume a reservation,
grant graduation, or demonstrate that either signer is protected.

## Intended operator deployment

1. An independent human/operator provisions the Q8 test host and protected
   monotonic CAS witness. The trainer must not control its transport roster,
   signer private key, witness, snapshots, or administrative account.
2. Another examiner freezes the exact host ID, Git source SHA-256 and external
   audit-bundle SHA-256 and provisions **two distinct Ed25519 public keys**
   through a separately trusted channel. Never derive these trust pins from
   the evidence file under inspection.
3. The operator runs Q8-A1–A10 against the actual protected test service and
   captures each independent before/after witness index/head and artifact
   trace digest. In particular, **restore both service filesystem and signer
   during A3**; restoring only the trainer is an invalid antirollback test.
4. The operator and independent witness sign the same canonical observation
   body, but with different role-specific domain separators. This intake
   module accepts the bundle only with both matching pinned public keys.
5. Human reviewers independently inspect the trace bytes, audit-bundle
   integrity, actual signer custody, independent operator and witness
   administration, host leases, and the nonrollback boundary. The module's
   return value deliberately has **no authorization fields**.

## V1 envelope

The root JSON object contains exactly:

- `schema`: `labyrinth.q8.external-observation.v1`.
- `host_id`, `source_sha256`, `audit_bundle_sha256`: exact independent pins.
- `observations`: ten unique rows (`Q8-A1` through `Q8-A10`).
- `operator_signature_hex`, `witness_signature_hex`: raw Ed25519 signatures
  as lower-case 128-character hexadecimal strings.

Each observation contains exactly `case_id`, `outcome`, `before_index`,
`after_index`, `before_head_sha256`, `after_head_sha256`, and `trace_sha256`.
The index is the independent monotonic witness's committed index. Refusal,
unknown and fail-closed cases claim unchanged head/index; A9 claims exactly
one CAS winner; A5 reconciliation allows at most one reported advance.

The envelope and signed message use ASCII JSON with sorted keys, compact
separators, a newline terminator and no nonfinite values. Noncanonical wire
reserialization is rejected, preserving an unambiguous evidence-byte digest.
The two signatures cover the
**unsigned** object with role-separated prefixes:

```text
labyrinth.q8.operator-observation.v1\0 || canonical_json_body
labyrinth.q8.witness-observation.v1\0  || canonical_json_body
```

Only `cryptography` Ed25519 verification is accepted; its absence is a hard
failure. No local HMAC or alternative signature fallback exists. Duplicate
JSON keys, extra fields, forged signatures, reused signer keys, mismatched
pins, absent/duplicate cases, nonmonotonic counters and inconsistent case
outcomes are rejected. `inspect_signed_q8_evidence(raw_bytes, externally_pinned_keys)`
returns immutable classification
`SIGNED_SOURCE_ENVELOPE_NOT_PHYSICAL_Q8`, not `PASS`.

## Why a verified envelope is still insufficient

> An attacker controlling both private keys can fabricate two internally
> consistent sets of observations. Even genuine independent signatures can
> attest false readings if the signers did not actually exercise a protected
> host. The parser cannot establish custody, separation of duties, immutable
> storage, exclusive host lease, or truthful experiment execution.

The existing PR72 cloneable SQLite+witness design is a concrete falsifier.
Do not feed this result to the Minotaur graduation gate as an authority token.
External Q8 physical reviewer evidence remains mandatory, and Q5/Q6 real
pretrained learning plus Q7 sealed final exam are separate gates.

## Local source test command

`python -m unittest tests.test_q8_host_authority_probe -v`

The tests generate two ephemeral signing keys in test scope only; they do not
start the host service, use the user's device credentials, or run physical Q8.
