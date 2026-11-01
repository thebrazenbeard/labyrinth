# Q6 remote witness client — SOURCE ONLY, no installed antirollback

## Scope and disposition

`labyrinth/q6_signed_remote_witness_client_r121.py` implements the operator-side
transport and signed-response verifier for the `Q6IndependentMonotonicWitness`
interface in `q6_durable_slot_authority_r120.py`. It **does not** implement,
install, possess, or authenticate a separately operated monotonic witness
server. It contains no prompt, gold, model, trainer, deployment command or
graduation endpoint. The `source_only_unqualified` field remains true and
`independent_host_attested` remains false.

**Live evidence at source construction:** Labyrinth Draft #105
`1214abc6bd5beb1d184432da1442a91946c2b952` includes only an injected
witness protocol. Bus #404/#406/#407 provide coordination statements, not an
independent signer, append-only endpoint, protected prompt store or host lease.
No authenticated witness endpoint, signer public key or client credential
bundle was available in this environment. No host service was installed or
modified, and no protected prompt was revealed.

## Wire and cryptographic contract

- Operator supplies a fixed `https://HOST[:PORT]/q6/witness/r121` endpoint, a
  CA certificate, client certificate/key, server leaf certificate SHA-256
  pin, and an **independently established** 32-byte Ed25519 signer key.
- TLS 1.3 with hostname validation, trusted CA and client certificate is
  mandatory. The response must come from the pinned server leaf certificate.
  No plaintext, redirect, downgrade or proxy fallback is supported.
- For every `READ` or `CAS`, the client generates a fresh random 128-bit
  lowercase hex nonce and POSTs canonical JSON with exact keys `schema`,
  `operation`, `nonce`, `expected`, `new`. The heads contain exact nonnegative
  `commit_index` and lowercase 64-hex `head_sha256` fields.
- HTTP 200 `application/json` returns at most 4096 bytes of canonical JSON
  with exact keys `schema`, `operation`, `nonce`, `request_sha256`, `ok`, `head`,
  `signature_hex`. The response's `request_sha256` must hash the **full request
  bytes**, including old/new heads, so a previously signed ACK for a different
  CAS cannot be transplanted. Duplicate fields, unknown fields, invalid
  types, malformed heads, stale nonce or wrong operation fail closed.
- The Ed25519 signature covers
  `b"labyrinth.q6-independent-witness-reply.source-r121.v1\\0"` followed by
  canonical response JSON **without** `signature_hex`. Its public key is
  externally pinned, not supplied by the responding server.
- A successful signed CAS ACK must name the exact new head and must advance
  exactly one global event. A signed `ok=false` is refusal, not authorization.
  Timeout, mTLS failure, wrong signer, wrong certificate, HTTP failure,
  malformed response or uncertain acknowledgment raises
  `Q6DurableOutcomeUnknown`. The durable core must reconcile its local signed
  journal, never retry a protected prompt reveal.

The wire format is a **new candidate protocol**. No independent server
currently implements this exact schema, and assigning its keys to a local
same-process test server does not convert it into an authority.

## Local verification

`python -B -m unittest tests.test_q6_signed_remote_witness_client_r121 -q`

The tests cover signed success/refusal, wrong signer, signed request/nonce
replay, mutated CAS preimage, mismatched ACK head, malformed canonical JSON,
duplicate keys, invalid typed heads, insecure endpoint rejection, one-event
CAS, and a **real loopback TLS 1.3 client-certificate-authenticated handshake**
using ephemeral self-generated certificates. The loopback server persists
its high-water in a Python dictionary and is intentionally **not independent,
durable, authenticated by an external principal or rollback resistant**.
Those are source/transport tests, not a physical operational qualification.

## Required external evidence before any protected Q6 reveal

1. An independently operated signer, witness service and truly append-only
   high-water database installed in a host/account/failure domain where the
   trainer cannot read keys, impersonate the operator, modify the service,
   restore snapshots or reset the witness. Signed source Git commits and a
   local Python object alone do not meet this condition.
2. An authenticated owner grant binding a particular endpoint, TLS identities,
   client key custody, server certificate and Ed25519 signer fingerprint,
   plus service version, host identity, monotonic storage and rollback policy.
   Do not generate or claim such a grant in a test or GitHub comment.
3. Physical two-host experiments: restore the entire trainer/Q6 local DB to a
   previously signed state while **retaining the witness high-water**, then
   demonstrate that no second BEGIN or protected prompt reveal occurs; test
   lost ACK, process death, competing instances, service/key restart and
   independently frozen audit readback. The independent examiner owns the
   prompt bank and requires the signed BEGIN and witness pin before release.
4. Separate current R19-01 GPU/NAS host lease, data admission, Qwen artifact
   identity and external evaluator rights. This client grants none of them.

**Disposition: SOURCE PARTIAL PASS; physical Q6 antirollback and graduation
remain HARD HOLD.**
