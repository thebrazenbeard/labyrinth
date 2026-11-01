# External cognitive TRAIN rights/exclusion admission — private consumer v1

Status: **HOLD for genuine training** until independently administered rights,
protected-evaluation, host/GPU, NAS/Executor, and physical-training grants are
actually installed. This change installs **no** keys, issuer, lease, evaluator,
private corpus, gold answers, route, or grant. Do not use the test's disposable
Ed25519 signers as production authority.

`labyrinth.external_cognitive_admission_v1.verify_external_cognitive_admission`
is a **consumer**, not an issuer. It verifies two independently pinned Ed25519
attestations against the *actual 15-row private TRAIN export* and the complete
private semantic exclusion set:

1. The external rights steward must sign the canonical private source identity,
   corpus SHA-256, exact private manifest SHA-256, protected roster SHA-256,
   explicit candidate permission, license identifier, and private source-
   authorship/license/consent/privacy-review evidence commitments. The four
   actual private evidence byte sequences must be supplied; their SHA-256 values
   must match the signed commitments. Cryptography verifies authenticity and
   byte integrity, **not** the legal adequacy of the evidence. The independent
   steward remains responsible for substantive license/consent review.
2. A *different* protected evaluator custodian must sign the same dataset and
   same complete roster digest, its count, `SEALED_COMPLETE` status, and a
   reviewed cross-system no-train-exposure ledger. Protected evaluation/gold
   answers are never accessed by this module. The signer must independently
   establish roster completeness and exposure-ledger accuracy.
3. The consumer reruns all canonical 15-row TRAIN checks, recomputes the private
   manifest, checks semantic collisions with the supplied complete roster, and
   checks exact source commit and source-file digest expected by its operator.
4. Both signatures use role-specific domain-separated Ed25519 messages.
   A signed claim is canonical sorted compact JSON with no trailing newline,
   unknown/duplicate keys, or extra fields. Both claims must have nonces of
   16–128 safe characters, issuance no more than 60 seconds in the future,
   expiry after the trusted clock but within 24 hours of issuance. A private,
   durable SQLite registry consumes both nonces in one transaction; the next
   replay is rejected. Use a host-trusted clock.
5. The trusted pin file must be installed **outside Git** by an independent
   administrator, with a different pinned Ed25519 public key and issuer ID for
   each role and without group/world write access. The verifier **does not**
   cryptographically establish ownership or independence of that file: the
   physical authority must authenticate its provisioning and ACL separately.
   Pins embedded in claims, Git, the public bus, and test fixtures are **not**
   independent trust anchors.

The external trust-anchor file has canonical JSON shape:

```json
{
  "schema": "labyrinth.external-cognitive-trust-anchors.v1",
  "steward": {"id": "<independently provisioned rights issuer ID>", "ed25519_public_key_hex": "<32-byte pinned public key, lowercase hex>"},
  "evaluator": {"id": "<different evaluator custodian ID>", "ed25519_public_key_hex": "<different 32-byte pinned public key, lowercase hex>"}
}
```

The `source` object in the signed rights claim contains repository, producer
commit, producer file SHA-256, dataset SHA-256, objective ID, contract SHA-256,
and cycle index, all bound to the provided full private manifest. Required
rights fields are `permission=GRANTED_FOR_CANDIDATE`, license identifier, four
`private://...` evidence references, and `evidence_sha256` with exactly
`source_authorship`, `license`, `consent`, `privacy_review` keys. The evaluator
claim is separately signed with `roster_status=SEALED_COMPLETE` and
`cross_system_exposure_ledger=REVIEWED_NO_TRAIN_EXPOSURE`. These are signed
operator attestations, not local self-authorizing booleans.

**Privacy restriction:** Do not publish these claims, corpus hashes, manifest
hashes, prompt/semantic/source hashes, exclusion roster hashes or counts,
answer labels, seed/cycle/source secrets, evidence blobs, or the nonce registry
on a public PR/chat/bus. Low-entropy 15-row choices can be recovered from public
commitments by offline enumeration. Keep every sensitive input, runtime log and
test-output artifact in privately controlled custody. No live evidence is
included in this repository.

A successful return is a deliberately opaque `PrivateExternalAdmissionVerification`
whose `private_evidence_verified()` must only be called in the authenticated
private operator context. Its public representation remains `GPU_HOLD`, and
`authorizes_training`, `authorizes_host_execution`, and `authorizes_exam` are
**always false**. Source verification is not an exclusive GPU lease, a NAS
route-owner grant, an external rights issuance, or qualification of actual
model-weight learning. No genuine training process should start on this return
alone.

Adversarial synthetic test: `python -m unittest tests.test_external_cognitive_admission_v1 -q`.
