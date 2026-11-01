# R160 — Independent Q5–Q9/Q6 evaluator-custody admission: source hard hold

**Status: `UNVERIFIED_REJECTED`; NO protected evaluator is installed.**

The independently appointed examiner and separately controlled protected-bank
custodian, key-registry operator, rights steward, host transport identity and
Q6 monotonic witness have **not** been authenticated in this repository.
Labyrinth Draft #105 provides a SQLite/Unix source prototype; its adversarial
review explicitly found that a forged same-process witness can duplicate an
exposure after local rollback. Draft #106 improves cognitive tokenizer source
parity only. Model-training Draft #184 returns `UNVERIFIED_REJECTED` for
caller-provided cognitive TRAIN data. Bus #406's independent-examiner and
weight-change request is **not** an actual external authorization.

`labyrinth.independent_evaluator_custody_r160` is an additive, metadata-only
**source-inspection** seam, not a protected prompt service or authority
broker. The exact compact canonical JSON schema carries opaque role/store IDs,
claimed Ed25519 public key/generation/signature, and SHA-256 commitments for
the private bank, train exclusion roster, pretraining precommitment, frozen
model, Q6 witness head, rights decision, and examiner policy. It cannot
authenticate any of them. Every structurally valid caller-provided claim
returns `UNVERIFIED_REJECTED`, with all authorization/attestation flags false;
`require_independent_evaluator_admission` always raises a `HARD HOLD`.
Malformed metadata, repeated keys, extra authority flags, or attempted raw
prompt/gold fields are refused before any report. Returned data contains only
the SHA-256 of the caller's *claim bytes*, no identities, signature, bank
commitment, prompt, gold, predictions, or raw labels. No network, disk,
evaluator, physical model, or protected bank access occurs.

## Missing external issuer and evidence

Before creating any positive admission path, a separately appointed issuer
and operator must independently provide, verify and retain all of the following
*outside trainer/model rollback reach*:

1. **Examiner and bank custody:** actual issuer appointment/authority, a
   separately operated private prompt+gold store, host/service identity and
   physical access roster. Trainer and model-selection principals must lack
   private-bank read, key-management and final-unseal rights.
2. **Signer trust root:** independently pinned non-exportable signing provider,
   public-key generation, revocation register and an authenticated transport
   principal derived from OS credentials or mTLS — never JSON-provided role
   labels or public keys alone.
3. **Chronology and separation:** externally timestamped pretraining Q5 plan
   and predecessor retention receipt; frozen model and control-arm SHA pins;
   independent exclusion commitments for train, prior dev, previously spent
   final and transfer cases; semantic human review as necessary. A late
   signature cannot repair a retrospective precommitment.
4. **Q6 antirollback:** independent monotonic compare-and-advance witness
   outside trainer/service snapshot domain, one-shot exposure/nonce register,
   durable lost-ACK reconciliation and a signed service identity/restore
   qualification. A software witness passed in-process does not suffice.
5. **Rights and graduation:** rights-holder and independent data-steward
   immutable ADMIT for the exact sources/purposes; signed evaluator policy,
   preregistered thresholds and clean controls; examiner-owned one-shot final
   unseal only after immutable model freeze. Q9 promotion, model serving and
   deployment are separate grants; an exam score cannot confer them.

For a real physical claim, preserve exact issuer/host identities and key
generations, signed raw receipt bytes plus independent verifier pins, bank
storage ACL/lease attestations, Q6 A1–A10 adversarial restore/clone traces,
before/after externally anchored head and immutable audit logs. The issuer
must decide admission; source code, CI, ChatGPT agents and owner administrative
access cannot appoint themselves.

## Synthetic verification

```sh
python -B -m unittest tests.test_independent_evaluator_custody_r160 -v
```

Twelve synthetic tests challenge self-signed/fabricated receipts, a self-owned
bank and retrospective precommitment, spoofed transport/witness head, role
collision, malformed/duplicate JSON, and raw gold/authority-field smuggling.
Green tests prove only that this local **rejection** boundary is fail-closed.
They neither exercise a protected bank nor replace independent physical
examiner/custodian acceptance.
