# Q8 protected host authority: physical acceptance contract (R57)

**Status: HOLD.** This is a test and deployment acceptance contract, not a
provider, an attestation, a release permit, or evidence of host custody. The
source contracts and test doubles in this repository do not satisfy Q8 by
themselves.

## Exact boundary

Labyrinth already implements a fail-closed source-level reservation protocol:

- `external_monotonic_authority_contract.py` pins an Ed25519 public key, the
  complete reservation intent, and a compare-and-swap predecessor. Its client
  refuses lost-ACK replay and requires a second read of the new head.
- `authority_service_core_r20.py` validates provisioned Journey identity and
  consumes event/exam/nonce identifiers transactionally. However, its SQLite
  file and Ed25519 signing seed are cloneable by anyone controlling the service
  account or storage rollback domain.
- `authority_unix_service_r22.py` derives the peer UID from Linux
  `SO_PEERCRED` instead of request JSON and checks operator file ancestry.
  Those checks **do not** prove a separate OS account, unshared administrative
  control, a protected signer, or an antirollback storage boundary.

The source-only clone attack is explicit in
`tests/test_external_monotonic_authority_contract.py`: restoring both the
provider's old state **and its signer** can produce a second apparently valid
reservation. Adding another mutable in-process signature, HMAC, SQLite column,
or receipt wrapper cannot resolve this failure.

## Independently operated service requirements

The operator of the actual host authority, not the model/trainer, must:

1. **Bind identity at the transport boundary.** The service owns an immutable
   UID/mTLS-identity-to-principal roster and resolves the caller from OS or
   authenticated-channel credentials. A principal in caller JSON is never
   accepted as authority. The trainer must not manage the service account,
   roster, socket parent, or administrative controls.
2. **Protect the signing key.** The private signing operation must be owned by
   a separately administered provider or nonexportable signing service. The
   verification public key and key rotation generation must be independently
   pinned by the host operator. A signed receipt from a cloned key is not
   sufficient.
3. **Anchor a monotonic CAS head outside rollback reach.** A durable,
   independently operated store must atomically compare exact
   `(provider, run, epoch, fence, commit index, head hash)` and consume each
   event, exam, and operation nonce once. Restoring the trainer and service
   filesystem images must **not** restore the minimum accepted head or the
   consumed-event register. No unprotected replica may become an authority.
4. **Freeze provisioning independently.** Candidate bytes, scope, prior
   checkpoint, exam lineage, source commit/tree, operator policy, trainer
   peer, and migration witness come from an independently pinned operator
   roster before the request. Replaying a valid signature for another
   candidate or restoring an old roster must fail.
5. **Reconcile uncertain outcomes.** Loss of the reserve ACK means UNKNOWN,
   never automatic retry. A read-only reconciliation must return one protected
   committed operation (or remain UNKNOWN); it cannot consume again. A
   restart or failover cannot emit a duplicate reservation.
6. **Separate observation and permission.** Q8 reservation evidence may
   authorize only its declared reservation step. It cannot by itself create a
   pretrained-learning claim, private examination result, graduation token,
   merge permit, deployment authorization, or transferred model consent.

## Required external acceptance tests

Execute these against a separately administered **test** instance with the
independent operator and witness present. Record the exact host, service,
source, signer key generation, counter generation, event roster, and test
artifact digests. These are not executable from this repository alone.

| Test | Adversarial action | Required independently observed result |
| --- | --- | --- |
| Q8-A1 | Restore an older trainer checkpoint | Current protected CAS head unchanged; old request refused |
| Q8-A2 | Restore an older service DB/volume snapshot | Independent monotonic witness retains later head; replay refused |
| Q8-A3 | Clone the service image and its software signer | Clone cannot sign as the uniquely pinned protected provider or pass witness acceptance |
| Q8-A4 | Reuse a consumed event, exam, or nonce after restart | One durable receipt only; second reservation refused |
| Q8-A5 | Lose ACK after commit and restart the service | Read-only reconciliation succeeds or remains UNKNOWN; no retry/write |
| Q8-A6 | Change candidate, scope, checkpoint, source, or private-exam lineage | Host-provisioned binding mismatch refused before CAS |
| Q8-A7 | Supply another principal in JSON or connect from an unprovisioned UID | Transport-bound identity is used; no access or reservation |
| Q8-A8 | Rotate signer or roll back operator roster without approved generation | Old generation refused; explicit independently witnessed rotation required |
| Q8-A9 | Race two workers at the same expected head | At most one atomic winner; loser sees a conflict or UNKNOWN, never two grants |
| Q8-A10 | Deny signer, store, network, or witness during reconciliation | Fail closed; no fallback to local SQLite/HMAC or trainer-declared permissions |

For each row, capture the independently signed or operator-verified **before
and after** protected witness heads, service audit event, actual refusal code,
and a separately reviewed replay/rollback trace. A Linux unit test or CI log
cannot substitute for this evidence.

## Current evidence classification and handoff

The repo's source-only tests exercise useful policy, signature and Linux
transport behavior; they do **not** attest the host operator, signer custody,
rollback isolation, or a live lease. At R57 review, the Q8 host service and
independent external monotonic witness remain unqualified, and the TRAININGDAY
R19-01 exclusive host/GPU lease release and authenticated NAS route still
lack the necessary owner receipts. Do not start host validation based solely
on a running process, a connected tunnel, a successful CI badge, or the
service's `SOURCE_ONLY` status string.

**Release decision:** Q8 = HOLD until the operator independently supplies and
an examiner verifies all required physical acceptance evidence. Even a
genuine Q8 PASS does not clear Q5/Q6 pretrained learning, Q7 protected final
exam, or Q9 graduation/merge/deployment authority.
