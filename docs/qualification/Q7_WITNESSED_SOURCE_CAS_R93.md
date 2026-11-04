# R93 Q7: Witnessed Public Source Chain — NOT a Protected Final Examination

**Classification:** `SIGNED_Q7_CHAIN_SOURCE_ONLY_NO_TEMPORAL_CUSTODY`.

`labyrinth/q7_witnessed_cas_intake_r93.py` composes the existing exact-byte
Ed25519 Q7 source verifier with a separate, **externally pinned** witness-key
and two monotonically numbered public receipt head transitions. It is a strict
source-envelope consumer, not an external witness implementation. Its callers
are untrusted until an actual independent host/service enforces the pinned
source, key roles, chronology, one-shot state and rollback protections.

## Public wire contract

The host owner must provide four distinct, **independently pinned** Ed25519
role keys: precommit signer, anchor signer, training-start signer and witness
signer. The precommit/phase source-verification functions authenticate their
respective signed public bytes. The separate witness must sign a canonical
domain-separated JSON record with exactly these fields:

- `schema`, `host_id`, `exam_id`, `event_id`, `precommit_sha256`;
- `anchor_receipt_sha256`, `training_start_receipt_sha256`;
- `before_index`, `anchor_index`, `after_index`;
- `before_head_sha256`, `anchor_head_sha256`, `after_head_sha256`;
- `custody_verified=false`, `graduation_authorized=false`;
- `witness_signature_hex`.

The index sequence must advance exactly `N → N+1 → N+2`. Each hash head is
computed as SHA-256 over `labyrinth.q7.witnessed-chain-head.v1\0`, the prior
head's 32 bytes, the next index as eight big-endian bytes, and the signed
phase-receipt digest's 32 bytes. Thus a swapped public receipt or changed
signed precommit cannot preserve the expected independently pinned after-head
without invalidating the chain. The actual prior index/head and final head
must be supplied out-of-band by an external owner, never derived from the
same witness envelope under inspection.

No sealed exam questions, answers, labels, sample identities, private seeds or
optimizer evidence are represented in this public witness schema. Only the
already public hash commitment and receipt identifiers appear.

## Tested source attacks

`tests/test_q7_witnessed_cas_intake_r93.py` exercises:

1. Positive exact-signature, four-role and public binding with immutable
   nonauthorizing result flags.
2. Independently computed append-head known-form calculation, index
   contiguity, wrong predecessor head and wrong final head.
3. Reusing an exam event **when the independent consumed-event roster is
   present**; foreign host/exam/objective/source key substitution.
4. Post-construction role-key collapse via `object.__setattr__`, which is
   rejected by an exact local snapshot and revalidation.
5. Replayed/swapped source phase receipts, a forged signature, JSON wire
   mutation, duplicate fields, signed nonfinite JSON precommit flags and
   attempted positive authority flags. Nested parser exceptions are normalized
   to the public Q7 witness error type without accepting malformed source.
6. Identical replay with a **cloned empty external roster/head** is accepted
   as source bytes and still returns explicit temporal custody `false`.

## The missing physical service (blocking Q7)

> A signed chronological **claim** is not proof that the commitment existed
> before training. Four signatures can all be issued after training by a
> single operator controlling four private keys. A cloned or reverted ledger
> can replay the same committed event with the same plausible source head.

To qualify actual Q7, an independently controlled host owner must provide:

1. Non-exportable role-bound keys and an identity-bound, protected source
   pin registry (not just caller-supplied `PinnedQ7Witness`).
2. A host-asserted **before-training** append time/order; a protected sealed
   bank inaccessible to the trainee, optimizer and Minotaur.
3. Atomic `compare-and-swap(expected_head, expected_index, event_id)` and a
   durable, globally one-shot spent-event register; restart/clone/rollback
   and concurrent one-winner tests on the *actual host*.
4. An independently authenticated readback of the after-head and spent-event
   state, plus the host's actual lease/recovery/rollback identity.
5. Separate human/owner release and external evaluation evidence before a
   final decision. Passing this source verifier never authorizes graduation,
   merge, deployment, model-weight changes or a Q7 physical PASS.

The module intentionally has no signing endpoint, host connection, database,
credential loading, private question bank, or graduation call. Source-only
verification cannot solve protected host authority by multiplying wrappers.
