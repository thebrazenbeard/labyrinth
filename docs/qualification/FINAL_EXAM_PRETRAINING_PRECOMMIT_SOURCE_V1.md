# Evaluator-owned final exam pre-training projection (source candidate)

Status: **SOURCE-ONLY / NO FINAL EXAM RUN / NO GRADUATION**.

The existing `sealed_semantic_exam_protocol.build_sealed_plan` contains the
approved-for-source-testing *synthetic* 400-situation proposal. Its public plan
contains selected seeds and other experimental details; therefore it must not
be released as the pre-training trainee/Minotaur handoff.

`labyrinth.final_exam_precommit` adds a **smaller, Ed25519-signed projection**:

1. The separate evaluator constructs a `SignedSealedPlan`, holding the oracle,
   private case identifiers, all prompt forms, labels, and bank key privately.
2. Before any training the evaluator verifies the HMAC-protected source plan,
   rejects any unexpected plan fields, and signs only exam/event/objective IDs,
   scope/TRAIN/source hashes, the full plan hash, and signer public key hash.
3. The trainer sees **only** this projection, not the sealed plan itself. No
   seeds, case IDs, prompts, labels, answer key or bank HMAC are exported.
4. A separately trusted coordinator checks the pinned Ed25519 verification
   key and expected exam/event/objective/scope/plan identities **and** the
   independently recorded source and training-inventory SHA-256 pins.
   `verify_pretraining_precommit` requires the keyword arguments
   `expected_source_sha256` and `expected_training_inventory_sha256` to be
   valid, exact matches; omitted or malformed expectations fail closed.
   These expectations must not be derived from the untrusted projection being
   checked. The coordinator
   must persist both pretraining order and consumed exam events *outside* the
   trainer and Minotaur. This helper reads a supplied spent-event roster but
   **cannot** enforce its durability, independence or freshness.
5. Only after candidate freeze should the separate evaluator execute and
   score a privately held examination; that physical procedure is not present
   in this module.

Source tests use a known synthetic exam bank, **not** secret final questions.
An exact schema allowlist fails closed if upstream `SignedSealedPlan` gains an
unexpected field; deliberate schema review is required to upgrade.

## Explicit hostile failure modes and qualification ceiling

> A trainer with access to the evaluator's process, private question bank,
> signing key, or underlying filesystem can read or reproduce the exam.
> Ed25519 authenticity does not prove host/process isolation.

> The signing key can sign the projection **after** training. A local clock,
> self-reported timestamp, or the `external_anchor_verified=False` field does
> not establish pre-training order. An independently controlled external
> timestamp/append-only anchor and actual readback are required.

> An operator can pass a fresh empty `consumed_event_sha256s` set, restore its
> snapshot, or clone the signer. This source helper is deliberately **not** a
> monotonic global spent-event service and cannot satisfy Q8.

> The existing sealed scorer accepts signed predictions generated outside
> its process and does not attest genuine pinned-model inference. This
> projection adds no weight, training, transfer, retention or custody evidence.

Potential release gates remain **Q7 NOT RUN**, **Q8 FAIL/HOLD**, **Q9 HOLD**.
To test locally:

```bash
python -m unittest discover -s tests -p test_final_exam_precommit.py -v
```
