# Q7 public pre-training phase-order source interface

Status: **SOURCE-ONLY / UNATTESTED / Q7 NOT RUN / Q8 HOLD**.

`labyrinth.q7_temporal_custody_source` composes the existing bounded
`final_exam_precommit` verifier with two separately signed, public-only
events: an examiner-side `anchor`, followed by a trainer-owner
`training_start`. Both bind the same prior signed precommit. The second
phase binds the hash of the first signed event. The verifier accepts only
exact canonical schemas, exact Ed25519 signature bytes, three distinct
public key pins, two independently expected receipt hashes, and external
source/training-inventory/sealed-plan identity pins.

This gives a future separately operated service an auditable *source-level*
wire contract for a training handoff. No private case identifiers, prompts,
labels, seeds, or gold answers appear in either public phase. The return
value's `temporal_custody_verified` and `graduation_authorized` remain
`False` even for a cryptographically valid chain.

## Acceptance and known falsifiers

Source tests reject changed event IDs; inconsistent precommit identities;
wrong external digest pins; reused signing identities; unauthorized
graduation flags; reordered, duplicated, or tampered phase records; forged
predecessors; and duplicate signed JSON fields.

> **Hostile: real chronology is not proven.** One actor can mint both
> signatures *after training*, and the verifier will accept those bytes if
> given matching externally expected pins. Numeric sequence `1 -> 2` is
> only a source convention. Signatures and code alone cannot authenticate
> clock order, key custody, independent roles, or an append-only ledger.

> **Hostile: external expectations are not supplied by this library.** If
> Labyrinth/Minotaur controls the expected hashes or signing keys, their
> values cannot establish independent evidence. A legitimate host service
> must protect those pins outside the trainee's write authority, enforce
> atomic one-shot event use, and attest elapsed order with independent audit.

> **Hostile: no final-exam execution.** The source check does not run a
> pretrained model, control final questions, score a sealed exam, validate
> transfer/retention, or authorize graduation. Q7 stays NOT RUN.

To execute the source tests:

```sh
python -m unittest discover -s tests -p test_q7_temporal_custody_source.py -v
```
