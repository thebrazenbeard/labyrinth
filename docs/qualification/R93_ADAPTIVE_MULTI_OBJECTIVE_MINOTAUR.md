# R93 adaptive multi-objective Minotaur TRAIN scheduling

**Status: SOURCE_ONLY_UNATTESTED.** This is a trainer-side coordinator for
approved, separate objectives. It is not a model trainer, learned policy, host
lease, independent evaluator, consent/permission service, neural identity,
protected examination, or graduation authority.

## Executable boundary

`labyrinth.adaptive_minotaur_r93.schedule_adaptive_training` accepts 2–8
`ObjectiveLane`s, each binding one `ObjectiveContract` to one currently TRAINING
`JourneyState`. It accepts only the existing executable provider routes:

- `source_priority`: R92 audit-validity TRAIN provider and its exact rubric;
- `program_reasoning`: pre-approved semantic register-program provider;
- `arithmetic`: approved legacy arithmetic TRAIN provider, with its own R92
  toy optimizer available separately.

It revalidates every contract and snapshots the journey before reading its
failed-development state. A lane whose previous Minotaur decision was FAIL is
ranked before a fresh lane, but only when its source record contains an exam
event, linked sealed case and prompt commitments, and failure status. All
objectives still receive **every required TRAIN room** through a deterministic
round-robin order. Identity and exact prompt collisions are forbidden across
both current batches and all lanes' historical TRAIN/exam commitments. The
returned `PreparedTrainingRoom` has trainer-only gold; `trainee_sequence()`
provides prompt-only inference views.

At admission, each scheduled room now retains a source-only SHA-256 digest of
the complete trainer-side generated material, including the prompt, gold,
case identity, tier and metadata, **plus immutable primitive trainee-view
fields**. Both `ScheduledTrainRoom.trainee_view` and
`AdaptiveTrainSchedule.trainee_sequence()` fail closed if the nested
`PreparedTrainingRoom.challenge` is mutated after scheduling (even though its
dataclass is frozen). A reproduced attack replaced an admitted program prompt
with unrelated TRAIN text and previously caused the scheduler to emit the
altered view silently. The regression also checks gold/metadata/case-ID
substitution. Returned learner prompts are built from the admitted primitive
snapshot, avoiding a second read of mutable challenge text after its digest
check. The digest only detects in-process inconsistency with the
source-owned admitted record; it is not a signature, independent admission
register, or trusted host custody.

Independent hostile review exposed a second mutation family: changing the
admitted learner-view tuple could forge prompts without touching the trainer
digest, and changing the room's objective ID could misroute a learner prompt.
The scheduled room is now a tuple-backed record, not a frozen dataclass;
`object.__setattr__` cannot rewrite its route fields or admitted view tuple.
Learner-view access additionally checks admitted primitive fields against
the current prepared source view. These are bounded Python-level integrity
checks, not protection against an attacker fabricating an entire new record,
controlling both source and records, or bypassing the caller's routing API.

The program/audit pair demonstrates actual multi-objective routing. A measured
public development failure on the audit lane changes the next routing order,
cycle, room identities and audit remediation instructions, while preserving
the separate program objective. The mathematical arithmetic objective's
independent R92 loop performs real four-parameter SGD, a matched zero-update
sham, heldout/transfer development scoring and checkpoint reload. **There is
no shared multi-skill neural optimizer in this R93 module.** It would be false
to infer that success on arithmetic makes the audit/program model improve.

## Adversarial and causal checks

`tests/test_adaptive_minotaur_r93.py` exercises:

1. Two approved provider routes in one program, with distinct TRAIN identities,
   no prompt/answer leakage through the trainee-view API, and full room counts.
2. An actual public-scored Minotaur FAIL inducing a new audit cycle and a
   different objective scheduling priority.
3. A fabricated prior failure with its exam seals removed, refused.
4. Wrong scope, unsupported rubric, or cloned objective identity, refused.
5. The public rank-1-or-rank-2 shortcut below the 75% audit TRAIN target and
   exact single-line response-format rejection.
6. Physical *toy* parameter change vs zero-update sham and reload through the
   independently approved arithmetic objective loop.
7. Post-admission nested prompt, gold, metadata and case-ID mutation is
   rejected before creating a learner-facing prompt sequence.
8. Hostile in-place replacement of admitted learner-view fields or the
   objective-route identity is rejected by the tuple-backed record.

Every finding is local/source or toy CPU evidence. Validation of a
self-consistent journey cannot establish that an independent evaluator really
ran it. An attacker who can fabricate the entire journey plus sealed history
may still impersonate failure. This requires a separately owned, signed,
nonrollback examiner/host register, and source data cannot substitute for it.

## Explicitly unavailable

- Generic natural-language objective synthesis or empathy mastery.
- Pretrained HF/Qwen/GGUF/LoRA parameter updates and broad transfer.
- Shared model identity, consent, permissions, inference delegation, or
  non-plastic memory/identity/config writes.
- Independently controlled Q5/Q6 benchmarks, Q7 private final, Q8 host CAS,
  Q9 graduation, or production deployment.

Plastic model-component writes remain a separate authorized trainer operation;
this scheduler never authorizes a write. `SOURCE_ONLY_UNATTESTED` is not a
permission class or an attestation from a privileged operator.
