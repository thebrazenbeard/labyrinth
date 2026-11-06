# Five-arm regenerative MiniLM source contract (experimental, v1)

**Status: SOURCE ONLY / NOT PHYSICAL / NO GRADUATION AUTHORITY.**

This opt-in module precommits the ordered TRAIN batches and corresponding
correct, zero-update, inverted, zero-learning-rate and class-preserving
randomized-label arms to one declared two-cycle Maze. Its selector and
label-transformations are imported from the actual MiniLM runner, rather than
reimplemented in the audit. Crucially, the verifier **reconstructs the Maze
and every arm's labels from the pinned scope parameters and actual runner
functions**, so a malicious caller cannot substitute all-zero "correct"
and "randomized" assignments merely by recalculating its own manifest digest.
It currently covers the **unmeasured** selector;
measured Minotaur tactic-conditioned schedules need a different separately
signed precommitment. No call here imports a neural backend or starts training.

The reference starting source for this isolated candidate was Labyrinth Draft
PR #62 at GitHub head `e417631f8afbbdc4de0130cce9a13ae965df9bc7`
(tree `008a65cb43a00c106772ca4eb82c4bca4827d43b`), both source CI
workflows passing. This is a versioned reference, **not a live-head oracle**.

## Claimed evidence, precisely bounded

- The plan binds one base-model SHA-256, initial trainable-state digest **and
  initial optimizer-state digest shared across all five arms**,
  declared scope, source SHA/tree, two-cycle ordered TRAIN exposure IDs,
  exact five-arm labels, a learning-rate/time budget and opaque salted private
  transfer/final roster commitments. Repeating the source plan is deterministic.
- Each **caller-reported** arm must start from the same trainable digest,
  visit the identical ordered training cases, and use the declared optimizer
  semantics. `zero_update` must not call `step`; `zero_lr` **must** call
  `step` but have a zero effective learning rate. Randomized-label assignments
  must match the source runner's exact class-preserving permutation.
- Each arm reports its cycle-zero trainable and optimizer-state digests and
  the cycle-one start digests. Both must remain identical across regeneration:
  **the Maze resets; the learner's tensors and optimizer moments do not**.
  These are still caller-asserted hashes, not independently read model bytes.
  A stepped zero-learning-rate control must show optimizer-moment evolution
  while its trainable tensor digest remains unchanged; a zero-update control
  must show neither tensor nor optimizer-state evolution.
- A separately supplied evaluator roster is vetoed for exact TRAIN prompt
  overlap, approved generator sentiment adjectives (even if those words were
  not sampled in the specific seed), and declared training template families.
  Private raw prompts are not embedded in the public plan.
- For retention, the plan freezes case IDs and the maximum allowed count
  of **previously correct → incorrect** deteriorations. New successes cannot
  offset lost mastered cases. A fixed **75% prior baseline accuracy floor**
  prevents tiny previously-mastered subsets from hiding behind a 5%-of-roster
  absolute deterioration cap. Duplicate private evaluation commitments across
  nominally independent seeds are rejected.

The source gate returns `SOURCE_ONLY_UNATTESTED`,
`SOURCE_OVERLAP_VETO_ONLY`, or
`SOURCE_RETENTION_ACCOUNTING_ONLY` with every physical/evaluator-custody and
graduation authorization flag **false**. In particular, a well-formed digest
is not a readback of a model, optimizer, or bank.

## Demonstrable falsifiers

Run:

```bash
python -m unittest tests.test_five_arm_experiment_contract -v
python -m unittest discover -s tests -q
```

Tests reject mismatched arm visits, a randomized-label arm silently using
correct/inverted labels, an unstepped zero-LR optimizer, a stepped zero-update
optimizer, a neural/optimizer reset between Maze cycles, forged initial
digests, a changed source plan **even after rehashing an all-zero five-arm
label map or reordered visits**, repeated evaluators
across seeds, and a retention case where all formerly mastered examples fail
but aggregate accuracy stays unchanged.

## Gates still required outside this module

1. **Owner and lease:** TRAININGDAY R19-01 must explicitly grant a bounded
   exclusive host workload and the Executor/NAS route must be read back.
2. **Physical provenance:** A separate process/host must hash real pretrained
   bytes, initial and final adapter/head tensors, all optimizer states and
   per-step visits; this module currently trusts caller-supplied values.
3. **Causal qualification:** Execute five matched arms from the same initial
   model and comparable budgets with independently controlled held-out forms.
   Preregister a positive correct-label advantage and uncertainty bounds.
4. **Transfer and retention:** Independent family-level semantic judging,
   no TRAIN-derived prompts or familiar public fixtures, paired prior-skill
   observations, and a separately controlled retention policy. Source lexical
   checks cannot establish semantic novelty.
5. **Protected final and authority:** A bank and examiner unavailable to the
   trainer, one-shot host-owned monotonic state, separate keys/principals and
   rollback resistance. The Minotaur cannot approve its own graduation.

This is not a substitute for a physical experiment or a claim that Rezon
distributed whole-model reviews imply shared-model tensor/gradient inference.
