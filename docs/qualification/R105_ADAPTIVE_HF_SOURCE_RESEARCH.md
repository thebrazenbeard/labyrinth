# R105 adaptive Minotaur-to-HF source research

**Status: SOURCE_ONLY_UNATTESTED — merge, promotion and deployment HOLD.**
This successor demonstrates an opt-in source path from feedback-conditioned
Minotaur TRAIN rooms into the existing small-checkpoint HF LoRA pilot. It is
not an independently witnessed physical training campaign, a private exam,
or Q5–Q9 graduation evidence.

## Opt-in experimental boundary

`run_pilot` accepts a trainer-private `MinotaurHfTrainRoster` only with its
explicit unauthenticated adaptive research opt-in, an exact arithmetic
training scope and cycle zero. Predecessor/resume inputs are not accepted in
this mode. Ordinary legacy v2/v3 pilots retain their existing source and
report behavior. The HF pilot still requires a local safetensors checkpoint
under its **1 GiB, nonquantized** limit; this route does **not** train the
intended Qwen3.5-4B GGUF used with KoboldCPP.

An adaptive roster now supplies detached public route provenance: the
initial checkpoint commitment and ordered TRAIN feedback rows (room/case
identity, prompt digest, binary feedback score and before/after checkpoint
commitments). The manifest never publishes raw supervision answers or bare
per-answer hashes. Its source verifier replays Minotaur's dynamic corridor
and policy decisions, then independently regenerates each admitted TRAIN
challenge and its trainer-private gold. Practice arithmetic, workshop
transfer and contrast prompts are all handled explicitly. A deterministic
public heldout diagnostic is constructed without known arithmetic-semantic
overlap with the admitted TRAIN source.

The distinct `labyrinth.hf-lora-adaptive-source.v1` report schema requires
the source manifest and full ordered `(case_id, SHA256(prompt))` optimizer
visit receipts in **all three** sham/candidate/corrupted-label arms. The
offline auditor rejects missing, altered, reordered, or incompatible
adaptive-source claims. A separate-process adapter reload path must verify
that report before loading model weights. These are internal consistency
checks; the manifest is still self-reported source data, not an external
signature or direct observation of token IDs or gradients.

In particular, the final self-reported TRAIN feedback score cannot be
independently confirmed from the manifest alone: no later policy transition
constrains it, and the learner's response remains outside this public
record. A host-owned, append-only feedback/exposure witness is still needed.
Legacy v3 audits are intentionally backward compatible and may omit the
new visit fields entirely; this separate adaptive schema does not silently
upgrade those historical reports to strict exposure custody.

## Q6 local mutation integrity

The Q6 toy study now retains complete primitive snapshots of all four
admitted cohorts, including private scoring answers and metadata. It
rechecks them around optimizer, scorer, checkpoint and publication
callbacks. Changing heldout gold, transfer prompts, partition identities or
unvisited TRAIN examples makes the local study abort instead of scoring or
reporting against altered material. This only addresses source consistency
inside the same Python process; it cannot establish complete historical
exposure custody or an independent evaluator.

## Physical and ecosystem hold

No release claim follows from source-level tests. Qwen3.5 Vera needs its
own authenticated host/GPU lease, a live approved Executor/NAS route, a
compatible pretrained loader and complete quantized optimizer-step fit,
independently admitted training/evaluation data, matched controls, saved
adapter and restart evidence. A future adapter must be bound to the exact
base/tokenizer and proven through GGUF conversion and KoboldCPP runtime
identity before Rezon or VeraOS cutover. The R100 Minotaur arithmetic roster
must **not** be substituted for TrainingDay R18's separately admitted
integer-operation-chain corpus.

Source Git and CI success do not grant those host/evaluator rights. Keep
all source PRs Draft and do not merge or deploy until each separate physical,
custody and owner-authority gate is demonstrated at the relevant exact head.
