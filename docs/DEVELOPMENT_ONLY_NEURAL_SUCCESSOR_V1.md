# Entrant W0 and regenerative neural successor — development-only v1

**Status:** tested source prototype with explicit neural-execution HOLD. Not a
model training run, physical host attestation, independent examination, or
qualification. No private graduation, Q5–Q9, or promotion contracts change.

## Two independent entrance measurements

`labyrinth.entrance_state_v1` records two explicitly separate axes:

1. **Exact immutable state identity:** streamed SHA-256 and length for every
   referenced base weight, tokenizer/config, initial trainable LoRA adapter,
   and optional optimizer/scheduler/RNG files. Large pretrained weights are
   referenced by their existing absolute local path, never copied. Missing
   trainable or optimizer-state families are named as gaps, not invented.
2. **Observed training-safe capability baseline:** bounded public development
   prompts, scorer, skill, exact prompt digest, integer score and W0 state
   digest. This observation is explicitly **operator self-reported**. The
   bytes hash authenticates *which file bytes were found*, not the ability of
   the model, the correctness of scoring, or independent inference custody.

The manifest can be published once to a new local file. A separate Python
process re-hashes every referenced model/tokenizer/adapter/auxiliary file and
the source receipt. A changed adapter can be compared to W0 using exact bytes,
but `bytes_changed=true` is not proof of beneficial training. Persistent
optimizer/scheduler/RNG state is required for a *full-state* resume claim.
Even if all these file families are present and rehashed, arbitrary bytes do
**not** prove they can be loaded or that the optimizer's progress can resume.
The verifier reports auxiliary file capture separately and leaves
`full_resume_state_available=false` until a real trainer loader and replay
test establish recovery. `w0_state_sha256` identifies model parameters;
`w0_captured_training_state_sha256` additionally commits to all captured
auxiliary bytes without treating them as executable training state.

## Minotaur source cycle 1+ preflight

`labyrinth.hf_development_only_successor_v1` creates a versioned **TRAIN-only**
candidate from a fully completed `TrainingCorridorRunner.regenerate(journey)`:

- Existing arithmetic scope and bounded HF budget, explicit `ResumeContext`.
- Identical predecessor report bytes SHA, candidate LoRA digest, and declared
  retained Journey checkpoint; locally pinned context is not external custody.
- All prior *public TRAIN* prompt witnesses, not merely case IDs or prompt
  hashes; recompute arithmetic semantics, including each contrast expression's
  three exposed atoms. Previous TRAIN identities and prompt commitments must
  match Journey's sealed history exactly.
- All new TRAIN rooms, their scored feedback bits, per-room checkpoint digests,
  full order and optimizer visit schedule, and exact Minotaur policy/admissions.
- Prior development exam case/prompt *commitments only*. Never serialize
  evaluator gold, private final cases, or feedback from private examination.
- Explicit `DEVELOPMENT_ONLY`, `NOT_ESTABLISHED`, and a source digest that is
  only a local self-consistency check. The source is rechecked against caller-
  held inputs. A mutually fabricated manifest plus inputs is **not** caught.

The v1 module intentionally refuses neural execution:
`run_local_neural_successor(...)` raises
`DevelopmentOnlyNeuralExecutionBlocked`. This is critical honesty, not a
runtime feature. Existing `hf_pilot.py` and `hf_resume_bridge.py` cannot yet
replay a later adaptive policy plus prior public diagnostic history into the
three matched neural optimizer arms without losing semantic anti-reuse checks.
Furthermore, staging currently allows a caller to put an arbitrary future
checkpoint hash into a route. That must be replaced with a real post-training
candidate weight receipt and rebind before a running neural successor exists.

## Next implementation gate — additive, not independent-review gate

1. Capture EntranceState W0 from exact official pretrained weights and actual
   initial adapter, plus TRAIN-safe measured baseline. Preserve source/rights
   provenance; pin a new local owner-held write-once receipt.
2. Version and persist complete public TRAIN policy/feedback/semantic history
   with actual **post-SGD** saved adapter digest W1. Each chamber's small
   optimizer update (a training *spark*) needs actual before/after tensors,
   step counts and optimizer/loss receipts; weight change is not skill gain.
   A staging-time synthetic
   hash is insufficient. Bind the Minotaur development verdict, its per-skill
   scores, and failed weakness profile to the same W1 state and case roster.
3. Regenerate a structurally fresh, weakness-targeted TRAIN maze, rejecting
   previous TRAIN and all development-exam prompt and arithmetic equivalences.
   Increased difficulty should be justified by observed ability/weakness and
   specificity, not merely an arbitrary cycle counter.
4. Extend the HF source report and audit with a *distinct* DEV_ONLY cycle-1+
   schema. Implement exact predecessor adapter readback and all three
   compute-matched arms from W1; use a new output root and save W2. Record
   whether optimizer/scheduler/RNG state was actually resumed or restarted.
5. Verify W2 in a fresh process, measure new local developmental heldouts and
   delayed transfer with controls; repeat through a bounded cycle budget. An
   Ariadne thread points to W0→W1→W2 receipt identities; private graduation
   continues to require its separate protected evaluator and owner custody.

Only after step 4 has real working source/tests can this be described as a
neural multi-cycle executable saga. Until then, ordinary local development
may continue using existing supported pilots and toy journeys without an
independent examiner's signature; promotion remains protected.

## Reproduction

```bash
python -m unittest -q tests.test_entrance_state_v1 tests.test_hf_development_only_successor_v1
```

The tests use tiny fake local `.safetensors` byte fixtures and a deterministic
toy arithmetic operator. They perform no downloads, heavy neural training,
network calls, host mutations or private evaluator activity. The EntranceState
test invokes the verifier in a separate Python process and confirms bytes
were rehashed after reopening the manifest.
