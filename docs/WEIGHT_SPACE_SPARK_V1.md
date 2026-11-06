# Sparks of Weight-Space Programming in the Labyrinth

**Status:** Implemented source-only physical prototype, **not** genuine Vera
neural advancement. The operator's ordinary development workflow can exercise
this directly; a private Minotaur reviewer is not needed to create, inspect,
or compare toy candidate artifacts.

## Why this is a separate evolutionary weapon

Patrick's hypothesis is that a reasoning model can *reason about the function
of weights* and deliberately author smaller edits, treating the architecture
as a canvas rather than assuming every improvement requires many gradient
steps. That is not the same as swapping a prompt, changing a system persona,
or claiming a GGUF has been created from thin air.

Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab's
[**Sparks of Weight-Space Programming** (November 2, 2026)](https://rikiyat.github.io/blog/weight-space-programming/)
reports experiments in which an LLM analyzed and directly modified network
weights. The authors' primary
[released logs](https://github.com/RikiyaT/lm_understand-logs)
include Experiment 8's Qwen3.5-0.8B trials. The 50-pass run chose a targeted
projection-subtraction edit to reduce looping; the authors report MATH-500
sample accuracy rising from 7.9% to 48.1% in that **particular** protocol.
Other experiments and baselines are also in their logs.

Important distinctions from that paper:

- The published 0.8B result is **not** evidence that a proposed edit will
  improve Qwen3.5-4B, Vera's conversational qualities, or any GGUF.
- The experiment permitted practice-set forward passes but not gradient
  updates. A locally installed runtime/weight access path is still required
  to inspect actual tensors and evaluate edited candidates.
- The reported improvements are benchmark-specific. Persona, consistency,
  truthfulness, general reasoning, and regression need their own measurements.
- No code was imported or copied from the authors; this module is an original
  bounded mechanism inspired by their described rank-one directional edits.

## Concrete implementation

`labyrinth.weight_space_spark_v1` accepts an untrusted model-generated
`WeightSpaceEditProposal` (a rationale, exact parent state hash, SHA-256
of public TRAIN source, exact shard/tensor, chosen columns, finite unit
direction, and projection gain). The model cannot supply the
`WeightSpaceEditPolicy`: operator code separately selects the physical
shards and tensors that are allowed, how many elements may change, a maximum
relative Frobenius change, and a strict size ceiling.

For a rank-two **F32** matrix `W` with rows corresponding to the components
of unit vector `u`, the module edits selected columns `C`:

\[
W'[:,j] = W[:,j] - \alpha u\,(u^{\mathsf T}W[:,j]),\quad j \in C.
\]

It preserves all other entries and checks physical F32 rounding. This
is one *mechanism*, not the only conceivable weight-space method.

Sequence:

1. Fresh-process readback of **complete** W0/Wt full-state manifest.
2. Validate selected TRAIN-only proposal data, source-owned policy,
   normalized direction, column budget and finite numeric bounds.
3. Read only the exact small whitelisted safetensors shard and validate
   the full safetensors header already covered by the campaign verifier.
4. Compute the edit in memory, reject no-op, NaN/overflow, excessive
   fraction, or relative magnitude **before** touching candidate storage.
5. Verify the complete parent again. Atomically claim a **fresh**
   candidate directory with exclusive `mkdir` and copy the entire genome.
   Existing/user directories are never overwritten.
6. Change only the candidate's chosen tensor bytes. Capture a new,
   write-once **full** child manifest including all base, tokenizer/config,
   adapter (if any), optimizer, scheduler, RNG, and cursor files.
7. Reverify the child and the untouched parent in fresh processes.
   Return a source-only receipt with proposal hash and physical ancestry.

If a write fails midway, its newly created candidate directory is retained
for incident review, **not** silently deleted or treated as eligible.

Both HF `full_weight_edit` and `lora_adapter` families may be proposed,
but **only a small F32 two-dimensional safetensors tensor** is implemented
for this version. No quantized GGUF edit, BF16/FP16 math, sharded multi-GB
loader, CUDA, inference, optimizer restore, or model-guided direction
inference is implemented. Optimizer/scheduler/RNG/cursor are *copied*, not
advanced by SGD, and this does not prove they can safely resume after a
direct non-gradient mutation.

## Tie it into the epic

The prospective Minotaur round should include **multiple proposal families**
from the *same Wt parent*:

1. unchanged parent control;
2. ordinary authorized LoRA/SGD candidate (if host training is genuinely
   available);
3. one or more direct weight-space candidates from different model hypotheses;
4. hybrid candidates only after preserving exact provenance and testing.

Use `evolutionary_population_v1.prepare_population` to compare physical
candidate manifests, then supply held-out identity, natural dialogue,
memory/correction, transfer, and regression evidence **separately**.
The journal may record the selected ancestor; it cannot pronounce model
improvement from a changed SHA. The next harder Labyrinth inherits only the
selected physical genotype and permitted public experience.

The checked-in
`tests/test_weight_space_spark_v1.py` actually writes tiny synthetic
F32 safetensors, creates two distinct direct-edit siblings, rejects forbidden
edits, validates full-state lineage and unchanged W0, and demonstrates their
compatibility with source-only population preparation. It never claims
the model was loaded or that a language model got better.

## Follow-on qualification

Before real Qwen or Vera weights:

- Qualify the actual model/adapter lineage and exact tensor architecture;
  design a streamed BF16/FP16 edit path with tensor index, shape and maximum
  compute budget, plus rollback to W0.
- Use genuine, source-authorized local model inference to discover behavioral
  failure features and propose a **testable hypothesis**; an LLM's verbal
  explanation is not proof the suspected direction carries that feature.
- Pre-register unrelated blind Minotaur suites, resistant to practice
  contamination and evaluator self-grading; test identity, persona, memory,
  naturalness and harms, not just math.
- Keep D:\\VERA and user-host identity/permissions source-first. Do not
  launch TrainingDay through the `SOURCE_ONLY_HARD_HOLD` adapter. No cloud
  compute costs are authorized.

The important shift is **weight-space editing becomes a competing,
experimentally falsifiable mutation operator**, not an imaginary shortcut
where an uninspected model can write arbitrary billions of values and be
declared the winner.
