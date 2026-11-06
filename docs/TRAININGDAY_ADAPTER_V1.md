# Labyrinth → TrainingDay source-bound handoff v1

**Evidence:** source inspection plus deterministic local mock-artifact tests. No
Qwen weight update, physical GPU execution, verified training-state restart or
private examination took place while building this adapter.

## Exact current source contract

- `thebrazenbeard/model-training` main at
  `2b83430dd36076c5f65624709968e372b170ea36`.
- Entrypoint: `trainingday.lowmem_experiment.one_step_kernel(base_binding_path)`.
- Official base: `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
  text loader class `Qwen3_5ForCausalLM`.
- Its very first operation is `require_physical_launch_authority`; the upstream
  source plan says `SOURCE_ONLY_HARD_HOLD` and `authorizes_training=False`.
  This adapter does not bypass the authority boundary or launch hardware.

`labyrinth.trainingday_adapter_v1.inspect_trainingday_handoff()` inspects an
*explicitly mock-only* local bundle with a source/version pin, complete
base/tokenizer/adapter/optimizer/scheduler/RNG/TRAIN-data roster, owner-supplied
expected SHA-256 for **every file**, and write/readback receipts. It reads
each artifact in bounded 1 MiB chunks from a no-follow file descriptor; large
optimizer bytes are streamed, never buffered into memory. It rejects missing,
symlinked, out-of-tree, unexpected or changed files and any attempt to assert
physical training, resume or graduation from this source-only inspection.

The expected digests are supplied independently by a trusted caller. Copying
the manifest's own hashes back as `trusted_sha256` would only show internal
self-consistency. A mock receipt (including a reported `fresh_process` reload)
is not a genuine new-process loader probe or authenticated worker witness.

## Required before a genuine Labyrinth W0 → Wt neural successor

1. Use `campaign_full_state_v1` to capture and fresh-process rehash full
   sharded HF weights, adapter, optimizer, scheduler, RNG and data cursor.
   This adapter's mock artifact roster alone does not parse all tensors.
2. Obtain separately authorized host launch rights and an actual low-memory
   TRAIN-only trainer that writes a complete, physically verified Wt. Do not
   call protected `one_step_kernel` through alternate routes.
3. Record pre-effect append-only Ariadne intent and post-effect trainer proof;
   ambiguous crashes must not automatically replay optimizer steps.
4. Load W0 and Wt in independently observed fresh processes and compare
   causal behavioral outcomes, matched unchanged/corrupted controls and
   predeclared unseen transfer. Minotaur development and formal private
   graduation remain distinct.

Until then the only accepted classification is
`SOURCE_ONLY_LOCAL_MOCK_FILE_INTEGRITY` with all neural/resume/graduation booleans
false. Direct `invoke_trainingday_trainer` raises
`TrainingDayPhysicalExecutionHold` even when callers supply flags or handles.

## Reproduction

```bash
python -m unittest -q tests.test_trainingday_adapter_v1
```
