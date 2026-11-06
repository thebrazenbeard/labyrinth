# R57 offline separate-process MiniLM artifact byte preflight

Status: `SOURCE_ONLY_UNATTESTED`. This is an additive source-level test tool,
not an independently conducted final examination, a model evaluation, or a
training/graduation authorization.

`scripts.verify_minilm_fresh_process_r57` is a separately executable Python CLI
using only the standard library. It does **not** download, import, initialize,
or run a pretrained model. Its bounded purpose is to read six locally staged
artifact files and compare the raw bytes with an externally supplied canonical
manifest and independently obtained exact source/head/tree/run pins.

## Staging contract

Place these six roles under one purpose-specific read-only bundle directory:

| Role | Required relative path |
| --- | --- |
| Frozen-base weights | `base/model.safetensors` |
| Tokenizer | `tokenizer/tokenizer.json` |
| LoRA adapter weights | `adapter/adapter_model.safetensors` |
| Classification head | `head.safetensors` |
| Full optimizer state | `optimizer-state/state.safetensors` |
| RNG state | `optimizer-state/rng.json` |

This layout is an **evaluator staging contract**. The current physical MiniLM
training runner does not automatically export all six roles in this layout,
and the existence of a staged file does not prove it came from training.
The source and training owners must agree on an independently curated staging
step before this check can be used as part of a larger physical qualification.

`manifest.json` is a canonical, newline-terminated, sorted-key, compact JSON
object with exactly `schema`, `source_head`, `source_tree`, `run_id`,
`trainer_pid`, and `files`. Its `schema` must be
`labyrinth.minilm-fresh-process-byte-readback.v1`. Each of the six `files`
objects contains exactly the fixed `path`, lowercase 64-hex `sha256`, and
positive integer `size_bytes`. The `source_head` and `source_tree` are exact
40-character lowercase Git SHAs, and `run_id` has a bounded character set.
The manifest must be no larger than 64 KiB and reject duplicate JSON keys.

```text
python -m scripts.verify_minilm_fresh_process_r57 \
  --bundle-root ./staged-minilm-eval \
  --manifest-file ./staged-minilm-eval/manifest.json \
  --expected-manifest-sha256 <independently-obtained-sha256> \
  --expected-source-head <independently-obtained-git-commit> \
  --expected-source-tree <independently-obtained-git-tree> \
  --expected-run-id <independently-obtained-run-id>
```

The CLI exits nonzero on missing/mismatched pins, bad framing, symlinks,
unexpected or privileged fields, missing roles, or changed file bytes. It
rejects symlinks in **every lexical ancestor**, not just the final bundle,
manifest, or artifact name; an alias above `bundle/` is not accepted. Paths
containing `..` are also refused. A platform whose customary temporary-path
prefix itself passes through a symlink must use a physically located,
non-symlink staging directory and pass that lexical path directly.

This is a static-file preflight, **not** atomic directory-descriptor custody:
an adversary concurrently replacing parent directories could race the
component checks and subsequent file opens. The file-descriptor identity
checks detect ordinary last-file swaps but do not close all ancestor races.
Only read-only, owner-controlled staging is suitable, and no host authority
can be derived from these checks.

The CLI reports its own process ID and whether it differs from the **declared**
trainer PID. That comparison is diagnostic; it does not authenticate who
actually trained the model. Running this verifier as a separate Python
subprocess is tested directly without loading a model. The invocation above
is from a source checkout; the current wheel packaging does not distribute
development `scripts` as an installed CLI API.

## Trust boundary and remaining Q5/Q6 requirements

- All expected pins are caller-supplied. A caller can fabricate matching pins
  and fake staged files. No external host signer or monotonic witness is used.
- Re-reading file bytes is stronger than trusting an in-memory score but does
  **not** execute the checkpoint, verify semantic transfer, or reproduce its
  predictions. The current runner still reloads a new object within its own
  Python process; a separately controlled *model inference evaluator* remains
  to be built and physically qualified.
- The six roles do not attest original frozen-base provenance or full
  optimizer/RNG restoration. The caller may mutate staged files before both
  manifest and expected pins are established; concurrency and bundle custody
  are not protected by this source preflight.
- `model_loaded`, `optimizer_executed`,
  `independent_evaluator_custody_verified`, `private_evaluation_verified`,
  `can_authorize_training`, `can_authorize_graduation`, and
  `can_authorize_deployment` are explicitly **false**, even when every byte
  matches. Q5–Q9 and host/GPU lease ownership remain independent HOLD gates.
