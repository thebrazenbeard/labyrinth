# Labyrinth campaign full genome state v1 (development-only)

`labyrinth.campaign_full_state_v1` captures an experiment's **physical file
identity**, not an inference claim, training receipt, learning result or private
graduation verdict. The intended use is to keep candidate ancestry honest when
an evolutionary campaign forks two or more children and later selects a parent
for the next generation.

## What is pinned

The `capture_campaign_state(...)` boundary takes **absolute, canonical local
paths** to every constituent and refuses missing parts:

- A Hugging Face safetensors base with configuration, tokenizer files, and
  every weight shard. Multiple shards require an index whose `weight_map`
  references exactly the actual tensor names and files. Safetensors headers,
  tensor offsets and byte lengths are checked; arbitrary bytes with a matching
  filename are refused. GGUF hybrids are unsupported. Valid headers do *not*
  establish a loadable model or authentic pretrained provenance.
- Either an actual LoRA adapter with `adapter_model.safetensors` and
  `adapter_config.json` for `family="hf_safetensors_lora"` and
  `mutation_kind="lora_adapter"`, or no adapter for
  `family="hf_safetensors_full"` and `mutation_kind="full_weight_edit"`.
- Nonempty optimizer, scheduler and RNG directories, with SHA-256 and byte size
  recorded for every regular file. A local data-cursor file is mandatory.
- Path-independent full-content `state_sha256`, a path-binding manifest digest,
  exact `origin_state_sha256`, `parent_state_sha256` and `generation_index`.

The earlier entrance inventory's 512 MiB *pilot* ceiling applies to small
auxiliary evidence. This campaign inventory reuses entrance's streaming file
hasher and canonical-path checks but allows up to 64 GiB optimizer, 8 GiB
scheduler and 8 GiB RNG artifacts. This is a byte capacity, not a promise that
the chosen trainer supports resuming these files.

## Concrete operations

```python
from pathlib import Path
from labyrinth.campaign_full_state_v1 import (
    capture_campaign_state, save_campaign_state_once,
    verify_campaign_state, verify_generation_ancestry,
)

w0 = capture_campaign_state(
    base_dir=Path("/abs/qwen-base"),
    adapter_dir=Path("/abs/adapter-w0"),
    optimizer_dir=Path("/abs/optimizer-w0"),
    scheduler_dir=Path("/abs/scheduler-w0"),
    rng_dir=Path("/abs/rng-w0"),
    data_cursor_path=Path("/abs/cursor-w0.json"),
    family="hf_safetensors_lora", mutation_kind="lora_adapter",
)
save_campaign_state_once(w0, Path("/abs/receipts/w0.json"))
readback = verify_campaign_state(Path("/abs/receipts/w0.json"), fresh_process=True)
```

The verified worker starts in a **new Python interpreter and PID** and reads
every real source file. The caller independently rehashes before and after the
worker, pins its source SHA and rejects a mismatched report. This is stronger
than comparing unverified declared hash strings, but the worker and parent
are still local source-controlled processes, not independent witnesses.

For children, pass `parent_manifest=w0` to `capture_campaign_state` *after*
the relevant independent training work. Save each manifest once. Then call:

```python
verify_generation_ancestry(
    Path("/abs/receipts/w0.json"),
    (Path("/abs/receipts/left.json"), Path("/abs/receipts/right.json")),
    Path("/abs/receipts/right.json"),
    Path("/abs/receipts/generation2-entry.json"),
)
```

This rechecks parent, both siblings, selected champion and next entry in fresh
processes. It refuses mismatched origins, changed immutable LoRA bases,
unchanged child adapters, duplicate child states, or a next entry whose full
content differs from the selected champion (including its optimizer, scheduler,
RNG and data cursor). For full-weight edits, tokenizer/config remain fixed and
actual safetensors bytes must differ.

## Evidence and authority boundaries

Every success remains `SOURCE_ONLY_LOCAL_FILE_CONTENT_INTEGRITY`, and the
lineage result is `SOURCE_VERIFIED_LINEAGE_NOT_CAUSAL_TRAINING`. All records
explicitly set `neural_update_proven=False`,
`restorable_training_state_proven=False`, and `graduation_authorized=False`.

**What is still missing before a real claim:**

1. An authenticated trainer must bind each child's actual *pre-update* bytes to
   the parent, prove the optimizer step and write-once successor, and record
   the complete update and rights/host custody. A claimed parent digest by
   itself cannot prove the child actually started there.
2. An actual model-and-trainer **process-death restore**, with correct
   optimizer/scheduler/RNG/cursor behavior, must be tested separately.
3. A real independently controlled examiner must measure fresh behavioral
   changes with appropriate controls. Development case scores do not certify
   independent fitness or generalization; a selected child cannot grade itself.
4. Host-owned monotonic/immutable witness authority and the formal signed
   graduation gate remain separate. Local hash chains are not anti-rollback.

No paths in this module cause training, host dispatch, model download, network
use, private data ingestion, promotion or deployment.
