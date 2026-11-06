# From local GGUF runtime W0 to a distinct, trainable HF W0

**Status:** read-only physical interoperability check + CLI. This is a source
candidate for the next Labyrinth Draft PR slice, not an installed host runner.

The architecture question has a straightforward but consequential answer:
**a 3 GB GGUF you can converse with and an HF model you can update are two
different physical objects.** They may be related, but neither the filenames
nor their claimed family/revision establishes that relationship. An HF model
isn't turned into the same GGUF simply by knowing the name of its architecture.

The new `labyrinth.dual_track_w0_entrance_v1` reuses three existing physical
contracts without changing them:

1. `gguf_entrance_fingerprint_v1` reads the entire GGUF base's bytes and,
   optionally, a separate GGUF LoRA's bytes; parses bounded v3 metadata and
   computes SHA-256 for both. An operator may compare against a previously
   retained manifest hash. **GGUF file SHA and GGUF manifest SHA differ.**
2. `campaign_full_state_v1` fresh-process-verifies one complete HF W0
   safetensors base, tokenizer/config, optimizer, scheduler, RNG and cursor.
   Its `state_sha256` is a digest of the *whole genome*, not a GGUF file hash.
   HF is optional. When omitted the report explicitly remains inference-only.
3. `local_full_state_probe_v1` checks the syntax of an operator-supplied
   `127.0.0.1` KoboldCPP/llama.cpp/OpenAI-compatible endpoint. The CLI never
   contacts it, changes a model, or infers that it loaded the declared GGUF.

The resulting descriptor carries the two physical fingerprints, explicitly
unproven relation and future local-only route. All of its claims about model
identity, successful load, conversion equivalence, LoRA application, actual
trainability, neural advancement and graduation remain `false`. There is no
secret check of weights or automatic inference of missing LoRA ancestry.

## Local read-only command

From Windows, using an installed Python with Labyrinth imported:

```powershell
python -m labyrinth.dual_track_w0_entrance_v1 `
  --runtime-gguf 'D:\VERA\models\gguf\model_q5_k_s.gguf' `
  --backend koboldcpp `
  --loopback-url 'http://127.0.0.1:5001/api/v1/generate'
```

If separate GGUF LoRA and complete HF full-state manifest locations are
physically discovered, the optional arguments are `--lora-gguf` and
`--hf-full-state-manifest`; use their **actual current paths**, never the
obsolete `C:\VERA` references. If an externally retained pin exists, use
`--expected-gguf-manifest-sha256` and/or `--expected-hf-state-sha256`.

For Lappy **inside WSL2**, use the actual mounted Linux path, typically
`/mnt/d/VERA/...`, not the literal PowerShell `D:\` spelling. Confirm paths
and idle workloads before hashing multi-gigabyte files. This command reads
every GGUF byte and all referenced HF state bytes; it performs no CUDA or
network operation, but its storage bandwidth and wall time are not zero.

One could alternatively use a small Python call:

```python
describe_dual_track_w0(
    runtime_gguf_path=Path('/absolute/verified/base.gguf'),
    trainable_hf_manifest_path=Path('/absolute/verified/hf-w0/manifest.json'),
)
```

## What the actual sources establish

- The freshly inspected Draft [Labyrinth PR #125](https://github.com/thebrazenbeard/labyrinth/pull/125)
  reports a source-only evolutionary saga with physical HF manifests and GGUF
  parser/loopback candidates. It explicitly has **not** loaded a private
  model or trained a production Qwen checkpoint.
- [model-training's TrainingDay WSL2 specification](https://github.com/thebrazenbeard/model-training/blob/main/docs/TRAININGDAY_LAPPY_WSL2.md)
  describes Lappy's local WSL2 NVIDIA CUDA route, separate from GitHub's
  public identity bootcamp. Its dated runtime evidence confirms only *tiny
  synthetic* optimizer steps; it does not establish 4B full-model fit.
- [model-training's pinned public base manifest](https://github.com/thebrazenbeard/model-training/blob/main/docs/artifacts/qwen35-4b-public-base-20261102.json)
  is for `rodrigomt/Qwen3.5-4B-Uncensored-Aggressive` at a specific revision.
  A different training hypothesis names official `Qwen/Qwen3.5-4B` at another
  revision. These are **not** interchangeable sources. A local GGUF name
  cannot resolve which upstream weights it derives from.
- [llama.cpp's official HF→GGUF conversion path](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py),
  [quantization documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md),
  and [LoRA converter](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py)
  establish *possible* transformation mechanisms, not that our particular
  bytes were ever converted, paired, or loaded.

## Actual next host-bound workflow

1. Read-only identify which GGUF and optional LoRA files really exist on
   `D:\VERA`, exact SHA and metadata. Never assume a stale manifest path.
2. Separately locate an HF model and verify its complete W0 manifest and
   upstream revision. If there is no such eligible checkpoint, remain GGUF
   inference-only without inventing an optimizer state.
3. If linking HF to GGUF, *record and reproduce* the conversion/merge/quantize
   tool SHA, source HF shard hashes, parameters, output GGUF hash, tokenizer,
   adapter dependencies and model-behavior controls. A mere claim of shared
   architecture is insufficient.
4. Run a genuinely local model service and match the public identical
   W0/control/W1 prompt cohort using the existing local probe API. Preserve
   exact service launch arguments and proof of what bytes were loaded;
   localhost text responses alone cannot establish this.
5. Physical TRAIN effects and hidden Minotaur examination are separate:
   eligible genuine updates follow the parent genome, public TRAIN permit,
   Ariadne write-ahead log and heldout evidence. This dual-track CLI neither
   starts nor authorizes any of them.

Tests: `python -m unittest -q tests.test_dual_track_w0_entrance_v1`
