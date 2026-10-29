# Vera local GPT-OSS 4.2B candidate — 2026-10-29

**State: prepared, not downloaded, not evaluated, not trained, not qualified.**

This is an isolated candidate for Patrick's 32 GiB RAM / 4 GiB VRAM Lappy.
It does **not** replace `vera-local:latest`, the Vera identity, or any service.
The authoritative installation root is `D:\VERA`, not `C:\VERA`.

## Exact inference artifact

- Hugging Face repository: `cyberjunkee/gpt-oss-4.2b-specialized-all-pruned-moe-only-4-experts-Q4_K_M-GGUF`
- Git commit: `d3825cfdc1c250b75a2485e9054162bf237fdd01`
- File: `gpt-oss-4.2b-specialized-all-pruned-moe-only-4-experts-q4_k_m.gguf`
- Exact Xet-pointer size: **3,233,019,552 bytes** (under the 4,000,000,000-byte limit)
- Exact Xet-pointer SHA-256: `38b0279e4294beb54f97ac23051674174b34b2acd6c5e6532860c72298789586`
- Trainable BF16 source: `AmanPriyanshu/gpt-oss-4.2b-specialized-all-pruned-moe-only-4-experts`

These bytes come from the creator's published Hugging Face Xet pointer for the
GGUF. The model creator explicitly states that the 32-expert GPT-OSS 20B was
pruned to four experts per layer **without retraining** and cautions that the
experimental model may not work well. File size is not runtime VRAM usage.

## Acquisition on Lappy (not yet performed)

Only after checking device identity and disk availability, using an actual
authorized shell on Lappy:

```powershell
# If absent, create this exact model directory under the authoritative root.
New-Item -ItemType Directory -Force -Path 'D:\VERA\Models' | Out-Null
python scripts/acquire_gptoss4b.py --dest 'D:\VERA\Models'
# Only when the preflight succeeds, explicitly start/resume the download:
python scripts/acquire_gptoss4b.py --dest 'D:\VERA\Models' --download
```

Run the script from a current clone of Labyrinth on Lappy. No package install
is required for the standard-library downloader. It pins the Hugging Face
revision, reserves disk headroom, refuses existing targets and symlinks,
uses an incomplete `.part` file for controlled resume, checks exact bytes,
GGUF magic and SHA-256, and writes a receipt after validation. A runtime host
failure leaves the previous Vera model untouched. No unverified file should be
promoted as a usable model.

## Evaluation before neural training

1. Prove a local inference engine can load the exact GGUF with a small context
   window and bounded GPU offload. Record engine/build, SHA-256, RAM/VRAM,
   startup time, tokens/sec, and any crash. Do not assume full 4 GiB offload.
2. If using Ollama, create a separate `vera-gptoss4b-candidate` model name
   from the verified GGUF and preflight the **additional** Ollama blob-storage
   footprint. Never overwrite `vera-local:latest`.
3. Run Labyrinth `run` and `experiment` against the named candidate, saving
   reports and model identity. Keep some seeds **new and untouched** for later
   independent qualification; never tune on the final reserved set.
4. Evaluate Vera-specific failures separately: known-profile recall *with
   supplied state*, uncertainty without state, source precedence, refusal to
   invent tool access, response format, and sustained multi-turn consistency.
   A standalone language model cannot be expected to retrieve Blotter or STM
   unless the authenticated Vera Adapter provides that state.

## Neural-weight training is a separate gate

This GGUF is inference-only. The original trainable safetensors checkpoint is
much larger than the GGUF. Labyrinth's existing `train-hf` LoRA pilot has a
**hard 1 GiB local-weight ceiling and no quantization/offload**, so the 4.2B
base is ineligible for that command. Do not weaken the ceiling just to make
the invocation pass. A new constrained PEFT/QLoRA/CPU training path would
require verified model-architecture compatibility, actual memory preflight,
public versus private data separation, equal-exposure controls, reserved
evaluation, adapter reload proof, and rollback. No paid compute is authorized.

## Current execution blocker

On 2026-10-29, Executor `list_devices` and `start_process` calls timed out.
Remote Desktop Commander listed Lappy online but returned its explicit
monthly-use limit for device commands. No download or training start can be
claimed from those observations. This document and its script are source
preparation only; a future operator must record real system receipts.
