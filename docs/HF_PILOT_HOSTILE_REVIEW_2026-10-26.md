# Optional HF LoRA pilot: source review, fixes, and unpassed runtime gate

**Date:** 2026-10-26. **Scope:** source-level review of `labyrinth/hf_pilot.py` plus dependency-free contract tests. The requested independent reviewer was a native `lora_pilot_design` agent (requested model not separately runtime-attested here). Review-subject SHA-256: `hf_pilot.py` = `fa1eaa7a71861575e61e6d4125633b5c7519e2fc26378ae9fa86540741f6fd7e`, `cli.py` = `6b96b7db62baa5f290617454767a05d646392ae3a241187a74c1a5fadd377775`. The review also covered an earlier version of the test file; file revisions changed as tests were added. No neural training was executed.

## Material hostile challenges

> **GPU-size trap:** A 1 GiB safetensors file is not a 1 GiB GPU-allocation bound. Loading to FP32 expands smaller-dtype tensors, and forward/backward activations cost more. Remediation: `run_pilot` now checks free CUDA memory against a conservative weight-expansion-plus-overhead estimate before loading. This preflight is **not** a successful GPU memory qualification. CPU-first tiny-model execution is preferred on the target laptop.

> **Invalid placebo:** The source's original `sham` performed zero optimizer steps **and** zero training-case visits. It is merely an untrained-adapter control. The report now explicitly records `sham_is_compute_matched=false`, and this limitation remains **open** for causal learning claims. An equal-visit/equal-compute, zero-effective-update arm is still required.

> **Stochastic control asymmetry:** Restoring trainable LoRA tensors alone does not reset dropout or RNG-dependent optimization. The training loop now reseeds Python/PyTorch for candidate and corrupted-label runs. This improves control alignment; source review alone does not establish identical floating-point trajectories.

> **Changed bytes are not valid parameters:** Hash inequality can occur for a corrupt update; the loop now rejects non-finite trainable tensors after each optimizer step. Actual optimizer correctness is still untested in the scratch environment.

> **Fake structural sample size:** Tier 4 currently selects from exactly four train templates and four distinct eval templates. Many seeds vary numbers, not structure. Report results grouped by structural fingerprint and introduce independent blind grammars before transfer claims.

## Exact evidence levels

- **Source:** local safetensors-only path; HF config/tokenizer file hash commitments; masked prefix tokens; train-only case generation; opt-in LoRA code path; model tensor digests; separate reload CLI; non-overwrite outputs; bounded local model size. Implementation present, not executed with a real HF checkpoint.
- **Test:** `python -m unittest discover -s tests -q` passed **70/70** tests after integration; tests use stdlib/mocks and establish public interface/guards, not actual LoRA optimization.
- **Unverified runtime:** model load/backprop on target laptop; GPU/CPU memory use and throughput; PEFT compatibility; adapter save/reload with a real model; baseline-versus-candidate inference after new-process reload. Do not mark these PASS.
- **Unverified semantics:** improved unseen reasoning, matched-compute sham, multi-seed uncertainty, independent blind task families, negative transfer controls.
- **Privacy:** local reports can contain absolute checkpoint paths. Keep them private; sanitize such paths before any public upload.

**Next falsification:** an explicitly selected small *already-local* HF safetensors checkpoint or a deliberately generated tiny local model, isolated Python environment, two bounded CPU updates, independently observed tensor changes, then a separate process reload with identical held-out predictions. Only after that mechanism test should pretrained-model gain claims be attempted.
