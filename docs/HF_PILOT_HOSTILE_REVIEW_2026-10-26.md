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
- **Runtime at review time:** model load/backprop, PEFT compatibility and adapter persistence were not yet executed when the source-only hostile review was issued. A subsequent **separately bound** CPU-only runtime smoke (E-016 in the research log) did establish these mechanisms on a randomly initialized tiny model, including numerical tensor equality after fresh-process reload. GPU memory feasibility and representative CPU/GPU throughput remain unqualified.
- **Unverified semantics:** improved unseen reasoning, matched-compute sham, multi-seed uncertainty, independent blind task families, negative transfer controls.
- **Privacy:** local reports can contain absolute checkpoint paths. Keep them private; sanitize such paths before any public upload.

**Next falsification:** an explicitly selected small *already-local* HF safetensors checkpoint or a deliberately generated tiny local model, isolated Python environment, two bounded CPU updates, independently observed tensor changes, then a separate process reload with identical held-out predictions. Only after that mechanism test should pretrained-model gain claims be attempted.

## Subsequent executed smoke / hostile follow-up

The proposed *tiny random model* mechanism gate above was exercised on the authorized target workstation on 2026-10-26 using exact source commit `6f7b2924d7f6178623c9d014dece3762890bcf82`. Both CPU train and separate Python-process reload exited 0. The first SHA-256 of LoRA tensor values differed from the post-training digest, while the **frozen** reloaded LoRA tensors reproduced the trained digest exactly (`75f12dbabc0424a932abaa01181d0246971df49a8d1cdf5cf9f962d5dbc3db58`). Held-out baseline, candidate and controls all scored **0/2**. The reviewer expressly classified this as a **real neural update/persistence mechanism**, not pretrained adaptation or reasoning gain.

> **Still-open hostile finding:** Matching trained tensors proves what survived serialization, not what reasoning improved. The current untrained-adapter `sham` is not compute-matched, and four fixed tier-4 holdout structures are insufficient for broad transfer claims. The tiny smoke result must not be promoted into a trained-reasoning or safety qualification.
