# MeZO-family optimizers for Labyrinth: source review and experimental gate

**Status (2026-10-30):** Research and an implementation/qualification protocol, **not a neural-training result**. Target baseline: `thebrazenbeard/labyrinth` `main@39a95ac1d12cce372de980b8cad2bc41016e1705`. MeZO, AdaMeZO, and LOZO+ are candidates, not proven improvements to Minotaur. Do not promote an adapter or assert reasoning transfer from this document or unit tests.

## What is being tested

Labyrinth's existing `labyrinth/hf_pilot.py` trains a small *local Hugging Face causal LM's LoRA parameters* using AdamW/backpropagation. It retains a frozen baseline, a zero-effective-update sham, a deranged-label control, train/evaluation split checks, base/frozen/buffer integrity checks, new-path adapter export, and a separate-process reload requirement. The current limit is **at most 1 GiB of local safetensors**, not general GGUF training or a guarantee of fitting on a 4 GiB GPU. Its current controls are valuable and must remain available.

The [original MeZO paper](https://papers.neurips.cc/paper_files/paper/2023/hash/a627810151be4d13f907ac898ff7e948-Abstract-Conference.html) studies both full-parameter and parameter-efficient LM tuning. A **LoRA-only MeZO pilot is not full-model MeZO**: only the explicitly trainable adapter tensors are perturbed; frozen pretrained weights and buffers must remain intact. It may reduce autograd activation and AdamW-state memory, but its memory advantage over the already-small LoRA AdamW baseline may be modest. It does not make a quantized GGUF file trainable, expand the allowed base checkpoint, or establish superior reasoning.

## Reproducible two-sided estimator

For trainable parameter vector `theta`, fixed example batch `B`, sampled perturbation `z ~ N(0, I)`, scale `epsilon > 0`, and learning rate `eta`:

```text
loss_plus  = L(theta + epsilon*z; B)
loss_minus = L(theta - epsilon*z; B)
directional_derivative = (loss_plus - loss_minus) / (2*epsilon)
theta_next = theta - eta * directional_derivative * z
```

The positive and negative forwards must see the **same examples, token labels, model mode, and stochastic conditions**. Dropout is disabled or its randomness must be controlled. Perturbation seeds and a stable named-parameter iteration order are needed to reproduce `z` without a permanent model-sized noise allocation. Both forward losses and the directional derivative must be finite. A failed forward or invalid gradient estimate must not leave a partially perturbed checkpoint. Report the exact total loss-forward queries (normally two per direction per training step), not simply update counts.

The [official Princeton implementation](https://github.com/princeton-nlp/MeZO/blob/main/large_models/trainer.py) regenerates noise from a seed and performs in-place perturbations. It does not promise bit-identical parameter restoration under every floating-point dtype, every epsilon, or interrupted execution. An *adapter-only implementation* may instead snapshot the trainable adapter tensors for exact failure rollback and zero-LR-sham identity; **that snapshot adds memory proportional to trainable parameter size** and must be disclosed rather than called inference-memory-equivalent MeZO. Such a safeguarded implementation is a local variant, not a faithful proof of the paper's peak-memory bounds.

## Alternatives to compare, not assume superior

| Candidate | Primary paper / source implementation | Mechanism and caution |
| --- | --- | --- |
| **MeZO** | [NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/a627810151be4d13f907ac898ff7e948-Abstract-Conference.html); [official code](https://github.com/princeton-nlp/MeZO) | Two-sided seeded ZO-SGD without backprop; noisy estimator and sometimes many more update steps than first-order training. |
| **MeZO-SVRG** | [ICML 2024](https://proceedings.mlr.press/v235/gautam24a.html); [official code](https://github.com/amazon-science/mezo_svrg) | Variance reduction introduces additional reference-loss/gradient-estimation work. |
| **LOZO / LOZO-M** | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9ccc9d814d3dee4750debaf23061e733-Abstract-Conference.html); [official code](https://github.com/optsuite/LOZO) | Low-rank perturbation and optional momentum; benefit depends on gradient geometry and selected rank. |
| **Sparse-MeZO** | [NeurIPS 2025 source](https://arxiv.org/abs/2402.15751); [official code](https://github.com/NUS-HPC-AI-Lab/SparseMeZO) | Restricts updates to a selected weight subset; mask choice and preservation of frozen tensors require checks. |
| **AdaMeZO** | [ICML 2026](https://proceedings.mlr.press/v306/cai26a.html); [official code](https://github.com/shawnnn3di/AdaMeZO) | Reconstructs Adam-like moment information from seeded historic perturbations and scalar directional projections. Authors report up to 70% fewer forward passes on their study workloads; replay computation and temporary tensors may offset gains locally. |
| **LOZO+** | [ICML 2026](https://proceedings.mlr.press/v306/fang26k.html) | Greedy loss-feedback low-rank subspace selection with adaptive threshold; candidate selection adds cost. The proceedings page does not provide a verified official implementation link. |

The [2026 *On Adaptivity in Zeroth-Order Optimization* preprint](https://arxiv.org/abs/2605.03869) challenges the assumption that ZO-Adam necessarily beats a well-tuned ZO-SGD baseline. Its findings are not a Labyrinth benchmark. Use MeZO as the simple reference before introducing history windows, sparse masks, or selection heuristics.

**Hardware caution:** An inference-compatible quantized base is not necessarily a mutable, trainable HF model. For quantized/frozen-base research see [SensZOQ](https://github.com/GarlGuo/SensZOQ), [QuZO](https://github.com/lloo099/QuZO), and [QZO](https://github.com/maifoundations/QZO); none establishes the proposed Labyrinth target can be trained on Patrick's 4 GiB VRAM. No hardware measurement is contained here.

## Falsifiable integration and evaluation gate

1. **Correctness first.** Unit-test a known quadratic finite-difference direction and multi-tensor parameter update against an independently computed oracle. Check reproducibility across seeds, invalid/zero epsilon, nonfinite loss, thrown-forward rollback, no autograd/backward and strictly trainable-parameter updates. Document dtype/cancellation limits.
2. **Keep the established baseline intact.** Existing AdamW LoRA remains the default. Introduce MeZO only by an explicit optimizer selection; immutable old receipts remain parseable. Record optimizer kind, `epsilon`, direction seeds, step count, total loss forwards, training example visits and zero-update behavior in a versioned receipt.
3. **Matched experimental arms.** Use the *same initial base and adapter state*, train-only labels/cases, tokenizer, decoding, and case order for correct supervision, sham and corrupted-label arms; frozen baseline has no weight updates. For MeZO match **two loss-forward evaluations per step** even in the sham. Deranged labels can have different tokenizer lengths on the same cases, so matching visits and forward calls does **not** establish equal token-level compute exposure. Compare the reported `input_tokens_seen` for each arm and explicitly mark this confound. Compare against AdamW at both equal sample exposure and explicitly reported computation/time budgets; never say the backward and forward costs are identical.
4. **Persistence and operating bounds.** Verify original safetensors, frozen parameters and persistent buffers did not mutate; save candidate adapter only to a new output directory, hash its tensors/files, reload in another process, and compare predictions. Record peak host RAM, CUDA allocated/reserved memory if applicable, latency, versions, hardware, and failed attempts. No implicit download, dependency install, service restart, or promotion.
5. **Real task transfer.** Precommit held-out arithmetic, operation chains, unseen program structures/lengths and source-priority counterfactuals; separate strict semantic correctness from answer-format adherence and report per-family paired disagreements. Use multiple seeds and explicit uncertainty; tune only on development cases. Public generator holdouts are *not* privately blind. `program_maze.py` currently has only four hard-coded training and four hard-coded evaluation instruction structures; multiple numeric variants do not increase structural diversity.
6. **Falsifiers.** No actual parameter change is a failed update. A successful weight/hash change with no held-out semantic gain is not learned transferable reasoning. A sham or deranged-label arm matching the candidate destroys a causal attribution. Failed fresh-process reload defeats a persistence claim. Any tested final set subsequently used to tune parameters ceases to be untouched final evaluation.

The repository's [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md) and graduation gate remain authoritative for higher-level claims. This is a **research design** pending exact-head code tests, an actual local-model MeZO training run, hardware measurements and independent evaluation. Existing AdamW, LBFGS and toy training remain separate controls; none should be replaced merely because the optimizer is new.
