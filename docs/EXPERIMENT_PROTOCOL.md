# Labyrinth: experimental qualification protocol

Status: **design and prototype controls**, not a claim that a language model has learned reasoning. This protocol was written after independent challenges to the initial 2026-10-26 prototype. It must be frozen with a particular source commit and model checkpoint **before** a final evaluation. This document does not itself pre-register a completed experiment.

## Separate three questions

1. **Can the harness generate, score, isolate and replay experiments?** Unit/integration tests and independent oracles answer this.
2. **Did a backend change persistent parameters?** Compare before/after exact checkpoint hashes, update receipts, and predictions after fresh-process reload without demonstrations. A different hash alone is insufficient.
3. **Did the update improve transferable reasoning?** Reserved structural holdouts, unchanged prompts/decoding, independent controls, sufficient sample size, and replicated paired gains are required. Neither a falling loss nor better few-shot scores answers this.

The `train-toy` command addresses question 2 for an explicitly hand-engineered four-coefficient arithmetic policy. It cannot answer question 3 for a neural language model. The `experiment` command compares prompts, not training interventions.

## Frozen evaluation contract

Before the final set is exposed, record:

- source commit and generator implementation hashes;
- base model architecture/tokenizer/weight digest and runtime build;
- exact train/development/final case IDs, structural fingerprints and oracle commitments;
- training optimizer, epochs, example order, sample/token budget and random seeds;
- decoding configuration, instruction template, output budget and format metric;
- primary accuracy metric, failure/abstention handling, minimum effect, uncertainty method, family slices and stopping rule;
- whether the candidate selection procedure ever consumed evaluation results.

Use *development* splits for tuning. A final split becomes development data once its results change code, checkpoints, prompts, or selection. A deterministic public generator is reproducible, **not secret**. Do not claim untouched final-set independence without an evaluator/operator boundary.

## Controlled comparisons

For genuine weight adaptation, use the **same initial base checkpoint** in all intervention arms:

| Arm | Intervention | Inference context |
| --- | --- | --- |
| Frozen baseline | No optimizer update | No demonstrations |
| Correct supervision | Declared optimizer and correct train-only labels | No demonstrations |
| Sham update | Equal case/token/step budget, no effective parameter updates | No demonstrations |
| Corrupted supervision | Equal steps and plausibly corrupted train-only targets | No demonstrations |
| Few-shot secondary | No weight update | Demonstrations disclosed and budgeted |

Any curriculum improvement additionally requires an equal-budget **uniform sampling** control. Compare candidate and controls on the **same final challenge IDs**, and report paired disagreements and per-family results. Randomize or counterbalance arm execution order where a backend may have hidden external state. A fresh Python object does not guarantee a remote model server is stateless.

For prompt-only `experiment` comparisons, every shuffled label must differ from its correct label. If a multiset-preserving derangement is impossible, fail rather than present a sham identical to the candidate. The neutral placebo is character-length matched only, not token- or semantic-structure matched. Those limitations remain experimental confounds.

## Transfer slices

- **Within-template:** fresh numbers with familiar structures; this tests instance generalization only.
- **Structural:** new register-program operation compositions, with canonical structure fingerprints disjoint across train and evaluation.
- **Length:** longer register programs than those seen during training.
- **Representation:** meaning-preserving rewrites of serialization/instructions, only after independent validation of semantic equality.
- **External:** independently authored task families or established benchmarks with contamination/licensing controls.

Never replace these slices with a single pooled score. The register-program evaluator has a simple deterministic interpreter baseline; a model's scientific question is whether it learned the procedure, not whether a computer can execute it.

## Measurement and falsifiers

- Publish paired accuracy deltas, correct/total, discordant counts, format adherence, per-family breakdowns and uncertainty intervals; account for repeated templates/clusters when estimating independence.
- Pre-specify the minimum relevant positive effect and uncertainty bound before looking at the final data. Correct multiple primary comparisons; do not treat retries or model generations on the same prompt as new independent tasks.
- An unchanged checkpoint after a claimed update is a **failed update**. A fresh-process loss of apparent gain is **failed persistence**. Falling train loss with flat structural holdout score is **failed transfer**, not success.
- A candidate that wins only with its demonstrations has shown *context conditioning*, not persistent parameter learning.
- A candidate that improves only output-format adherence has not yet shown improved reasoning accuracy; report both.
- A result dependent on arm order, reused/overlapping cases, invalid shuffled labels, or unverified model version is **unqualified**.

## Optional language-model training boundary (not implemented)

A local SFT/LoRA backend should be **optional** and depend on explicit already-local model/tokenizer files. No implicit model downloads, arbitrary shell execution from model output, or requirement for CUDA in the core harness. A training receipt should contain consumed train-only examples, initialization and exported adapter hashes, tokenizer/base-model hashes, optimizer specification, completed update count, seed, loss summary and package/runtime versions. Freeze the base weights, update only declared adapter parameters, save to a new path, and verify clean-process reload. The **scratch test runner** initially lacked PyTorch, Transformers and PEFT; the separate target workstation has these packages but its active PyTorch interpreter is CPU-only. Neither finding qualifies an LLM training run.

Reference starting points: [Transformers offline loading](https://huggingface.co/docs/transformers/main/installation), [PEFT quicktour](https://huggingface.co/docs/peft/en/quicktour), and [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer). Consult pinned versions when implementation begins; current documentation is not an immutable environment specification.
