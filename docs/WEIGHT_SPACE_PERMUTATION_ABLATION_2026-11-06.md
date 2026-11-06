# Weight-space merging: permutation counterexample, exact CPU pilot

Status: **SOURCE-ONLY / RESEARCH**. No Qwen weights, private data, owner hosts,
gradient training, checkpoint creation, or model deployment were used.

## Falsifier

Two networks can implement *exactly the same function* yet have incompatible
raw parameter coordinates. Reordering hidden neurons, while applying the same
permutation to first-layer outputs and second-layer inputs, preserves the
function mathematically. Averaging the two unaligned parameter vectors does
not generally preserve it. Consequently, simple arithmetic between unrelated
repository models or adapters is not a reliable way to combine their abilities.

This is the symmetry addressed by [Git Re-Basin](https://github.com/samuela/git-re-basin).
The [Model Soups](https://github.com/mlfoundations/model-soups) and
[Task Vectors](https://github.com/mlfoundations/task_vectors) results require
careful assumptions about shared model lineages and/or fine-tuned origins.
Task vectors are *differences between an existing base and a fine-tuned model*;
they do not synthesize the missing pretrained knowledge from source code.

## Actual local experiment

The repository's existing `boolean_neural_specs.build()` compiles a 3-bit
full adder to a two-layer exact-integer ReLU network (3 inputs, 16 hidden
neurons, 2 outputs, 98 scalar weights and biases). We reverse its 16 hidden
neurons and carry each neuron's incoming weights, bias and outgoing weights
together. This deliberately changes numerous numerical coordinates while
leaving the complete input/output function invariant.

We evaluate all eight Boolean inputs, with a separate plain full-adder oracle
already in the Labyrinth tests:

| Arm | Input/output mismatches over 8 exhaustive inputs |
|---|---:|
| Original network | 0 |
| Hidden-permuted network | 0 |
| **Naïve 50:50 coordinate-wise mean** | **7** |
| Exact-neuron-aligned 50:50 mean | 0 |

The naïve mean yields `(0, 0)` on all eight inputs, despite the original
full adder having seven nonzero outputs. The implementation evaluates means
using exact rational fractions so that rounding cannot conceal an error.
It also tests 40 additional deterministic shuffled permutations, an unchanged
identity control, non-equal weights, ABI mismatch, malformed indices and
optimized Python guards.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_weight_space_permutation -v
PYTHONDONTWRITEBYTECODE=1 python -O -m unittest tests.test_weight_space_permutation -q
```

This pilot can **align exact copies** of a small model that differ solely by
hidden-unit permutation. It cannot align arbitrary models; it explicitly
refuses changed neurons, distinct output biases and mismatched ABIs. It does
not perform general optimal assignment, feature matching, learned alignment,
or propose to average unrelated Qwen/GGUF models.

## Research routes for VERA — tested meaning and feasibility

| Route / primary source | What it contributes | Critical cost or missing proof |
|---|---|---|
| [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) and [logs](https://github.com/RikiyaT/lm_understand-logs) | Agent-guided sparse edits to *pretrained* Qwen3.5-0.8B, motivated by repetition pathology | Exact run still used 158 editor turns, 80 Python calls, 50 forward-pass batches, separate GPU evaluation; the harness is not public and the result is not a 4B or scratch-model proof |
| [Tracr](https://github.com/google-deepmind/tracr), [ALTA](https://github.com/google-deepmind/alta), [torchwright](https://github.com/physicsrob/torchwright), [Loom](https://github.com/mkturkcan/Loom) | Can compile **specified restricted computations** into actual numerical neural parameters without ordinary SGD | Cannot transmute a mixed software portfolio into fluent open-world language; Loom uses nonstandard top-2 argmax attention, torchwright approximates operations and emits FP32 HF artifacts |
| [Text-to-LoRA](https://github.com/SakanaAI/text-to-lora) | Trained hypernetwork maps textual task descriptions to adapters for existing LLMs | Requires **over 16GB VRAM** to run demo and reports approximately **five H100 days** to train hypernetwork; no Lappy 4GB reproduction |
| [D2NWG](https://github.com/sorobedio/DNNWG) and [WeightCLIP](https://github.com/HSG-AIML/weightCLIP) | Train generative models/latent spaces over existing model-weight zoos and dataset representations | Depend on pre-existing trained model zoos and nontrivial generative model training; not a from-zero LLM compiler |
| [Git Re-Basin](https://github.com/samuela/git-re-basin), [Task Vectors](https://github.com/mlfoundations/task_vectors), [Model Soups](https://github.com/mlfoundations/model-soups) | Alignment-aware merging, adaptation arithmetic, averaging similar fine-tuned models | Our exhaustive pilot directly shows naïve coordinates can be misleading even when both networks are perfectly correct |

## Next bounded $0 experiment

Treat GitHub repositories as **governed capability specifications**, not
pretrained language data by fiat. Compile only a small typed mathematical or
Boolean capability into numerical weights. Separately expose it as a callable
tool to an already-qualified small pretrained conversational model. Measure
the unchanged base model against: (A) external deterministic capability, (B)
same-base LoRA when an authorized checkpoint exists, and (C) a specifically
targeted, reversible sparse weight edit with a matched random/sham control.
Score identity, personality, factuality, consistency, recovery and resource
cost on held-out dialogues; do not reuse those test answers to select edits.

If the compiled capability corrects tool-invoking behavior but does not change
ordinary conversation without explicit routing, that is a valid positive
finding *for modular capability integration*, not evidence of semantic
weight-space intelligence synthesis. If sparse edits only improve a looping
decoding mode and hurt normal-mode performance, retain that negative result.

Promotion stays on **HOLD** until exact provenance, rights, genuinely isolated
evaluation, owner-host capacity and actual model behavior are demonstrated.
