# Weight-Space Programming and Neural Compilation — Research Gate (2026-11-05)

Status: **RESEARCH / EXPERIMENTAL**, not Vera model creation or runtime qualification.
Owner: Labyrinth research branch. This note separates independently testable mechanisms from attractive but unsupported extrapolation.

## Thesis

A neural model's parameters need not be obtained exclusively through stochastic gradient descent. For **specified algorithms**, ordinary source programs can sometimes be compiled into weights and executed as a neural network. That makes source → intermediate representation → neural weights a legitimate research branch alongside training, retrieval, LoRA, and native program execution.

Crucial distinction: a neural computer that executes a defined program is not thereby a pretrained open-world language model. Original-from-scratch neural components are technically real; a fluent ~3 GB Vera compiled from a GitHub portfolio remains a research question.

## Direct primary-source routes

| Route | Public upstream | What its authors demonstrate/claim | What remains unproven here |
|---|---|---|---|
| Restricted-language to attention/MLP circuits | [Tracr](https://github.com/google-deepmind/tracr) | RASP programs compiled to transformer weights without learning a conventional task dataset | General source languages, arbitrary exact compositions |
| Interpreter + compiler | [ALTA](https://github.com/google-deepmind/alta) | Program-specified transformer computation and interpreter traces | Real-world codebase compilation and broad language fluency |
| Python computation-graph compile | [torchwright](https://github.com/physicsrob/torchwright) | Python-defined operations emitted as HF-compatible transformer checkpoints | Numerical errors through arbitrary chains, general semantics |
| Instruction-set neural machine | [Loom](https://github.com/mkturkcan/Loom) | Analytically built transformer executing an instruction set | Ordinary LLM understanding or favorable efficiency versus native software |
| Hardware logic to neural operations | [verilog2nn](https://github.com/Enter-tainer/verilog2nn) | Logic circuit converted to ReLU network weights | Efficient transformer ABI or general reasoning |
| Interpreter + specialization | [transformer-vm](https://github.com/Percepta-Core/transformer-vm) | Author-claimed WebAssembly execution and a route for program specialization into weight matrices | Independent verification of correctness, performance, storage, and security |
| Transformer with firmware/data state | [Qwenputer](https://github.com/divagr18/qwenputer) | Neural CPU layers and program/RAM held in activation state before a pre-existing Qwen | Fast inference or removal of pretrained language-model dependency |

These are different mechanisms. A runtime calling Python functions through a neural-facing wrapper does **not** by itself demonstrate source code compiled into neural weights. The minimum honest artifact must expose actual parameter matrices, a weight-space inference path, and proof that the intended computation is performed through those matrices.

## The November 2026 weight-space editing claim

[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) reports that an LLM can inspect and edit another model's weights using a bounded read/write/feedback loop. [Logs](https://github.com/RikiyaT/lm_understand-logs) and a [published modified Qwen3.5-0.8B checkpoint](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming) allow closer inspection. The large reported Qwen math-score gain occurs in a looping thinking-mode baseline; do not infer new mathematical competence.

An [independent reproduction](https://github.com/leofmarciano/weightspace) reports MATH-500 thinking-on baseline 8.4% and edited 52.4%, but with thinking **off** baseline 46.4% versus edited 42.0%. GSM8K thinking-off is reported 56.6% versus 57.2%. The original authors also report failure to produce comparable new math ability in a non-looping Gemma-3 270M control. These results support targeted behavior repair, not generic low-cost intelligence synthesis.

## Hostile challenges and design constraints

> [Tracr issue #25](https://github.com/google-deepmind/tracr/issues/25) reports a reproducible discrepancy between compiled network and source RASP expression that its own validation missed. A compiler's own PASS is not independent semantic proof. Use a separate reference interpreter/oracle and full composition tests.

> A tiny number of nonzero weights does not imply a tiny loadable or fast dense model. Dense tensor allocation, runtime kernel choice, quantization, activation/state memory, and fixed model overhead determine the resident cost. Inspect [Qwenputer size analysis](https://github.com/divagr18/qwenputer/blob/main/docs/q35-size.md).

> [A public weight-editing demonstration](https://github.com/ValerioDolci/frontier-weight-edit) claims a small targeted weight modification can embed hidden preference while leaving ordinary benchmarks largely normal. This is author-reported, not independently qualified here. Treat all weight files as executable supply-chain artifacts, with hash/provenance, rollback, and adversarial behavior tests.

> 'No gradient steps' does not necessarily mean 'no data fitting' or 'no upstream learned system': closed-form regression and pretrained frontier agents still involve learned representations or optimization.

> [wsp-lab](https://github.com/ArioMoniri/wsp-lab) reports negative results on reliably predicting the effect of weight edits from descriptors/deltas, and uses hardware/budget profiles unsuitable for the zero-cost target. Negative findings should be preserved, not averaged away.

## Narrow, falsifiable pilot

1. Compile **two different deterministic operations** into explicit numerical neural parameters from a typed specification, with no optimizer, training loop, pretrained language model, or hidden Python delegation in the inference path.
2. Compose the two through a documented input/output ABI and compare its result against a separate ordinary reference implementation. Verify each component and both orders where meaningful.
3. Prove the output depends on parameters: alter/remove weights and observe expected failure; serialize/reload parameters and re-evaluate; record parameter count, resident bytes and CPU latency.
4. Include negative and out-of-domain controls. Do not label self-generated task examples independent; reserve unseen test cases/seeds.
5. Emit reproducible source/weight digests and claim the **specific tested function only**. Measure quantized execution separately, if attempted.

### Integration boundaries

- **Labyrinth:** owns small original compiled-weight experiments, independent oracles and adversarial challenge generation.
- **model-training:** may later consume independently qualified weights and manifests; its existing Qwen/LoRA line remains separate.
- **BT2 / Vera runtime:** consumes a *candidate capability* only after explicit interface, provenance, safety, authority, and deployment gates. Source tests are not runtime installation.
- **Plasticity/governance:** installation, firmware/state update, learned weight change, memory admission, and self-model changes must be separately typed and authorized.

### Claim ledger

| Proposition | Current verdict |
|---|---|
| Defined algorithms can be compiled into neural weights without SGD | **SURVIVES** at narrow published-demo level; must reproduce here |
| A stateful neural interpreter can accept new programs as input/state without a new weight file | **SURVIVES** conceptually and in published demonstrations; local evidence pending |
| Source graph, compiler output, and neural behavior are automatically equivalent | **FAILS**; explicit counterexample/validator weakness |
| Portfolio code alone can be automatically compiled into fluent autonomous Vera | **UNKNOWN** |
| A useful end-to-end Vera will fit in ~3 GB and operate well on Lappy 4 GB VRAM | **UNKNOWN**, requires measured prototype |
| Direct weight editing repaired a pathological Qwen inference mode | **SUPPORTED** by independent reproduction; not evidence of new general math ability |

Zero monetary spend; never auto-download frontier weights, alter active model files, run owner-host training, deploy, or attach sensitive personal materials in this public-repo experiment.

## Provenance

Public upstream pointers, critical independent reproductions, and research classification; the authors' experiments were **not** independently rerun for this initial note. Subsequent pilot commits must report exact test command, platform, artifact hashes, and observed results.