# Qwenputer GDN RAM: source-only feasibility gate (2026-11-06)

**Decision: HOLD for a 3 GB conversational Vera model on Lappy.** A bounded, independently implemented one-hot delta-rule RAM fixture passes. This is evidence that one specified mathematical write/read primitive can be represented exactly with integers; it is **not** evidence that Qwenputer, any Qwen3.5 checkpoint, or a 3 GB GGUF has been compiled, loaded, quantized, or evaluated here. No model files were downloaded, patched, trained, or deployed.

## Primary-source custody

| Source | Pinned source blob SHA-1 | Observed claim or contract |
| --- | --- | --- |
| [Qwenputer `LICENSE`](https://github.com/divagr18/qwenputer/blob/main/LICENSE) | `ccdbffd15bbbf20bc8127e113326cae649e8059f` | MIT license for code; third-party game ROMs excluded. Our fixture uses neither their code nor ROMs. |
| [`chipformer/q35/ir.py`](https://github.com/divagr18/qwenputer/blob/main/chipformer/q35/ir.py) | `0e380ce0fef075a8b5408e5a493ab4c723878e3c` | Intermediate representation for SwiGLU and Gated DeltaNet, including one-hot writes to state. |
| [`chipformer/q35/emit.py`](https://github.com/divagr18/qwenputer/blob/main/chipformer/q35/emit.py) | `dc7d67bb611095505d20376165f2f18a251ce4a7` | Analytical tensor/matrix synthesis with ballast and normalizations; still demands exact tensor placement and precision. |
| [`chipformer/q35/merge.py`](https://github.com/divagr18/qwenputer/blob/main/chipformer/q35/merge.py) | `bb28a63694fd35a1ebb9e985e470f800dcd13e9c` | Reads pretrained Qwen weights, widens layers/embeddings, appends chip layers and tokenizer entries. **Not** a tiny sparse patch into unchanged Qwen. |
| [`chipformer/q35/lib/macros.py`](https://github.com/divagr18/qwenputer/blob/main/chipformer/q35/lib/macros.py) | `80a6f2b7bcf12d21234c43dfe7b79a5b0345446b` | CHIP-8 extended bignum macros (`BADD`, `BMUL`, `BSUB`, `BRSUB`, `BZERO`, `BTRIM`, `BLD`, `BSTA`), not free-form conversational semantics. |
| [`docs/q35-size.md`](https://github.com/divagr18/qwenputer/blob/main/docs/q35-size.md) | `efdf8340aa3266e7b8f5d907fb951990c2f70520` | Author-measured parameter counts, physical BF16 sizes, dense-zero distinction. |
| [`pyproject.toml`](https://github.com/divagr18/qwenputer/blob/main/pyproject.toml) | `0c6eebd19cb3e2373f514ad2c0fe6bcf867bb384` | Python >=3.11; NumPy and safetensors; heavier optional Hugging Face/torch stack. |

These are inspected **source/author claims** only, not an independent reproduction of the upstream project. In particular, the reported Qwenputer 147-suite tests and arithmetic/tool demonstrations were not rerun. The optimized CHIP-8 variant inserts **12 chip layers** (hidden 2944); the repo also discusses a distinct RISC-V design adding 11 layers. Do not conflate them.

## What this bounded test actually executes

The independent `labyrinth/gdn_ram_feasibility.py` fixture uses 0/1 integer keys and values and models the delta-state rule directly:

\[
S' = S + \beta k (v-k^\mathsf{T}S)^\mathsf{T}, \qquad \mathrm{read}(q)=q^\mathsf{T}S.
\]

For a one-hot row key and beta 1, a write overwrites only the selected row; beta 0 retains state. The fixture computes the full dense equation rather than assigning a Python row directly. `micro_ram_control()` runs 128 seeded writes to a 16-address by 8-bit store and compares **every** read to an independent dictionary-memory oracle after **every** write: 2,048 correlated row comparisons (not 2,048 statistically independent trials). Input and CPU bounds are explicit; multi-hot keys, malformed values and non-binary beta are rejected. The test is deterministic and needs only Python stdlib.

**Work-amplification falsifier:** initial argument bounds also admitted 64 rows × 64 bits × 4,096 transactions, producing an estimated 1,107,296,256 dense cell contributions from exhaustive readback alone plus write work. The reviewed version now computes `transactions * width * (rows² + 2*rows)` **before allocation**, rejects estimates above 2,000,000 cell steps, and tests that the hostile oversized configuration is rejected before constructing RAM. Default control estimates 294,912 bounded cell steps. The limit measures this implementation's work shape, not a hardware instruction-count guarantee.

It intentionally omits the *actual* Qwen3.5 GDN sequence, finiteness/rounding behavior, RMSNorm, SwiGLU, attention, tokenizer, hidden-state routing, and generation. Passing here cannot prove Qwenputer's activation precision, macros, transformer integration, or semantic competence. To falsify this fixture further, mutate any unaddressed state row, make beta partially overwrite, or change key validation and observe disagreement or negative-control failure.

## Quantitative footprint gate

Qwenputer upstream reports its stock/pretrained 4B base ~9.34 decimal GB BF16; original overlay 7.577B params / 15.15 GB BF16; optimized 12-layer overlay 6.756B params / 13.53 GB BF16; optimized *text-only* overlay **6.421B params / 12.86 GB BF16**. Added chip layers alone contribute approximately **1.558B dense parameters**, ~3.116 GB BF16, although only ~0.8 million are nonzero. Sparse zeros compress transfer/storage, but do not disappear from the stock dense runtime representation.

Using upstream rounded counts, the following are **conditional dense-format size scenarios**, not universal compression lower bounds. Each excludes all quantization metadata, scales, tokenizer, executable, activations, KV/GDN state and accuracy effects:

| Configuration | Decimal bytes minimum | 3 GB decimal target | 3 GiB binary target |
| --- | ---: | --- | --- |
| 6.421B text-only parameters, every scalar in a *dense 4-bit format* | 3,210,500,000 | Fails by 210.5 MB | Under by only ~10.7 MB **before any overhead** |
| 1.558B exact BF16 chip layer + remaining 4.863B ideal 4-bit | 5,547,500,000 | Fails | Fails |
| Dense stock optimized text-only BF16 (upstream observed) | ~12,860,000,000 | Fails | Fails |

There is no demonstrated 3 GB GGUF export or exact-preserving quantization of generated chip layers in this audit. A *dense-Q4 scenario* **does not rule out** Q2, Q3, compressed sparse tensors, different architectures, or hybrid symbolic routes—nor does it prove executable fit or function preservation: chip exactness may fail when zero/ballast/routing terms are quantized. A 32 GB system RAM target and 4 GB VRAM are separate constraints; no inference memory, GPU allocation, throughput or Lappy host measurement was taken. Upstream's transfer-compression figure is not resident RAM. Stock 13.53 GB is the author-optimized overlay, not the stale 15.15 GB original.

## Comparisons and a useful next falsifier

- **Labyrinth Draft #126–129:** explicit bounded symbolic/neural branch state, replay and source gates. This fixture adds only the mathematical delta-memory primitive; it does not integrate a transformer layer or supplant host custody.
- **Labyrinth Draft #130:** exact 98-parameter adder with a hidden-neuron permutation falsifier; naive weight averaging fails 7/8 cases. Neither that exact compiled circuit nor this RAM microtest is an open-language foundation model.
- **Model-training Draft #350:** narrow detached synthetic weight-patch/byte-recovery path, with training-set controls still declarative. This fixture writes ephemeral **activation state**, not Qwen weights, so cannot be represented as proof of a trained model or validated sparse Qwen edit.

Before a Qwenputer transfer to Vera, a separate isolated test would need a frozen instruction→IR macro contract; byte/AST-level upstream code and license provenance; exact semantic reference traces for bounded programs; a CPU-only finite-precision model of the relevant GDN layer; a quantized-vs-BF16 differential falsifier, and signed size/memory checks against **the exact proposed 3 GB artifact**. Keep general conversation/identity evaluation and real Qwen benchmarks separate from CHIP-8 correctness. No permission for downloading 4B weights, host GPU use, merging, deployment, or purchasing compute is inferred.

**Evidence state:** `SOURCE_INSPECTION` for upstream pinned files; `AUTHOR_REPORT` for upstream performance/counts; `TEST` only for our local synthetic integer equation and arithmetic; `UNKNOWN/HOLD` for model fit, quantization fidelity, runtime behavior and a conversational Vera model.
