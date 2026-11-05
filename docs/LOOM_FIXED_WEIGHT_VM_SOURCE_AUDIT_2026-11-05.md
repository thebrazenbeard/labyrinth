# Wave 3: Loom source audit and sealed model feasibility

**2026-11-05, source inspection only.** No upstream repo or user repo edits, no local Lappy execution, no weight artifact produced. GitHub connector: `mkturkcan/Loom` public `main` and `LICENSE` Apache-2.0 (blob `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`). Source identifiers below are **blob SHA-1**, not branch HEAD; a development PR must pin its actual commit before copying/adapting code.

## Concrete primary source

| File | Git blob | Actual evidence |
|---|---|---|
| [README](https://github.com/mkturkcan/Loom/blob/main/README.md) | `5060015b5fdf117ffbd4634fc1f647dcf7f96040` | Author claims 8 analytically generated transformer layers, 21 opcodes, no SGD, 8MB/15MB/28MB weights, C→ISA, over 140 passing tests, CPU examples. Claims not independently replicated here. |
| [loom_v1.py](https://github.com/mkturkcan/Loom/blob/main/loom_v1.py) | `9a08008117e5740a039cf2e77d225a3fb63a27b0` | Canonical `LoomConfig` and `LoomComputer`, `_build_layers()` appends 8 hand-constructed attention/FFN layers; weighs are `torch.zeros(...)` arrays populated by explicit assignments, not optimizer output. |
| [subleq.py](https://github.com/mkturkcan/Loom/blob/main/subleq.py) | `dcbfa68b540d155f8c1970aad18ffe0f470c238f` | `TransformerLayer` is a **plain Python class, not `torch.nn.Module`**; raw tensor attributes `Q/K/V` (plus numbered head variants), `W1/b1/W2/b2`. `forward` applies attention from these tensors followed by ReLU FFN and residual. `NeuralSUBLEQ.step` loops over neural layers; `run` is Python supervisor checking PC. |
| [c_compiler.py](https://github.com/mkturkcan/Loom/blob/main/c_compiler.py) | `ee6041614f40a0b3335f58f76a1582f1a5467167` | `compile_c(source, s=32,m=64,n=1024,N=8)` returns `(cfg, memory, commands, meta)`; C subset has loops/conditionals/arrays and no pointers/recursion/division/modulo. `compile_and_run` creates a fresh computer and runs `.step` repeatedly, with Python PC HALT and a time/step cap. |
| [extended_isa_v4.py](https://github.com/mkturkcan/Loom/blob/main/extended_isa_v4.py) | `811227ba13f17fd440e4a8c557e123f127c0bb28` | `ExtendedNeuralComputerV4` and `ExtendedConfigV4` are compatibility **aliases** to `loom_v1.py`; old import names in tests are coherent. |
| [test_comprehensive.py](https://github.com/mkturkcan/Loom/blob/main/test_comprehensive.py) | `79cb28dadf710915098225f625d6343bc85c2970` | Standard compiled-C fixture `s=32,m=160,n=512,N=8`, `run_c` uses neural `.step` and reads variables after halt. Contains GCD, array, nested control flow tests. **At bottom**: `TestCompiledC.test_array_sum = None`, disabling a test with an acknowledged bad expected result (`150` doesn't fit in signed 8-bit). |

No `requirements.txt`, `pyproject.toml`, or `setup.py` was found at **root paths** tested (404). The README asks `pip install torch numpy`, tests additionally import `pytest`. Source imports `torch`, `torch.nn.functional`, `numpy`, standard-library modules; pin actual versions in our isolated experiment, do not assume upstream lockfile.

## Resource estimate for Patrick's Lappy

Source formula for `LoomConfig(s=32,m=160,n=512,N=8)`:

- `logn=9`; row allocation `nrows_cmds=27`, `nrows_memory=8`, `nrows_scratchpad=43`, `nrows_pc=9`, `nrows_pos_enc=9`, `nrows_buffer=41`, `nrows_tag=8`, `nrows_indicator=1`. **Thus `d_model=146`.**
- Activation/machine state is `146 × 512` `float32`: `299,008 bytes` = ~292 KiB. This is **not** the total model weight size, interpreter runtime, or peak RSS.
- Upstream README claims ~8 MB of weights for this shape, but actual serialization size depends on all per-head Q/K/V + FFN weight and bias tensors. It was **not measured** in our session. Lappy's 32 GB host RAM is ample for these static arrays under ordinary memory conditions; RTX 3050 4 GB hardware is historically known, not freshly probed. Start **CPU** to avoid collisions with TrainingDay workloads.
- Performance risk is substantially higher than storage risk: `_attn_head` forms `n × n` score matrices, invokes `topk(2)`, and iterates Python over each of the `n=512` columns to scatter results, for every nonzero attention head, over 8 layers, **per ISA instruction**. Hundreds of instructions may take conspicuous time on CPU. Start with ≤20-instruction opcode checks and measured budgets; defer 5000-step GCD stress until timing is known.
- Default `compile_c` config is `n=1024` with `m=64`, different from the standard `n=512,m=160` test fixture, so always pass explicit matching dimensions to compiler, network and state loader. `n` must remain a representable column space, and signed values wrap at `N=8`.

## Exact single-sealed-weight-file complication

`LoomComputer(cfg)` computes its weights on construction, but its plain tensor-bearing layers have **no `nn.Module.state_dict()`** in inspected source. Therefore a *single file* is **possible as a new adapter**, not demonstrated as existing functionality:

1. Enumerate each layer's actual tensor attributes (`Q/K/V` or `Q1..Q3/K1..K3/V1..V3`, `W1/b1/W2/b2`) and serialize their fp32 bytes as named tensors in a non-executable format, e.g. `safetensors`; embed exact config and ISA version in canonical metadata or require a separately fixed loader implementation.
2. Write a strict loader that validates expected layer count, heads, names, tensor shapes, dtype, metadata, size bounds and SHA-256; reconstruct a `LoomComputer(cfg)` and **replace every generated tensor with loaded bytes**. Because constructor regenerates weights, use a weight-zero negative control and input-output divergence to prove inference used file bytes, not constructor defaults.
3. Keep the `program/data state X` **outside the frozen weight file**. A model file contains interpreter weights, not three compiled C programs; each program is an independent ISA/data-state artifact or ephemeral input.
4. Fresh process with no compiler import should `load_weights()`, `init_state()` from validated ISA instructions, repeat `computer.step(X)` and decode output. The inference loop may inspect PC, timing and resource budgets; it may **not interpret and execute ISA opcodes in host Python**. Security: no pickle/untrusted `torch.load`, no dynamic `exec` or arbitrary source code inside inference path.
5. A serialized `.safetensors` is a **single model weight artifact**, still requiring trusted PyTorch source runner. It is not a `.gguf` or a self-contained model file for KoboldCPP. Standard softmax Transformer conversion also changes numerical semantics; Loom uses top-2 hard argmax with tie logic. Any future GGUF requires explicit architecture support and re-testing.

### Potential silent truncation must be patched at adapter boundary

In `subleq.py`, `init_state` initializes columns of `memory` only while `i < cfg.m`, then `break`; likewise commands only while `cfg.s + cfg.m + i < cfg.n`, then `break`. This **silently drops** excess values instead of rejecting them. Before calling it, require:

```
0 <= len(memory) <= cfg.m
0 < len(commands) <= cfg.n - cfg.s - cfg.m
all ISA fields valid and encodable in cfg.logn bits
all memory values in signed N-bit range (unless explicit wrap requested)
all branch targets within program columns or HALT sentinel
```

This avoids false successful load with a truncated program, out-of-domain addresses, or accidental infinite loops.

## Actual 3-program gate (no truth-table shortcut)

### Exact proof experiment

1. Baseline upstream core/test collection, record exclusions (especially dynamically disabled `test_array_sum`). Pin SHA, license, dependency versions, CPU host observation.
2. Construct ONE `LoomComputer(LoomConfig(s=32,m=160,n=512,N=8))`; serialize all neural weights to one safe file. Hash the file. Immediately terminate builder process.
3. In fresh worker A, load the **sealed** file, compile a simple arithmetic program (e.g. `int main(){int x=7;int y=9;int z=x+y;return z;}`), then `init_state` and run neural `step` to HALT. Check `__retval == 16`.
4. In fresh worker B, load *same* file and compile a loop program (e.g. `sum_to_5` using `while`), check `__retval == 10`, compare result to independent Python reference.
5. After file hash frozen, a reviewer authors and compiles **previously unseen C source** such as GCD by subtraction. New C source and ISA command bytes differ from A/B, but weight artifact **must be byte-identical**. Test positive pair examples under signed 8-bit constraints, HALT within bound, inspect actual iteration trace and output. Start with a short pair to keep time bounded.
6. Compare A/B/C same SHA-256 weights, distinct ISA/program state SHA-256. Recreate worker in fresh process with compiler module unavailable at inference time; only a precompiled ISA program is fed to the neural interpreter. Re-run. Record exact `step` count, wall duration, RSS, GPU memory if used, and full on-disk size.
7. Fail expected cases: hash corruption, omitted tensor, wrong tensor shape, incompatible ISA/config, zeroed attention weight, malicious/unsupported C syntax, nonhalting C loop beyond budget, signed overflows not declared, program with `len(commands) > n-s-m`, too many variables.

**Claims allowed if PASS:** One original analytically-generated fixed-weight neural ISA interpreter successfully executes three compiled programs through tensor neural forward operations; loaded checkpoint remains unchanged, no neural training. **Claims not allowed:** general language intelligence, new unconstrained learned reasoning, native GGUF support, real Vera runtime qualification, Lappy GPU acceleration/efficiency superiority (unless measured independently), broad C correctness outside the tested subset.

## Hostile conclusions

> The most likely false positive is **constructor fallback**: the loader might construct fresh hard-coded weights and appear to pass even when serialized bytes are ignored. Zeroed/altered file tests and hash-diff fresh runs are mandatory.

> The next is **host interpreter leakage**: `compile_and_run` has a C→ISA compiler plus Python driving loop. The driver is allowed to orchestrate neural `.step`, but if any result is computed by a Python ISA emulator, that no longer proves in-weights execution.

> The third is **numerical/semantic inflation**: exact argmax/top-2 and 8-bit bipolar arithmetic are specific to Loom; a generic dense GGUF graph or quantized softmax model will not automatically preserve them. Also, one can represent program semantics faithfully while still being wildly slower than ordinary CPU code.

**Recommended next act:** After the bounded fixed-weight ALU pilot, implement a separate, source-pinned Loom `safetensors` exporter/loader and three fresh-process programs **only after** measuring a 1–3 instruction CPU baseline. A real Loom integration belongs in a future isolated `model-training` PR; Labyrinth remains its evaluator.
