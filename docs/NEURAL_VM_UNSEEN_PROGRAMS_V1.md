# Wave 2: A real neural interpreter, not a dressed-up truth table

**Research date:** 2026-11-05. **Evidence:** live GitHub source inspection only. No model weights were compiled or executed on Lappy during this subtask. User's $0 rule controls. This is a proposed experiment, not a deployment plan.

## Finding: prioritize a source-defined neural VM

The strongest near-term step beyond finite-domain function compilation is to create a **fixed-weight neural interpreter whose weights encode instruction semantics, while previously unseen programs are loaded as runtime state**. That is a different unit of generalization: the weights need not memorize all functions or be regenerated for each new source program. The machine can implement control flow, indexed memory and arithmetic (within its defined ISA) using the same neural layers for different programs.

Three complementary authentic source implementations:

1. [**Loom**](https://github.com/mkturkcan/Loom), Apache-2.0. The main README declares eight analytically generated Transformer layers implementing 21 opcodes. `c_compiler.py` translates a restricted C subset (integer variables, conditions, `while`, `for`, arrays, no recursion/pointers/division), `loom_v1.py` computes weights directly, `subleq.py` implements the key attention/FFN forward operations, and `test_comprehensive.py` exercises C→ISA→neural execution. A backwards-compatibility `extended_isa_v4.py` shim reexports the canonical `loom_v1.py`, so its old test imports are not automatically stale. Repository reports 8–28 MB weights at three scales, but I did not independently validate byte size. **Strongest direct match** to original, compact, program-agnostic neural functionality without pretrained language weights.
2. [**torchwright**](https://github.com/physicsrob/torchwright), Apache-2.0 v0.2.0. Compiles an explicitly wired Python operation DAG into stock Phi-3 Transformer weights; `compile_hf_bundle` exports a standard `transformers` model without `trust_remote_code`. The `binary_increment` example uses attention and carry propagation; `fibonacci` shows feedback via generated tokens. Claims 16 layers, 512 hidden dimensions and ~10s CPU compile for binary increment. **Better eventual standard checkpoint compatibility**, but different compiled program normally means a *different* weight artifact; not a fixed interpreter. Only FP32 supported; static KV cache and approximation-error budgets. Normal Hugging Face output is a **directory**, not intrinsically a single `.gguf` file.
3. [**Tracr**](https://github.com/google-deepmind/tracr) (archived) and [**ALTA**](https://github.com/google-deepmind/alta), both Apache-2.0. Real source program→attention/MLP weights with compiler and symbolic reference interpreter, but custom/research Transformer runtimes and bounded languages, less suitable as the immediate reusable program-agnostic VM. Their compiler/intermediate-representation design is useful for typed wiring and verification.

[**Qwenputer**](https://github.com/divagr18/qwenputer) is an architectural bridge to fluent LMs, not the first lightweight build: it inserts 11 untrained RISC-V-capable Transformer layers in front of a **pretrained 4B Qwen** and uses Gated DeltaNet state as RAM. Its own README acknowledges huge dense checkpoint overhead despite ~0.7M nonzero circuit weights, plus slow inner neural inference; independent Lappy fit/performance is unproven. It is no evidence that a compact general-language model can be compiled from ordinary software source alone.

## One bounded $0 experiment: `NEURAL_VM_UNSEEN_PROGRAMS_V1`

**Purpose:** verify that a single immutable, original, analytically generated small neural model can execute *multiple programs not present when the model was built*, using recurrent state/program memory, with no optimizer steps, no pretrained LM, and no host execution of the program's operations during inference.

**Proposed owner:** `thebrazenbeard/model-training`, `experiments/neural_vm/`; later read-only evaluation interface in `thebrazenbeard/labyrinth`. Do not modify Lappy's existing TrainingDay Qwen runs. Do not write private code/data into public repository. During early work run entirely CPU in a pinned local Python venv; no need to occupy the RTX 3050 or alter WSL/system configuration.

### Test sequence

1. Freeze exact upstream Loom **commit and Apache-2.0 attribution**, exact package versions and CPU capability. Run its own unit tests first. **Hostile pre-check:** `test_comprehensive.py` contains a known mismatched expectation in `test_array_sum` and later sets `TestCompiledC.test_array_sum = None` to suppress it; do not report `140+ pass` as independently established until test collection and exclusions are audited. Their `argmax` mode is piecewise/discrete, not differentiable standard softmax semantics.
2. Build one 8-layer, `N=8` numerical ISA machine using `LoomComputer`/`LoomConfig`, with `torch.no_grad()` and **no optimizer step or pretrained checkpoint**. Pick a small config that actually fits two short compiled programs (the repo's compiled-C tests use `s=32,m=160,n=512,N=8`, not the 8-memory-slot quickstart). Serialize **only fixed matrices plus config and ABI** into one safe weight container (e.g., `safetensors` with embedded JSON metadata if the loader supports it); no `torch.save` pickle. Record SHA-256, bytes, tensor count, and *actual* resident RAM. Do not call it GGUF unless an exact GGUF loader has been implemented and probed.
3. In independent process A, compile a C subtraction-GCD program such as the repo's established example (positive operands 48,18) into ISA **program/data state**, load immutable weight container and initial state, repeatedly invoke neural `step(X)` until program counter HALT or a strict step cap. Compare final machine memory to a separate Python oracle. Keep model weight digest unchanged.
4. In independent process B, compile an unrelated array-sum or linear search C program into ISA, load the **same sealed weight file** with a different program/data-state digest, execute neural steps, verify output. Disallow a host interpreter of the C/ISA instruction stream in the inference process; the driver may only load state, call neural `step` and observe HALT as existing Loom mechanics require.
5. **Crucial transfer test:** after freezing and hashing the weight container, author a **third previously unseen** short C program with nested conditional/loop/array access inside the supported subset. Compile it using the fixed documented C→ISA translator, execute with *identical weights* and a new program-state hash. Correct result establishes program-level reuse across the chosen ISA, a stronger claim than truth-table enumeration. It does **not** establish open-world reasoning.
6. Rerun all programs under step cap with independent input combinations that do not overflow signed 8-bit range unless wrapped behavior is the tested specification. Wrong answers, nontermination, wrong HALT semantics or stale state count as fails. Restore weights/compile program in fresh processes to avoid accidental Python memory/state reuse.
7. Negative controls: perturb one critical attention/FFN weight; corrupt weights checksum; swap program states; induce a loop exceeding step budget; give an unsupported operation (e.g. C division), out-of-range arithmetic, malformed memory shape, and a program whose compiler exceeds ISA capacity. Each should fail obviously and with the correct error class.
8. Performance: record compile time for new programs, neural instruction throughput, median and p95 end-to-end task latency, peak CPU RSS, checkpoint bytes, and compare with straightforward Python execution. Keep any GPU measurement opt-in and tied to a fresh host-idle probe. Expect the neural VM to be **slower** than Python initially.

**Exact acceptance**: `same_weight_sha256_A == B == C`, `program_state_sha256_A != B != C`, `optimizer_steps == 0`, model executed with a fresh-process forward path for A/B/C, correct outputs across explicitly enumerated seeds/domain, negative controls fail as designed, no paid compute. An evidence receipt must identify source refs, exact tensor hashes, ISA/compiler version, state-loader implementation, runtime backend, test exclusions, numerical precision, and host probe. If stateful file packing is unimplemented, output a **multi-file source prototype**, report single-file checkpoint as `UNKNOWN`, and do not fudge it with ZIP packaging.

## Typed ABI candidate (minimal)

```text
NEURAL_VM_V1 {
    weight_artifact: {schema, safe_format, sha256, tensors, architecture, dtype: float32,
                      ISA: Loom-21-v4, exact_compiler_sha, no_grad_provenance}
    program: {c_source_sha256, ISA_binary_sha256, ISA_version,
              instruction_count, tested_language_subset, memory_address_map}
    machine_state: {shape: [d_model,n], dtype: float32, encoding: bipolar_minus1_plus1,
                    registers, PC, memory, program, initial_sha256, final_sha256}
    step: float32[d_model,n] -> float32[d_model,n]
    supervisor: {host_role: init_state + loop_neural_step + halt/budget + decode,
                 forbidden_role: execute_ISA_opcodes}
    result: {halted, step_count, decoded_output, oracle_comparison,
             weight_sha256_before_after, memory_peak_bytes, wall_seconds}
}
```

The **machine state is mutable program/data/activation state**; its changes are not neural weight updates. A newly installed C program is a configurable/procedural capability, not evidence the network learned that program's domain semantics by gradient descent. Source code is not automatically trusted or granted permission to act on host files. Only an explicit compiler-admitted subset is supported.

## Hostile challenges and failure distinctions

> **Potential pseudo-neural interpreter:** Loom's host-side loop checks PC and calls `.step()` repeatedly. If a benchmark quietly delegates opcode execution to Python or a Torch custom op unrelated to weights, it fails the "in-weights interpreter" claim. Trace the actual tensor path and instrument forbidden calls. Host-level scheduling/stop checking is acceptable but must be described.

> **Tests may overclaim:** The repository's own `test_array_sum` is disabled dynamically after an acknowledged incorrect expected answer; measure actual collected/passed/skipped tests and run unsuppressed target cases. Their `argmax` attention differs from stock Transformer softmax. A small serializable checkpoint is **not** automatically a standard `transformers`/KoboldCPP/GGUF model.

> **No learned language:** Successful C programs on a fixed neural VM establish algorithmic instruction execution in weights. They do not create fluency, commonsense, agency, or a 3 GB personal Vera. For a model that is both conversational and computational, the learned language/semantic subsystem remains a separately evidenced component. Qwenputer demonstrates a possible link, with an expensive dense-model footprint rather than a 3 GB solution.

> **8-bit capacity and correctness:** `N=8` wraps signed arithmetic. Choosing a large array sum and expecting mathematical 150 (outside +127) is wrong. Do not erase correct wrap semantics to make tests green. Branch behavior and numerical stability under repeated steps need long-horizon stress, plus explicit step caps to avoid infinite loops.

> **Single file ≠ self-contained executable:** A `.safetensors` file can hold model matrices (and metadata), but still needs trusted Python/PyTorch interpreter and matching architecture. It is a single **weight artifact**, not a working standalone Vera runtime. A native GGUF/llama.cpp/KoboldCPP route would require a separate architecture/export/quantization port and independent bitwise/behavioral requalification.

## Strategic implication for the wider ecosystem

This route separates **fixed computational laws** (analytic neural weight compiler) from **new procedures** (program source compiled to mutable ISA data) and from **learned semantics** (a future language/identity/skill model). Patrick's repositories can legitimately supply instruction sets, deterministic kernels, test oracles, and governed capability definitions, *without turning every repo into training text*. A Labyrinth suite can try to falsify each compiled capability. A future BT2/HC runtime can arbitrate whether a capability is routable or authorized. The three boundaries must stay distinct.

**No repository mutation, Lappy execution, weight serialization, or PR performed by this research subtask.**
