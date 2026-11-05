# Stage 3 — fixed-weight neural ALU executing new program data

This prototype explores a materially different implication of weight-space
programming: an immutable neural network can execute **multiple independently
selected short programs as instruction DATA**, with no weight edits, training,
LLM, optimizer, NumPy, PyTorch, internet or external services. **$0 runtime.**

## Reproduce

```bash
python -m unittest discover -s tests -p 'test_fixed_weight_alu.py' -v
python -m scripts.verify_fixed_weight_alu
python -m scripts.verify_fixed_weight_alu --write artifact-fixed-isa.json
```

The fixed neural network is **5 inputs → 32 ReLU hidden units → 3 outputs**,
with **291 exact-integer numeric parameters**. Its input and output contracts:

| Inputs | Meaning |
| --- | --- |
| `opcode_hi`, `opcode_lo` | 2-bit ISA opcode |
| `operand` | 1-bit immediate / bit index |
| `reg_lo`, `reg_hi` | 2-bit state register, low bit first |
| **Outputs** | `next_lo`, `next_hi`, `valid` |

The tiny ISA's four possible opcodes are:

| Opcode | Semantics |
| --- | --- |
| `00` | NOP, preserve register |
| `01` | Add 0 or 1 (argument) modulo 4 |
| `10` | XOR register bit 0 or 1 (selected by argument) |
| `11` | Reserved/invalid; `valid=0` from the neural network |

The compiler enumerates the complete 5-bit Boolean truth table **once** and
generates exact one-hot ReLU detector weights and output readout coefficients.
The runtime supervisor fetches the next encoded word (three bits), concatenates
it with the current register (two bits), performs neural forward inference,
checks its `valid` output, and advances the program counter. **The supervisor
does not parse opcodes or implement ADD/XOR/NOP**. The ISA reference interpreter
exists solely in the compiler/test code and is absent from forward inference.

## Actual fixed-weight reuse

The command saves and reloads one model, confirms all **32 instruction/state
combinations**, runs three different encoded instruction sequences, and checks
that the model-file bytes and weight hash remain unchanged. The third program
is constructed **after** the model is frozen and reloaded. Both the program
and checkpoint state carry distinct hashes, independent of the model weights.

The `HostSupervisor.snapshot()` returns ISA/schema, program SHA, weight SHA,
PC and state bits. `restore()` replays the instruction prefix through the
numeric neural network and rejects altered state or mismatched provenance.
This is a small, bounded custody proof, **not a secure signature** or
tamper-proof external authority.

The test suite covers all 32 instruction/state combinations, three diverse
programs, both invalid instructions and out-of-domain inputs, mutated numeric
weights changing results, source-interpreter calls forbidden after freeze,
strict PC checks, snapshot replay, corrupt weight manifests, and loading
weights in an independent Python interpreter without compiling them again.

## Measured resource reporting

`scripts.verify_fixed_weight_alu` prints the exact serialized weight-file byte count, measured CPU
and wall-clock time for 10,000 in-process neural steps, and the peak allocation
observed by Python's `tracemalloc`. **Tracemalloc is not a full process-RSS or
GPU VRAM measurement**, and these timings are only local measurements of
Python implementation overhead. No speed, energy, efficiency, compression,
or general-language capability advantage is claimed.

> **Hostile boundary:** This is a *tiny truth-table-synthesized neural ALU with
> external instruction fetch*—not a full neural stored-program computer or
> autonomous VM. The PC, instruction storage, snapshot validation and step
> scheduling live on the Python host. The 2-bit state register has just four
> values; 2^n one-hot compilation scales exponentially. It does, however,
> demonstrate one reusable fixed numerical weight circuit performing different
> computational programs without retraining or changing its weights.
