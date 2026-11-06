# Stage 4 — numerical conditional PC pilot (stacked on Draft PR #127)

This source-only experiment adds **one fixed 5→32→2 integer/ReLU circuit**
for a 2-bit program counter. Inputs are a program-provided `skip_if_zero`
bit, the **pre-instruction** two-bit register, and the two-bit current PC.
The circuit outputs `PC+2 mod 4` when the skip bit is set and the register
is zero, otherwise `PC+1 mod 4`. Its truth table is compiled into fixed
numeric weights with no gradients or runtime reference-interpreter dispatch.

The existing frozen ALU circuit from #127 separately evaluates one 3-bit
instruction in the same step. The supervisor fetches the four-bit word from
four host-owned program slots, passes inputs into both circuits, then stores
their numeric outputs as the next register and next PC. Different programs,
including programs first sampled **after** frozen weights are persisted, can
execute against the same circuits without recompiling or modifying weights.

Reproduce from the stacked branch:

```bash
python -m unittest discover -s tests -p 'test_fixed_weight_branch.py' -v
python -m unittest discover -s tests -p 'test_fixed_weight_alu*.py' -v
python -O -m unittest discover -s tests -p 'test_fixed_weight_branch.py' -v
```

### Branch-state custody and reproducible recovery

The host supervisor records a monotonically increasing step counter, and
`snapshot()` returns a provenance receipt containing the exact branch-program
hash, fixed ALU-weight hash, branch-weight hash, execution step count, PC,
and register state. An additional `execution_sha256` binds the bytecode of
the current numerical forwards, supervisor step/validation, and bit/dot
helpers; it detects direct method monkeypatches or code changes even when
the weight hashes remain identical. This is Python-runtime-version-specific
and requires a matching execution implementation, not just matching weights.
`restore()` checks the dependencies and replays that many
steps **through the numerical circuits** before accepting the recovered PC
and register. Recovery is bounded to 4,096 prior steps to prevent unbounded
replay of an untrusted receipt; longer-lived sessions require a new recovery
strategy. Test coverage includes fresh-process recovery after checkpoint
reload, resumed loop behavior, source-oracle functions patched to fail,
swapped program and weight identities, tampered PC/state/steps/hashes, and
future-path numerical-forward monkeypatches that would otherwise replay a
valid prefix but diverge on continuation.

> This is *reconstructible state*, not authenticated custody: hashes are
> unkeyed and the receipt is controlled by the host. It establishes that a
> supplied state is consistent with a known program and numerical circuits,
> not that the program or its execution actually occurred independently.
> The code fingerprint is not a trusted measurement of the operating system,
> all imported dependencies, arbitrary globals, or a compromised process.

The branch circuit has 5 input values, 32 one-hot ReLU detector units, and
2 output values (258 exact-integer coefficients and biases). The canonical
manifest content hash is pinned as
`bc81ed5e7b9d8bd5a02b312ac6296848c6e3f103f1ab17dc1072be31822220db`.
`load_branch` checks both the internal manifest checksum and this independent
expected SHA (unless the caller explicitly supplies another trusted SHA),
so simply modifying weights and recomputing the self-declared checksum fails.
This hash is an **identity check**, not cryptographic authorship or host
attestation; a caller changing the trusted pin can authorize altered weights.
Coefficients are bounded and checkpoint save refuses to overwrite a path.

## Falsifiers and strict boundaries

- Exhaustive branch input truth table: 32/32 combinations.
- A test-created, seeded post-freeze set: 100 four-instruction programs,
  24 transitions each, compared with a separate arithmetic oracle while
  the production branch and ALU specification functions are patched to fail.
- Changing **one numeric branch coefficient** changes observed control flow
  with unchanged host supervisor and program; the original circuit remains
  immutable, and its hash differs from the edited copy.
- Recomputed-checksum tampering and wrong trusted pins are rejected, as are
  invalid bit inputs, huge integer weights, out-of-range execution budgets,
  unsupported ALU opcodes and a second write to an existing checkpoint.

> **This is not a standalone neural VM.** The host still owns the actual PC,
> fetches the instruction word, stores state, schedules steps, and bounds
> execution. There is no HALT instruction: the four-slot PC wraps around,
> with a hard 4,096-step execution cap per `run_steps` call. Programs are
> not learned from conversations and no dialogue quality, pretrained-Qwen,
> self-identity, personality, GGUF behavior, hardware speedup or general
> reasoning improvement is demonstrated. One-hot synthesis scales
> exponentially with input-bit width. Future experiments must measure the
> growth against conventional computation and use held-out conversation
> tests before any behavioral claim.
