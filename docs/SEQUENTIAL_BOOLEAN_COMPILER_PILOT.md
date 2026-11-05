# Stage 2: sequential neural composition (no gradients)

**Claim proved:** two separately truth-table-compiled Boolean neural networks
can be connected *sequentially* by algebraically folding their adjoining
numerical affine weight layers. The result is one standalone, multi-layer ReLU
network, whose inference uses only numeric matrix–vector products, biases and
ReLU—no Python source function, interpreter dispatch or intermediate Boolean
oracle. Pure Python 3.10+ standard library; $0 dependencies, compute or API.

## Reproduce

```bash
python -m unittest discover -s tests -p 'test_sequential_boolean_compiler.py' -v
python -m scripts.verify_sequential_boolean_compiler
python -m scripts.verify_sequential_boolean_compiler --write NEW_ARTIFACT.json
```

The demonstration compiles `XOR(a,b)` to 2→4→1, compiles `NOT(xor_bit)` to
1→2→1, then fuses **the output affine of XOR with the input affine of NOT**.
The composed architecture is **2→4→2→1**, with actual ReLU hidden layers.
Expected result XNOR(00)=1, XNOR(01)=0, XNOR(10)=0, XNOR(11)=1.

## The algebra

Given a layer `z=W_left h + b_left` followed by
`ReLU(W_right z+b_right)`, the exact folded layer is

`ReLU((W_right W_left) h + (W_right b_left+b_right))`.

This is a *numerical circuit linker*, not a Python call graph. The ABI matches
output names/width of the first net to input names/width of the second.
Tests independently validate exhaustive reference vs fused weights, separate
two-model inference vs fused inference, wrong ABI rejection, a mutated-weight
negative control, nonbinary input refusal, no source callback in inference,
artifact digest checking and fresh-process reload. The linker **also validates
each component's binary output contract exhaustively before folding**, so a
bad intermediate `(2,)` cannot be silently transformed into a plausible final
binary output. This validates numerical source components separately; it does
not synthesize or enumerate a new combined function truth table.

> **Hostile limit:** both components are only bounded Boolean truth tables,
> needing `2^n` hidden detectors per function. This proves architecture-level
> numerical composition for exact finite circuits; it does not establish a
> scalable source-to-transformer compiler, learned language understanding,
> gradient-free generalized intelligence, GGUF compatibility, speedups or
> safe untrusted execution. Hashes are not signatures.

This complements, rather than replaces, the separate parallel neural-linker
pilot. It is an experimental Labyrinth module pending independent review and runtime qualification; it is not deployed to Vera.
