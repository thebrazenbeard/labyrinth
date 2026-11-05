# Weight-space compilation: bounded, from-scratch pilot

This is an experimental, **training-free, pretrained-model-free** neural compiler.
It compiles exact Boolean specifications into **actual numerical ReLU-network
weight tensors**, links two independently generated networks using an explicit
input/output ABI, and verifies the result against separate reference functions.

The concrete case is a full adder, composed from independently specified
three-input `parity3` and `majority3` circuits. All eight possible inputs are
tested. The fused network has architecture **3 → 16 ReLU → 2**, using ordinary
affine weights and biases. Inference never invokes the reference functions.

## Run on CPU ($0, no model downloads)

Requires Python 3.10+ **standard library only**. No NumPy, PyTorch, LLM APIs,
gradients, optimizer, pretrained model, package installation, or network access.

```bash
python -m unittest discover -s tests -p 'test_boolean_neural_compiler.py' -v
python scripts/verify_boolean_neural_compiler.py
```

`verify.py` prints reference-vs-neural results and reproducible weight hashes,
then (if run with `--write PATH`) saves a canonical JSON manifest. The
`load_manifest` function checks format, ABI, shapes and the serialized digest;
the tests also reload that manifest in a **separate Python process**. No
artifacts are written automatically and no cloud services are contacted.

## How it works

For every binary input pattern `p`, compile a ReLU hidden neuron:

`h_p(x) = ReLU(sum_i((2*p_i - 1)*x_i) + 1 - sum_i(p_i))`

The neuron is `1` exactly when input `x == p`, otherwise `0` for every binary
input. An output matrix encodes the complete truth table. This is exact
**synthesis**, not learned generalization. The linker concatenates disjoint
hidden channels and creates a block-diagonal readout; it rejects ABI conflicts.

## Known limitations / hostile review

- **Exponential scaling:** `2^n` detector neurons per `n`-input function;
  fails with more than ten input bits by design. This does not solve general
  arithmetic scaling or arbitrary software compilation.
- **No learned semantics:** producing exact neural weights for Boolean programs
  does **not** establish a self-created conversational model, intelligence,
  semantic grounding, creative reasoning or open-domain transfer.
- **No compression claim:** manifest and network size are small because the
  tasks are tiny. A source-program length is not resident weight size.
- **No speed advantage:** Python reference code is simpler and likely faster;
  benchmark accuracy, runtime and memory separately before making claims.
- **No low-precision GPU qualification:** exact Python integer arithmetic here
  avoids floating-point rounding, quantization and cache/state bugs. A GGUF,
  GPU, or actual FP16 implementation is a separate test obligation. The tests
  check only Python floating-point equivalence for this tiny network.
- **Trust boundary:** the compiler executes the supplied Python reference
  during synthesis. Only run trusted, inspectable functions in an isolated
  process; there is no safe sandbox for untrusted code.
- **Artifact authenticity:** the manifest checksum detects accidental changes,
  but cannot authenticate an attacker-controlled replacement; production needs
  signed provenance and controlled sources.

### External research context

Related but distinct approaches: Google DeepMind
[Tracr](https://github.com/google-deepmind/tracr),
[ALTA](https://github.com/google-deepmind/alta),
[torchwright](https://github.com/physicsrob/torchwright),
[Loom](https://github.com/mkturkcan/Loom), and
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
This implementation is an original small pedagogical pilot, **not** a claimed
replication or integration of those frameworks.

## Next useful experiment

Implement an honest sequential or attention-based neural compiler with a
specified ABI, show two nontrivial circuits compose, then perform both
holdout and quantized runtime testing. Preserve source/weights/state as
separate, versioned artifacts. Compare with regular Python on latency and
memory. Treat model construction as one possible component of Vera rather
than evidence that GitHub code directly compiles into language understanding.
