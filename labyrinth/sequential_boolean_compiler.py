"""Compile bounded Boolean specifications and compose true neural layers.

Standard library only. No gradients, pretrained model, runtime source callbacks,
external packages, or symbolic dispatch at inference. Not a general compiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from typing import Callable, Iterable


FORMAT = "bounded-boolean-sequential-relu-v1"
MAX_BITS = 10


def _canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _dot(weights: tuple[int, ...], values: tuple[int, ...]) -> int:
    return sum(a * b for a, b in zip(weights, values))


@dataclass(frozen=True)
class DenseLayer:
    """Numeric affine transform and optional ReLU; no embedded user program."""

    weights: tuple[tuple[int, ...], ...]  # [output_width, input_width]
    biases: tuple[int, ...]  # [output_width]
    activation: str

    def __post_init__(self) -> None:
        if self.activation not in ("linear", "relu"):
            raise ValueError("unsupported activation")
        if type(self.weights) is not tuple or not self.weights:
            raise ValueError("empty numeric weight matrix")
        if type(self.biases) is not tuple or len(self.biases) != len(self.weights):
            raise ValueError("biases do not match matrix")
        width = len(self.weights[0])
        if width < 1:
            raise ValueError("zero input width")
        if any(type(row) is not tuple or len(row) != width or
               any(type(weight) is not int for weight in row) for row in self.weights):
            raise ValueError("weight matrix must be rectangular exact integers")
        if any(type(value) is not int for value in self.biases):
            raise ValueError("bias vector must contain exact integers")

    @property
    def input_width(self) -> int:
        return len(self.weights[0])

    @property
    def output_width(self) -> int:
        return len(self.weights)

    def forward(self, values: tuple[int, ...]) -> tuple[int, ...]:
        if len(values) != self.input_width:
            raise ValueError("layer width mismatch")
        linear = tuple(_dot(row, values) + bias for row, bias in zip(self.weights, self.biases))
        return tuple(max(0, value) for value in linear) if self.activation == "relu" else linear


@dataclass(frozen=True)
class NeuralProgram:
    name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    layers: tuple[DenseLayer, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("invalid program name")
        if type(self.input_names) is not tuple or type(self.output_names) is not tuple:
            raise ValueError("ABI must be tuples")
        if (type(self.layers) is not tuple or
            any(type(layer) is not DenseLayer for layer in self.layers)):
            raise ValueError("layers must be immutable numeric DenseLayer tuple")
        if not self.input_names or not self.output_names or not self.layers:
            raise ValueError("empty program")
        for names in (self.input_names, self.output_names):
            if any(type(name) is not str or not name for name in names) or len(set(names)) != len(names):
                raise ValueError("duplicate or invalid ABI labels")
        if len(self.input_names) != self.layers[0].input_width or len(self.output_names) != self.layers[-1].output_width:
            raise ValueError("program ABI does not match first/last layer")
        for before, after in zip(self.layers, self.layers[1:]):
            if before.output_width != after.input_width:
                raise ValueError("incompatible sequential neural tensor shapes")
        if self.layers[-1].activation != "linear":
            raise ValueError("binary output requires linear readout")

    def evaluate(self, inputs: Iterable[int]) -> tuple[int, ...]:
        values = tuple(inputs)
        if len(values) != len(self.input_names) or any(type(value) is not int or value not in (0, 1) for value in values):
            raise ValueError("input outside declared Boolean ABI")
        for layer in self.layers:
            values = layer.forward(values)
        if any(value not in (0, 1) for value in values):
            raise ArithmeticError("compiled output outside Boolean range")
        return values

    def manifest(self) -> dict:
        return {
            "format": FORMAT,
            "name": self.name,
            "abi": {"inputs": list(self.input_names), "outputs": list(self.output_names)},
            "architecture": [len(self.input_names)] + [layer.output_width for layer in self.layers],
            "layers": [
                {"weights": [list(row) for row in layer.weights],
                 "biases": list(layer.biases), "activation": layer.activation}
                for layer in self.layers
            ],
        }

    def sha256(self) -> str:
        return sha256(_canonical(self.manifest())).hexdigest()

    def save(self, path: Path) -> None:
        payload = self.manifest()
        payload["sha256"] = self.sha256()
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load(path: Path) -> NeuralProgram:
    """Load exact neural weights, checking checksum, schema, shape and ABI.

    Hash alone is NOT an authenticity/security proof; the publisher can replace
    both artifact and digest. Production needs a trusted signing authority.
    """
    with path.open("rb") as stream:
        raw = stream.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("manifest too large")
    payload = json.loads(raw, object_pairs_hook=_reject_duplicates)
    if type(payload) is not dict:
        raise ValueError("bad manifest root")
    digest = payload.pop("sha256", None)
    if type(digest) is not str or digest != sha256(_canonical(payload)).hexdigest():
        raise ValueError("manifest checksum mismatch")
    if set(payload) != {"format", "name", "abi", "architecture", "layers"} or payload["format"] != FORMAT:
        raise ValueError("unsupported manifest format")
    abi = payload["abi"]
    if type(abi) is not dict or set(abi) != {"inputs", "outputs"} or any(type(abi[k]) is not list for k in ("inputs", "outputs")):
        raise ValueError("bad ABI")
    if not 1 <= len(abi["inputs"]) <= MAX_BITS:
        raise ValueError("unsupported Boolean input width")
    if type(payload["layers"]) is not list or not 1 <= len(payload["layers"]) <= 16:
        raise ValueError("unsupported layer count")
    if (type(payload["architecture"]) is not list or
        len(payload["architecture"]) != len(payload["layers"]) + 1 or
        any(type(width) is not int or width < 1 for width in payload["architecture"])):
        raise ValueError("invalid architecture vector")
    layers = []
    for entry in payload["layers"]:
        if type(entry) is not dict or set(entry) != {"weights", "biases", "activation"}:
            raise ValueError("bad layer schema")
        layers.append(DenseLayer(tuple(tuple(row) for row in entry["weights"]),
                                 tuple(entry["biases"]), entry["activation"]))
    program = NeuralProgram(payload["name"], tuple(abi["inputs"]),
                            tuple(abi["outputs"]), tuple(layers))
    if payload["architecture"] != [len(program.input_names)] + [layer.output_width for layer in program.layers]:
        raise ValueError("architecture mismatch")
    if program.sha256() != digest:
        raise ValueError("manifest reconstruction changed canonical content")
    return program


def compile_boolean(name: str, input_names: tuple[str, ...], output_names: tuple[str, ...],
                    function: Callable[[tuple[int, ...]], tuple[int, ...]]) -> NeuralProgram:
    """Exhaustively synthesize one-hidden-layer ReLU weights from a truth table."""
    n, m = len(input_names), len(output_names)
    if not 1 <= n <= MAX_BITS or m < 1:
        raise ValueError("Boolean domain exceeds bounded compiler contract")
    inputs = tuple(product((0, 1), repeat=n))
    w1 = tuple(tuple(2 * bit - 1 for bit in pattern) for pattern in inputs)
    b1 = tuple(1 - sum(pattern) for pattern in inputs)
    outputs = tuple(function(pattern) for pattern in inputs)
    if any(type(row) is not tuple or len(row) != m or
           any(type(bit) is not int or bit not in (0, 1) for bit in row) for row in outputs):
        raise ValueError("reference outside Boolean output contract")
    w2 = tuple(tuple(row[col] for row in outputs) for col in range(m))
    return NeuralProgram(name, input_names, output_names,
                         (DenseLayer(w1, b1, "relu"), DenseLayer(w2, (0,) * m, "linear")))


def compose_sequential(name: str, first: NeuralProgram, second: NeuralProgram) -> NeuralProgram:
    """Real neural composition by folding adjacent affine transforms.

    Let z=W_first @ h + b_first and y=ReLU(W_second @ z+b_second).
    Algebraically y=ReLU((W_second @ W_first) @ h +
                         W_second @ b_first + b_second).
    Thus the first network's final affine and second network's initial affine
    become ONE real numerical layer. No symbolic calls, hidden Python function,
    hand-written XNOR formula, or output lookup occur in composed inference.
    """
    if first.output_names != second.input_names:
        raise ValueError("sequential neural ABI mismatch")
    # This is a finite-Boolean-linker contract, not a general-purpose layer
    # composition operation. Algebraic folding is only valid under its output
    # range promise: validate EACH constituent network independently first.
    # Crucially, this evaluates already-compiled numeric weights (never Python
    # source callbacks) and does NOT construct a combined truth table or
    # compile a new composite source function.
    for candidate in (first, second):
        if len(candidate.input_names) > MAX_BITS:
            raise ValueError("component input exceeds bounded semantic domain")
        for domain_input in product((0, 1), repeat=len(candidate.input_names)):
            try:
                candidate.evaluate(domain_input)
            except ArithmeticError as error:
                raise ValueError(f"component {candidate.name} violates binary output contract") from error
    left, right = first.layers[-1], second.layers[0]
    if left.activation != "linear" or right.activation != "relu":
        raise ValueError("fusion requires affine -> ReLU boundary")
    width = left.input_width
    w_fold = tuple(tuple(_dot(right_row, tuple(left.weights[j][i] for j in range(left.output_width)))
                         for i in range(width)) for right_row in right.weights)
    b_fold = tuple(_dot(row, left.biases) + bias for row, bias in zip(right.weights, right.biases))
    fused = DenseLayer(w_fold, b_fold, "relu")
    return NeuralProgram(name, first.input_names, second.output_names,
                         first.layers[:-1] + (fused,) + second.layers[1:])
