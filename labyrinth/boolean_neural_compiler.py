"""Bounded Boolean truth-table program -> exact ReLU-network weight compiler.

No gradients, pretrained weights, LLM, PyTorch or NumPy. Strict finite domain.
This is not a general-purpose symbolic compiler or language-model trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from typing import Callable, Iterable


FORMAT_VERSION = "boolean-relu-v1"
MAX_INPUT_BITS = 10  # exponential circuit size: fail before runaway growth


def _canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _validate_matrix(value: tuple[tuple[int, ...], ...], ncols: int) -> None:
    if type(value) is not tuple or any(type(row) is not tuple or len(row) != ncols or
                                        any(type(item) is not int for item in row) for row in value):
        raise ValueError("expected rectangular matrix of exact integer weights")


def _validate_vector(value: tuple[int, ...], size: int) -> None:
    if type(value) is not tuple or len(value) != size or any(type(v) is not int for v in value):
        raise ValueError("expected vector of exact integer biases")


def _dot(row: tuple[int, ...], x: tuple[int, ...]) -> int:
    return sum(a * b for a, b in zip(row, x))


@dataclass(frozen=True)
class NeuralArtifact:
    """Affine -> ReLU -> affine network; immutable exact numerical tensors."""

    name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    w1: tuple[tuple[int, ...], ...]  # [hidden, inputs]
    b1: tuple[int, ...]  # [hidden]
    w2: tuple[tuple[int, ...], ...]  # [outputs, hidden]
    b2: tuple[int, ...]  # [outputs]

    def __post_init__(self) -> None:
        if type(self.input_names) is not tuple or type(self.output_names) is not tuple:
            raise ValueError("ABI names must be immutable tuples")
        if type(self.name) is not str or not self.name:
            raise ValueError("invalid artifact name")
        n, m, h = len(self.input_names), len(self.output_names), len(self.w1)
        if n < 1 or m < 1 or h < 1:
            raise ValueError("empty input/output/hidden ABI")
        if any(type(label) is not str or not label for label in (*self.input_names, *self.output_names)):
            raise ValueError("invalid ABI label")
        if len(set(self.input_names)) != n or len(set(self.output_names)) != m:
            raise ValueError("duplicate ABI names")
        _validate_matrix(self.w1, n)
        _validate_vector(self.b1, h)
        if len(self.w2) != m:
            raise ValueError("wrong output matrix shape")
        _validate_matrix(self.w2, h)
        _validate_vector(self.b2, m)

    def evaluate(self, inputs: Iterable[int]) -> tuple[int, ...]:
        x = tuple(inputs)
        if len(x) != len(self.input_names):
            raise ValueError("input violates ABI shape")
        if any(type(bit) is not int or bit not in (0, 1) for bit in x):
            raise ValueError("only binary inputs in the declared domain are valid")
        # Genuine two-affine-layer ReLU neural inference. The reference program
        # is never dispatched or evaluated after compilation.
        hidden = tuple(max(0, _dot(row, x) + bias) for row, bias in zip(self.w1, self.b1))
        output = tuple(_dot(row, hidden) + bias for row, bias in zip(self.w2, self.b2))
        if any(bit not in (0, 1) for bit in output):
            raise ArithmeticError("network output violated binary contract")
        return output

    def manifest(self) -> dict:
        return {
            "format": FORMAT_VERSION,
            "name": self.name,
            "abi": {"inputs": list(self.input_names), "outputs": list(self.output_names)},
            "architecture": [len(self.input_names), len(self.w1), len(self.output_names)],
            "activation": "relu",
            "dtype": "exact-integer",
            "w1": [list(row) for row in self.w1],
            "b1": list(self.b1),
            "w2": [list(row) for row in self.w2],
            "b2": list(self.b2),
        }

    def sha256(self) -> str:
        return sha256(_canonical_bytes(self.manifest())).hexdigest()

    def write_manifest(self, path: Path) -> None:
        payload = self.manifest()
        payload["sha256"] = self.sha256()
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> NeuralArtifact:
    """Reconstruct weights after checksum, schema, shape and ABI validation.

    Hashes detect accidental modification, not malicious replacement of both
    artifact and hash. Production needs trusted signatures/authorization.
    """
    raw = path.read_bytes()
    if len(raw) > 2_000_000:
        raise ValueError("unreasonably large Boolean neural manifest")
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    if type(payload) is not dict:
        raise ValueError("invalid manifest root")
    digest = payload.pop("sha256", None)
    if type(digest) is not str or digest != sha256(_canonical_bytes(payload)).hexdigest():
        raise ValueError("manifest checksum mismatch")
    expected = {"format", "name", "abi", "architecture", "activation", "dtype", "w1", "b1", "w2", "b2"}
    if set(payload) != expected:
        raise ValueError("unknown or missing manifest fields")
    if payload["format"] != FORMAT_VERSION or payload["activation"] != "relu" or payload["dtype"] != "exact-integer":
        raise ValueError("unsupported neural artifact format")
    abi = payload["abi"]
    if type(abi) is not dict or set(abi) != {"inputs", "outputs"}:
        raise ValueError("invalid manifest ABI")
    if type(payload["name"]) is not str or not payload["name"]:
        raise ValueError("invalid manifest name")
    if type(abi["inputs"]) is not list or type(abi["outputs"]) is not list:
        raise ValueError("invalid manifest input/output ABI")
    if any(type(x) is not str or not x for x in (*abi["inputs"], *abi["outputs"])):
        raise ValueError("invalid manifest ABI label")
    n = len(abi["inputs"])
    if not 1 <= n <= MAX_INPUT_BITS:
        raise ValueError("manifest exceeds bounded Boolean ABI")
    artifact = NeuralArtifact(
        payload["name"], tuple(abi["inputs"]), tuple(abi["outputs"]),
        tuple(tuple(row) for row in payload["w1"]), tuple(payload["b1"]),
        tuple(tuple(row) for row in payload["w2"]), tuple(payload["b2"]),
    )
    architecture = payload["architecture"]
    if type(architecture) is not list or len(architecture) != 3 or any(type(x) is not int for x in architecture) or architecture != [n, len(artifact.w1), len(artifact.output_names)]:
        raise ValueError("manifest architecture mismatch")
    if artifact.sha256() != digest:
        raise ValueError("manifest reconstruction changed canonical content")
    return artifact


def compile_boolean(
    name: str,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    reference: Callable[[tuple[int, ...]], tuple[int, ...]],
) -> NeuralArtifact:
    """Truth-table synthesis into two real affine matrices with ReLU between.

    Hidden unit for pattern p: ReLU(sum_i((2p_i-1)x_i) + 1 - sum_i(p_i)).
    Over binary inputs that is EXACTLY the indicator for x == p.
    """
    n, m = len(input_names), len(output_names)
    if not 1 <= n <= MAX_INPUT_BITS:
        raise ValueError(f"input bit width must be 1..{MAX_INPUT_BITS}")
    if m < 1:
        raise ValueError("need at least one output")
    patterns = tuple(product((0, 1), repeat=n))
    w1 = tuple(tuple(2 * bit - 1 for bit in pattern) for pattern in patterns)
    b1 = tuple(1 - sum(pattern) for pattern in patterns)
    columns = []
    for pattern in patterns:
        result = reference(pattern)
        if type(result) is not tuple or len(result) != m or any(type(bit) is not int or bit not in (0, 1) for bit in result):
            raise ValueError(f"reference output outside Boolean ABI for {pattern}")
        columns.append(result)
    w2 = tuple(tuple(column[i] for column in columns) for i in range(m))
    return NeuralArtifact(name, input_names, output_names, w1, b1, w2, (0,) * m)


def link_parallel(name: str, components: Iterable[NeuralArtifact]) -> NeuralArtifact:
    """Link neural hidden subspaces with same-input parallel ABI only."""
    parts = tuple(components)
    if not parts:
        raise ValueError("cannot link zero components")
    abi = parts[0].input_names
    if any(p.input_names != abi for p in parts):
        raise ValueError("component input ABI mismatch")
    outputs = tuple(out for p in parts for out in p.output_names)
    if len(set(outputs)) != len(outputs):
        raise ValueError("output ABI collision")
    w1 = tuple(row for p in parts for row in p.w1)
    b1 = tuple(value for p in parts for value in p.b1)
    b2 = tuple(value for p in parts for value in p.b2)
    w2_rows = []
    offset, hidden_total = 0, len(w1)
    for p in parts:
        for row in p.w2:
            w2_rows.append((0,) * offset + row + (0,) * (hidden_total - offset - len(p.w1)))
        offset += len(p.w1)
    return NeuralArtifact(name, abi, outputs, w1, b1, tuple(w2_rows), b2)
