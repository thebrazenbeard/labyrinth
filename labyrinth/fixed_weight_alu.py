"""A fixed-weight, truth-table-synthesized neural ALU instruction stepper.

Two-bit opcode, one-bit argument, two-bit register state -> two new state bits
and a valid-operation bit. All computation of opcode semantics at *runtime*
happens inside numerical affine->ReLU->affine weights. Standard library only.
The host fetches encoded instructions and advances PC; it never decodes opcodes.

This is NOT a general neural VM, transformer, or learned reasoning model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from typing import Iterable


ISA = "tiny-alu-v1"
FORMAT = "exact-relu-truthtable-5in-v1"
SNAPSHOT_SCHEMA = "neural-supervisor-snapshot-v1"
MAX_WORDS = 4096
INPUT_ABI = ("opcode_hi", "opcode_lo", "operand", "reg_lo", "reg_hi")
OUTPUT_ABI = ("next_lo", "next_hi", "valid")


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key in neural weights")
        result[key] = value
    return result


def _assert_bits(value: Iterable[int], width: int, label: str) -> tuple[int, ...]:
    bits = tuple(value)
    if len(bits) != width or any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        raise ValueError(f"{label} must be exactly {width} binary integers")
    return bits


def _dot(weights: tuple[int, ...], inputs: tuple[int, ...]) -> int:
    return sum(weight * x for weight, x in zip(weights, inputs))


def opcode_reference(bits: tuple[int, ...]) -> tuple[int, ...]:
    """Compiler-only specification; never called by the execution supervisor.

    ISA opcode 00 NOP (argument ignored); 01 ADD operand modulo 4;
    10 XOR one register bit indexed by operand; 11 INVALID.
    Register value is little-endian two-bit unsigned integer.
    """
    op_hi, op_lo, operand, reg_lo, reg_hi = bits
    opcode = (op_hi << 1) | op_lo
    register = reg_lo | (reg_hi << 1)
    if opcode == 0:  # NOP
        next_register, valid = register, 1
    elif opcode == 1:  # ADD one-bit immediate (0 or 1), modulo 4
        next_register, valid = (register + operand) % 4, 1
    elif opcode == 2:  # XOR one register bit selected by argument
        next_register, valid = register ^ (1 << operand), 1
    else:  # Reserved opcode 11; invalid signalled BY compiled neural output
        next_register, valid = register, 0
    return (next_register & 1, (next_register >> 1) & 1, valid)


@dataclass(frozen=True)
class NeuralStepper:
    """Numeric weights only, with immutable tensor rows and explicit ABI."""

    w1: tuple[tuple[int, ...], ...]  # [hidden=32, in=5]
    b1: tuple[int, ...]
    w2: tuple[tuple[int, ...], ...]  # [out=3, hidden=32]
    b2: tuple[int, ...]

    def __post_init__(self) -> None:
        hidden = len(self.w1)
        if hidden != 32 or len(self.b1) != hidden or len(self.w2) != 3 or len(self.b2) != 3:
            raise ValueError("wrong neural architecture")
        if (type(self.w1) is not tuple or type(self.b1) is not tuple or
            type(self.w2) is not tuple or type(self.b2) is not tuple):
            raise ValueError("weights must be immutable tuples")
        for matrix, width in ((self.w1, 5), (self.w2, hidden)):
            if any(type(row) is not tuple or len(row) != width or
                   any(type(weight) is not int for weight in row) for row in matrix):
                raise ValueError("nonintegral or malformed neural matrix")
        if any(type(value) is not int for value in self.b1 + self.b2):
            raise ValueError("biases must be exact integers")

    def forward(self, input_bits: Iterable[int]) -> tuple[int, int, int]:
        x = _assert_bits(input_bits, 5, "neural input")
        h = tuple(max(0, _dot(row, x) + bias) for row, bias in zip(self.w1, self.b1))
        y = tuple(_dot(row, h) + bias for row, bias in zip(self.w2, self.b2))
        if any(bit not in (0, 1) for bit in y):
            raise ArithmeticError("neural output outside binary ABI")
        return y

    def manifest(self) -> dict:
        return {
            "format": FORMAT,
            "isa": ISA,
            "architecture": [5, 32, 3],
            "input_abi": list(INPUT_ABI),
            "output_abi": list(OUTPUT_ABI),
            "activation": "relu",
            "dtype": "python-exact-integer",
            "w1": [list(row) for row in self.w1],
            "b1": list(self.b1),
            "w2": [list(row) for row in self.w2],
            "b2": list(self.b2),
        }

    def sha256(self) -> str:
        return sha256(_canonical(self.manifest())).hexdigest()

    def save(self, path: Path) -> None:
        payload = self.manifest()
        payload["sha256"] = self.sha256()
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def compile_stepper() -> NeuralStepper:
    """One-time exhaustive Boolean synthesis; no gradient optimization."""
    patterns = tuple(product((0, 1), repeat=5))
    w1 = tuple(tuple(2 * bit - 1 for bit in pattern) for pattern in patterns)
    b1 = tuple(1 - sum(pattern) for pattern in patterns)
    table = tuple(opcode_reference(pattern) for pattern in patterns)
    w2 = tuple(tuple(row[out] for row in table) for out in range(3))
    return NeuralStepper(w1, b1, w2, (0, 0, 0))


def load_stepper(path: Path) -> NeuralStepper:
    """Reload weights alone; no compiler or opcode implementation required."""
    with path.open("rb") as stream:
        raw = stream.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("neural weight manifest too large")
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    if type(payload) is not dict:
        raise ValueError("invalid manifest root")
    stored = payload.pop("sha256", None)
    if type(stored) is not str or stored != sha256(_canonical(payload)).hexdigest():
        raise ValueError("neural weight checksum mismatch")
    expected = {"format", "isa", "architecture", "input_abi", "output_abi",
                "activation", "dtype", "w1", "b1", "w2", "b2"}
    if set(payload) != expected:
        raise ValueError("unknown neural weight manifest fields")
    if (payload["format"] != FORMAT or payload["isa"] != ISA or
        payload["activation"] != "relu" or payload["dtype"] != "python-exact-integer"):
        raise ValueError("unsupported neural weight format")
    if any(type(v) is not int for v in payload["architecture"]) or payload["architecture"] != [5, 32, 3]:
        raise ValueError("architecture mismatch")
    if payload["input_abi"] != list(INPUT_ABI) or payload["output_abi"] != list(OUTPUT_ABI):
        raise ValueError("neural ABI mismatch")
    return NeuralStepper(tuple(tuple(row) for row in payload["w1"]), tuple(payload["b1"]),
                         tuple(tuple(row) for row in payload["w2"]), tuple(payload["b2"]))


class UnsupportedInstructionError(Exception):
    """Valid encoding outside supported ISA, reported by the neural valid bit."""


@dataclass(frozen=True)
class EncodedProgram:
    """Only encoded instruction DATA; never compiled into or stored with weights."""

    name: str
    words: tuple[tuple[int, int, int], ...]  # op-hi, op-lo, argument
    initial_state: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("program name required")
        if type(self.words) is not tuple or not 1 <= len(self.words) <= MAX_WORDS:
            raise ValueError("program must be a nonempty immutable instruction sequence")
        for word in self.words:
            if type(word) is not tuple:
                raise ValueError("encoded instructions must be tuples")
            _assert_bits(word, 3, "instruction word")
        _assert_bits(self.initial_state, 2, "initial register")

    def sha256(self) -> str:
        data = {"format": "encoded-neural-program-v1", "isa": ISA,
                "name": self.name, "words": [list(w) for w in self.words],
                "initial_state": list(self.initial_state)}
        return sha256(_canonical(data)).hexdigest()


@dataclass
class HostSupervisor:
    """Fetches instruction bytes, applies fixed neural step, advances PC.

    No opcode decode, arithmetic semantics, fallback branch by instruction type,
    or reference spec at runtime. The only branch is on neural `valid` output.
    """

    weights: NeuralStepper
    program: EncodedProgram
    pc: int = 0
    state: tuple[int, int] = field(default_factory=lambda: (0, 0))

    def __post_init__(self) -> None:
        self._validate_runtime()

    def _validate_runtime(self) -> None:
        if type(self.weights) is not NeuralStepper or type(self.program) is not EncodedProgram:
            raise ValueError("invalid fixed neural weights/program identity")
        if type(self.pc) is not int or not 0 <= self.pc <= len(self.program.words):
            raise ValueError("supervisor PC outside program bounds")
        _assert_bits(self.state, 2, "supervisor register")

    @classmethod
    def start(cls, weights: NeuralStepper, program: EncodedProgram) -> HostSupervisor:
        return cls(weights, program, 0, program.initial_state)

    def step(self) -> tuple[int, int]:
        self._validate_runtime()  # guard against field mutation between steps
        if self.pc == len(self.program.words):
            raise StopIteration("program complete")
        instruction = self.program.words[self.pc]  # no semantic decode
        next_lo, next_hi, valid = self.weights.forward(instruction + self.state)
        if valid != 1:
            raise UnsupportedInstructionError(f"neural valid bit rejected instruction at pc={self.pc}")
        self.state = (next_lo, next_hi)
        self.pc += 1
        return self.state

    def run(self) -> tuple[int, int]:
        self._validate_runtime()
        while self.pc < len(self.program.words):
            self.step()
        return self.state

    def snapshot(self) -> dict:
        """Execution state is a separate provenance object, not weight mutation."""
        return {"schema": SNAPSHOT_SCHEMA, "isa": ISA,
                "program_sha256": self.program.sha256(),
                "weights_sha256": self.weights.sha256(),
                "pc": self.pc, "state": list(self.state)}

    @classmethod
    def restore(cls, weights: NeuralStepper, program: EncodedProgram, state: dict) -> HostSupervisor:
        """Validate snapshot by replay of numeric prefix, not trust in stored state."""
        if type(state) is not dict or set(state) != {"schema", "isa", "program_sha256", "weights_sha256", "pc", "state"}:
            raise ValueError("invalid state snapshot fields")
        if state["schema"] != SNAPSHOT_SCHEMA or state["isa"] != ISA:
            raise ValueError("snapshot schema/ISA mismatch")
        if state["weights_sha256"] != weights.sha256() or state["program_sha256"] != program.sha256():
            raise ValueError("snapshot provenance mismatch")
        if type(state["pc"]) is not int or not 0 <= state["pc"] <= len(program.words):
            raise ValueError("snapshot PC invalid")
        bits = _assert_bits(state["state"], 2, "snapshot register")
        validated = cls.start(weights, program)
        for _ in range(state["pc"]):
            validated.step()
        if validated.state != bits:
            raise ValueError("snapshot state does not match replayed neural execution")
        return validated
