"""Tiny fixed-weight conditional program-counter circuit, stacked on the ALU pilot.

The numerical network computes PC+1 or PC+2 (modulo four) depending on a
program bit and whether the *previous* 2-bit register equals zero. The host
still owns instruction memory, the PC register, and execution scheduling.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from types import CodeType

from labyrinth import fixed_weight_alu as alu_module
from labyrinth.fixed_weight_alu import (
    NeuralStepper, UnsupportedInstructionError,
    _assert_bits, _canonical, _reject_duplicate_keys,
)

FORMAT = "exact-relu-neural-pc-5in-v1"
INPUT_ABI = ("skip_if_zero", "reg_lo", "reg_hi", "pc_lo", "pc_hi")
OUTPUT_ABI = ("next_pc_lo", "next_pc_hi")
CANONICAL_SHA256 = "bc81ed5e7b9d8bd5a02b312ac6296848c6e3f103f1ab17dc1072be31822220db"
MAX_ABS_COEFFICIENT = 1024
SNAPSHOT_SCHEMA = "neural-branch-supervisor-snapshot-v1"
MAX_RECOVERY_STEPS = 4096
EXECUTION_ABI = "cpython-neural-branch-execution-bytecode-v1"
_EXECUTION_BUILTINS = ("max", "sum", "zip", "tuple", "any", "len", "type", "int",
                       "range", "str", "isinstance", "enumerate")
_BASELINE_BUILTINS = {name: getattr(builtins, name) for name in _EXECUTION_BUILTINS}


def branch_reference(bits: tuple[int, ...]) -> tuple[int, int]:
    """Synthesis/test oracle only; never called by the execution supervisor."""
    skip, reg_lo, reg_hi, pc_lo, pc_hi = _assert_bits(bits, 5, "branch input")
    pc = pc_lo | (pc_hi << 1)
    step = 2 if skip and not (reg_lo | reg_hi) else 1
    next_pc = (pc + step) % 4
    return next_pc & 1, next_pc >> 1


@dataclass(frozen=True)
class NeuralBranchCircuit:
    w1: tuple[tuple[int, ...], ...]
    b1: tuple[int, ...]
    w2: tuple[tuple[int, ...], ...]
    b2: tuple[int, ...]

    def __post_init__(self) -> None:
        if (type(self.w1) is not tuple or type(self.b1) is not tuple or
            type(self.w2) is not tuple or type(self.b2) is not tuple):
            raise ValueError("branch weights must be immutable tuples")
        if len(self.w1) != 32 or len(self.b1) != 32 or len(self.w2) != 2 or len(self.b2) != 2:
            raise ValueError("invalid branch circuit architecture")
        if any(type(row) is not tuple or len(row) != 5 or
               any(type(v) is not int for v in row) for row in self.w1):
            raise ValueError("invalid branch input weights")
        if any(type(row) is not tuple or len(row) != 32 or
               any(type(v) is not int for v in row) for row in self.w2):
            raise ValueError("invalid branch output weights")
        if any(type(v) is not int for v in self.b1 + self.b2):
            raise ValueError("invalid branch biases")
        if any(abs(v) > MAX_ABS_COEFFICIENT for row in self.w1 + self.w2
               for v in row) or any(abs(v) > MAX_ABS_COEFFICIENT for v in self.b1 + self.b2):
            raise ValueError("branch coefficient exceeds bounded integer domain")

    def forward(self, bits: tuple[int, ...]) -> tuple[int, int]:
        x = _assert_bits(bits, 5, "branch input")
        hidden = tuple(max(0, sum(a * b for a, b in zip(row, x)) + bias)
                       for row, bias in zip(self.w1, self.b1))
        output = tuple(sum(a * b for a, b in zip(row, hidden)) + bias
                       for row, bias in zip(self.w2, self.b2))
        if any(bit not in (0, 1) for bit in output):
            raise ArithmeticError("branch circuit output outside bit ABI")
        return output

    def manifest(self) -> dict:
        return {"format": FORMAT, "architecture": [5, 32, 2],
                "input_abi": list(INPUT_ABI), "output_abi": list(OUTPUT_ABI),
                "activation": "relu", "dtype": "python-exact-integer",
                "w1": [list(row) for row in self.w1], "b1": list(self.b1),
                "w2": [list(row) for row in self.w2], "b2": list(self.b2)}

    def sha256(self) -> str:
        return sha256(_canonical(self.manifest())).hexdigest()

    def save(self, path: Path) -> None:
        payload = self.manifest()
        payload["sha256"] = self.sha256()
        # Exclusive creation: do not overwrite an existing checkpoint/symlink.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def compile_branch() -> NeuralBranchCircuit:
    patterns = tuple(product((0, 1), repeat=5))
    w1 = tuple(tuple(2 * bit - 1 for bit in pattern) for pattern in patterns)
    b1 = tuple(1 - sum(pattern) for pattern in patterns)
    outcomes = tuple(branch_reference(pattern) for pattern in patterns)
    w2 = tuple(tuple(outcome[index] for outcome in outcomes) for index in range(2))
    return NeuralBranchCircuit(w1, b1, w2, (0, 0))


def load_branch(path: Path, expected_sha256: str = CANONICAL_SHA256) -> NeuralBranchCircuit:
    """Check both file integrity and an independently pinned expected weight SHA.

    A self-declared checksum alone cannot authenticate an edited circuit.
    Noncanonical experiments must explicitly supply their trusted expected SHA.
    """
    with path.open("rb") as stream:
        raw = stream.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("branch weight manifest too large")
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    if type(payload) is not dict:
        raise ValueError("invalid branch manifest root")
    stored = payload.pop("sha256", None)
    if type(stored) is not str or stored != sha256(_canonical(payload)).hexdigest():
        raise ValueError("branch manifest checksum mismatch")
    if (type(expected_sha256) is not str or len(expected_sha256) != 64 or
        any(ch not in "0123456789abcdef" for ch in expected_sha256)):
        raise ValueError("expected branch SHA must be lowercase SHA256 hex")
    if stored != expected_sha256:
        raise ValueError("branch manifest differs from pinned expected weights")
    if set(payload) != {"format", "architecture", "input_abi", "output_abi",
                        "activation", "dtype", "w1", "b1", "w2", "b2"}:
        raise ValueError("unknown branch manifest fields")
    if (payload["format"] != FORMAT or payload["architecture"] != [5, 32, 2] or
        payload["input_abi"] != list(INPUT_ABI) or
        payload["output_abi"] != list(OUTPUT_ABI) or
        payload["activation"] != "relu" or payload["dtype"] != "python-exact-integer"):
        raise ValueError("unsupported branch format/ABI")
    return NeuralBranchCircuit(tuple(tuple(row) for row in payload["w1"]),
                               tuple(payload["b1"]), tuple(tuple(row) for row in payload["w2"]),
                               tuple(payload["b2"]))


@dataclass(frozen=True)
class BranchProgram:
    name: str
    # Four slots in a 2-bit PC ring. Each word is branch-bit + ALU ISA word.
    words: tuple[tuple[int, int, int, int], ...]
    initial_state: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("branch program name required")
        if type(self.words) is not tuple or len(self.words) != 4:
            raise ValueError("branch program needs exactly four instruction slots")
        for word in self.words:
            if type(word) is not tuple:
                raise ValueError("branch word must be immutable")
            _assert_bits(word, 4, "branch word")
        _assert_bits(self.initial_state, 2, "initial register")

    def sha256(self) -> str:
        return sha256(_canonical({"format": "neural-branch-program-v1", "name": self.name,
                                   "words": [list(word) for word in self.words],
                                   "initial_state": list(self.initial_state)})).hexdigest()


@dataclass
class BranchSupervisor:
    """Host-managed execution; branch and ALU semantics are numeric only."""

    alu: NeuralStepper
    branch: NeuralBranchCircuit
    program: BranchProgram
    pc: int = 0
    state: tuple[int, int] = (0, 0)
    steps: int = 0

    @classmethod
    def start(cls, alu: NeuralStepper, branch: NeuralBranchCircuit,
              program: BranchProgram) -> BranchSupervisor:
        return cls(alu, branch, program, 0, program.initial_state, 0)

    def _validate(self) -> None:
        if type(self.alu) is not NeuralStepper or type(self.branch) is not NeuralBranchCircuit or type(self.program) is not BranchProgram:
            raise ValueError("invalid branch execution dependencies")
        if type(self.pc) is not int or not 0 <= self.pc < 4:
            raise ValueError("branch PC outside four slots")
        _assert_bits(self.state, 2, "branch supervisor register")
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("invalid branch execution count")

    def step(self) -> tuple[int, tuple[int, int]]:
        self._validate()
        skip, *instruction = self.program.words[self.pc]
        # Branch reads the register BEFORE this instruction. PC stays in host.
        old_state = self.state
        next_pc_lo, next_pc_hi = self.branch.forward((skip,) + old_state +
                                                     (self.pc & 1, self.pc >> 1))
        next_lo, next_hi, valid = self.alu.forward(tuple(instruction) + old_state)
        if valid != 1:
            raise UnsupportedInstructionError(f"neural ALU rejected opcode at pc={self.pc}")
        self.state = (next_lo, next_hi)
        self.pc = next_pc_lo | (next_pc_hi << 1)
        self.steps += 1
        return self.pc, self.state

    def run_steps(self, count: int) -> tuple[int, tuple[int, int]]:
        if type(count) is not int or not 0 <= count <= 4096:
            raise ValueError("bounded step count required; no implicit infinite run")
        for _ in range(count):
            self.step()
        return self.pc, self.state

    def snapshot(self) -> dict:
        """A replay-verifiable host-state receipt, not a cryptographic signature."""
        self._validate()
        if self.steps > MAX_RECOVERY_STEPS:
            raise ValueError("branch state exceeds bounded recovery horizon")
        return {"schema": SNAPSHOT_SCHEMA, "program_sha256": self.program.sha256(),
                "alu_sha256": self.alu.sha256(), "branch_sha256": self.branch.sha256(),
                "execution_sha256": execution_semantics_sha256(),
                "steps": self.steps, "pc": self.pc, "state": list(self.state)}

    @classmethod
    def restore(cls, alu: NeuralStepper, branch: NeuralBranchCircuit,
                program: BranchProgram, receipt: dict) -> BranchSupervisor:
        """Replay numeric transitions and reject impossible or swapped state."""
        if type(alu) is not NeuralStepper or type(branch) is not NeuralBranchCircuit or type(program) is not BranchProgram:
            raise ValueError("invalid branch recovery dependencies")
        expected = {"schema", "program_sha256", "alu_sha256", "branch_sha256",
                    "execution_sha256", "steps", "pc", "state"}
        if type(receipt) is not dict or set(receipt) != expected:
            raise ValueError("invalid branch snapshot fields")
        if receipt["schema"] != SNAPSHOT_SCHEMA:
            raise ValueError("branch snapshot schema mismatch")
        if receipt["execution_sha256"] != execution_semantics_sha256():
            raise ValueError("branch execution semantics changed since snapshot")
        if (receipt["program_sha256"] != program.sha256() or
            receipt["alu_sha256"] != alu.sha256() or
            receipt["branch_sha256"] != branch.sha256()):
            raise ValueError("branch snapshot provenance mismatch")
        if (type(receipt["steps"]) is not int or
            not 0 <= receipt["steps"] <= MAX_RECOVERY_STEPS):
            raise ValueError("branch snapshot recovery bound invalid")
        if type(receipt["pc"]) is not int or not 0 <= receipt["pc"] < 4:
            raise ValueError("branch snapshot PC invalid")
        state = _assert_bits(receipt["state"], 2, "branch snapshot register")
        recovered = cls.start(alu, branch, program)
        recovered.run_steps(receipt["steps"])
        if recovered.pc != receipt["pc"] or recovered.state != state:
            raise ValueError("branch snapshot state disagrees with numeric replay")
        return recovered


def _code_identity(code: CodeType) -> dict:
    """Represent executed CPython code without path/line-number artifacts."""
    return {"bytecode": code.co_code.hex(),
            "exceptions": getattr(code, "co_exceptiontable", b"").hex(),
            "argcount": code.co_argcount, "posonlyargcount": code.co_posonlyargcount,
            "kwonlyargcount": code.co_kwonlyargcount, "flags": code.co_flags,
            "names": list(code.co_names), "varnames": list(code.co_varnames),
            "freevars": list(code.co_freevars), "cellvars": list(code.co_cellvars),
            "consts": [{"code": _code_identity(value)} if isinstance(value, CodeType)
                       else {"literal": repr(value)} for value in code.co_consts]}


def execution_semantics_sha256() -> str:
    """Detect common in-process code drift; NOT authenticated execution.

    The receipt is intentionally interpreter-bytecode-version-specific.
    Runtime monkeypatches of any listed function change this fingerprint.
    This is advisory custody, not a defense against hostile process control.
    """
    # Code-object hashes alone miss shadows of Python builtins: for example,
    # changing this module's `max` global changes future hidden activations
    # without changing NeuralBranchCircuit.forward.__code__. Fail closed when
    # critical primitive lookups change relative to this process's import.
    for module_globals in (globals(), alu_module.__dict__):
        for name, expected in _BASELINE_BUILTINS.items():
            resolved = module_globals.get(name, getattr(builtins, name))
            if resolved is not expected:
                raise ValueError(f"branch execution primitive binding changed: {name}")
    watched = {
        "branch.forward": NeuralBranchCircuit.forward,
        "alu.forward": NeuralStepper.forward,
        "supervisor.step": BranchSupervisor.step,
        "supervisor.run_steps": BranchSupervisor.run_steps,
        "supervisor.start": BranchSupervisor.start.__func__,
        "supervisor.validate": BranchSupervisor._validate,
        "branch.bit_validation": _assert_bits,
        "alu.bit_validation": alu_module._assert_bits,
        "alu.dot": alu_module._dot,
    }
    functions = {}
    for label, func in watched.items():
        code = getattr(func, "__code__", None)
        if not isinstance(code, CodeType):
            raise ValueError(f"uninspectable branch execution function: {label}")
        functions[label] = {"code": _code_identity(code),
                            "defaults": repr(getattr(func, "__defaults__", None)),
                            "kwdefaults": repr(getattr(func, "__kwdefaults__", None))}
    return sha256(_canonical({"abi": EXECUTION_ABI, "functions": functions})).hexdigest()
