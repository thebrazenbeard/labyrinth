"""Seeded, combinatorial register traces for the bounded Minotaur V2 tokenizer.

This is deliberately independent of :mod:`program_maze`: it samples ordered
instructions rather than selecting from a fixed template catalogue.  Labels
remain fields on the returned objects and are never rendered into model input.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import re


_MODULUS = 97
_OPERATIONS = ("ADD", "SUB", "MUL")
_REGISTERS = ("A", "B")
_FAULTS = ("target_delta", "non_target_change", "register_swap",
           "final_target_mutation", "no_change")
_SPLIT_NAMES = ("train", "dev", "structure_disjoint_transfer")


@dataclass(frozen=True)
class CurriculumConfig:
    """Small, strict limits keep every example usable by tokenize_observed_trace_v2."""
    seed: int = 20261027
    train_cases: int = 48
    dev_cases: int = 24
    transfer_cases: int = 24

    def validate(self) -> "CurriculumConfig":
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be a non-boolean 31-bit integer")
        for count in (self.train_cases, self.dev_cases, self.transfer_cases):
            # Paired clean/faulty cases need an even count; five programs gives
            # more than the four structures that motivated this curriculum.
            if type(count) is not int or count % 2 or not 10 <= count <= 512:
                raise ValueError("each split needs an even case count from 10 through 512")
        return self

    def count_for(self, split: str) -> int:
        return {"train": self.train_cases, "dev": self.dev_cases,
                "structure_disjoint_transfer": self.transfer_cases}[split]


@dataclass(frozen=True)
class CombinatorialCase:
    case_id: str
    split: str
    question: str
    steps: tuple[str, ...]
    first_error_step: int | None
    canonical_sequence: str
    fault_family: str


@dataclass(frozen=True)
class CombinatorialSplit:
    """A split plus an opaque SHA-256 commitment to ordered case identities."""
    cases: tuple[CombinatorialCase, ...]
    commitment: str


def _rng(seed: int, split: str, index: int) -> random.Random:
    tag = f"minotaur-combinatorial-v1\0{seed}\0{split}\0{index}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(tag).digest(), "big"))


def _canonical(program: tuple[tuple[str, str, str], ...]) -> str:
    return ";".join(f"{op} {target} {'R' if operand in _REGISTERS else 'C'}"
                    for op, target, operand in program)


def _execute(program: tuple[tuple[str, str, str], ...], initial: tuple[int, int]
             ) -> tuple[tuple[int, int, int], ...]:
    """Generator-side modular executor; parser reexecution below is separate."""
    registers = {"A": initial[0], "B": initial[1]}
    states = []
    for step, (op, target, operand) in enumerate(program, 1):
        value = registers[operand] if operand in _REGISTERS else int(operand)
        if op == "ADD":
            registers[target] = (registers[target] + value) % _MODULUS
        elif op == "SUB":
            registers[target] = (registers[target] - value) % _MODULUS
        elif op == "MUL":
            registers[target] = (registers[target] * value) % _MODULUS
        else:  # Internal programs never reach this branch.
            raise ValueError("unknown operation")
        states.append((step, registers["A"], registers["B"]))
    return tuple(states)


def interpret_question_mod97(question: str) -> tuple[tuple[int, int, int], ...]:
    """Parse and independently re-execute the rendered program modulo 97."""
    if type(question) is not str:
        raise ValueError("question must be text")
    initial = re.search(r"^Initial registers: A=(\d+), B=(\d+)$", question, re.MULTILINE)
    operations = re.findall(r"^(\d+)\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$",
                           question, re.MULTILINE)
    if initial is None or not 1 <= len(operations) <= 6:
        raise ValueError("invalid bounded register program")
    registers = {"A": int(initial.group(1)), "B": int(initial.group(2))}
    if any(not 0 <= value < _MODULUS for value in registers.values()):
        raise ValueError("initial registers outside modular domain")
    output = []
    for position, (printed_step, op, target, operand) in enumerate(operations, 1):
        if int(printed_step) != position:
            raise ValueError("instruction numbering is not consecutive")
        value = registers[operand] if operand in _REGISTERS else int(operand)
        if not 0 <= value < _MODULUS:
            raise ValueError("literal outside modular domain")
        # Kept intentionally separate from _execute so the rendered artifact,
        # not generator state, establishes every returned ground truth state.
        result = {"ADD": registers[target] + value,
                  "SUB": registers[target] - value,
                  "MUL": registers[target] * value}[op]
        registers[target] = result % _MODULUS
        output.append((position, registers["A"], registers["B"]))
    return tuple(output)


def _question(initial: tuple[int, int], program: tuple[tuple[str, str, str], ...]) -> str:
    lines = [
        "Execute this register program in order. Each instruction updates its target using the current operand.",
        "After every instruction take the nonnegative remainder modulo 97.",
        f"Initial registers: A={initial[0]}, B={initial[1]}",
    ]
    lines.extend(f"{index}. {op} {target} {operand}"
                 for index, (op, target, operand) in enumerate(program, 1))
    return "\n".join(lines)


def _make_program(rng: random.Random, program_index: int) -> tuple[tuple[str, str, str], ...]:
    length = 1 + program_index % 6
    items = []
    for _ in range(length):
        op, target = rng.choice(_OPERATIONS), rng.choice(_REGISTERS)
        operand = rng.choice(_REGISTERS) if rng.randrange(2) else str(rng.randrange(2, _MODULUS))
        items.append((op, target, operand))
    # Every program has an ADD or SUB instruction; that provides a guaranteed
    # genuine state transition for the no-change fault family.
    forced = program_index % length
    op, target, operand = items[forced]
    if op == "MUL":
        items[forced] = ("ADD", target, operand)
    return tuple(items)


def _faulty_states(clean: tuple[tuple[int, int, int], ...],
                   program: tuple[tuple[str, str, str], ...], family: str,
                   program_index: int) -> tuple[tuple[tuple[int, int, int], ...], int]:
    if family == "final_target_mutation":
        step = len(clean)
    elif family == "no_change":
        candidates = [i for i, instruction in enumerate(program)
                      if instruction[0] in ("ADD", "SUB")]
        step = candidates[program_index % len(candidates)] + 1
    else:
        step = 1 + (program_index * 5 + 1) % len(clean)
    changed = list(clean)
    number, a, b = changed[step - 1]
    target = program[step - 1][1]
    if family == "target_delta":
        delta = 2 + (program_index * 7) % 31
        a, b = ((a + delta) % _MODULUS, b) if target == "A" else (a, (b + delta) % _MODULUS)
    elif family == "non_target_change":
        delta = 3 + (program_index * 11) % 29
        a, b = (a, (b + delta) % _MODULUS) if target == "A" else ((a + delta) % _MODULUS, b)
    elif family == "register_swap":
        a, b = b, a
        if (a, b) == clean[step - 1][1:]:
            a = (a + 1) % _MODULUS
    elif family == "final_target_mutation":
        # One field on the final state is altered, but it is intentionally not
        # a universal +1 signature that a classifier could memorize.
        delta = 2 + (program_index * 13) % 31
        a, b = ((a + delta) % _MODULUS, b) if target == "A" else (a, (b + delta) % _MODULUS)
    elif family == "no_change":
        a, b = ((clean[step - 2][1], clean[step - 2][2]) if step > 1 else (0, 0))
    else:
        raise ValueError("unknown fault family")
    changed[step - 1] = (number, a, b)
    if changed[step - 1] == clean[step - 1]:
        raise RuntimeError("fault did not alter observed state")
    return tuple(changed), step


def _format_states(states: tuple[tuple[int, int, int], ...]) -> tuple[str, ...]:
    return tuple(f"Step {step}: A={a}, B={b}" for step, a, b in states)


def _case_id(split: str, question: str, steps: tuple[str, ...], fault_family: str) -> str:
    payload = "\0".join((split, question, "\n".join(steps), fault_family)).encode()
    return "comb-" + hashlib.sha256(payload).hexdigest()[:32]


def _make_split(config: CurriculumConfig, split: str, forbidden: set[str]) -> CombinatorialSplit:
    count = config.count_for(split)
    cases: list[CombinatorialCase] = []
    seen: set[str] = set()
    # A retry counter rather than a fixed template list is what makes this a
    # combinatorial sample population; collisions are rejected structurally.
    program_index = 0
    while len(cases) < count:
        rng = _rng(config.seed, split, program_index)
        program = _make_program(rng, program_index)
        canonical = _canonical(program)
        program_index += 1
        if canonical in seen or canonical in forbidden:
            continue
        seen.add(canonical)
        initial = (rng.randrange(1, _MODULUS), rng.randrange(1, _MODULUS))
        question = _question(initial, program)
        generated = _execute(program, initial)
        clean = interpret_question_mod97(question)
        if generated != clean:
            raise RuntimeError("generator/parser differential check failed")
        clean_steps = _format_states(clean)
        clean_id = _case_id(split, question, clean_steps, "none")
        cases.append(CombinatorialCase(clean_id, split, question, clean_steps, None,
                                       canonical, "none"))
        family = _FAULTS[((len(cases) // 2) - 1) % len(_FAULTS)]
        faulty, first_error = _faulty_states(clean, program, family, program_index - 1)
        faulty_steps = _format_states(faulty)
        faulty_id = _case_id(split, question, faulty_steps, family)
        cases.append(CombinatorialCase(faulty_id, split, question, faulty_steps, first_error,
                                       canonical, family))
    ordered = tuple(cases)
    commitment = hashlib.sha256("\n".join(case.case_id for case in ordered).encode()).hexdigest()
    return CombinatorialSplit(ordered, commitment)


def validate_curriculum_case(case: CombinatorialCase) -> None:
    """Verify labels solely by parser reexecution of model-visible content."""
    if not isinstance(case, CombinatorialCase):
        raise ValueError("CombinatorialCase required")
    expected = interpret_question_mod97(case.question)
    if len(case.steps) != len(expected):
        raise ValueError("trace length disagrees with program")
    observed = []
    for position, line in enumerate(case.steps, 1):
        match = re.fullmatch(r"Step (\d+): A=(\d+), B=(\d+)", line)
        if match is None or int(match.group(1)) != position:
            raise ValueError("malformed observed trace")
        values = tuple(map(int, match.groups()))
        if not 0 <= values[1] < _MODULUS or not 0 <= values[2] < _MODULUS:
            raise ValueError("observed value outside modular domain")
        observed.append(values)
    mismatch = next((index for index, (got, want) in enumerate(zip(observed, expected), 1)
                     if got != want), None)
    if mismatch != case.first_error_step:
        raise ValueError("stored label disagrees with independently reexecuted trace")


def build_combinatorial_splits(config: CurriculumConfig = CurriculumConfig()
                               ) -> dict[str, CombinatorialSplit]:
    """Build train, dev, and structure-disjoint transfer data from a seed."""
    config.validate()
    result: dict[str, CombinatorialSplit] = {}
    occupied: set[str] = set()
    all_ids: set[str] = set()
    for split in _SPLIT_NAMES:
        group = _make_split(config, split, occupied)
        structures = {case.canonical_sequence for case in group.cases}
        ids = {case.case_id for case in group.cases}
        if len(structures) <= 4 or len(ids) != len(group.cases):
            raise RuntimeError("split lacks combinatorial diversity or unique case identities")
        if occupied & structures or all_ids & ids:
            raise RuntimeError("split overlap detected")
        if sum(case.first_error_step is None for case in group.cases) * 2 != len(group.cases):
            raise RuntimeError("correct/incorrect cases are not balanced")
        for case in group.cases:
            validate_curriculum_case(case)
        occupied.update(structures)
        all_ids.update(ids)
        result[split] = group
    return result
