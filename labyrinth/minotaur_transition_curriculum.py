"""Causal, step-level register-transition corpus for Minotaur V4.

The visible artifact is a program plus an observed trace.  A faulty trace is
not made by changing one clean row and copying the remainder: after its first
wrong state, every later row is replayed from that wrong state.  The oracle
always compares the observation to a fresh execution of the original program.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import re


_MODULUS = 97
_OPS = ("ADD", "SUB", "MUL")
_REGS = ("A", "B")
_FAMILIES = ("target_delta", "non_target_change", "register_swap",
             "subtle_final_error", "unchanged_op")
_SPLITS = ("train", "dev", "structure_disjoint_transfer")


@dataclass(frozen=True)
class CurriculumConfig:
    """Case counts include both members of each clean/faulty pair.

    Request 256 cases for a split when 128 distinct canonical programs are
    desired.  The small lower bound keeps component and length diversity real.
    """
    seed: int = 20261027
    train_cases: int = 256
    dev_cases: int = 256
    transfer_cases: int = 256

    def validate(self) -> "CurriculumConfig":
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 31:
            raise ValueError("seed must be a non-boolean 31-bit integer")
        for count in (self.train_cases, self.dev_cases, self.transfer_cases):
            if type(count) is not int or count % 2 or not 10 <= count <= 1024:
                raise ValueError("each split needs an even case count from 10 through 1024")
        return self

    def count_for(self, split: str) -> int:
        return {"train": self.train_cases, "dev": self.dev_cases,
                "structure_disjoint_transfer": self.transfer_cases}[split]


@dataclass(frozen=True)
class TransitionCase:
    case_id: str
    split: str
    question: str
    steps: tuple[str, ...]
    # These are supervision fields, never included in question or steps.
    first_error_step: int | None
    canonical_sequence: str
    fault_family: str
    transition_targets: tuple[tuple[int, int], ...]
    transition_mask: tuple[int, ...]


@dataclass(frozen=True)
class TransitionSplit:
    cases: tuple[TransitionCase, ...]
    commitment: str


def _rng(seed: int, *parts: object) -> random.Random:
    payload = "minotaur-transition-v4\0" + "\0".join(map(str, (seed,) + parts))
    return random.Random(int.from_bytes(hashlib.sha256(payload.encode()).digest(), "big"))


def _canonical(program: tuple[tuple[str, str, str], ...]) -> str:
    return ";".join(f"{op} {target} {'R' if operand in _REGS else 'C'}"
                    for op, target, operand in program)


def _question(initial: tuple[int, int], program: tuple[tuple[str, str, str], ...]) -> str:
    lines = [
        "Execute this register program in order. Each instruction updates its target using the current operand.",
        "After every instruction take the nonnegative remainder modulo 97.",
        f"Initial registers: A={initial[0]}, B={initial[1]}",
    ]
    lines.extend(f"{i}. {op} {target} {operand}" for i, (op, target, operand) in enumerate(program, 1))
    return "\n".join(lines)


def replay_mod97(question: str) -> tuple[tuple[int, int], ...]:
    """Independently parse and execute model-visible program text."""
    if type(question) is not str:
        raise ValueError("question must be text")
    initial = re.search(r"^Initial registers: A=(\d+), B=(\d+)$", question, re.MULTILINE)
    instructions = re.findall(r"^(\d+)\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$", question, re.MULTILINE)
    if initial is None or not 1 <= len(instructions) <= 6:
        raise ValueError("invalid bounded register program")
    state = {"A": int(initial.group(1)), "B": int(initial.group(2))}
    if any(not 0 <= value < _MODULUS for value in state.values()):
        raise ValueError("initial registers outside modular domain")
    states = []
    for expected_number, (number, op, target, operand) in enumerate(instructions, 1):
        if int(number) != expected_number:
            raise ValueError("instruction numbering is not consecutive")
        value = state[operand] if operand in _REGS else int(operand)
        if not 0 <= value < _MODULUS:
            raise ValueError("literal outside modular domain")
        if op == "ADD":
            state[target] = (state[target] + value) % _MODULUS
        elif op == "SUB":
            state[target] = (state[target] - value) % _MODULUS
        else:
            state[target] = (state[target] * value) % _MODULUS
        states.append((state["A"], state["B"]))
    return tuple(states)


def _parse_observed(steps: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    states = []
    for position, line in enumerate(steps, 1):
        match = re.fullmatch(r"Step (\d+): A=(\d+), B=(\d+)", line)
        if match is None or int(match.group(1)) != position:
            raise ValueError("malformed observed trace")
        a, b = int(match.group(2)), int(match.group(3))
        if not 0 <= a < _MODULUS or not 0 <= b < _MODULUS:
            raise ValueError("observed value outside modular domain")
        states.append((a, b))
    return tuple(states)


def reconstruct_first_failure(question: str, steps: tuple[str, ...]) -> int | None:
    """Return the first observed state differing from true program execution.

    This deliberately does *not* treat a later trace row as independently
    correct relative to its preceding observed row.  A propagated bad state is
    still wrong whenever it differs from original-program execution.
    """
    expected, observed = replay_mod97(question), _parse_observed(steps)
    if len(expected) != len(observed):
        raise ValueError("trace length disagrees with program")
    return next((i for i, (got, want) in enumerate(zip(observed, expected), 1)
                if got != want), None)


def _apply(state: tuple[int, int], instruction: tuple[str, str, str]) -> tuple[int, int]:
    op, target, operand = instruction
    values = {"A": state[0], "B": state[1]}
    value = values[operand] if operand in _REGS else int(operand)
    values[target] = {"ADD": values[target] + value, "SUB": values[target] - value,
                      "MUL": values[target] * value}[op] % _MODULUS
    return values["A"], values["B"]


def _make_program(rng: random.Random, ordinal: int) -> tuple[tuple[str, str, str], ...]:
    # Ordinal only ensures all six lengths occur; all operation choices are
    # independently random and no fault position is derived from it.
    length = 1 + ordinal % 6
    return tuple((rng.choice(_OPS), rng.choice(_REGS),
                  rng.choice(_REGS) if rng.randrange(2) else str(rng.randrange(0, _MODULUS)))
                 for _ in range(length))


def _mutate(state: tuple[int, int], family: str, target: str, delta: int,
            prior: tuple[int, int]) -> tuple[int, int]:
    a, b = state
    if family == "target_delta" or family == "subtle_final_error":
        return ((a + delta) % _MODULUS, b) if target == "A" else (a, (b + delta) % _MODULUS)
    if family == "non_target_change":
        return (a, (b + delta) % _MODULUS) if target == "A" else ((a + delta) % _MODULUS, b)
    if family == "register_swap":
        return b, a
    if family == "unchanged_op":
        return prior
    raise ValueError("unknown fault family")


def _faulty_trace(program: tuple[tuple[str, str, str], ...], initial: tuple[int, int],
                  clean: tuple[tuple[int, int], ...], family: str,
                  fault_rng: random.Random) -> tuple[tuple[tuple[int, int], ...], int] | None:
    # Position is keyed separately from program ordinal/template.  The final
    # family is explicitly a subtle end-of-trace transcription error.
    position = len(program) - 1 if family == "subtle_final_error" else fault_rng.randrange(len(program))
    prior = initial if position == 0 else clean[position - 1]
    target = program[position][1]
    deltas = list(range(1, _MODULUS))
    fault_rng.shuffle(deltas)
    for delta in deltas:
        altered = _mutate(clean[position], family, target, delta, prior)
        if altered == clean[position]:
            continue
        observed = list(clean[:position]) + [altered]
        state = altered
        for instruction in program[position + 1:]:
            state = _apply(state, instruction)
            observed.append(state)
        # Retain only causal examples: every post-failure row remains wrong
        # compared with the original true execution.
        if all(got != want for got, want in zip(observed[position:], clean[position:])):
            return tuple(observed), position + 1
    return None


def _format(states: tuple[tuple[int, int], ...]) -> tuple[str, ...]:
    return tuple(f"Step {i}: A={a}, B={b}" for i, (a, b) in enumerate(states, 1))


def _case_id(split: str, question: str, steps: tuple[str, ...], family: str) -> str:
    payload = "\0".join((split, question, "\n".join(steps), family)).encode()
    return "transition-" + hashlib.sha256(payload).hexdigest()[:32]


def _make_split(config: CurriculumConfig, split: str, forbidden: set[str]) -> TransitionSplit:
    cases: list[TransitionCase] = []
    structures: set[str] = set()
    ordinal = 0
    while len(cases) < config.count_for(split):
        generator = _rng(config.seed, split, "program", ordinal)
        program = _make_program(generator, ordinal)
        ordinal += 1
        canonical = _canonical(program)
        if canonical in structures or canonical in forbidden:
            continue
        initial = (generator.randrange(_MODULUS), generator.randrange(_MODULUS))
        question = _question(initial, program)
        clean = replay_mod97(question)
        pair_number = len(cases) // 2
        chooser = _rng(config.seed, split, "fault", canonical, pair_number)
        family = chooser.choice(_FAMILIES)
        faulty_result = _faulty_trace(program, initial, clean, family, chooser)
        if faulty_result is None:
            continue
        faulty, first = faulty_result
        clean_steps, faulty_steps = _format(clean), _format(faulty)
        mask = tuple(1 for _ in clean)
        cases.extend((
            TransitionCase(_case_id(split, question, clean_steps, "none"), split, question,
                           clean_steps, None, canonical, "none", clean, mask),
            TransitionCase(_case_id(split, question, faulty_steps, family), split, question,
                           faulty_steps, first, canonical, family, clean, mask),
        ))
        structures.add(canonical)
    ordered = tuple(cases)
    return TransitionSplit(ordered, hashlib.sha256("\n".join(c.case_id for c in ordered).encode()).hexdigest())


def validate_transition_case(case: TransitionCase) -> None:
    """Check labels and transition supervision solely against visible content."""
    if not isinstance(case, TransitionCase):
        raise ValueError("TransitionCase required")
    expected = replay_mod97(case.question)
    if case.transition_targets != expected or case.transition_mask != (1,) * len(expected):
        raise ValueError("transition targets or mask disagree with program")
    if reconstruct_first_failure(case.question, case.steps) != case.first_error_step:
        raise ValueError("stored first failure disagrees with independent replay")


def build_transition_splits(config: CurriculumConfig = CurriculumConfig()) -> dict[str, TransitionSplit]:
    """Build seeded, paired, canonical-structure-disjoint corpus splits."""
    config.validate()
    result: dict[str, TransitionSplit] = {}
    occupied: set[str] = set()
    ids: set[str] = set()
    for split in _SPLITS:
        group = _make_split(config, split, occupied)
        structures = {case.canonical_sequence for case in group.cases}
        group_ids = {case.case_id for case in group.cases}
        if len(structures) != len(group.cases) // 2 or len(group_ids) != len(group.cases):
            raise RuntimeError("pairing or identity collision")
        if occupied & structures or ids & group_ids:
            raise RuntimeError("split overlap detected")
        if sum(c.first_error_step is None for c in group.cases) * 2 != len(group.cases):
            raise RuntimeError("clean/faulty balance failure")
        for case in group.cases:
            validate_transition_case(case)
        occupied.update(structures)
        ids.update(group_ids)
        result[split] = group
    return result
