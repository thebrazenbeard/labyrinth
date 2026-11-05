"""Train-only, externally verifiable state-transition supervision.

This module derives the target sequence from the *printed* register program,
not from language-model reasoning or hidden evaluator answers. It deliberately
does not import holdout generators or accept an evaluation split.
"""

from __future__ import annotations

import re


_INITIAL = re.compile(r"^Initial registers: A=(\d{1,2}), B=(\d{1,2})$", re.M)
_INSTRUCTION = re.compile(r"^(\d+)\. (ADD|SUB|MUL) ([AB]) ([AB]|\d{1,2})$", re.M)
_NUMBERED_ROW = re.compile(r"^\d+\. .+$", re.M)


def verified_state_trace(record: dict[str, str]) -> str:
    """Return a state target for a verified canonical *train-only* example.

    The independent interpreter checks the existing final label. A corrupted
    label or incompatible prompt raises ValueError rather than laundering a
    contradictory supervision signal into the trace arm.
    """
    if not isinstance(record, dict) or set(record) != {"case_id", "split", "prompt", "answer"}:
        raise ValueError("unexpected training record shape")
    if (record["split"] != "train" or not isinstance(record["case_id"], str)
            or not record["case_id"]):
        raise ValueError("state traces require a train-only case")
    prompt, answer = record["prompt"], record["answer"]
    if not isinstance(prompt, str) or len(prompt) > 4096:
        raise ValueError("invalid prompt")
    if not isinstance(answer, str) or re.fullmatch(r"(?:0|[1-9]\d?)", answer) is None:
        raise ValueError("invalid label")
    matches = list(_INITIAL.finditer(prompt))
    if len(matches) != 1:
        raise ValueError("expected exactly one initial state")
    registers = {"A": int(matches[0][1]), "B": int(matches[0][2])}
    if max(registers.values()) > 96:
        raise ValueError("initial register exceeds modulus")
    instructions = list(_INSTRUCTION.finditer(prompt))
    if not 1 <= len(instructions) <= 8:
        raise ValueError("expected 1..8 instructions")
    if len(instructions) != len(_NUMBERED_ROW.findall(prompt)):
        raise ValueError("unrecognized numbered instruction")
    if [int(m[1]) for m in instructions] != list(range(1, len(instructions) + 1)):
        raise ValueError("instruction numbers are not sequential")
    lines: list[str] = []
    for number, operation, target, operand in (m.groups() for m in instructions):
        value = registers[operand] if operand in registers else int(operand)
        if operation == "ADD":
            registers[target] = (registers[target] + value) % 97
        elif operation == "SUB":
            registers[target] = (registers[target] - value) % 97
        else:
            registers[target] = (registers[target] * value) % 97
        lines.append(f"STATE {number}: A={registers['A']}, B={registers['B']}")
    if int(answer) != registers["A"]:
        raise ValueError("train answer contradicts the program")
    return "\n".join([*lines, f"ANSWER: {registers['A']}"])
