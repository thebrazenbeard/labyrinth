"""Random-initialized, instruction-local neural modular transition learner.

Arithmetic outputs are produced by trainable state logits, NOT by a symbolic
transition routine. The *unchanged* register uses a documented structural copy
prior; the updated register must be learned. No model has graduation authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

MOD = 97
MAX_STEPS = 6
OPS = {"ADD": 0, "SUB": 1, "MUL": 2}
INITIAL = re.compile(r"Initial registers: A=(\d+), B=(\d+)")
INS = re.compile(r"([1-9]\d*)\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)")
STEP = re.compile(r"Step ([1-9]\d*): A=(\d+), B=(\d+)")
PREAMBLE = (
    "Execute this register program in order. ADD, SUB, and MUL update",
    "Execute this register program in order. Each instruction updates",
    "Execute operations.",
)


@dataclass(frozen=True)
class TransitionConfig:
    seed: int = 20261027
    width: int = 64
    max_steps: int = 6
    modulus: int = 97
    temperature: float = 1.0

    def validate(self) -> "TransitionConfig":
        if (type(self.seed) is not int or not 0 <= self.seed < 2**31
                or type(self.width) is not int or not 16 <= self.width <= 128
                or type(self.max_steps) is not int or self.max_steps != 6
                or type(self.modulus) is not int or self.modulus != 97
                or type(self.temperature) is not float
                or not 0.05 <= self.temperature <= 10.0):
            raise ValueError("invalid bounded transition configuration")
        return self


@dataclass(frozen=True)
class TransitionFeatures:
    initial_states: Any
    operations: Any
    targets: Any
    operand_is_register: Any
    operand_values: Any
    mask: Any
    step_numbers: Any


def _parse_question(question: str) -> tuple[tuple[int, int], list[tuple[int, int, bool, int]]]:
    if type(question) is not str or not 1 <= len(question) <= 8192:
        raise ValueError("invalid program")
    lines = question.splitlines()
    if not lines or not any(lines[0].startswith(x) for x in PREAMBLE):
        raise ValueError("unsupported preamble")
    initials = [i for i, line in enumerate(lines) if INITIAL.fullmatch(line)]
    if len(initials) != 1 or not 1 <= initials[0] <= 2:
        raise ValueError("invalid initial state")
    if any(INS.fullmatch(s) for s in lines[:initials[0]]):
        raise ValueError("instruction before initial state")
    starting = INITIAL.fullmatch(lines[initials[0]])
    assert starting
    initial = (int(starting[1]), int(starting[2]))
    if any(not 0 <= x < MOD for x in initial):
        raise ValueError("invalid initial residue")
    result = []
    for line in lines[initials[0] + 1:]:
        found = INS.fullmatch(line)
        if found is None:
            if line.startswith("What is the final value of A?") and line == lines[-1]:
                continue
            raise ValueError("malformed instruction")
        if int(found[1]) != len(result) + 1 or len(result) >= MAX_STEPS:
            raise ValueError("invalid operation sequence")
        operand = found[4]
        as_reg = operand in ("A", "B")
        value = (0 if operand == "A" else 1) if as_reg else int(operand)
        if not as_reg and not 0 <= value < MOD:
            raise ValueError("invalid literal")
        result.append((OPS[found[2]], 0 if found[3] == "A" else 1, as_reg, value))
    if not result:
        raise ValueError("no operations")
    return initial, result


def _teacher_states(initial: tuple[int, int], instructions: Sequence[tuple[int, int, bool, int]]
                    ) -> list[tuple[int, int]]:
    """Supervision-only arithmetic oracle; NEVER invoked inside the neural model."""
    state = list(initial)
    output = []
    for op, target, as_reg, value in instructions:
        operand = state[value] if as_reg else value
        new = ((state[target] + operand) if op == 0 else
               (state[target] - operand) if op == 1 else state[target] * operand) % MOD
        state[target] = new
        output.append(tuple(state))
    return output


def encode_transition_cases(cases: Sequence[Any], config: TransitionConfig, torch: Any
                            ) -> tuple[TransitionFeatures, Any]:
    config.validate()
    if (not isinstance(cases, (list, tuple)) or not 1 <= len(cases) <= 4096):
        raise ValueError("bounded nonempty transition cases required")
    n, steps = len(cases), config.max_steps
    initial = torch.zeros((n, 2), dtype=torch.long)
    operations = torch.zeros((n, steps), dtype=torch.long)
    targets = torch.zeros((n, steps), dtype=torch.long)
    as_reg = torch.zeros((n, steps), dtype=torch.bool)
    operand_values = torch.zeros((n, steps), dtype=torch.long)
    mask = torch.zeros((n, steps), dtype=torch.bool)
    step_numbers = torch.zeros((n, steps), dtype=torch.long)
    expected = torch.zeros((n, steps, 2), dtype=torch.long)
    for i, case in enumerate(cases):
        start, instructions = _parse_question(case.question)
        if (type(case.steps) is not tuple or len(case.steps) != len(instructions)):
            raise ValueError("trace length mismatch")
        for j, line in enumerate(case.steps, 1):
            match = STEP.fullmatch(line)
            if match is None or int(match[1]) != j or any(
                    not 0 <= int(match[k]) < MOD for k in (2, 3)):
                raise ValueError("malformed observed step")
        initial[i] = torch.tensor(start)
        for j, (op, target, register, value) in enumerate(instructions):
            operations[i, j], targets[i, j] = op, target
            as_reg[i, j], operand_values[i, j] = register, value
            mask[i, j], step_numbers[i, j] = True, j + 1
        expected[i, :len(instructions)] = torch.tensor(
            _teacher_states(start, instructions), dtype=torch.long)
    return (TransitionFeatures(initial, operations, targets, as_reg,
                               operand_values, mask, step_numbers), expected)


def derive_first_mismatch(predicted_states: Any, observed_steps: tuple[str, ...]) -> int | None:
    if (type(observed_steps) is not tuple or not 1 <= len(observed_steps) <= MAX_STEPS
            or tuple(predicted_states.shape) != (len(observed_steps), 2)):
        raise ValueError("invalid first-mismatch input")
    for i, line in enumerate(observed_steps, 1):
        match = STEP.fullmatch(line)
        if match is None or int(match[1]) != i:
            raise ValueError("invalid observed state")
        actual = (int(match[2]), int(match[3]))
        if any(not 0 <= value < MOD for value in actual):
            raise ValueError("invalid state residue")
        predicted = tuple(int(x) for x in predicted_states[i - 1].tolist())
        if predicted != actual:
            return i
    return None


def build_transition_model(config: TransitionConfig = TransitionConfig()) -> Any:
    config.validate()
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("Transition V4 neural model requires PyTorch") from e
    torch.manual_seed(config.seed)
    nn = torch.nn
    fourier_width = 9
    input_width = 3 * config.width + 2 * fourier_width + 2

    class TransitionNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.residue_embedding = nn.Embedding(MOD, config.width)
            self.op_embedding = nn.Embedding(3, config.width)
            self.affine = nn.Sequential(
                nn.Linear(input_width, config.width * 2),
                nn.GELU(),
                nn.Linear(config.width * 2, config.width),
                nn.GELU(),
                nn.Linear(config.width, MOD),
            )

        def _numeric(self, residues: Any) -> Any:
            x = residues.to(torch.float32) / MOD
            wave = [x]
            for harmonic in (1, 2, 4, 8):
                wave.extend((torch.sin(2 * math.pi * harmonic * x),
                             torch.cos(2 * math.pi * harmonic * x)))
            return torch.stack(wave, dim=-1)

        def _single(self, prev: Any, op: Any, target: Any,
                    as_reg: Any, operand_index: Any) -> Any:
            targeted = prev.gather(1, target[:, None]).squeeze(1)
            operand = torch.where(as_reg, prev.gather(
                1, operand_index.clamp(0, 1)[:, None]).squeeze(1), operand_index)
            embedded = torch.cat([
                self.residue_embedding(targeted), self.residue_embedding(operand),
                self.op_embedding(op), self._numeric(targeted), self._numeric(operand),
                as_reg.float()[:, None], target.float()[:, None]
            ], dim=-1)
            learned_target_logits = self.affine(embedded) / config.temperature
            copied = torch.full((prev.shape[0], 2, MOD), -14.0,
                                dtype=learned_target_logits.dtype, device=prev.device)
            copied.scatter_(2, prev.unsqueeze(-1), 14.0)
            target_mask = torch.arange(2, device=prev.device)[None, :] == target[:, None]
            return torch.where(target_mask[:, :, None], learned_target_logits[:, None, :],
                               copied)

        def forward(self, features: TransitionFeatures, teacher_states: Any = None
                    ) -> dict[str, Any]:
            batch, max_steps = features.mask.shape
            if (max_steps != MAX_STEPS or features.initial_states.shape != (batch, 2)
                    or features.operations.shape != (batch, MAX_STEPS)):
                raise ValueError("invalid transition feature shapes")
            if teacher_states is not None and teacher_states.shape != (batch, MAX_STEPS, 2):
                raise ValueError("invalid teacher target shape")
            prev = features.initial_states
            all_logits, predicted = [], []
            for j in range(MAX_STEPS):
                logits = self._single(prev, features.operations[:, j], features.targets[:, j],
                                      features.operand_is_register[:, j],
                                      features.operand_values[:, j])
                guesses = logits.argmax(dim=-1)
                present = features.mask[:, j]
                next_state = torch.where(present[:, None], guesses, prev)
                predicted.append(torch.where(present[:, None], guesses,
                                             torch.zeros_like(guesses)))
                all_logits.append(torch.where(present[:, None, None], logits,
                                              torch.zeros_like(logits)))
                if teacher_states is not None:
                    prev = torch.where(present[:, None], teacher_states[:, j, :], prev)
                else:
                    prev = next_state
            return {"state_logits": torch.stack(all_logits, dim=1),
                    "predicted_states": torch.stack(predicted, dim=1),
                    "mask": features.mask}

    return TransitionNet()
