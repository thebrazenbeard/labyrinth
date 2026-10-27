"""Minotaur V2: random-only, advisory representation for observed traces.

This deliberately small experiment is separate from ``minotaur_scratch`` V1.
It has no checkpoint loader, signing capability, or release authority.  Torch
is imported only by functions that construct or train a model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import re
import time
from typing import Any


SCHEMA = "labyrinth.minotaur.scratch-process-verifier.v2"
MAX_STEPS = 6
MAX_REGISTER_VALUE = 96
VOCAB_SIZE = 160

PAD = 0
CLS = 1
INITIAL_STATE = 2
OBSERVED_STATE = 3
STEP_BASE = 8                 # 8..13 encode the position of each observation
OP_TOKENS = {"ADD": 16, "SUB": 17, "MUL": 18}
REGISTER_TOKENS = {"A": 20, "B": 21}
OPERAND_REGISTER = 22
OPERAND_LITERAL = 23
NUMBER_BASE = 32              # 32..128 encode the modulo-97 register values


@dataclass(frozen=True)
class V2Config:
    """Bounded architecture configuration; all parameters start random."""
    seed: int = 812
    width: int = 64
    heads: int = 4
    layers: int = 2
    max_tokens: int = 96
    abstain_threshold: float = 0.70

    def validate(self) -> "V2Config":
        if (type(self.seed) is not int or not 0 <= self.seed < 2**31
                or type(self.width) is not int or not 16 <= self.width <= 128
                or type(self.heads) is not int or not 1 <= self.heads <= 8
                or self.width % self.heads or type(self.layers) is not int
                or not 1 <= self.layers <= 3 or type(self.max_tokens) is not int
                or not 48 <= self.max_tokens <= 192
                or type(self.abstain_threshold) is not float
                or not 0.50 <= self.abstain_threshold <= 0.99):
            raise ValueError("invalid bounded V2 scratch configuration")
        return self


def tokenize_observed_trace_v2(question: str, steps: tuple[str, ...]) -> list[int]:
    """Encode ordered program steps and their explicit observed states.

    Each state is preceded by ``OBSERVED_STATE`` and a distinct step token.
    The gold correctness/error label is never included in this sequence.
    """
    if type(question) is not str or type(steps) is not tuple or not 1 <= len(steps) <= MAX_STEPS:
        raise ValueError("expected a bounded register program and its step trace")
    initial = re.search(r"Initial registers: A=(\d+), B=(\d+)", question)
    operations = re.findall(r"^\d+\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$",
                            question, flags=re.MULTILINE)
    if initial is None or len(operations) != len(steps):
        raise ValueError("program/trace structure invalid")
    initial_a, initial_b = int(initial[1]), int(initial[2])
    if not (0 <= initial_a <= MAX_REGISTER_VALUE and 0 <= initial_b <= MAX_REGISTER_VALUE):
        raise ValueError("initial register outside modular domain")
    tokens = [CLS, INITIAL_STATE, NUMBER_BASE + initial_a, NUMBER_BASE + initial_b]
    for index, ((operation, target, operand), line) in enumerate(zip(operations, steps), 1):
        state = re.fullmatch(r"Step (\d+): A=(\d+), B=(\d+)", line)
        if state is None or int(state[1]) != index:
            raise ValueError("invalid step trace or order")
        observed_a, observed_b = int(state[2]), int(state[3])
        if not (0 <= observed_a <= MAX_REGISTER_VALUE and 0 <= observed_b <= MAX_REGISTER_VALUE):
            raise ValueError("trace register outside modular domain")
        if operand in REGISTER_TOKENS:
            operand_kind, operand_value = OPERAND_REGISTER, REGISTER_TOKENS[operand]
        else:
            literal = int(operand)
            if not 0 <= literal <= MAX_REGISTER_VALUE:
                raise ValueError("literal outside bounded domain")
            operand_kind, operand_value = OPERAND_LITERAL, NUMBER_BASE + literal
        tokens.extend((STEP_BASE + index - 1, OP_TOKENS[operation], REGISTER_TOKENS[target],
                       operand_kind, operand_value, OBSERVED_STATE,
                       NUMBER_BASE + observed_a, NUMBER_BASE + observed_b))
    if len(tokens) > 96:
        raise ValueError("token budget exceeded; never silently truncate")
    return tokens


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("V2 scratch model requires local PyTorch") from exc
    return torch


def build_v2_model(config: V2Config = V2Config()) -> Any:
    """Create a fresh V2 network; callers control deterministic seeding."""
    config.validate()
    torch = _torch()
    nn = torch.nn

    class V2Minotaur(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tokens = nn.Embedding(VOCAB_SIZE, config.width, padding_idx=PAD)
            self.positions = nn.Embedding(config.max_tokens, config.width)
            layer = nn.TransformerEncoderLayer(
                d_model=config.width, nhead=config.heads, dim_feedforward=4 * config.width,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers,
                                                 enable_nested_tensor=False)
            self.correctness_head = nn.Sequential(nn.LayerNorm(config.width),
                                                  nn.Linear(config.width, 2))
            self.earliest_error_head = nn.Sequential(nn.LayerNorm(config.width),
                                                     nn.Linear(config.width, MAX_STEPS))

        def forward(self, ids: Any) -> dict[str, Any]:
            if ids.ndim != 2 or ids.shape[1] > config.max_tokens:
                raise ValueError("V2 ids must be a bounded [batch, tokens] tensor")
            if bool((ids[:, 0] != CLS).any()):
                raise ValueError("V2 inputs must begin with CLS")
            padding = ids == PAD
            positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            encoded = self.encoder(self.tokens(ids) + self.positions(positions),
                                   src_key_padding_mask=padding)
            pooled = encoded[:, 0, :]
            return {"correctness_logits": self.correctness_head(pooled),
                    "earliest_error_logits": self.earliest_error_head(pooled)}

    return V2Minotaur()


def pad_v2_tokens(sequences: list[list[int]], config: V2Config, torch: Any) -> Any:
    """Create an explicit zero-padded batch; padding is masked by the model."""
    config.validate()
    if not sequences or any(not sequence or len(sequence) > config.max_tokens for sequence in sequences):
        raise ValueError("invalid V2 token batch")
    ids = torch.zeros((len(sequences), max(map(len, sequences))), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return ids


def v2_loss(outputs: dict[str, Any], first_error_steps: Any, torch: Any) -> Any:
    """Train correctness for every trace and location only for incorrect traces."""
    if first_error_steps.ndim != 1 or not bool(((first_error_steps >= 0) &
                                                (first_error_steps <= MAX_STEPS)).all()):
        raise ValueError("first-error labels must be in [0, 6]")
    correctness = (first_error_steps == 0).to(dtype=torch.long)
    correct_loss = torch.nn.functional.cross_entropy(outputs["correctness_logits"], correctness)
    wrong = first_error_steps != 0
    if not bool(wrong.any()):
        return correct_loss
    location_loss = torch.nn.functional.cross_entropy(
        outputs["earliest_error_logits"][wrong], first_error_steps[wrong] - 1)
    return correct_loss + location_loss


def advisory_v2(outputs: dict[str, Any], *, threshold: float = 0.70) -> list[dict[str, Any]]:
    """Return bounded neural advice.  It never grants a release or signs a result."""
    if type(threshold) is not float or not 0.50 <= threshold <= 0.99:
        raise ValueError("advisory threshold must be in [0.50, 0.99]")
    torch = _torch()
    correctness = torch.softmax(outputs["correctness_logits"].detach(), dim=-1)
    locations = torch.softmax(outputs["earliest_error_logits"].detach(), dim=-1)
    result = []
    for row in range(correctness.shape[0]):
        is_correct = int(correctness[row].argmax().item()) == 1
        if is_correct:
            confidence = float(correctness[row, 1].item())
            step = None
        else:
            location_confidence, location = locations[row].max(dim=-1)
            confidence = float((correctness[row, 0] * location_confidence).item())
            step = int(location.item()) + 1
        disposition = ("ADVISE_ALL_CORRECT" if is_correct else "ADVISE_EARLIEST_ERROR")
        if confidence < threshold:
            disposition, step = "ABSTAIN", None
        result.append({"disposition": disposition, "confidence": confidence,
                       "earliest_error_step": step, "graduation_authority": "none",
                       "qualification": "NOT_ESTABLISHED"})
    return result


def parameter_digest_v2(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _fallback_gradient_smoke(config: V2Config, max_seconds: float) -> dict[str, float | int | str]:
    """Dependency-free forward/backward check for the two V2 output heads.

    This is a tiny mean-token embedding network, used only when local Torch is
    unavailable. It keeps the smoke test real (analytic cross-entropy gradients
    update random neural parameters) without pretending to test the optional
    Transformer implementation.
    """
    from .guardian_benchmark import generate_process_cases

    started = time.perf_counter()
    rng = random.Random(config.seed)
    width = config.width
    embeddings = [[rng.uniform(-0.05, 0.05) for _ in range(width)] for _ in range(VOCAB_SIZE)]
    correctness_head = [[rng.uniform(-0.05, 0.05) for _ in range(width)] for _ in range(2)]
    error_head = [[rng.uniform(-0.05, 0.05) for _ in range(width)] for _ in range(MAX_STEPS)]

    def digest() -> str:
        flat = embeddings + correctness_head + error_head
        return hashlib.sha256(";".join(repr(value) for row in flat for value in row).encode()).hexdigest()

    def softmax(logits: list[float]) -> list[float]:
        largest = max(logits)
        values = [math.exp(value - largest) for value in logits]
        total = sum(values)
        return [value / total for value in values]

    before = digest()
    cases = generate_process_cases(6002, 2)[:8]
    loss = 0.0
    for case in cases:
        ids = tokenize_observed_trace_v2(case.question, case.steps)
        feature = [sum(embeddings[token][column] for token in ids) / len(ids)
                   for column in range(width)]
        target = case.first_error_step or 0
        correct_target = int(target == 0)
        correct_probs = softmax([sum(row[i] * feature[i] for i in range(width))
                                 for row in correctness_head])
        loss -= math.log(correct_probs[correct_target])
        correct_gradient = correct_probs[:]
        correct_gradient[correct_target] -= 1.0
        for row, gradient in zip(correctness_head, correct_gradient):
            for column in range(width):
                row[column] -= 0.001 * gradient * feature[column]
        if target:
            error_probs = softmax([sum(row[i] * feature[i] for i in range(width))
                                   for row in error_head])
            error_target = target - 1
            loss -= math.log(error_probs[error_target])
            error_gradient = error_probs[:]
            error_gradient[error_target] -= 1.0
            for row, gradient in zip(error_head, error_gradient):
                for column in range(width):
                    row[column] -= 0.001 * gradient * feature[column]
    elapsed = time.perf_counter() - started
    if not math.isfinite(loss) or elapsed > max_seconds:
        raise RuntimeError("V2 fallback gradient smoke exceeded its bounded budget")
    after = digest()
    if before == after:
        raise RuntimeError("V2 fallback gradient smoke did not update parameters")
    return {"elapsed_seconds": elapsed, "optimizer_steps": 1,
            "initial_weight_sha256": before, "trained_weight_sha256": after,
            "lineage": "random_from_seed", "qualification": "NOT_ESTABLISHED"}


def gradient_smoke_v2(config: V2Config = V2Config(width=16, heads=2, layers=1),
                      *, max_seconds: float = 30.0) -> dict[str, float | int | str]:
    """One real, bounded CPU gradient update using no pretrained parameters."""
    if type(max_seconds) is not float or not 0.1 <= max_seconds <= 30.0:
        raise ValueError("smoke time budget must be in [0.1, 30.0] seconds")
    from .guardian_benchmark import generate_process_cases

    config.validate()
    try:
        torch = _torch()
    except RuntimeError:
        return _fallback_gradient_smoke(config, max_seconds)
    started = time.perf_counter()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(config.seed)
    model = build_v2_model(config)
    cases = generate_process_cases(6002, 2)[:8]
    ids = pad_v2_tokens([tokenize_observed_trace_v2(c.question, c.steps) for c in cases],
                        config, torch)
    labels = torch.tensor([c.first_error_step or 0 for c in cases], dtype=torch.long)
    before = parameter_digest_v2(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = v2_loss(model(ids), labels, torch)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite V2 smoke loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    elapsed = time.perf_counter() - started
    if elapsed > max_seconds:
        raise RuntimeError("V2 gradient smoke exceeded its time budget")
    after = parameter_digest_v2(model)
    if before == after:
        raise RuntimeError("V2 gradient smoke did not update parameters")
    return {"elapsed_seconds": elapsed, "optimizer_steps": 1,
            "initial_weight_sha256": before, "trained_weight_sha256": after,
            "lineage": "random_from_seed", "qualification": "NOT_ESTABLISHED"}
