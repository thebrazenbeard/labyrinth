"""Minotaur v0: an original, randomly initialized neural process verifier.

This is a narrow TransformerEncoder, NOT a pretrained language model and NOT
an admission authority. No Hugging Face weights or adapters are loaded.
Public synthetic register traces are training/development data only.
Optional torch/safetensors imports remain lazy so core Labyrinth needs neither.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .guardian_benchmark import generate_process_cases

SCHEMA = "labyrinth.minotaur.scratch-process-verifier.v1"
NUM_BASE = 16
VOCAB_SIZE = 128
OPS = {"ADD": 4, "SUB": 5, "MUL": 6}
REGS = {"A": 7, "B": 8}


@dataclass(frozen=True)
class ScratchConfig:
    seed: int = 411
    width: int = 64
    heads: int = 4
    layers: int = 2
    max_steps: int = 6
    max_tokens: int = 96

    def validate(self) -> "ScratchConfig":
        if (type(self.seed) is not int or not 0 <= self.seed < 2**31
                or type(self.width) is not int or not 16 <= self.width <= 128
                or type(self.heads) is not int or not 1 <= self.heads <= 8
                or self.width % self.heads or type(self.layers) is not int
                or not 1 <= self.layers <= 3 or type(self.max_steps) is not int
                or self.max_steps != 6 or type(self.max_tokens) is not int
                or not 48 <= self.max_tokens <= 192):
            raise ValueError("invalid bounded scratch transformer configuration")
        return self


def tokenize_process_case(question: str, steps: tuple[str, ...]) -> list[int]:
    """Only the program and observed trace enter the neural input, not the gold label."""
    if type(question) is not str or type(steps) is not tuple or not 1 <= len(steps) <= 6:
        raise ValueError("expected a bounded register program and its step trace")
    initial = re.search(r"Initial registers: A=(\d+), B=(\d+)", question)
    operations = re.findall(r"^\d+\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$",
                            question, flags=re.MULTILINE)
    if initial is None or len(operations) != len(steps):
        raise ValueError("program/trace structure invalid")
    a, b = (int(initial[1]), int(initial[2]))
    if not (0 <= a <= 96 and 0 <= b <= 96):
        raise ValueError("initial register outside modular domain")
    tokens = [1, 2, NUM_BASE + a, NUM_BASE + b]
    for i, ((operation, target, operand), line) in enumerate(zip(operations, steps), 1):
        state = re.fullmatch(r"Step (\d+): A=(\d+), B=(\d+)", line)
        if state is None or int(state[1]) != i:
            raise ValueError("invalid step trace or order")
        new_a, new_b = int(state[2]), int(state[3])
        if not 0 <= new_a <= 96 or not 0 <= new_b <= 96:
            raise ValueError("trace register outside modular domain")
        if operand in REGS:
            op_kind, op_value = 9, REGS[operand]
        else:
            literal = int(operand)
            if not 0 <= literal <= 96:
                raise ValueError("literal outside bounded domain")
            op_kind, op_value = 10, NUM_BASE + literal
        tokens.extend((3, OPS[operation], REGS[target], op_kind, op_value,
                       NUM_BASE + new_a, NUM_BASE + new_b))
    if len(tokens) > 96:
        raise ValueError("token budget exceeded; never silently truncate")
    return tokens


def encoded_process_cases(seed: int, count_per_slice: int,
                          slices: tuple[str, ...]) -> list[dict[str, Any]]:
    allowed = {"fresh_numbers", "new_composition", "longer_program"}
    if (type(seed) is not int or not 0 <= seed < 2**31
            or type(count_per_slice) is not int or not 1 <= count_per_slice <= 128
            or type(slices) is not tuple or not slices
            or len(set(slices)) != len(slices) or not set(slices) <= allowed):
        raise ValueError("invalid dataset seed, count or slice boundary")
    cases = generate_process_cases(seed, count_per_slice=count_per_slice)
    out = []
    for case in cases:
        if case.slice_name in slices:
            tokens = tokenize_process_case(case.question, case.steps)
            out.append({"case_id": case.case_id, "slice": case.slice_name,
                        "tokens": tokens, "label": case.first_error_step or 0})
    return out


def _torch():
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError("scratch training requires local PyTorch and safetensors") from exc
    return torch, load_file, save_file


def _model(config: ScratchConfig, torch: Any):
    nn = torch.nn

    class ScratchMinotaur(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokens = nn.Embedding(VOCAB_SIZE, config.width, padding_idx=0)
            self.positions = nn.Embedding(config.max_tokens, config.width)
            layer = nn.TransformerEncoderLayer(
                d_model=config.width, nhead=config.heads,
                dim_feedforward=4 * config.width,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers,
                                                 enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(config.width),
                                      nn.Linear(config.width, config.max_steps + 1))

        def forward(self, ids: Any):
            pad = ids == 0
            pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            embedded = self.tokens(ids) + self.positions(pos)
            states = self.encoder(embedded, src_key_padding_mask=pad)
            return self.head(states[:, 0, :])

    return ScratchMinotaur()


def _parameter_digest(model: Any) -> str:
    """Stable digest of every initialized/trained neural parameter."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _features(cases: list[dict[str, Any]], length: int, torch: Any):
    ids = torch.zeros((len(cases), length), dtype=torch.long)
    labels = torch.empty((len(cases),), dtype=torch.long)
    for i, item in enumerate(cases):
        values = item["tokens"]
        if len(values) > length:
            raise ValueError("scratch model input exceeds max_tokens")
        ids[i, :len(values)] = torch.tensor(values, dtype=torch.long)
        labels[i] = item["label"]
    return ids, labels


def _evaluate(model: Any, cases: list[dict[str, Any]], config: ScratchConfig,
              torch: Any) -> dict[str, Any]:
    ids, gold = _features(cases, config.max_tokens, torch)
    model.eval()
    with torch.no_grad():
        predictions = torch.cat([
            model(ids[j:j + 32]).argmax(dim=-1)
            for j in range(0, len(ids), 32)
        ])
    pred = predictions.cpu().tolist()
    golds = gold.tolist()
    wrong_cases = sum(label != 0 for label in golds)
    correct_cases = len(golds) - wrong_cases
    false_passes = sum(p == 0 and g != 0 for p, g in zip(pred, golds))
    false_rejects = sum(p != 0 and g == 0 for p, g in zip(pred, golds))
    return {
        "total": len(cases),
        "accuracy": sum(p == g for p, g in zip(pred, golds)) / len(cases),
        "correct_case_count": correct_cases, "wrong_case_count": wrong_cases,
        "first_error_accuracy": (sum(p == g and g != 0 for p, g in zip(pred, golds))
                                 / wrong_cases if wrong_cases else None),
        "false_approval_rate": false_passes / wrong_cases if wrong_cases else None,
        "false_rejection_rate": false_rejects / correct_cases if correct_cases else None,
        "prediction_sha256": hashlib.sha256(bytes(pred)).hexdigest(),
        "qualification": "NOT_ESTABLISHED",
    }


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_scratch(output_dir: str | Path, config: ScratchConfig = ScratchConfig(),
                *, epochs: int = 8, train_per_slice: int = 96,
                eval_per_slice: int = 24, batch_size: int = 32) -> dict[str, Any]:
    """Initialize ALL model weights from a seed and optimize actual neural weights."""
    config.validate()
    if (type(epochs) is not int or not 1 <= epochs <= 30
            or type(train_per_slice) is not int or not 1 <= train_per_slice <= 128
            or type(eval_per_slice) is not int or not 1 <= eval_per_slice <= 128
            or type(batch_size) is not int or not 2 <= batch_size <= 128):
        raise ValueError("invalid bounded scratch training budget")
    dest = Path(output_dir)
    if not dest.is_absolute() or dest.exists() or dest.is_symlink() or not dest.parent.is_dir():
        raise FileExistsError("scratch output must be a new absolute directory")
    train = encoded_process_cases(1101, train_per_slice, ("fresh_numbers", "new_composition"))
    dev = encoded_process_cases(2202, eval_per_slice, ("fresh_numbers", "new_composition"))
    transfer = encoded_process_cases(3303, eval_per_slice, ("longer_program",))
    ids = [c["case_id"] for group in (train, dev, transfer) for c in group]
    if len(ids) != len(set(ids)):
        raise ValueError("train/development/transfer case IDs overlap")
    torch, _, save_file = _torch()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    model = _model(config, torch)
    init_digest = _parameter_digest(model)
    baseline = {"dev": _evaluate(model, dev, config, torch),
                "transfer": _evaluate(model, transfer, config, torch)}
    x, y = _features(train, config.max_tokens, torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    batches = 0
    for epoch in range(epochs):
        perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(
            config.seed + epoch))
        model.train()
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[idx])
            loss = torch.nn.functional.cross_entropy(logits, y[idx])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite scratch training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batches += 1
    final_digest = _parameter_digest(model)
    if init_digest == final_digest:
        raise RuntimeError("no scratch neural parameter updates observed")
    trained = {"dev": _evaluate(model, dev, config, torch),
               "transfer": _evaluate(model, transfer, config, torch)}
    dest.mkdir(mode=0o700)
    weights = dest / "minotaur.safetensors"
    save_file({key: value.cpu().contiguous() for key, value in model.state_dict().items()},
              str(weights))
    report = {
        "schema": SCHEMA, "config": asdict(config), "epochs": epochs,
        "optimizer_steps": batches,
        "lineage": {"initialization": "random_from_seed",
                    "pretrained_checkpoint": None, "adapter": None,
                    "training_data": "public_synthetic_register_traces"},
        "train_cases": len(train), "dev_cases": len(dev),
        "transfer_cases": len(transfer),
        "initial_weight_sha256": init_digest,
        "trained_weight_sha256": final_digest,
        "safetensors_sha256": _hash_file(weights),
        "evaluations": {"frozen_initial": baseline, "trained": trained},
        "eval_prediction_sha256": {
            name: trained[name]["prediction_sha256"] for name in ("dev", "transfer")},
        "graduation_authority": "none", "qualification": "NOT_ESTABLISHED",
        "limitations": [
            "Randomly initialized small process-verifier, not an LLM or pretrained foundation model.",
            "Trained on public finite synthetic structures; not an independent private exam.",
            "Advisory predictions have no authority to graduate Labyrinth trainees.",
        ],
    }
    (dest / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                                     encoding="utf-8")
    return report


def reload_scratch(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir)
    file = root / "minotaur.safetensors"
    manifest = root / "report.json"
    if any(item.is_symlink() for item in (root, file, manifest)):
        raise ValueError("symlinked scratch model artifact rejected")
    report = json.loads(manifest.read_text(encoding="utf-8"))
    if (report.get("schema") != SCHEMA or
            report.get("lineage", {}).get("initialization") != "random_from_seed"
            or report.get("lineage", {}).get("pretrained_checkpoint") is not None
            or _hash_file(file) != report.get("safetensors_sha256")):
        raise ValueError("scratch model manifest or weight hash invalid")
    config = ScratchConfig(**report["config"]).validate()
    torch, load_file, _ = _torch()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = _model(config, torch)
    model.load_state_dict(load_file(str(file), device="cpu"), strict=True)
    digest = _parameter_digest(model)
    if digest != report.get("trained_weight_sha256"):
        raise ValueError("scratch model parameter digest mismatch")
    dev = encoded_process_cases(2202, report["dev_cases"] // 4,
                                ("fresh_numbers", "new_composition"))
    transfer = encoded_process_cases(3303, report["transfer_cases"] // 2,
                                     ("longer_program",))
    observations = {"dev": _evaluate(model, dev, config, torch),
                    "transfer": _evaluate(model, transfer, config, torch)}
    hashes = {name: observations[name]["prediction_sha256"] for name in observations}
    if hashes != report.get("eval_prediction_sha256"):
        raise ValueError("scratch inference changed on checkpoint reload")
    return {"trained_weight_sha256": digest,
            "eval_prediction_sha256": hashes,
            "fresh_process": None, "qualification": "NOT_ESTABLISHED"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Strict scratch Minotaur checkpoint reload")
    parser.add_argument("--reload", required=True)
    args = parser.parse_args()
    print(json.dumps(reload_scratch(args.reload), sort_keys=True))
