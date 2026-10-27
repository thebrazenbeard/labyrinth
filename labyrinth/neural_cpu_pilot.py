"""Bounded CPU supervised update of an existing pretrained causal LM.

An intentionally small, dependency-light research pilot, not a claim of
transferable reasoning. Run each arm in its own isolated output directory.
Only train-split records may be passed to this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import time

if __package__:
    from .state_traces import verified_state_trace
else:  # Direct execution on an isolated host without installing a package.
    from state_traces import verified_state_trace


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def layer_digest(layer) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(layer.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read_training(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    if not 1 <= len(raw) <= 1024 * 1024:
        raise ValueError("training file size must be 1..1048576 bytes")
    records = json.loads(raw)
    if not isinstance(records, list) or not 1 <= len(records) <= 256:
        raise ValueError("expected 1..256 train-split examples")
    ids = set()
    for item in records:
        if not isinstance(item, dict) or set(item) != {"case_id", "split", "prompt", "answer"}:
            raise ValueError("unexpected training record fields")
        if item["split"] != "train":
            raise ValueError("evaluation examples cannot enter training")
        if not isinstance(item["case_id"], str) or not item["case_id"] or item["case_id"] in ids:
            raise ValueError("duplicate or invalid case ID")
        ids.add(item["case_id"])
        if not isinstance(item["prompt"], str) or not 1 <= len(item["prompt"]) <= 4096:
            raise ValueError("invalid prompt")
        if not isinstance(item["answer"], str) or re.fullmatch(r"[0-9]{1,3}", item["answer"]) is None:
            raise ValueError("expected nonnegative decimal answer")
    return records


def attest_generated_train_only(records: list[dict[str, str]], *,
                                generator_seed: int) -> list[dict[str, str]]:
    """Authenticate ordered SFT inputs against the source-bound train generator.

    The ``split`` field alone is user-controlled: a held-out record can be
    relabelled.  Reconstruct every expected record from the *train* generator,
    then compare all fields before a model or optimizer is loaded.  Do not
    silently downgrade to split-field-only checking if source is unavailable.
    """
    if type(generator_seed) is not int or not isinstance(records, list) or not 1 <= len(records) <= 256:
        raise ValueError("invalid train generator attestation arguments")
    try:
        from labyrinth.program_maze import generate_program_challenges
    except ImportError as exc:
        raise ValueError("train generator source unavailable for attestation") from exc
    expected = generate_program_challenges(generator_seed, len(records), "train")
    for index, (record, challenge) in enumerate(zip(records, expected)):
        required = {"case_id": challenge.id, "split": "train",
                    "prompt": challenge.prompt, "answer": challenge.answer}
        if record != required:
            raise ValueError(f"training record {index} does not match train generator")
    return records


def supervised_target(item: dict[str, str], *, policy: str, arm: str,
                      step: int) -> str:
    """Derive train-only supervised suffix, never an evaluator oracle.

    Corrupted-state targets perturb every register observation and final
    answer while leaving instruction counters and syntax unchanged.
    """
    if policy not in ("answer", "trace") or arm not in ("trained", "sham", "corrupted"):
        raise ValueError("unknown supervised target policy or arm")
    if type(step) is not int or step < 0:
        raise ValueError("invalid step")
    if policy == "answer":
        value = item["answer"]
        if arm == "corrupted":
            value = str((int(value) + 1 + step % 7) % 97)
        return "ANSWER: " + value
    text = verified_state_trace(item)
    if arm == "corrupted":
        offset = 1 + step % 7
        text = re.sub(
            r"(\bA=|\bB=|ANSWER: )(\d+)",
            lambda m: m.group(1) + str((int(m.group(2)) + offset) % 97),
            text,
        )
    return text


_ONE_LINE_REQUEST = (
    "What is the final value of A? Respond with exactly one line: ANSWER: <integer>."
)
_TRACE_REQUEST = (
    "Show the updated values of both registers after each numbered instruction, "
    "in order, using one line per instruction: STATE N: A=<integer>, B=<integer>. "
    "Then finish with exactly one line: ANSWER: <integer>."
)


def training_prefix(item: dict[str, str], *, policy: str) -> str:
    """Align the written task instruction with its supervised output contract."""
    if policy == "answer":
        return item["prompt"] + "\n"
    if policy != "trace" or not item["prompt"].endswith(_ONE_LINE_REQUEST):
        raise ValueError("trace policy requires an explicit known final-answer prompt")
    return item["prompt"][:-len(_ONE_LINE_REQUEST)] + _TRACE_REQUEST + "\n"


def train(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # A larger pretrained CPU model needs a bounded longer pilot to revisit
    # more independent train-only cases; still cap this at 128 optimizer steps.
    if not 1 <= args.steps <= 128 or not 1 <= args.threads <= 8:
        raise ValueError("steps or CPU threads outside safe pilot bounds")
    base = Path(args.base).resolve(strict=True)
    data = Path(args.train_data).resolve(strict=True)
    output = Path(args.output).resolve()
    if not base.is_dir() or not (base / "model.safetensors").is_file():
        raise ValueError("a local safetensors pretrained-model directory is required")
    if output.exists():
        raise FileExistsError("pilot refuses to overwrite an existing run")
    examples = attest_generated_train_only(
        read_training(data), generator_seed=args.train_generator_seed)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True,
                                               trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True,
                                                trust_remote_code=False,
                                                use_safetensors=True)
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("pilot supports model.model.layers transformer blocks only")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    block = model.model.layers[-1]
    for parameter in block.parameters():
        parameter.requires_grad_(True)
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not 100_000 <= trainable_count < parameter_count:
        raise ValueError("not enough trainable neural parameters or model not frozen")
    before = layer_digest(block)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=0.0)
    model.train()
    losses = []
    supervised_tokens = []
    started = time.monotonic()
    for step in range(args.steps):
        item = examples[step % len(examples)]
        prefix = training_prefix(item, policy=args.target_policy)
        suffix = supervised_target(item, policy=args.target_policy, arm=args.arm,
                                   step=step) + (tokenizer.eos_token or "")
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        ids = tokenizer(prefix + suffix, add_special_tokens=False).input_ids
        if ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("tokenizer prefix boundary changes on concatenation")
        if len(ids) > (384 if args.target_policy == "trace" else 256) or len(prefix_ids) >= len(ids):
            raise ValueError("train example token length outside bounded pilot")
        supervised_tokens.append(len(ids) - len(prefix_ids))
        input_ids = torch.tensor([ids], dtype=torch.long)
        labels = input_ids.clone()
        labels[:, :len(prefix_ids)] = -100
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, labels=labels).loss
        if not torch.isfinite(loss):
            raise ValueError("non-finite loss")
        losses.append(float(loss.detach()))
        if args.arm != "sham":
            loss.backward()
            optimizer.step()
        else:
            # Call with no gradients: same number of case visits and step calls,
            # but no parameter update. The receipt says zero gradient steps.
            optimizer.step()
    after = layer_digest(block)
    if (before == after) != (args.arm == "sham"):
        raise ValueError("parameter-change invariant failed")

    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    receipt = {
        "schema": "labyrinth.cpu-neural-pilot.v2" if args.target_policy == "trace" else "labyrinth.cpu-neural-pilot.v1",
        "arm": args.arm, "seed": args.seed, "lr": args.lr,
        "target_policy": args.target_policy,
        "supervised_token_count": sum(supervised_tokens),
        "supervised_token_lengths": supervised_tokens,
        "optimizer": "AdamW", "optimizer_step_calls": args.steps,
        "gradient_steps": 0 if args.arm == "sham" else args.steps,
        "case_visits": args.steps, "unique_training_cases": len(examples),
        "training_data_sha256": file_digest(data),
        "train_generator": "labyrinth-program-v1", "train_generator_seed": args.train_generator_seed,
        "base_model_sha256": file_digest(base / "model.safetensors"),
        "base_tokenizer_sha256": file_digest(base / "tokenizer.json"),
        "model_parameters": parameter_count, "trainable_parameters": trainable_count,
        "last_block_sha256_before": before,
        "last_block_sha256_after": after,
        "checkpoint_sha256": file_digest(output / "model.safetensors"),
        "losses": losses, "elapsed_seconds": round(time.monotonic() - started, 3),
        "torch_version": torch.__version__,
    }
    (output / "train_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)


def verify(args) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = Path(args.checkpoint).resolve(strict=True)
    receipt = json.loads((checkpoint / "train_receipt.json").read_text(encoding="utf-8"))
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), local_files_only=True,
                                                trust_remote_code=False,
                                                use_safetensors=True)
    tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                        trust_remote_code=False)
    actual = layer_digest(model.model.layers[-1])
    expected = receipt["last_block_sha256_after"]
    if actual != expected or file_digest(checkpoint / "model.safetensors") != receipt["checkpoint_sha256"]:
        raise ValueError("fresh-process neural checkpoint reload digest mismatch")
    print(json.dumps({"fresh_process_reload": "PASS", "layer_digest": actual,
                      "vocab_size": len(tok), "checkpoint_sha256": receipt["checkpoint_sha256"]},
                     sort_keys=True), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Real pretrained CPU SFT, resource-bounded")
    sub = parser.add_subparsers(dest="command", required=True)
    ptrain = sub.add_parser("train")
    ptrain.add_argument("--base", required=True)
    ptrain.add_argument("--train-data", required=True)
    ptrain.add_argument("--train-generator-seed", required=True, type=int,
                        help="Source-bound train-generator seed; checked before neural loading")
    ptrain.add_argument("--output", required=True)
    ptrain.add_argument("--arm", choices=("trained", "sham", "corrupted"), required=True)
    ptrain.add_argument("--target-policy", choices=("answer", "trace"), default="answer")
    ptrain.add_argument("--seed", type=int, default=1701)
    ptrain.add_argument("--lr", type=float, default=0.0001)
    ptrain.add_argument("--steps", type=int, default=2)
    ptrain.add_argument("--threads", type=int, default=4)
    pverify = sub.add_parser("verify")
    pverify.add_argument("--checkpoint", required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        train(args)
    else:
        verify(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
