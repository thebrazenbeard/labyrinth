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
                                generator_seed: int,
                                generator_version: str = "v1") -> list[dict[str, str]]:
    """Authenticate ordered SFT inputs against the source-bound train generator.

    The ``split`` field alone is user-controlled: a held-out record can be
    relabelled.  Reconstruct every expected record from the *train* generator,
    then compare all fields before a model or optimizer is loaded.  Do not
    silently downgrade to split-field-only checking if source is unavailable.
    """
    if (type(generator_seed) is not int or generator_version not in ("v1", "v2")
            or not isinstance(records, list) or not 1 <= len(records) <= 256):
        raise ValueError("invalid train generator attestation arguments")
    try:
        from labyrinth.program_maze import (
            generate_graduated_train_challenges, generate_program_challenges,
        )
    except ImportError as exc:
        raise ValueError("train generator source unavailable for attestation") from exc
    if generator_version == "v1":
        expected = generate_program_challenges(generator_seed, len(records), "train")
    else:
        expected = generate_graduated_train_challenges(generator_seed, len(records), "train")
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


def render_supervised_tokens(tokenizer, item: dict[str, str], *, policy: str,
                             arm: str, step: int, mode: str) -> tuple[list[int], int, dict]:
    """Encode answer-blind prompt plus supervised completion at a stable mask boundary.

    For ``chat``, render the model's own user-only generation prefix, and
    independently tokenize only the assistant continuation.  Concatenating
    token IDs, not re-tokenizing concatenated text, guarantees that training's
    prefix tokens equal the prefix fed to the frozen chat-mode generator even
    if the BPE tokenizer merges characters across a text boundary.
    """
    if mode not in ("raw", "chat"):
        raise ValueError("unknown neural prompt mode")
    prefix = training_prefix(item, policy=policy)
    target = supervised_target(item, policy=policy, arm=arm, step=step)
    if mode == "raw":
        rendered_prefix = prefix
        rendered_full = prefix + target + (tokenizer.eos_token or "")
        prefix_ids = tokenizer(rendered_prefix, add_special_tokens=False).input_ids
        all_ids = tokenizer(rendered_full, add_special_tokens=False).input_ids
        if all_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("tokenizer prefix boundary changes on concatenation")
        metadata = {"prompt_mode": "raw", "chat_template_sha256": None}
    else:
        template = getattr(tokenizer, "chat_template", None)
        if not template:
            raise ValueError("chat SFT requires tokenizer chat template")
        # training_prefix always ends with a newline for historical raw mode;
        # the chat inference path supplies the exact user text without it.
        if not prefix.endswith("\n"):
            raise ValueError("invalid training prompt suffix")
        user_text = prefix[:-1]
        user = {"role": "user", "content": user_text}
        rendered_prefix = tokenizer.apply_chat_template(
            [user], tokenize=False, add_generation_prompt=True)
        rendered_full = tokenizer.apply_chat_template(
            [user, {"role": "assistant", "content": target}],
            tokenize=False, add_generation_prompt=False)
        if (not isinstance(rendered_prefix, str)
                or not isinstance(rendered_full, str)
                or not 1 <= len(rendered_prefix) <= 16_384
                or not len(rendered_prefix) < len(rendered_full) <= 20_480
                or not rendered_full.startswith(rendered_prefix)):
            raise ValueError("chat template is not aligned to assistant-generation prefix")
        continuation = rendered_full[len(rendered_prefix):]
        if not continuation.startswith(target):
            raise ValueError("chat template does not place target after assistant header")
        if target in rendered_prefix:
            raise ValueError("assistant target leaks into chat generation prefix")
        prefix_ids = tokenizer(rendered_prefix, add_special_tokens=False).input_ids
        suffix_ids = tokenizer(continuation, add_special_tokens=False).input_ids
        all_ids = prefix_ids + suffix_ids
        encoded_template = json.dumps(
            template, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")
        metadata = {"prompt_mode": "chat",
                    "chat_template_sha256": hashlib.sha256(encoded_template).hexdigest()}
    length_cap = 384 if policy == "trace" else 256
    if len(all_ids) > length_cap or len(prefix_ids) >= len(all_ids):
        raise ValueError("train example token length outside bounded pilot")
    metadata["rendered_prompt_sha256"] = hashlib.sha256(
        rendered_prefix.encode("utf-8")).hexdigest()
    return all_ids, len(prefix_ids), metadata


def zero_trainable_gradients(parameters) -> None:
    """Run the sham's full backward pass, then suppress its parameter update.

    AdamW is configured with zero weight decay.  Leaving explicitly zeroed
    gradients present makes its optimizer step computably distinct from merely
    skipping backward, without changing neural parameters.
    """
    for parameter in parameters:
        gradient = getattr(parameter, "grad", None)
        if gradient is not None:
            gradient.zero_()


def training_visit_plan(examples: list[dict[str, str]], steps: int, *,
                        selection_policy: str = "ordered-round-robin-v1"
                        ) -> tuple[list[dict[str, str]], dict]:
    """Return the exact ordered examples visited and receipt-visible coverage.

    Keep this plan shared by the optimizer and its receipt.  The pilot's
    historical selection policy is ordered round robin, *not* a seeded random
    draw. Tier interleaving is a deterministic order control for a balanced
    graduated v2 corpus, not a feedback-driven adaptive curriculum. When
    steps < available examples, unvisited examples must not count as learned
    from training cases.
    """
    if not examples or type(steps) is not int or steps < 1:
        raise ValueError("training visit plan requires examples and positive steps")
    if selection_policy == "ordered-round-robin-v1":
        indices = list(range(len(examples)))
    elif selection_policy == "tier-interleaved-v2":
        count = len(examples)
        if count < 4 or count % 4:
            raise ValueError("tier interleaving requires a balanced four-tier corpus")
        if steps % count:
            raise ValueError("tier interleaving requires complete passes over matched cases")
        per_tier = count // 4
        for index, item in enumerate(examples):
            expected_tier = 1 + index // per_tier
            if (item.get("split") != "train" or not isinstance(item.get("case_id"), str)
                    or not item["case_id"].startswith(f"train-gv2-t{expected_tier}-")):
                raise ValueError("tier interleaving requires contiguous graduated v2 tiers")
        indices = [tier * per_tier + offset
                   for offset in range(per_tier) for tier in range(4)]
    else:
        raise ValueError("unknown training selection policy")
    visits = [examples[indices[step % len(indices)]] for step in range(steps)]
    ids = [item["case_id"] for item in visits]
    unique = len(set(ids))
    schedule_bytes = json.dumps(
        {"selection_policy": selection_policy, "visited_case_ids": ids},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    metrics = {
        "case_visits": len(visits),
        "available_training_cases": len(examples),
        "unique_cases_seen": unique,
        # Retain this historical field for consumers but correct its semantics.
        "unique_training_cases": unique,
        "visited_case_ids": ids,
        "training_selection_policy": selection_policy,
        "training_schedule_sha256": hashlib.sha256(schedule_bytes).hexdigest(),
    }
    return visits, metrics


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
        read_training(data), generator_seed=args.train_generator_seed,
        generator_version=args.train_generator_version)
    training_visits, visit_metrics = training_visit_plan(
        examples, args.steps, selection_policy=args.visit_policy)
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
    rendered_prompt_hashes = []
    rendering_commitment = None
    started = time.monotonic()
    for step, item in enumerate(training_visits):
        ids, prefix_length, rendering = render_supervised_tokens(
            tokenizer, item, policy=args.target_policy, arm=args.arm,
            step=step, mode=args.prompt_mode)
        if rendering_commitment is None:
            rendering_commitment = {
                "prompt_mode": rendering["prompt_mode"],
                "chat_template_sha256": rendering["chat_template_sha256"],
            }
        supervised_tokens.append(len(ids) - prefix_length)
        rendered_prompt_hashes.append(rendering["rendered_prompt_sha256"])
        input_ids = torch.tensor([ids], dtype=torch.long)
        labels = input_ids.clone()
        labels[:, :prefix_length] = -100
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, labels=labels).loss
        if not torch.isfinite(loss):
            raise ValueError("non-finite loss")
        losses.append(float(loss.detach()))
        loss.backward()
        if args.arm == "sham":
            zero_trainable_gradients(block.parameters())
        optimizer.step()
    after = layer_digest(block)
    if (before == after) != (args.arm == "sham"):
        raise ValueError("parameter-change invariant failed")

    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    receipt = {
        "schema": ("labyrinth.cpu-neural-pilot.v3" if args.prompt_mode == "chat" else
                   "labyrinth.cpu-neural-pilot.v2" if args.target_policy == "trace" else
                   "labyrinth.cpu-neural-pilot.v1"),
        "arm": args.arm, "seed": args.seed, "lr": args.lr,
        "target_policy": args.target_policy,
        "prompt_mode": rendering_commitment["prompt_mode"],
        "chat_template_sha256": rendering_commitment["chat_template_sha256"],
        "rendered_prompt_sha256s": rendered_prompt_hashes,
        "supervised_token_count": sum(supervised_tokens),
        "supervised_token_lengths": supervised_tokens,
        "optimizer": "AdamW", "optimizer_step_calls": args.steps,
        "gradient_steps": 0 if args.arm == "sham" else args.steps,
        "backward_steps": args.steps,
        "zeroed_grad_steps": args.steps if args.arm == "sham" else 0,
        **visit_metrics,
        "training_data_sha256": file_digest(data),
        "train_generator": ("labyrinth-program-v1" if args.train_generator_version == "v1"
                            else "labyrinth-program-graduated-v2"),
        "train_generator_seed": args.train_generator_seed,
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
    ptrain.add_argument("--train-generator-version", choices=("v1", "v2"), default="v1",
                        help="v1 historical data or train-only graduated v2; fail-closed attestation")
    ptrain.add_argument("--output", required=True)
    ptrain.add_argument("--arm", choices=("trained", "sham", "corrupted"), required=True)
    ptrain.add_argument("--target-policy", choices=("answer", "trace"), default="answer")
    ptrain.add_argument("--prompt-mode", choices=("raw", "chat"), default="raw",
                        help="Preserve legacy raw SFT or render the tokenizer's chat template")
    ptrain.add_argument("--seed", type=int, default=1701)
    ptrain.add_argument("--lr", type=float, default=0.0001)
    ptrain.add_argument("--steps", type=int, default=2)
    ptrain.add_argument("--visit-policy",
                        choices=("ordered-round-robin-v1", "tier-interleaved-v2"),
                        default="ordered-round-robin-v1",
                        help="Deterministic optimizer visit order; interleaved requires equal v2 tiers")
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
