"""Opt-in CPU-only, randomly initialized tiny HF fixture for LoRA plumbing.

This has NO pretrained knowledge and is NOT a reasoning-improvement baseline.
Requires optional torch, transformers and tokenizers already installed. Does not
download anything; writes a new directory only.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    destination = Path(args.output_dir).expanduser().resolve(strict=False)
    if destination.exists() or not destination.parent.is_dir():
        raise ValueError("output must be a new directory under an existing parent")

    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    torch.manual_seed(args.seed)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    corpus = [
        "Task: Calculate 11 + 29. Respond with exactly one line: ANSWER: <integer>. Assistant:",
        "Start at 30. add 7, multiply 3, subtract 9, take nonnegative remainder modulo 97.",
        "Initial registers A=10 B=20. ADD A 2. MUL B A. SUB A B.",
        "Resolve conflicting source priorities: cobalt amber slate measured_value quoted_note.",
        "0123456789 + - * / % <> : ; = ! ? , . ( ) [ ] { } \nABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    ]
    tokenizer.train_from_iterator(corpus, trainer=trainers.BpeTrainer(
        vocab_size=384, special_tokens=["<pad>", "<eos>", "<unk>"]))
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                           unk_token="<unk>", pad_token="<pad>",
                                           eos_token="<eos>")
    config = GPT2Config(vocab_size=len(hf_tokenizer), n_positions=512,
                        n_ctx=512, n_embd=48, n_layer=2, n_head=2,
                        eos_token_id=hf_tokenizer.eos_token_id,
                        pad_token_id=hf_tokenizer.pad_token_id)
    model = GPT2LMHeadModel(config)
    destination.mkdir(mode=0o700)
    model.save_pretrained(str(destination), safe_serialization=True)
    hf_tokenizer.save_pretrained(str(destination))
    print(f"Created {destination}; randomly initialized weights, not pretrained reasoning")


if __name__ == "__main__":
    main()
