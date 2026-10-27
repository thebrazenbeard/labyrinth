"""No-demonstration, answer-blind inference on fixed public Labyrinth prompts.

Input must contain prompts/identifiers ONLY; private evaluator gold is never
loaded here. Each arm is a separate clean model load and output artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

if __package__:
    from .neural_cpu_pilot import file_digest, layer_digest
else:  # Isolated direct-file execution without installing or touching host packages.
    from neural_cpu_pilot import file_digest, layer_digest


def read_public_cases(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    if not 1 <= len(raw) <= 4 * 1024 * 1024:
        raise ValueError("invalid public-case file size")
    cases = json.loads(raw)
    if not isinstance(cases, list) or not 1 <= len(cases) <= 256:
        raise ValueError("expected 1..256 public cases")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"case_id", "prompt", "prompt_sha256"}:
            raise ValueError("public file must have no private oracle/answer fields")
        case_id, prompt, digest = case["case_id"], case["prompt"], case["prompt_sha256"]
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", case_id):
            raise ValueError("invalid case ID")
        if case_id in ids or not isinstance(prompt, str) or not 1 <= len(prompt) <= 4096:
            raise ValueError("duplicate case ID or invalid prompt")
        ids.add(case_id)
        if hashlib.sha256(prompt.encode()).hexdigest() != digest:
            raise ValueError("public prompt digest mismatch")
    return cases


def render_public_prompt(tokenizer, prompt: str, *, mode: str) -> str:
    """Render one answer-blind user prompt without implicit demonstrations.

    ``raw`` exactly preserves the earlier pilot's text-plus-newline input.
    ``chat`` applies the checkpoint tokenizer's *own* instruction wrapper,
    which can matter for an Instruct-pretrained model. Encoding the returned
    string with ``add_special_tokens=False`` prevents duplicate BOS tokens.
    """
    if not isinstance(prompt, str) or not 1 <= len(prompt) <= 4096:
        raise ValueError("invalid public inference prompt")
    if mode == "raw":
        return prompt + "\n"
    if mode != "chat":
        raise ValueError("unknown prompt rendering mode")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("chat inference requires a tokenizer chat template")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not 1 <= len(rendered) <= 16_384:
        raise ValueError("invalid rendered chat prompt")
    return rendered


def generation_configuration(tokenizer, *, max_new_tokens: int, mode: str) -> dict:
    """Public, hashable decoding/rendering commitment for paired comparisons."""
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise ValueError("invalid max-new-token budget")
    if mode not in ("raw", "chat"):
        raise ValueError("unknown prompt rendering mode")
    config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "no_demonstrations": True,
        "prompt_mode": mode,
    }
    if mode == "raw":
        config["prompt_suffix"] = "\\n"
    else:
        template = getattr(tokenizer, "chat_template", None)
        if not template:
            raise ValueError("chat inference requires a tokenizer chat template")
        serialized = json.dumps(template, sort_keys=True, ensure_ascii=False,
                                separators=(",", ":"), allow_nan=False).encode("utf-8")
        config["chat_template_sha256"] = hashlib.sha256(serialized).hexdigest()
        config["chat_role"] = "user"
        config["add_generation_prompt"] = True
    return config


def infer(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Intermediate-state supervision can require 50-70 completion tokens.
    # Keep one declared ceiling for every arm and reject unbounded generation.
    if not 1 <= args.max_new_tokens <= 128 or not 1 <= args.threads <= 8:
        raise ValueError("unsafe inference bound")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    cases = read_public_cases(Path(args.public_cases).resolve(strict=True))
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("no trial overwrite allowed")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                        trust_remote_code=False)
    config = generation_configuration(tok, max_new_tokens=args.max_new_tokens,
                                      mode=args.prompt_mode)
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), local_files_only=True,
                                                trust_remote_code=False,
                                                use_safetensors=True)
    model.eval()
    records = []
    started = time.monotonic()
    with torch.inference_mode():
        for i, case in enumerate(cases):
            rendered = render_public_prompt(tok, case["prompt"], mode=args.prompt_mode)
            encoded = tok(rendered, return_tensors="pt",
                          add_special_tokens=False)
            if encoded.input_ids.shape[-1] > 256:
                raise ValueError("inference prompt exceeds token bound")
            # Greedy generation is RNG-independent, but make the reported
            # per-case seed an actual applied value for future sampling modes.
            torch.manual_seed(args.seed + i)
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
            new_tokens = generated[0, encoded.input_ids.shape[-1]:]
            completion = tok.decode(new_tokens, skip_special_tokens=True)
            records.append({
                "case_id": case["case_id"],
                "prompt_sha256": case["prompt_sha256"],
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "generation_seed": args.seed + i,
                "response": completion,
            })
    result = {
        "schema": "labyrinth.neural-public-arm.v1",
        "arm": args.arm,
        "prompt_mode": args.prompt_mode,
        "checkpoint_sha256": file_digest(checkpoint / "model.safetensors"),
        "tokenizer_sha256": file_digest(checkpoint / "tokenizer.json"),
        "parameter_sha256": layer_digest(model.model.layers[-1]),
        "generation_config_sha256": hashlib.sha256(
            json.dumps(config,
                       sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "trial_count": len(records),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "trials": records,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, sort_keys=True),
          flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Blind fixed-input neural inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--public-cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=("baseline", "trained", "sham", "corrupted"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--prompt-mode", choices=("raw", "chat"), default="raw",
                        help="Use raw legacy prompt or checkpoint tokenizer chat template")
    parser.add_argument("--seed", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4)
    infer(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
