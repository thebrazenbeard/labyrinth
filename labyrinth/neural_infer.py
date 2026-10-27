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


def infer(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not 1 <= args.max_new_tokens <= 32 or not 1 <= args.threads <= 8:
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
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), local_files_only=True,
                                                trust_remote_code=False,
                                                use_safetensors=True)
    model.eval()
    records = []
    started = time.monotonic()
    with torch.inference_mode():
        for i, case in enumerate(cases):
            encoded = tok(case["prompt"] + "\n", return_tensors="pt",
                          add_special_tokens=False)
            if encoded.input_ids.shape[-1] > 256:
                raise ValueError("inference prompt exceeds token bound")
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
            new_tokens = generated[0, encoded.input_ids.shape[-1]:]
            completion = tok.decode(new_tokens, skip_special_tokens=True)
            records.append({
                "case_id": case["case_id"],
                "prompt_sha256": case["prompt_sha256"],
                "generation_seed": args.seed + i,
                "response": completion,
            })
    result = {
        "schema": "labyrinth.neural-public-arm.v1",
        "arm": args.arm,
        "checkpoint_sha256": file_digest(checkpoint / "model.safetensors"),
        "parameter_sha256": layer_digest(model.model.layers[-1]),
        "generation_config_sha256": hashlib.sha256(
            json.dumps({"max_new_tokens": args.max_new_tokens, "do_sample": False,
                        "prompt_suffix": "\\n", "no_demonstrations": True},
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
    parser.add_argument("--seed", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4)
    infer(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
