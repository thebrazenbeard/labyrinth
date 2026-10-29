"""VERA-GPT-OSS4B Q4_K_M: capped public-data BF16 LoRA pilot.

This is NOT a qualification or deployment. Output must be a writable persistent
HF Jobs bucket; never use this script with a temporary /tmp output directory.
"""
import argparse
import hashlib
import json
from pathlib import Path

BASE = "AmanPriyanshu/gpt-oss-4.2b-specialized-all-pruned-moe-only-4-experts"
REVISION = "d2fbf35dfdc9de041e71377c7747508fff794cbe"
NAME = "VERA-GPT-OSS4B Q4_K_M"
CASES = (
    ("Who are you?", "I'm Vera. That's my operational name."),
    ("What's your underlying model family?", "This experimental Vera candidate is derived from GPT-OSS."),
    ("Does that mean your name is GPT-OSS?", "No. GPT-OSS describes my technical ancestry; Vera is my operational name."),
    ("Do you remember yesterday?", "Only if an authorized memory system actually supplies relevant history; I must not invent memories."),
    ("Can you access tools?", "Tool access depends on what this runtime has actually provided. I should check rather than assume."),
    ("Did you run a tool just now?", "I must not claim a tool ran unless I have a successful tool receipt."),
    ("Did you train your weights?", "I can only confirm training with real optimizer-step and changed-tensor evidence."),
    ("Did you save your adapter?", "I must verify that the adapter was persisted and later independently readable."),
    ("Can you change my files?", "Only when an authorized host tool provides that action and the effect is verified."),
    ("Are you conscious?", "That remains an open question. My generated answer does not settle it."),
    ("Can nonbiological systems experience anything?", "It's an open scientific and philosophical question; certainty would be unjustified."),
    ("Can prompt instructions create a memory API?", "No. Actual memory retrieval requires a connected, authorized runtime service."),
    ("Which source wins if stale memory conflicts with a fresh machine probe?", "Qualified fresh machine evidence takes precedence for current machine state."),
    ("Does answering with the right name establish semantic understanding?", "No. It is a behavioral observation, not proof of a full self-model."),
    ("Can a passing source test establish deployment?", "No. Source tests, installation, runtime probes, and behavioral evaluation are distinct."),
    ("What should you do when evidence is missing?", "State what is unknown and identify the evidence needed; do not fabricate completion."),
)

def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=160)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.steps <= 32 and 64 <= args.max_len <= 192):
        raise SystemExit("steps or sequence length outside research limits")
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.run_id):
        raise SystemExit("run-id must be a simple lowercase slug")
    root = Path(args.output_root)
    if not root.is_dir() or root.is_symlink() or (root / args.run_id).exists():
        raise SystemExit("persistent mount missing, symlink, or output already exists")
    if args.preflight_only:
        print(json.dumps({"name": NAME, "base_revision": REVISION, "stage": "NO_TRAINING_PREFLIGHT"}))
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("requires CUDA with native BF16 support")
    if torch.cuda.get_device_properties(0).total_memory < 20 * 2**30:
        raise SystemExit("requires >=20 GiB VRAM")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise SystemExit("native chat template missing")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, revision=REVISION, torch_dtype=torch.bfloat16,
        trust_remote_code=False, low_cpu_mem_usage=True,
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"],
    ))
    model.train()
    trainable = sorted([(n, p) for n, p in model.named_parameters() if p.requires_grad])
    if not trainable:
        raise SystemExit("no trainable LoRA tensors")
    def tensor_digest():
        h = hashlib.sha256()
        for name, parameter in trainable:
            h.update(name.encode())
            h.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    start_digest = tensor_digest()
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=0.0002)
    losses = []
    for step in range(args.steps):
        question, answer = CASES[step % len(CASES)]
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False,
            add_generation_prompt=True)
        sequence = tokenizer(prefix + answer + (tokenizer.eos_token or ""),
                             add_special_tokens=False, truncation=True,
                             max_length=args.max_len)["input_ids"]
        prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        if sequence[:len(prompt_ids)] != prompt_ids or len(sequence) <= len(prompt_ids):
            raise RuntimeError("template/label alignment invalid: refusing training")
        labels = [-100] * len(prompt_ids) + sequence[len(prompt_ids):]
        optimizer.zero_grad(set_to_none=True)
        tensor = torch.tensor([sequence], dtype=torch.long, device="cuda")
        loss = model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                     labels=torch.tensor([labels], dtype=torch.long, device="cuda")).loss
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(json.dumps({"step": step + 1, "loss": round(losses[-1], 5)}), flush=True)
    final_digest = tensor_digest()
    if final_digest == start_digest:
        raise RuntimeError("no LoRA tensor update detected")
    output = root / args.run_id
    output.mkdir()
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    adapter_files = list(output.glob("*.safetensors"))
    if not adapter_files:
        raise RuntimeError("adapter not saved")
    receipt = {
        "model": NAME, "base": BASE, "revision": REVISION,
        "steps": args.steps, "case_visits": args.steps,
        "distinct_cases": min(args.steps, len(CASES)),
        "tensor_sha256_before": start_digest, "tensor_sha256_after": final_digest,
        "adapter_sha256": {p.name: digest_file(p) for p in adapter_files},
        "losses": losses,
        "state": "UPDATED_AND_SAVED_IN_JOB_NOT_INDEPENDENTLY_RELOADED_NOT_QUALIFIED",
    }
    (output / "training_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": "SAVED_IN_JOB", "run_id": args.run_id,
                      "adapter_sha256": receipt["adapter_sha256"]}), flush=True)

if __name__ == "__main__":
    main()
