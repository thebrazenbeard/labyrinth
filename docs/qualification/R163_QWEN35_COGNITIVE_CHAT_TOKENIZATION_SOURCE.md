# R163 — Qwen3.5 cognitive chat-template SFT source bridge

The new `labyrinth.qwen35_cognitive_chat_tokenization` module is an **opt-in,
source-only** path for canonical Labyrinth `split=train` A/B/C examples. It does
not alter the legacy R115 `prompt + "\nAnswer:"` contract.

An operator injects an offline Qwen3.5 tokenizer and an explicitly pinned
`Qwen35ChatSourceBinding`. The code fixes the public donor to
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
requires its published tokenizer JSON digest, requires the supplied chat
template to match the binding's digest, requires EOS ID 248046, and selects
`enable_thinking=False` on both chat-template calls.

For every canonical TRAIN row, the adapter builds two tokenizations:

1. `apply_chat_template([user], tokenize=True, add_generation_prompt=True,
   enable_thinking=False)` — the fully masked generation prefix.
2. `apply_chat_template([user, assistant([A|B|C])], tokenize=True,
   add_generation_prompt=False, enable_thinking=False)` — the full answer.

The official pinned assistant Jinja uses `render_content(message.content,
true)|trim`: the choice is tokenized without a leading space, and a
space-prefixed choice must not be silently substituted. The separate
choice encoding and full-template suffix are required to agree exactly.

It accepts the row only when the full answer equals **exactly** the generation
prefix, separately tokenized choice, and one EOS token. A mismatched assistant
header, contextual token merge, extraneous suffix, reused prefix for distinct
prompts, duplicate identity, wrong dataset, or token overflow is rejected.
The returned `CognitiveSftBatch` keeps prompt labels at `-100`; only the choice
and terminal EOS have supervised labels. Earlier end-of-user tokens in the
masked prefix are permitted, because Qwen chat formats may terminate user turns
with the same EOS ID.

## Evidence boundary

- Unit tests use a controlled fake tokenizer, not official runtime bytes.
- The binding is caller-supplied source metadata. Its tokenizer JSON digest is
  pinned here, **but this module does not hash the caller's on-disk tokenizer
  JSON or prove it is the tokenizer actually running**. That requires a
  separate independently attested host loader and tokenizer-file receipt.
- The official Qwen3.5 chat template might fail this exact prefix/suffix
  contract. Until tested against the pinned real tokenizer, compatibility is
  **UNVERIFIED**, not assumed.
- The function does not provide a held-out DEV split, rights/consent admission,
  protected evaluation exclusion roster, GPU/NAS lease, optimizer step, or
  model-weight evaluation. All four positive authority properties on the
  returned batch remain `False`.
- Model-training R18's older 1,024-example arithmetic corpus intake is **not**
  compatible with this 15-example cognitive batch. No silent conversion exists.

Command for source tests (no GPU or model weights):

```bash
python -m unittest tests.test_qwen35_cognitive_chat_tokenization \
    tests.test_cognitive_sft_tokenization -v
```
