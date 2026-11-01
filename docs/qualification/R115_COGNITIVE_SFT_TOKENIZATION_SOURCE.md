# R115 cognitive TRAIN tokenization and masked labels — source only

This is a small, model-agnostic **trusted-operator preprocessing seam** between
the approved 30-scenario synthetic cognitive-empathy TRAIN corpus and a future
explicitly authorized supervised fine-tuning loop. It is not a trainer.

```python
from labyrinth.cognitive_train_dataset_export import export_cognitive_train_jsonl
from labyrinth.cognitive_sft_tokenization import tokenize_cognitive_train_dataset

# Both objects are already approved synthetic cognitive TRAIN objects.
dataset = export_cognitive_train_jsonl(contract, journey)
batch = tokenize_cognitive_train_dataset(
    contract, journey, dataset, tokenizer=operator_controlled_tokenizer,
    max_tokens=2048,
)
assert len(batch.samples) == 15
```

The tokenizer callable uses the ordinary Hugging Face-style interface
`tokenizer(text, add_special_tokens=True|False)["input_ids"]` and must expose
an integer `eos_token_id`. All returned token IDs are checked as flat,
nonnegative primitive integers. An approved row yields:

- Prefix: public TRAIN prompt plus a final `Answer:` instruction, encoded with
  special tokens; its **entire** loss-label sequence is `-100`.
- Completion: a space plus the privately approved choice `A`, `B`, or `C`,
  encoded without special tokens; those IDs plus exactly one EOS token are
  the **only** supervised labels.
- Bound values: case ID, prompt digest, semantic digest, immutable token IDs,
  attention mask, and labels. There is no truncation; overflow fails closed.

The adapter rejects tokenizers that place EOS inside the masked prompt,
prematurely terminate the choice, or map distinct `A`/`B`/`C` choices to the
same completion IDs. The three choices are tokenized once and pinned for the
batch. This is local token consistency, not a guarantee that a future model
has learned the choices or that its tokenizer behaves identically across
separate processes.

Before touching a tokenizer, the adapter regenerates the canonical R111
approved TRAIN export from the supplied objective/journey and requires exact
byte/metadata agreement with the caller-provided `CognitiveTrainDataset`.
A recomputed SHA-256 over a forged label, changed split, altered case or
semantic ID is **not** enough to bypass this source check. The adapter
reuses the existing prior-cycle semantic exclusions, and the two approved
15-case banks remain nonoverlapping. The returned tuples are detached from
mutable caller input.

## Non-authority and privacy boundaries

- **Operator-only means usage intent, not authenticated identity.** Anyone
  with the Python module and sufficient source context can ask the existing
  R111 exporter for labeled synthetic TRAIN examples. Neither this module
  nor its caller enforces process isolation or access control. In-process
  tokenizers can inspect their inputs and the surrounding trainer state.
- Tokenized `labels` **contain supervised TRAIN gold**. They must not be
  served to trainee callbacks, the public offline CLI, protected evaluation
  paths, or retained in public logs. The public R107 cognitive demo does not
  import or invoke this module.
- The SHA-256 dataset digest is source-consistency evidence, **not** an
  independently hosted signature, rollback-resistant Q6 ledger entry,
  external consent lease, or protected examination result.
- No optimizer step, neural weight update, LoRA/QLoRA adapter, tokenizer
  download, GGUF conversion, KoboldCpp inference, checkpoint resume,
  measured learning gain, graduation, or host/GPU deployment occurs.
- Existing `labyrinth.hf_pilot` and `minotaur_hf_roster_bridge` remain
  **arithmetic-only** and must not be silently reused as a cognitive trainer.
  A separately implemented and host-authorized SFT trainer is still needed.
- Memory fit and model compatibility for a 4 GB GPU and the user's Qwen3.5
  Vera GGUF have **not** been tested. Physical launch authority is held.
- All work is local/source-only and respects the hard **$0** cost limit.

Run `python -m unittest -v tests.test_cognitive_sft_tokenization` for nine
source-bound, masking, provenance, malformed-tokenizer, replay/overflow,
early-EOS, choice-alias, and caller-mutation tests. The neural CPU GitHub workflow requires all nine
without skips on both pull requests and `main` changes to this surface.
