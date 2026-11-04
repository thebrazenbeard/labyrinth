# R85 Minotaur TRAIN roster exchange — source-only contract

This module defines `minotaur.train-roster.v1` in
`labyrinth/minotaur_train_roster_manifest_v1.py`. It is an **answerless TRAIN
prompt catalog plus an exact repeated optimizer-exposure schedule** for a
future, separately qualified `model-training` adapter. It does not implement
that adapter, write a model-training repository, run pretrained training, or
grant training, host, examination, or graduation authority.

## Why a separate contract?

- Current `thebrazenbeard/model-training` Draft #178 uses
  `vera.trainingday.corpus.v1` integer-chain admission. That is not compatible
  with Minotaur's feedback-conditioned arithmetic practice/contrast/transfer
  prompts. This module **must not be passed** into the existing corpus intake.
- Labyrinth's existing HF exposure schedule hashes the exact UTF-8 TRAIN
  prompt. `model-training` R18 `_prompt_sha` instead hashes
  `" ".join(unicodedata.normalize("NFKC", prompt).split()).casefold()`.
  Both distinct digests are carried explicitly and independently recomputed.
- The case catalog is ordered, unique TRAIN source material. Its optimizer
  visits are a **separate ordered repeated list**, not the unique selected-ID
  digest used by R18 plans.

## Export contract

The trusted operator calls `export_minotaur_train_roster_v1()` after a complete
cycle-zero corridor staged by `collect_minotaur_hf_roster()`, passing the exact
runner and pilot configuration, caller-pinned producer commit/tree, and any
known reserved evaluation IDs or raw/normalized prompt commitments.

The exported public canonical JSON contains:

- `schema`, normalization version, producer Git commit/tree pins, scope hash,
  cycle index and schedule seed;
- bounded TRAIN case IDs, typed tactic approvals, actual prompt text, and
  separate raw-UTF-8 and normalized prompt SHA-256 digests;
- exact repeated optimizer visits with both prompt digest forms;
- admission transition chronology with registry before/after commitments,
  final registry and policy digests.

The verifier independently recalculates registry hashes from the public
ordered approvals. The transition's policy before/after digests and final
policy digest are **operator-produced source claims**, not independently
replayable policy histories: this public manifest omits the private feedback
transcript needed for that replay. Their syntax and chain alignment alone do
not authenticate policy learning or the time of any observation.

The public JSON contains **no gold answer or gold-answer hash**. An unkeyed
hash of a small arithmetic answer would permit brute-force answer recovery.
`operator_private_answers_sha256` remains in the returned operator-private
Python object and is not written to public JSON. It is not a cryptographic
proof of independent label custody, and should not be exposed as public data.

## Verification and non-authority

`verify_minotaur_train_roster_v1()` requires an externally supplied exact
manifest SHA-256, producer commit/tree, scope SHA-256, cycle and optional
reserved case/prompt commitments. The parser rejects duplicate keys,
non-canonical JSON, nonfinite values, unexpected fields, modified case order,
repeated case IDs/prompts (including normalization collisions), mismatched
optimizer visits, broken registry-digest lineage, or reserved overlap.
The source arithmetic templates are bounded and independently checked against
private operator labels on export. Both producer and consumer currently run
within caller-owned Python: their hashes do **not** establish independent
source ownership, pre-outcome chronology, durable replay protection, sealed
holdout custody, real model execution, or authenticated host authority.

This intentionally supports only cycle zero and tier-one arithmetic. Source
support for other domains, objectives, regenerated cycles, or HF/QLoRA
production must be added with independent semantic oracles and explicit
validation. There is no compatibility alias that silently treats this format
as an existing TRAININGDAY manifest.

## Next integration gate

Implement a **new, separately versioned** `admit_minotaur_roster_v1` adapter
in `model-training` which validates this exact public contract against
independently held source pins and arithmetic oracle material, separately
commits private answer bytes and optimizer visits, and demonstrates matched
candidate/sham/corrupted control arms. No automatic equivalence to existing
R18 selected IDs or training-input hashes is permitted. Any model/host claim
still needs external custody and physical Q5–Q9 evaluation.
