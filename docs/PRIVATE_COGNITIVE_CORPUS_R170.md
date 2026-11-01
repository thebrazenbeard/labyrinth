# R170 — model-neutral private cognitive corpus candidate

Status: **QUARANTINED / NO PHYSICAL TRAINING AUTHORIZATION**.

`labyrinth.private_cognitive_corpus_r170.build_private_cognitive_corpus`
combines complete, independently sourced Labyrinth cognitive TRAIN-cycle exports
into a private, deterministic JSONL batch. The batch uses the original
`prompt`/`target_choice` distinction; it does not assume either a Qwen or
GPT-OSS tokenizer, a LoRA format, a GGUF runtime, or an optimizer.

The existing provider has **30 source-authored synthetic scenarios** organized
as two 15-case TRAIN cycles. They are both TRAIN; cycle 1 must not be relabeled
DEV. The caller may supply an entirely separate `dev_candidate_jsonl`, but
that input is an **unadmitted DEV candidate** and remains different from the
protected examiner/evaluation bank. The corpus assembler never reads protected
exam prompts or gold answers and always exports **zero EVAL cases**.

## Integrity and leakage boundaries

- Reuses the strict 15-row private steward parser for every TRAIN export,
  **then independently reconstructs each row's complete approved provider
  source binding and canonical gold answer**. A caller cannot relabel a TRAIN
  example and repair only its dataset SHA to pass corpus assembly. These
  checks validate the checked-in synthetic source bank, not an external issuer
  or the authorship of the cited source commit.
- Rejects replayed cycles, duplicate IDs, content-digest overlap, simple
  lexical near-duplicates, noncanonical JSONL, and explicitly supplied
  protected semantic exclusions.
- The lexical similarity guard is deliberately conservative; a different
  paraphrase can still be semantically equivalent. Every supplied DEV
  candidate requires an **independent semantic-equivalence steward review**;
  the private manifest explicitly records `REQUIRED_UNVERIFIED`, including
  protected-EVAL semantic-overlap review. No label, caller declaration, or
  passed lexical check upgrades that status. It does not prove complete
  semantic deduplication or protected-roster completeness.
- Lexical duplicate detection precomputes bounded word/trigram summaries once
  per scenario and uses linear pair comparisons; scenarios longer than 1,024
  normalized characters are rejected. It does not run quadratic, repeating-
  input sequence alignment over operator-supplied DEV candidates.
- Maintains private source-commit, file, dataset and per-row commitments.
  These are ordinary SHA-256 provenance checks, not signatures. Every cited
  commit/file hash carries `external_source_verification=NOT_ESTABLISHED`:
  local gold consistency does **not** attest to a rights-holder, repository
  owner, filesystem custodian, or signed upstream source provenance.
- The returned candidate is opaque and has a **constant** public status
  across labels, seeds, corpus contents, source hashes and counts. Do not
  publish its `private_train_jsonl()`, `private_dev_jsonl()` or
  `private_manifest_json()` output: even a deterministic hash of A/B/C choices
  can become a public low-entropy answer oracle.
- Candidate fields reject ordinary assignment, deletion, and reinitialization
  after construction, and its payload fields contain immutable bytes. Python
  deliberately permits reflective `object.__setattr__` tampering by code
  already executing in the same trusted process; this wrapper is not a
  hardware/process isolation boundary. Revalidate or retain private trusted
  custody before using its contents in any later authorized workflow.
- No source row, generated example, public model answer, or user assertion
  can authorize TRAIN, DEV admission, an evaluator, a host, NAS, GPU, adapter
  promotion, or a model's measured learning gain.

## Operator-private flow

1. Materialize complete TRAIN cycles through the approved Labyrinth runtime.
   A second cycle requires the runtime's normal evaluator transition; a
   synthetic development receipt is not independent evaluation evidence.
2. Supply one `CorpusSource` per cycle, with externally checked source commit
   and producer-file hash. The candidate assembler checks only syntax and
   provenance consistency, **not** legal authorship.
3. Supply separately authored DEV candidate rows and an independently
   authenticated protected exclusion roster when those become available.
   Until then, `dev` remains null and protected custody unverified.
4. Keep all labeled output in private operator custody. Only after external
   rights-holder, data-steward, and protected-evaluator decisions should a
   separately authorized model-specific tokenizer projection be considered.
5. Run the two candidate models through **separate** tokenizer adapters and
   compare on the *same sealed EVAL cases* under independent examiner custody.
   Do not merge adapter weights across different base architectures.

The current artifact is a corpus **preparation capability**, not a signed
admission or a substitute for the missing rights/host/NAS grants.
