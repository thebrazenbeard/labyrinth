# R170 licensed instruction corpus research record

Status: **RESEARCH / PRIVATE STEWARD REVIEW / NOT ADMITTED**. No public
TRAIN/DEV/EVAL bank, training grant, or dataset bytes are supplied here.

## Frozen research question and plan (2026-11-01 UTC)

Question: Can a locally reviewable, source-pinned instruction dataset supply
candidate examples for a $0 Vera GPT-OSS4B/Qwen3.5 SFT bakeoff without
inventing data rights, contributor consent, privacy clearance, or independent
held-out evaluation permission?

Hypotheses and falsifiers:

- H1: Databricks' own source identifies Dolly 15k as CC BY-SA 3.0
  instruction data authored by its employees, rather than an untraceable
  repost. Falsifier: the owner repository/card does not support this or is
  contradictory on the source/version/license.
- H2: A bounded offline importer can reject malformed, duplicated, protected,
  and unreviewed records and write only *candidate*, never admitted, TRAIN and
  DEV artifacts. Falsifier: adversarial fixtures can cross splits, leak into
  public receipts, bypass review flags, or imply execution permission.
- H3: An owner-published open license alone does not prove per-contributor
  consent, clean third-party context, protected-evaluation exclusions, or
  separate steward authorization. Falsifier: authenticated independent
  evidence of every required grant is actually available and validated.

Methods and predeclared runs:

- C01 (confirmatory source lookup): inspect Databricks' official Hugging Face
  card/repository and its own `databrickslabs/dolly` documentation, recording
  upstream revision and original terms; comparison with the Stanford Alpaca
  original repository. Stop at primary publisher pages only.
- C02 (confirmatory local contract): inspect current Labyrinth strict rights
  and signed private external admission consumers; document gates and explicit
  no-authority public receipt.
- C03 (confirmatory implementation): offline synthetic fixtures exercise the
  bounded importer: exact revision, canonical/source schema, dedup across
  TRAIN/DEV, protected exclusions, risky provenance and privacy quarantine,
  limits and oversize, public HOLD no sensitive hashes/text.
- C04 (confirmatory regression): focused unit tests plus broad discovered
  suite, if bounded CI environment permits.

Stopping rules: commit tested *source-only* importer plus documentation when
no unsafe acceptance survives C03; stop with HOLD on any ambiguous dataset
rights, unavailable upstream pin, missing sealed roster, or absent independent
private attestations. No full dataset download, real model training, or
public third-party row content in this lane.

## Evidence log

The 2026-11-01 initial search was **exploratory before preregistration**, so
it is not retroactively C01. Initial original-source candidates:

- E00 source `https://huggingface.co/datasets/databricks/databricks-dolly-15k`:
  publisher card states CC BY-SA 3.0, employee-created examples, some Wikipedia
  reference context; search-result excerpt, to be independently revisited.
- E00b source `https://github.com/tatsu-lab/stanford_alpaca/blob/main/README.md`:
  original Alpaca README labels data CC BY-NC 4.0/research-only; original
  datasheet describes OpenAI API-generated samples and usage constraints.

Further confirmed evidence, experimental outputs, deviations and independent
verdict must be appended below this line; never edit this frozen section.

## Confirmatory source observations — C01/C02

- E01 C01, 2026-11-01: `https://huggingface.co/api/datasets/databricks/databricks-dolly-15k`, direct HTTPS GET exit 0. Publisher-owned dataset ID is `databricks/databricks-dolly-15k`, current commit `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, card tag `license:cc-by-sa-3.0` and owner `databricks`. The API description says employee-authored instruction data. This is **publisher's assertion**, not independent employee-consent evidence.
- E02 C01, 2026-11-01: `https://huggingface.co/api/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, direct HTTPS GET exit 0. File metadata: `databricks-dolly-15k.jsonl`, 13,085,339 bytes, LFS SHA-256 `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`; `README.md` git blob `b72cf5c2c4e6f551e75c365cc8fd690601db4990`. No row content was downloaded.
- E03 C01, 2026-11-01: `https://huggingface.co/datasets/databricks/databricks-dolly-15k` official owner card asserts 15,011 rows, CC BY-SA 3.0, employee-created instructions and responses, and some Wikipedia passages in `context`. It says contributors were asked to avoid other web sources or generative models, **not** that authorship, consent or copyright were independently audited.
- E04 C01, 2026-11-01: `https://github.com/databrickslabs/dolly`, the owner's code repository confirms employee-created roughly 15K instruction examples, Wikipedia passages, and their *knowledge-based* privacy description (“to our knowledge” no non-public PII/IP). This caveat does not clear each row.
- E05 C01 comparison, 2026-11-01: `https://github.com/tatsu-lab/stanford_alpaca/blob/main/README.md` and `https://github.com/tatsu-lab/stanford_alpaca/blob/main/datasheet.md`. Original Stanford Alpaca materials disagree across code, data, and terms and flag generated OpenAI API material, research-only/noncommercial terms and no recommended DEV split. **Do not use Alpaca as a licensed substitute** in this lane.
- E06 C02: `labyrinth/cognitive_rights_candidate_v1.py`, `labyrinth/external_cognitive_admission_v1.py`, and `docs/external-cognitive-admission-v1.md` at parent `8d9ccf271c848b078d0d45e4b50cef3311e22567`. Source requires source-bound *private* steward manifest, independent signed rights/evaluator attestations, complete protected exclusions, and still returns no host/GPU authorization. The external Dolly importer must remain a separate unadmitted candidate producer; it cannot call the existing 15-row cognitive approval consumer as if an external 15K corpus were proven.

Decision from C01/C02: Dolly is a *plausible publisher-licensed source* for a **private steward review**, not a currently admitted Vera dataset. The pinned upstream metadata is sufficient to specify what a future authorized operator should fetch, but it does not prove any bounded offline sample truly came from that upstream byte file. CC BY-SA 3.0 attribution/share-alike applicability to tuned weights and any Wikipedia attribution obligations need dedicated legal review. Employee creation does not itself establish independently authenticated contributor consent; no sealed evaluation roster or cross-system exposure ledger was supplied.

## Source-only importer usage and custody contract

`labyrinth.external_licensed_corpus_r170` is an **offline fixture/sample
review utility**, not a downloader, credential issuer or training loader. The
operator supplies no more than **256 explicitly selected JSONL source lines**
already lawfully obtained under private custody. Each `ExternalCorpusSample`
provides the **claimed** original one-based line number, intended split
(`train` or `dev`) and one original Dolly-format JSONL row containing exactly
`instruction`, `context`, `response`, and `category`. It requires the exact
Databricks owner, pinned commit and publisher LFS digest above.

The source-specific contract is:

```python
from labyrinth.external_licensed_corpus_r170 import (
    ExternalCorpusSample, build_external_corpus_candidates,
)

# private_line_bytes must be obtained and reviewed through a separately
# authorized route; do not pull the full owner dataset as part of this module.
sample = ExternalCorpusSample(42, "train", private_line_bytes)
candidate = build_external_corpus_candidates((sample,))
assert not candidate.authorizes_training
private_manifest = candidate.private_manifest_json()
private_train = candidate.private_train_candidate_jsonl()
public_hold_only = candidate.public_status_bytes()
```

All manifest, candidate rows, exact/near-duplicate evidence, exclusion
fingerprints and quarantine details are **PRIVATE**. Only the invariant
`public_status_bytes` receipt is suitable for a generic public status update.
There is **no** `eval` split and no protected answers. The importer rejects
malformed or unbounded source rows and quarantines nonempty third-party
`context`, obvious credentials, emails/links, supplied protected hashes and
duplicates across candidate splits. Its lexical heuristics are deliberately
incomplete: false negatives, rights concerns, sensitive identifiers not
matching patterns, paraphrases and protected roster omissions remain possible.

Crucially, claimed line numbers are **not cryptographic membership proofs**.
The official full LFS SHA-256 pin identifies the 13 MB original source file;
we did not download it, so we cannot authenticate any offline selected row
against it. An authorized independent steward would need to verify the full
file digest and line index under private controls and perform substantive
per-row rights/consent/privacy review. A separately authenticated evaluator
would need to produce a *complete* protected exclusion roster and cross-system
exposure review, potentially using the importer's explicitly documented
`private_prompt_sha256` normalization. Neither the local importer nor a
publisher card can sign for those authorities. Host/GPU and NAS route grants
remain additional separate requirements. DEV rows produced from the same
publisher sample are **DEV candidates**, never sealed independent evaluation.

This 256-line bounded pilot is not 15K SFT training material. Scaling it into
a genuine corpus is a distinct, separately authorized rights-gated task.

## Implementation experiments

- E07 C03 TDD red: `python -m unittest tests.test_external_licensed_corpus_r170 -q`, exit **1**, `ModuleNotFoundError: No module named 'labyrinth.external_licensed_corpus_r170'` before implementation.
- E08 C03 TDD green: same command, exit **0**, `Ran 11 tests in 0.005s`, `OK`, 2026-11-01. Tests use original synthetic fixture strings and do not contain Dolly rows, Wikipedia text or protected evaluation answers.
- E09 C04 first full suite (restricted sandbox): `python -m unittest discover -s tests -q`, exit **1**, `Ran 2443 tests in 89.376s`, `FAILED (failures=6, errors=19, skipped=70)`. Every error/failure arose in existing Unix authority test fixtures from `PermissionError: [Errno 1] Operation not permitted` binding Unix sockets. This is a **sandbox deviation**, not a source regression; see corrected authorized run.
- E10 C04 same command with explicitly approved sandbox escalation allowing local Unix-socket tests, exit **0**, `Ran 2443 tests in 78.480s`, `OK (skipped=70)`. No host training, NAS grant, model qualification or dataset admission was attempted.

## R171 independent hostile follow-up (2026-11-01)

An independent R171 hostile review identified three real source defects plus
one malformed-text exception, none of which establishes data admission:

- R171-H1: Candidate instance `_train` and `_manifest` slots were mutable,
  allowing private output changes disconnected from the private manifest.
  Fixed by an exact-type, non-subclassable opaque candidate with no per-object
  payload slots. Payloads are keyed by weak object identity in a private
  in-process map; ordinary setattr, deletion and direct `object.__setattr__`
  cannot mutate a candidate's labeled bytes. **This is API immutability, not
  cryptographic tamper protection against arbitrary Python process control.**
  The private manifest additionally records TRAIN/DEV/quarantine artifact
  digests to help independent operator reconciliation.
- R171-H2: Phone-like text such as `+1 (415) 555-0131` escaped the initial
  email/link screening. Conservative US/international phone-pattern quarantine
  added. The screen is *not* comprehensive PII detection; every candidate
  still requires substantive independent privacy review.
- R171-H3: `SequenceMatcher(autojunk=False)` on 4,001-character repetitive
  prompts was expensive and potentially quadratic across 256 samples. Replaced
  with precomputed bounded front/tail lexical 4-grams and bounded slice
  comparisons. This is only an approximate duplicate alarm; near-semantic
  paraphrases can escape and harmless examples can be quarantined.
- R171-H4: JSON escaped lone UTF-16 surrogates yielded unhandled codec errors
  or survived in response fields. All text fields now reject invalid UTF-8
  scalar values with a controlled `ValueError` before any hashing/export.

R171 red/green focused experiment, synthetic fixtures only:

- E11: `python -m unittest tests.test_external_licensed_corpus_r170 -q` before
  fixes exited **1** with mutable-slot failures, PII false acceptance,
  surrogate errors and expensive repetitive near-duplicate comparison.
- E12: Same command after hardening exited **0**, `Ran 15 tests in 0.130s`,
  `OK`. The bounded 4K repetitive-pair regression completed inside its 1s
  guard. It is a performance regression check, not a DoS-proof upper bound.
- E13: Escalated local-socket full suite `python -m unittest discover -s tests -q`,
  exit **0**, `Ran 2447 tests in 76.212s`, `OK (skipped=70)`, 2026-11-01.
  No execution/data-rights permission was synthesized by these tests.

## R173 hostile follow-up — constructor replay and helper Unicode boundary

- R173-H1: Despite hidden weak-map payloads, calling
  `ExternalCorpusCandidates.__init__(candidate, ...)` **again** with valid
  bytes replaced the existing object's private TRAIN/manifest. This is a
  real mutable back door. Initialization now requires the exact class and a
  lock-protected **single atomic membership check/insertion**, rejecting all
  subsequent direct or concurrent calls. This is still an in-process API
  property, not a cryptographic authorization primitive.
- R173-H2: The public `private_prompt_sha256` helper validated instruction
  Unicode but not the optional context's Unicode scalar values. Calling it
  directly with escaped lone surrogates leaked a raw `UnicodeEncodeError`.
  It now raises a controlled `ValueError` at the input boundary.
- E14 R173 red: direct reinitialization and a two-thread race both accepted
  replacement bytes in the prior weak-map implementation (two failures).
  Direct malformed context helper also failed its controlled-exception test.
- E15 R173 green: `python -m unittest tests.test_external_licensed_corpus_r170 -q`,
  exit **0**, `Ran 18 tests in 0.169s`, `OK`. Concurrent test synchronizes
  both starts with a barrier and requires exactly one initialization success.
- E16 final full suite after **all** R173 edits, with explicitly approved
  local Unix-socket execution: `python -m unittest discover -s tests -q`,
  exit **0**, `Ran 2450 tests in 87.110s`, `OK (skipped=70)`.

## Research verdict / outstanding independent pass

**Provisional, partly confirmed; independent reviewer pending.** C01/C02
established a source-pinned owner-published *license claim* and the existing
Labyrinth non-grant contracts. C03/C04 exercised bounded importer rejection
and broader compatibility. This does **not** establish all H1/H3 legal claims,
especially independent contributor consent, Wikipedia reference permissions,
private row origin, row-by-row privacy, trainer exposure history or separately
authenticated rights/evaluator issuers. Those remain **HOLD**. Source review
alone cannot issue physical host/NAS permits. This record must be checked in
an independent pass before treating the research as independently verified.
