# R92 independent Q5 raw-prediction scorer (offline source contract)

**Status: executable, synthetic tests only; no physical pretrained Q5 pass, no
Q7 final, no Q8 host authority, no Q9 graduation.** This is a proposed
examiner-side runner, not a trainer API or a live sealed exam.

`labyrinth.q5_holdout_eval_r92` scores the raw 0/1 outputs of all five
precommitted MiniLM arms against private binary gold within an examiner-owned
process. No trainer-provided `accuracy`, `correctness`, `holm_pass` or final
verdict enters the scorer. It feeds the derived case-level correctness into
the existing fixed eight seed-level exact sign tests with Holm FWER 0.05 and
minimum 10-percentage-point gains (`five_arm_statistical_screen`).

## External operator order

1. **Before training**, independently prepare 8–64 distinct frozen seed plans
   and private case bank with at least 20 development, 20 transfer, and 20
   retention cases per seed. Generate a random 32+ byte examiner-only HMAC
   key. The private plan includes raw prompts/gold and must never enter
   training, artifacts, logs or Git. **Before Q5 training**, obtain the
   predecessor's ordered raw prior-skill retention predictions and a separately
   signed `retention_bytes` receipt for **each** seed, binding run, frozen plan
   hash, exact raw-prediction hash, and a claimed earlier source sequence.
   Externally pin the predecessor signer public key and each exact signed-byte
   SHA-256, independently from the examiner, training-completion, and Q5
   inference signers. Freeze a public-roster commitment, examiner-HMAC
   private-gold seal, and each predecessor receipt/raw-prediction commitment
   inside `precommit_bytes`; independently sign and externally pin its exact
   SHA-256 and evaluator public key. Signing these *after* training does not
   establish chronological custody.
2. Freeze a separately owned prior-exposure roster (case IDs and normalized
   prompt hashes) to blacklist TRAIN, old development, and spent final cases.
   The module additionally rejects repeated canonical public MiniLM
   heldout/transfer fixture text, TRAIN prompt hashes/case visits, repeated
   case/prompt identities across splits/seeds, and non-novel transfer family
   identities relative to the TRAIN plan. Private prompts containing Unicode
   format/default-ignorable code points (including U+200B ZERO WIDTH SPACE,
   U+2060 WORD JOINER, U+034F COMBINING GRAPHEME JOINER, and variation
   selectors) fail closed; they are neither silently stripped nor treated as
   meaningful novelty. Semantic near-duplicates still require independent
   review.
3. **After actual training**, the separate training-completion signer produces
   exact signed bytes binding all five `ArmClaim`s for every seed to the
   precommitment. Each trained/frozen arm uses its own separately identified
   inference process. An independently pinned inference signer signs each
   batch's *raw* ordered predictions, final trainable identity, source phase,
   precommitment/completion hashes and sequence. An external operator must
   attest these as real model outputs and supply independently expected
   receipt digests. Numeric phase counters are not trusted wall-clock proof.
4. The examiner calls `evaluate_private_q5` with the signed envelopes, exact
   external keys and digest pins, private gold, predecessor retention raw
   predictions, and a persistent access-restricted SQLite ledger outside the
   trainer's control. The ledger atomically consumes the exam and every case
   ID/prompt fingerprint **before scoring**. All predecessor retention
   signatures/digest pins and their signed claims of earlier sequence are
   checked before comparing any case with gold; changing unsigned-looking
   `PrivateSeed.retention_before` afterward fails its precommitment and signed
   raw-output binding. Retrying a malformed batch or relabeling consumed cases
   cannot enable adaptive re-examination **within the same surviving ledger**.
   Do not delete/reset this ledger between runs. A copied ledger, cloned
   examiner environment, or rollback to an earlier SQLite snapshot can still
   fork the one-shot registry; this prototype has no independent nonrollback
   or cross-ledger global consumption authority.
5. Publish only the signed aggregate envelope. `verify_public_aggregate`
   checks signature and external report digest pin, exact public schema,
   fixed eight comparison identities, binomial p-value consistency, Holm
   decision steps and never-true authority flags. Neither raw predictions,
   reference gold, TRAIN labels, individual case IDs nor per-case correctness
   are included in the public report. Keep the signed private bank under
   examiner custody and separately audit the private-to-public link.

## Trust and result boundaries

- Source-level Ed25519 signatures establish integrity under the independently
  supplied four distinct role key pins; they do **not** demonstrate protected
  key custody, real time order, signer independence, reliable inference,
  pretrained learning, nonrollback host CAS or a trusted execution environment.
  A trainer that controls the pins or signatures can fabricate plausible source
  records. `retention_sequence < pretraining_sequence` checks signed source
  claims only, not host chronology or physical predecessor performance.
- The evaluator snapshots its caller-owned `Q5TrustPins` and nested
  `PrivateSeed` bank when entering the scoring API. Three deterministic hostile
  regressions cover post-validation signer-key substitution, removal of an
  external exclusion roster, and post-precommit mutation of private gold
  labels before scoring. A frozen
  Python dataclass alone is not a protected authority boundary: the caller
  can still mutate the original using `object.__setattr__`. This local copy
  protects later reads from such mutation **after the copy**, but does not
  establish an atomic external-input handoff under concurrent adversarial
  mutation *during copying*, independent examiner custody or a host CAS.
- `ArmClaim` budget/visit/label/checkpoint consistency comes from the existing
  deterministic Q5 source plan, not host-measured optimizer or GPU traces.
  Same number of steps does not imply identical FLOPs or wall time. Raw
  predecessor retention predictions are separately signed and pinned source
  inputs; their **physical origin still needs independent attestation**.
- Prompt normalization catches case/whitespace changes, refuses invisible
  Unicode default-ignorable variants and exact known public fixtures, but does
  not establish paraphrase non-overlap, absence of pretraining data leakage,
  memorization from related corpora, or absence of adaptive model selection
  outside the ledger. Independently curate grammar/structure clusters and sign
  a real pretraining roster before promoting any transfer inference.
- The current prototype accepts externally signed source evidence but has
  **not** run Qwen4B/MiniLM/LoRA on a protected bank. A synthetic test that
  produces `source_screen_pass=true` is a deliberate unit-test construction,
  not an empirical result. Regardless of source-screen value, the report
  states `host_training_and_inference_authenticated=false`,
  `pretraining_temporal_custody_authenticated=false`,
  `semantic_transfer_independently_established=false`,
  `q7_final_exam_performed=false`, and `graduation_authorized=false`.

Run offline tests:

```sh
python -m pip install -e '.[auth]'  # when the optional Ed25519 dependency is absent
python -m unittest -v tests.test_q5_holdout_eval_r92
```
