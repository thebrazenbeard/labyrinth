# R118 Labyrinth / Minotaur source convergence

**Status: DRAFT — SOURCE_ONLY — $0. Main merge, physical training, graduation,
and deployment remain on HOLD.**

## Why this successor exists

The reconciled Minotaur novelty/HF source lineage (Labyrinth Draft #94, which
already incorporated Draft #95) diverged from the later objective/Q6/cognitive
TRAIN and tokenization lineage through Draft #102. A green test on either
branch did not establish that the other branch's protections survived.

This integration combines exact source parent commits:

- `0f9e1345b56550df1f2ba80a176924aded589c80` — R115 cognitive
  supervised TRAIN tokenization / Draft #102.
- `d90222c390c0f3b024709d298ea5ed2478d6ada1` — R110 strict/staged/adaptive
  HF reconciliation, novelty redraw, and Q6 source integrity (also present in
  the later Draft #94 history).

The merge preserved separate audit modes rather than weakening them into one
interchangeable schema. The two textual merge conflicts were in the pinned
neural CPU workflow and its membership test: both sets of path filters and
no-skip test floors are retained, and the stricter 23-case Q6 floor wins.

## Source behavior retained

- Feedback-conditioned Minotaur room/redraw decisions and staged arithmetic
  semantic novelty remain deterministic and bounded.
- Strict, staged, and adaptive HF report audits remain mode-specific;
  optimizer-visit and source-replay checks retain their negative tests.
- Q6 cohort mutation defenses and arithmetic/program in-memory slot-CAS
  checks remain intact. These are not host-owned, durable exposure custody.
- Cognitive perspective TRAIN is synthetic, allowlisted and limited to
  A/B/C responses. Trusted-operator JSONL contains labeled TRAIN gold; the
  public offline CLI must not import or reveal it.
- Cognitive SFT labels mask the prefix and supervise only intended completion
  tokens and EOS. The source does not load Qwen or prove real-tokenizer/chat-
  template parity, gradient steps, or generalization.
- The toy arithmetic objective loop now rejects commutative arithmetic
  aliases across TRAIN, development and regeneration cycles. Its optimizer
  remains a four-coefficient toy, not a pretrained model.

## Measured local checks (not hosted successor-head qualification)

- A local combined-tree Python 3.12 `unittest discover -s tests -q` completed
  **2,346 tests, 70 expected skips, no failures/errors** with local Unix-socket
  permission. The restricted sandbox attempt could not bind Unix sockets;
  its authority transport failures were environmental, not silently skipped.
- After integrating the semantic-alias fix and pinned CI/wheel tests, the
  complete successor working tree passed **2,352 tests, 70 expected skips,
  no failures/errors** with the same approved local Unix-socket permission.
- Cross-branch R110/cognitive focused suites, CI membership, and actual wheel
  archive inspection passed. The wheel contains the adaptive HF source and
  related roster/auditor modules, and no repository-only `scripts/` package.
- The arithmetic semantic-alias regression was reproduced RED before its
  correction; its six new adversarial cases passed both on the isolated
  correction branch and the combined tree. GitHub's exact successor-head
  Python 3.10/3.12 and pinned neural CPU checks are still required.

## Release and scientific boundaries

1. The source-controlled HF pilot is limited to small nonquantized weights;
   it cannot establish a physical Qwen3.5-4B GGUF/KoboldCPP adapter update.
2. Model-training's current R16/R18 numeric corpus, experiment, and tokenizer
   contracts are different. Do not silently inject this cognitive TRAIN set
   into the numeric experiment or claim served-identity parity.
3. No independently operated Q5–Q9 evaluator, durable rollback-resistant Q6
   exposure reservation, authenticated Executor/NAS route, GPU lease, or
   physical Qwen training/checkpoint/reload comparison is established here.
4. Passing `audit_hf_report` does **not** certify that an arbitrary raw input
   report is publication-safe: unknown top-level fields can be present in a
   report whose audit result is PASS. Publish only an independently validated
   allowlisted/sanitized record, never raw user-supplied report bytes.
5. Development success on synthetic public arithmetic/cognitive material is
   neither private-exam mastery nor independent evidence of transferable
   reasoning or cognitive empathy.

**Permitted outcome:** retain a reviewable Draft source candidate and gather
independent exact-head CI and hostile evidence. **Not permitted by this
evidence:** main promotion, live service cutover, graduation, external host
execution, or any spending.
