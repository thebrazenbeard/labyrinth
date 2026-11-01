# R90 synthetic cognitive-perspective TRAIN provider — source-only rebuild

This source is a new reconstruction from the specified contract after an
unpublished worktree was removed by workspace maintenance. It is **not** an
assertion that the former local tree or unpublished implementation was recovered.

The single immutable approved registry entry
`synthetic_cognitive_perspective_train_v1` materializes deterministic
TRAIN-only choice exercises for a canonical `ObjectiveContract` with domain
`empathy`, sole skill `cognitive_empathy`, one approved focus tag, and training
rubric ID `perspective_choice_train_v1`. All other learning scopes, including
reactive and emotional empathy, are rejected. The only grade supplied is an
operator-side transparent `100` for the expected choice or `0` otherwise;
there is no ability to assess genuine interpersonal empathy or to certify
mastery. The opaque `PreparedTrainingRoom.challenge.answer` stays
trainer-side; learner adapters receive `CognitiveTrainingRoom.trainee_view`
only, which has no answer key.

For bounded novelty the generator now holds **30 substantively distinct**
synthetic dilemmas in two approved banks of 15. A validated deterministic maze
uses up to 15 different cases per cycle. Cycle zero retains its original
**scenario index assignment**, but the R95 quality revision deliberately
changes its answer-choice wording and therefore its prompt bytes and current
content SHA values; no historical prompt-byte parity is claimed. Cycle one uses the second bank, so a fully exposed
15-room first cycle and 15-room successor can both be generated under
the operator's previous-semantic-exposure exclusion list. Further cycles
fail closed with an explicit bank-exhaustion error. This is finite source
catalog expansion, not a generative empathy model or proof of adaptive learning.
For each
generated prompt it reconstructs a **scenario-content** SHA-256 independently
of the per-case ID, A/B/C labels, or displayed option order. The SHA is over
NFKC/casefold/whitespace-normalized core behavioral dilemma and sorted normalized
choice texts, domain-separated by a versioned schema. The provider's cosmetic
setting preamble is deliberately excluded: changing a setting cannot bypass a
previously exposed core dilemma. All content digests must be
unique within a plan, and strictly validated operator-supplied forbidden
semantic hashes reject known reserved-exam or prior-training overlap.
Operator-supplied exact prompt SHA-256 and case ID exclusions add a second
strict boundary. This bounded lexical canonicalization **cannot detect all
semantic paraphrases**; no blind held-out evaluator is involved here.

R95 hostile review caught two shortcuts in the earlier choice wording: the
only option with “may/might” was always the answer (6000/6000), and choosing
the longest option was correct in 5800/6000 generated trials. The revised
30-scenario corpus uses semantically plausible hedged hard negatives, varied
gold wording, and a deliberate mix of short, medium, and long correct answers.
Deterministic 6,000-case probe tests reject the shortest/middle/longest length
rank, first hedge/unhedged, fewer absolute terms, and first non-"They" sentence
as perfect strategies; all three answer-length ranks and correct/incorrect
unique hedge cues must occur at substantive frequencies. A joint contingency
guard also requires every correct-option length rank to occur with both a
correct and an incorrect unique hedge cue and with non-unique hedging; this
rejects a simple combined cue such as "unique hedge means pick the longest."
These tests expose
simple shortcuts but do **not** prove that more sophisticated lexical models
cannot exploit this small static corpus. The previous v2 source-content commitments remain explicit
denylist aliases by underlying dilemma, so operator-provided old semantic
exposures are still blocked after wording changes. This compatibility does
**not** authenticate a complete history: an operator supplying only old raw
prompt hashes, or omitting prior semantic exposure evidence, cannot prove
cross-version freshness. External Q6 custody remains a release gate.

The trainer-side room stores a pinned local source-record commitment. Replacing
one approved scenario with another while retaining the case ID fails source
scoring unless the source record is regenerated; the unkeyed commitment can
itself be recomputed and replaced by someone controlling the Python process.
It catches accidental/replacement inconsistencies, **not** malicious same-
process source substitution. It is not an independent signer, tamper-proof
authority, or protected evaluator.

Generated records expose false-only properties for host execution, training
authorization, examination, and graduation. They do not carry custody or
physical-run evidence. This patch does not create pretrained weight updates,
checkpoints, measured learning gain, host leases, or authority to deploy.

The mandatory CPU workflow covers the module and its test file for both PR
events and pushes to `main`. At least 28 source-only tests are required without
skips, including seed `3`, all 99 nine-room seeds, 15-room stress, replay/heldout
denylists, focus/rubric/scope rejection, false-only authority, prompt/gold
tamper rejection, and answerless learner views. `tests/test_ci_membership.py`
asserts membership. The R95 extension adds source-only two-cycle novelty,
correct/wrong TRAIN scorer and third-cycle exhaustion regressions. Existing
Minotaur CPU gates remain mandatory.
