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

For bounded novelty the generator assigns 15 distinct base scenarios without
replacement by position in the validated deterministic maze. For each
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

Generated records expose false-only properties for host execution, training
authorization, examination, and graduation. They do not carry custody or
physical-run evidence. This patch does not create pretrained weight updates,
checkpoints, measured learning gain, host leases, or authority to deploy.

The mandatory CPU workflow covers the module and its test file for both PR
events and pushes to `main`. At least 28 source-only tests are required without
skips, including seed `3`, all 99 nine-room seeds, 15-room stress, replay/heldout
denylists, focus/rubric/scope rejection, false-only authority, prompt/gold
tamper rejection, and answerless learner views. `tests/test_ci_membership.py`
asserts membership. Existing Minotaur CPU gates remain mandatory.
