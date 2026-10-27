# Bounded propositional-logic curriculum and structural holdout

`labyrinth.propositional_logic` is a second, deliberately independent reasoning
domain. It does not call, import, or share an oracle with the modulo-97
register-program trace solver. Its public language is bounded to atoms `A`–`D`,
ASCII `!`, `&`, `|`, `>`, at most three formula nesting levels, 64 characters
per formula, eight proof claims, and 1,000 generated cases per request.

`generate_logic_challenges(seed, count, split)` produces balanced entailment
questions. `train` contains only modus ponens, conjunction introduction, and
implication-chain shapes; `holdout` contains only disjunction cases,
conjunction elimination, and negated-disjunction shapes. The structural names
are fingerprinted as metadata and are disjoint across splits. A SHA-256 domain
tag includes split, seed, index, and purpose, so the same numeric seed cannot
select the same task population in training and holdout. Final examinations
must nevertheless reserve previously unused seeds before any tuning.

The private `Challenge.answer` is `YES` or `NO`; trainee-facing prompts contain
only premises, conclusion, grammar, and output syntax—not the answer, a gold
field, or a positive/negative marker. `score_logic_response` accepts only a
complete `ANSWER: YES` or `ANSWER: NO` line. Prose, repeated answers, aliases,
and oversized output fail closed.

`generate_first_error_examples` creates alternating valid proof audits and
negative audits with a deliberately placed first invalid claim. The prompt
contains the proposed claims but not the private first-error index. The exact
`evaluate_first_error` examiner reparses each formula and uses exhaustive
truth-table entailment over at most four atoms; malformed premise or claim text
is an error at the first affected line. `score_first_error_response` accepts
only `FIRST_ERROR: NONE` or an in-range numbered error and recomputes the
oracle rather than trusting a generated label. The targeted tests use a separate
test-side postfix truth-table interpreter over the printed prompt as a check
against accidental shared-oracle mistakes.

This is a finite, public synthetic generator. The independent reference is
still source-resident and does not make this a blind or externally independent
exam. A custodian needs to pin source, reserve secret/unseen seeds, retain
private answer keys outside trainee storage, and run a separately controlled
examiner before treating any result as evidence. Passing these tests proves
deterministic grammar, scoring, and controls—not model reasoning, training
effect, or qualification.
