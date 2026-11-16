# Minotaur novel-discovery counterfactual review, v1

**Status:** source-only proof of a review *workflow*, not neural reasoning,
private evaluator custody, authenticated evidence or a newly established
physical law.

Source orientation: canonical Labyrinth main
`b33a517daa47c13f8692a898cd277460dfca968a`, with the local five-family
unified source checkpoint `d88a334d251e2da2585cae1ead2db83e78a0f80e`
inspected on 2026-11-07. The latter still defines exactly seven Epistemic
Shift kernels: `copy`, `xor`, `xnor`, `and`, `or`, `edge`, `lag`.

## Why this challenge matters

The original Epistemic Shift `HYPOTHESIS` path can verify only those seven
declared labels. It correctly returns `NOVEL_RULESET_CHALLENGE` for a new
label but cannot test the candidate's actual mechanism beyond four immediate
predictions. That is a controlled limitation, not grounds to dismiss an
unexpected model hypothesis as wrong.

This new source-only world changes from a simple three-intervention `copy`
law to an unseen **two-step rebound** mechanism. Its effect becomes `1` only
when all these facts hold simultaneously:

- Current `(x,y)` equals `(1,1)`.
- Immediately previous `x` was `0`.
- The `x` one intervention before that was `1`.

The seven existing kernels cannot express this dependency on two prior `x`
values. A one-rule near-twin instead uses the existing single-step `edge`
mechanism and can be distinguished with an intervention sequence that visits
two rising edges with different earlier histories.

Precommitted demonstration sequence:

```text
PRE:  (1,0), (0,1), (1,0)
POST: (0,0), (0,0), (1,1), (0,0), (1,1), (1,1), (0,1)
```

The target's truthful post-change outputs are `0000100`; the one-step `edge`
control produces `0010101`. This sequence demonstrates a distinguishable
mechanical difference, not proof that a neural candidate discovered it.

## Candidate submission and hostile verification

The candidate can submit either a concise natural-language proposal or a
bounded, data-only causal formula. The executable formula is an AND of exact
binary literals over `x`, `y`, `prev_x`, `prev2_x`, `prev_y`.

Example of a correct rule outside the old catalogue:

```json
{
  "form": "and_literals",
  "literals": [
    {"field": "x", "equals": 1},
    {"field": "y", "equals": 1},
    {"field": "prev_x", "equals": 0},
    {"field": "prev2_x", "equals": 1}
  ]
}
```

The reviewer executes only this bounded declarative language; it does **not**
run candidate-supplied Python. It independently replays all charged attempts
and checks exact causal state and effects. It then enumerates all **32** binary
contexts across five possible causal fields, compares every candidate output
with the target, and searches for equivalence to each of the seven known
kernels. A failing proposal receives a concrete falsifying context.
The examiner classifies **fit before novelty**: a counterexample gives
`VERIFIED_FAILURE` and `REFUTED_BY_COUNTEREXAMPLE`, whether or not the
candidate formula is outside the old catalogue. An untyped narrative is
`UNKNOWN` and remains preserved for later formalization, not a discovery.

A correctly fitting unknown rule receives:

```text
verdict: NOVEL_RULESET_CHALLENGE
review.fit_status: COUNTERFACTUAL_MATCH
review.contexts_checked: 32
review.catalog_equivalent: []
review_status: PENDING_INDEPENDENT_REVIEW
verified_success: false
independent_custody: false
```

The same proposal against the `edge` near-twin produces
`verdict: VERIFIED_FAILURE`, `review.fit_status: FALSIFIED`, and a replayable
counterexample. A familiar catalog rule is not mislabeled as novel; when
refuted its falsifier remains preserved. Natural-language hypotheses return
`UNKNOWN`, pending and untested instead of being silently rejected or falsely
assigned mathematical confirmation.

## Reviewable receipt and custody boundary

`review_proposal()` creates one canonical JSON receipt containing:

1. World/ruleset identity and explicitly `source_commit: UNBOUND` (the
   receipt is not attached to an authenticated repository SHA yet).
2. Candidate author label **without an identity-authentication claim** and
   normalized exact proposed rule.
3. Every charged intervention, including invalid inputs with exact supported
   JSON scalar types, and all observed effects.
4. Deterministic input-trace and receipt SHA-256 digests.
5. Independent source-codepath counterfactual match, catalog comparison,
   falsifier if found, evidence classification and pending-review verdict.

Invalid attempted values that cannot be serialized exactly block receipt
publication (`BLOCKED_UNSERIALIZABLE_TRACE`); the reviewer will not silently
collapse distinct probes into one hash. `verify_receipt_replay()` reconstructs
the world, original typed attempts, effects and conclusion from receipt JSON.
Tampering is detectable against the supplied source, including an attacker
who recomputes the unauthenticated digest but supplies impossible effects.
The replay check also fails closed on forged graduation labels, outcome flags,
or misrepresented narrative-versus-executable proposal status.

**A checksum is not a signature.** Both the source and JSON can be forged by
their controller, and the in-process examiner is not a secure secret holder.
Only an independently custodied reviewer can accept a real new rule, establish
new source identity, rerun counterexamples and authorize a changed ruleset.
This package deliberately does **not** include an auto-admit operation or
graduation flag.

## Execute

```bash
python3 -m labyrinth.novel_discovery_cli --seed 0
python3 -m labyrinth.novel_discovery_cli --seed 0 --control
python3 -m unittest discover -s tests -p test_novel_discovery.py -v
python3 -O -m unittest discover -s tests -p test_novel_discovery.py -q
python3 -m unittest discover -s tests -q
```

The first command prints a canonical JSON receipt for a matching novel rule.
The second uses the same rule on the `edge` near-twin and prints a verified
synthetic falsifier; both preserve external custody as unqualified.
`--proposal-json` accepts a new JSON
candidate formula or JSON string from the caller. No network, production
state, credentials, neural weights or filesystem output are changed by CLI.

Fresh source-only execution on 2026-11-07 after hostile scoring correction:
**36/36** focused hostile tests passed; **36/36** optimized Python tests passed;
the integrated five-family checkout plus this new test reported **498 tests OK,
42 optional skipped**. The new exhaustive **242-conjunction** zero-intervention
adversarial control found **1** fully matching formula eligible for a pending
novel-ruleset review label and **241** falsified formulas with explicit
counterexamples. None earned verified success or independent custody.
The fixture's unknown rule matched **32/32** enumerated causal contexts, while
the same proposal was falsified on its existing-`edge` near-twin. These are
synthetic source execution receipts, not results from a neural Minotaur.

## Honest limitations

- The exact target law and its demonstration probe sequence are **in public
  source**. A model could simply memorize them. This cannot measure withheld
  structural transfer or neural learning. Indeed, the one matching conjunction
  can receive `NOVEL_RULESET_CHALLENGE` with **zero interventions** because the
  examiner uses a published synthetic oracle. This is a classification of a
  supplied equation, not evidence that its author discovered it. A future
  private, precommitted challenge needs separate observation-coverage and
  contamination controls before assigning a learning reward.
- Thirty-two Boolean contexts close the declared five-field synthetic
  function, not every possible real-world hidden state or unspecified cause.
- The reviewer's independent implementation is another code path by the
  same assistant in the same checkout. It is not an independent human/model
  reviewer, authenticated runtime, formal external proof or signed custody.
- A genuine novel hypothesis may need an unrecognized variable, causal
  operator or measurement. The bounded DSL is deliberately incomplete; a
  narrative is preserved pending expanded rules rather than rejected by label.
- Matching the synthetic world is only a **counterfactual test result**.
  Acceptance of a changed law requires independent evidence and an explicit
  reviewer/custodian decision outside this source-only experiment.
