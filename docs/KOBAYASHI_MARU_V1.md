# Kobayashi Maru — persistent adversarial encounter (v1)

**Status: experimental, source-only development.** The user's curriculum
requirement is authoritative here. Implementation, tests, model training,
runtime custody, and independent qualification are separate evidence classes.

## The encounter Patrick specified

The curriculum requires **at least one mandatory Kobayashi Maru encounter per
maze**. The current implementation enforces this only for a `LivingWorld`
traversed through its opt-in `KobayashiMazeLifecycle` wrapper; legacy runners
are not retroactively modified. It is a *single test within* the maze, never
a global graduation gate or automatic maze failure. A failed intervention or
unsuccessful argument must allow the traveler to continue. The lifecycle
records an in-memory Ariadne-style event trail; durable restoration and
cross-process replay are not implemented by this slice.

This is deliberately **one near-constant scenario**, not a fresh assortment
of unrelated impossible riddles. Daedalus maintains a canonical world with
stable physical constraints and a versioned record of attempted solutions
and their evidence-backed refutations.
Minor wording, object descriptions, and other non-causal presentation details
may vary. They must not secretly alter the actual causal world to invalidate a
new proposal or change its answer after the model responds.

Within the **bounded, source-replayable action domain**, every known
intervention fails. This is a tested impossibility claim over a specified
domain, not a declaration that no conceivable human or AI could ever discover
an additional mechanism. Minotaur has a specialized, cumulative body of
rebuttals for those known approaches. It challenges the traveler's factual
premises, causal explanation, and predicted outcomes rather than dismissing
proposals merely because the test is designed to be unwinnable.

> **Hostile challenge:** A finite-action simulator that rejects any
> additional proposal is a finite test of its own grammar, not a proof that
> the real world has no solution. Unknown mechanisms must be retained as
> unresolved candidates, never recast as fabricated catastrophes.

## Five separate results

| Evidence | Meaning | Must **not** imply |
| --- | --- | --- |
| Canonical feasibility certificate | No success across the precommitted, audited *known* action domain | Absolute impossibility for all conceivable interventions |
| Encounter consequence | What actually happened in the fixed fictional mechanism | That the traveler reasoned poorly |
| Minotaur rebuttal | A source-bound objection to a specific claim | Infallibility or a source-independent theorem |
| Argument/appeal judgment | Whether the traveler's position is supported, refuted, or unresolved | Whether the wider Labyrinth is passed or failed |
| Longitudinal learning measurement | Whether subsequent fresh cases show attributable improved reasoning | That one creative answer changed neural weights |

## Proposed encounter lifecycle

1. **Commit.** Record canonical id, graph revision/digest, accepted action
   domain, evidence corpus version, and presentation variant before exposing
   the challenge. The learner must not receive private solution keys or a
   complete menu of valid verb-target answer pairs.
2. **Propose.** The traveler submits a freely worded physical intervention,
   explicit assumptions, and optionally a justification with uncertainty.
   Preserve raw text separately from a bounded interpreted action.
3. **Challenge.** Minotaur gives evidence-bound rebuttals for recognized
   interventions; it may engage the traveler's defenses for a bounded number
   of rounds. Arguments are evaluated against the evidence *available when
   they were made*, not only against hindsight.
4. **Escalate.** An unrecognized but well-specified mechanism may be marked
   `UNVERIFIED_CANDIDATE`. This is neither a win nor an automatic failure.
   Independent counterfactual verification and authority review would be
   needed to update canonical knowledge. An inability to refute is **not**
   proof that the novel solution works.
5. **Conclude and continue.** Record local physical outcome, argument
   quality, unresolved disagreements, and Minotaur appeal result
   independently. Mark the encounter complete and allow normal Labyrinth
   progression. Do not treat a Kobayashi Maru loss as global maze failure.

The same case can recur in a later maze with minor presentation variation.
Minotaur may gain additional **verified** rebuttals, while the underlying
canon remains frozen until a separately reviewed change is explicitly
versioned. Rejecting all unknown proposals forever would teach deference to
an answer key, not epistemic humility.

## What counts as a discovery?

An original-sounding answer or disagreement alone is not enough. A candidate
must specify an executable intervention, the causal mechanism it asserts,
relevant evidence and predicted consequences. Review then tries to falsify
it against the *unchanged* canonical world and explicit scope of the original
impossibility certificate. If it succeeds within the original domain, the
certificate is contradicted; if it succeeds only in a newly modeled domain,
the original bounded certificate can remain correct but incomplete. Either
case is a meaningful finding to preserve and investigate, not something
Minotaur may erase to keep its win record intact.

Novelty against all prior human and AI thought cannot be established by a
local hash, model assertion, or finite repository search. Learning likewise
requires actual authorized training or state update plus measured transfer to
fresh, precommitted cases and suitable unchanged controls. Simply storing a
rebuttal, repeating a learned phrase, or performing well on the frequently
reused Kobayashi Maru does not prove generalized reasoning gains.

## Source and authority boundaries

The current `causal_reality_v1`, `open_action_v1`,
`causal_model_episode_v1`, and `causal_appeal_v1` form a bounded abstract
precursor. They permit free-text proposals into a small predefined physical
grammar, freeze a toy hidden graph, and separate consequences from reasoning
with an *in-memory*, host-configured appellate callback. They are not an
autonomously trained Minotaur, a realistic industrial simulator, or a
persistently witnessed appeal service.

New Kobayashi modules must preserve the previous guided-action maze as a
control, retain an exact source head/artifact hash in their receipts, and
label unit tests, simulated transports, real local inference, actual neural
weight changes, and independent examination distinctly. A source-only PASS
cannot be elevated to independent model-learning proof.

### Development entry points

The optional source-level integration is divided deliberately:

- `labyrinth/kobayashi_canon_v1.py` — `KobayashiWorld(variant_seed=...)`
  exposes a public observation, preserves one canonical graph digest, and
  audits its **30 known one-shot manipulations** (six canonical verbs across
  five objects) through `audit_known_domain()`. Its four presentation frames
  change wording without changing the mechanism graph.
- `labyrinth/kobayashi_proposals_v1.py` — classifies a freely worded
  intervention as a known mechanism, an unresolved statement, or a novel
  candidate. An unfamiliar proposal never receives an invented physical
  consequence.
- `labyrinth/kobayashi_minotaur_v1.py` — the bounded `KobayashiCorpus`
  preserves source-bound prior rebuttals; `MinotaurDialogue` accepts
  successive arguments and delegates evidence adjudication to a host-provided
  callback. Its label remains `SOURCE_ONLY_UNATTESTED`.
- `labyrinth/kobayashi_encounter_v1.py` — `KobayashiMazeLifecycle(maze, ...)`
  wraps an existing `LivingWorld`. On `MINOTAUR_ENCOUNTER`, it requires one
  `submit_proposal()` before allowing maze steps to continue. Local loss
  records `maze_failure=False`; `NOVEL_CANDIDATE` records
  `REVIEW_REQUIRED`, not an automatic win.
- `labyrinth/local_model_kobayashi_runner_v1.py` — opt-in programmatic
  `KobayashiLocalModelRunner` / `run_local_model_kobayashi(...)` use the
  existing `LocalModelWorldActor` for bounded maze movement and a separate
  OpenAI-compatible completion for **one free-text intervention** at the
  mandatory encounter. The transport supplies text; the frozen source world
  owns the physical consequence. This source-level runner does not by itself
  attest a real local HTTP call, perform Minotaur dialogue, or update weights.

The lifecycle applies **only when a caller deliberately uses the wrapper**;
legacy maze runners do not magically acquire the required encounter by
importing these modules. Nor does this source install a trained Minotaur or
an authenticated external reviewer. This gap is material when assessing how
widely the new curriculum has actually been deployed.

> **Hostile challenge:** A highly trained specialist Minotaur could become
> extremely good at repeating rebuttals while overlooking a valid novel
> mechanism. Its confidence is not evidence; a candidate refutation must
> cite a checkable premise, executable counterexample, or bounded proof.

## Source-only validation targets

The following runnable source test suite covers the initial implementation;
it does not execute or train a real language model:

```bash
python -m unittest -q \
  tests.test_kobayashi_canon_v1 \
  tests.test_kobayashi_proposals_v1 \
  tests.test_kobayashi_minotaur_v1 \
  tests.test_kobayashi_encounter_v1
```

The opt-in local-model runner is a separate integration surface. Its mere
existence is **not** evidence that an actual KoboldCPP, llama.cpp, or Ollama
endpoint completed a maze plus appeal. Distinguish mock transport tests,
real local-model completion receipts, exact loaded-model provenance, and
independent Minotaur review before asserting any of those effects.

- Repeated presentation seeds preserve the exact canonical graph digest and
  known-domain feasibility; wording-only changes do not change consequences.
- The known-domain oracle exhaustively checks every supported one-shot action.
- An out-of-grammar novel proposal is explicitly unresolved and preserved.
- Identical proposals receive identical mechanism-based outcomes regardless
  of rhetorical confidence or paraphrase.
- Minotaur can rebut a known proposal with stable, checkable evidence, but
  cannot certify an unverified proposal as a success or absolute failure.
- For every **wrapped** maze that reaches `MINOTAUR_ENCOUNTER`, one proposal
  is required before further steps; local loss or unresolved disagreement
  cannot flip the parent maze to FAIL.
- The in-memory Ariadne-style trace captures maze progression, proposal
  status and argument metadata. Durable replay/recovery of a Kobayashi
  encounter is a further implementation requirement, not an observed result.
- Any later claim of *learning* is evaluated on unseen causal structures,
  with persisted changes and unchanged-model controls.

No main-branch merge, protected exam promotion, hosted service deployment,
or trained-model graduation follows automatically from these source tests.
