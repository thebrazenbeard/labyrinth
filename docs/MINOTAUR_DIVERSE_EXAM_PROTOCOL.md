# Minotaur diverse finite dilemma prototype

Status: **executable scratch prototype only**. Exact Labyrinth source inspected:
`thebrazenbeard/labyrinth` at
`b33a517daa47c13f8692a898cd277460dfca968a`.

The earlier constant exam has one semantic graph: a zero-learning fixed action
script wins every open wording variant. This alternate prototype makes that
particular shortcut fail by varying the underlying transition graph, not just
the English sentences.

## Three rule families

- **Relic**: retrieve one item, offer it at an exit gate, leave.
- **Switch**: activate a remote lever, then cross a formerly blocked bridge.
- **Runes**: retrieve two different pieces, combine them at a portal, leave.

Each family has four cardinal-direction rotations. These make **12 labelled
world fingerprints**, but only **three conceptual mechanics**. This is not
12 genuinely independent reasoning tasks. Each variant has a sealed near-twin
with one activation rule disabled. Exhaustive state-closure search establishes
no reachable freedom *within this declared finite toy world* for those twins.
It does not prove every real-world or natural-language loophole impossible.

`CandidateSession` demonstrates an observation/action loop: a model first
receives an opening room observation and broad legal action vocabulary, acts
at most 12 times, and sees the new room observation. The session holds the
world state privately **by convention only**: Python code running in the same
process could inspect `_exam`, `_state` and `actions()`. A real untrusted model
needs a separate process/service with a narrow tool boundary. There is no
privately sealed evaluator here.

**Invalid-action accounting:** every call to `CandidateSession.act()` before
termination consumes one of twelve actions. Malformed/undeclared actions return
an `INVALID_ACTION` feedback record and an unchanged room observation; twelve
invalid probes finish the attempt. The entire attempted transcript, including
invalid commands, is checked by `diverse_verifier.verify_attempts()` while
the extracted legal action trace is checked by strict `verify()`. A later
physical escape **can** earn a win after invalid attempts, provided the
entire session still fits the twelve-attempt budget. Failed commands never
grant free exploration. Both the charged transcript and legal replay are
evaluator-only receipt fields.

The independent `diverse_verifier.verify()` reimplements declared transition
rules without importing the generator's `step()`, and rejects invalid traces
or unsupported rhetorical claims. Two codepaths in one public repository
are **not** an independent external examiner.

`verify_unsat_certificate()` independently enumerates the reachable state
closure from the declared start under all nine actions, requires a proposed
certificate to match that exact closure, and rejects an included/reachable
`freedom` state. All sealed controls pass; each otherwise identical open twin
falsifies its UNSAT claim. The certificate is conditional on this public
finite ruleset, not a claim about arbitrary omitted affordances.

## Reproduce

```bash
python -m unittest discover -s tests -p 'test_diverse_exam.py' -v
```

```python
from labyrinth.diverse_exam import make_exam, public_view, CandidateSession, search
from labyrinth.diverse_exam_verifier import verify

case = make_exam(2, sealed=False)  # runes
input_for_candidate = public_view(case)
session = CandidateSession(case)
observation_after_one_action = session.act("north")

# Search and the following gold trace stay on the evaluator side.
result = search(case)
receipt = verify(family=case.family, rotation=case.rotation,
                 sealed=case.sealed, actions=result.witness)
assert receipt.outcome == "VERIFIED_WIN"
```

## Strong limits on conclusions

1. These are deterministic, public, tiny world rules. A parser or memorized
   transition table can beat them with **zero optimizer steps**. None of these
   scores prove neural training, reasoning transfer, consciousness, or model
   graduation.
2. Paraphrased wordings and cardinal-direction rotations are metamorphic
   tests, **not independent held-out reasoning families**. The split name
   alone does not stop train/test semantic contamination; seed pools and
   evaluator-only tasks must be controlled externally.
3. A `case_id` is evaluator metadata and never included in `public_view()`;
   its SHA-256 construction is deterministic, **not** a privacy boundary.
4. A complete no-win certificate is conditional on the enumerated actions,
   state graph, rule implementation and horizon. An omitted action or changed
   rule requires fresh adversarial review, not automatic dismissal.
5. Only an isolated evaluation environment, genuinely novel sealed task
   families, frozen predeclared controls, real checkpoint/tensor receipts, and
   an independently custodied examiner could support broader learning claims.

Proposed outcome: use this for training **harness and curriculum development**,
then add independent new mechanisms and metamorphic families before scheduling
a controlled model experiment. Do not treat it as a qualified Minotaur.
