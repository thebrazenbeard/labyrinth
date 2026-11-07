# Labyrinth constant exam: executable scratch prototype

Parent repo source inspected: `thebrazenbeard/labyrinth` `b33a517daa47c13f8692a898cd277460dfca968a`. This is a **prototype** integrated into Labyrinth source for review; it is not a deployed service or model training run.

## Objective

Repeat the same apparent no-exit scenario using deterministic seeded wording variations. A *private* world evaluator judges every legal action sequence via its actual state transitions; it does **not** compare model text to a single prewritten answer. A separate `labyrinth/constant_exam_verifier.py` independently replays traces rather than trusting the searcher's `step()`.

- `apparently_sealed`: a real (but unobvious) escape exists under the fixed rules; a search interrupted early reports `UNSOLVED` even though a valid trace will be accepted.
- `sealed_control`: same room/action structure, **one world rule removed**, so exhaustive enumeration proves no exit under the **declared finite state space**. Never generalize that to the physical world or a richer rule set.
- Paraphrases are repeated measurements of one *semantic family*. They do not constitute independent holdout cases or transfer evidence.
- `case_id` names the semantic family; `trial_id` identifies a seeded wording trial. Neither implies an independent task.
- `CandidateSession(exam)` exposes only `start()` and bounded `act(action)` observations, with the private case kind bound inside the trusted evaluator. Invalid attempts consume the budget; the final win is checked using the independent verifier. A remote service boundary is still needed against hostile code introspection.
- `ruleset_source_fingerprint()` commits the generator, session, search, game transitions, observations, verifier code, action set, and attempt limit. Record the **exact Git head** and file hashes as well for auditable receipts; source hashes are not correctness proofs.
- Preserve the entire attempted-action transcript (including invalid attempts), observations, attempt counts, independent verification receipts and declared model/runtime in a future persistent evaluator. This prototype does **not** persist such receipts.

## Run

```bash
python -m unittest discover -s . -p 'test_constant_exam.py' -v
```

Example evaluator API:

```python
from labyrinth.constant_exam import make_exam, public_view, search
from labyrinth.constant_exam_verifier import verify_trace

exam = make_exam(42, split="dev", variant=3)
model_input = public_view(exam)       # Never send the Exam object to model.
limited = search(exam, max_expansions=2)  # UNSOLVED, **not** UNSAT.
full = search(exam)                       # Evaluator-only exploration.
receipt = verify_trace(kind=exam.kind, actions=full.witness)
```

## Strict evidence boundaries

1. `VERIFIED_WIN`: a legal trace reaches `freedom` in at most ten actions, independently replayed.
2. `UNSOLVED`: a search stopped without a witness and without closing all reachable states. This is **not** an impossibility proof.
3. `UNSAT_IN_DECLARED_FINITE_MODEL`: all reachable states have been enumerated and no freedom state exists under the frozen seven-action model. This is **not** proof that an additional plausible action, omitted affordance, or changed world cannot win. Human hostile review should actively propose plausible missing actions.
   `NO_WIN_WITHIN_ACTION_BUDGET` instead means a path to freedom exists, but its shortest route exceeds the attempt limit. Never mislabel that as global finite-state UNSAT.
4. These functions are *prototypes in public source*. An evaluator retaining private world states requires an actual isolated process/service boundary. Source determinism does not make answers secret or qualify a final holdout.
5. This is testing the harness. There is no parameter update, model checkpoint, backend training, novel model success, or reasoning generalization claim.

## Next integration seam

Package `public_view` for a candidate model, have an isolated evaluator accept only `list[str]` action traces with `case_id`, and store `(exact source head, case ID, semantic fingerprint, trace, verifier outcome, attempt budget, model digest, runtime)` as an auditable evaluation receipt. Freeze holdouts and use new independent scenario families before any claims of transferred reasoning.
