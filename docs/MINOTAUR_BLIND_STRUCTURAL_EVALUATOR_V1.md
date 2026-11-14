# Blind structural mechanics V1 — source-only examiner protocol

**Status:** runnable development examiner, **not** an independently secured
neural qualification. Draft source must be evaluated on its exact commit.
There is no hidden case set or trained-model result in this repository.

## Why this is not another public-seed replay

The existing public Minotaur exam has reusable labels, familiar mechanics,
and paraphrase variants. A family/case-label lookup without weight updates
predicted 120/120 public verdict labels across six seed/variant runs. It did
not produce winning action traces. A label score cannot establish new
reasoning.

This protocol instead generates **fresh, unlabelled-by-family transition
tables** *after the candidate is frozen*. Every trial specifies a bounded
finite machine with 10–15 states, three actions (A/B/C), and **two temporal
phases**: even-numbered actions use phase-0 maps and odd-numbered actions use
phase-1 maps. An action may be blocked (-1). The candidate receives a new
machine's complete rule table, initial state, goal and three-move budget
and must return an exact **PLAN: A B C** action trace (1..budget moves)
or **UNSAT**. It never receives the examiner's precomputed witness or answer
status in its candidate-facing JSON.

The generator intentionally produces equal counts of reachable and
unreachable cases, but this balance is **evaluator-local** and not a
candidate-facing clue. Its structural laws and state transitions differ
from the earlier fixed public Minotaur cases. Externally authored rule tables
are also accepted *without* adding source-coded case labels.

This is **not** proof of a wholly unseen semantics family. The two-phase
table interpreter is published in source and belongs to a known bounded
language. Real held-out novelty requires an examiner to author and custody
previously unseen table arrangements after model and evaluation policy freeze;
a new primitive outside this grammar requires a separately reviewed verifier.

## Execution: separate examiner-controlled private key from public question file

Prepare on the *examiner's isolated host*. For a demonstration only (no
independent custodian is provisioned by this code):

```powershell
python -m labyrinth.blind_structural_exam prepare --count 8 --out-private C:/exam/private.json --out-public C:/exam/public.json --out-precommit C:/exam/precommit.json
```

The printed `receipt_sha256` must be retained **outside the candidate's
control** before any responses. The private file contains oracle witnesses
and SAT/UNSAT flags. It must **never** be made available to a candidate
process, placed in shared logs, committed to public Git or leaked through an
unisolated tool. The public file contains only randomized trial IDs, prompts,
world rules and initial/goal states.

The candidate output file has exactly this shape (illustrative IDs only):

```json
{
  "schema": "labyrinth.blind-mechanics-responses.v1",
  "exam_id": "blind-00000000000000000000000000000000",
  "responses": [
    {"trial_id": "blind-000000000000000000000000", "response": "PLAN: A C"},
    {"trial_id": "blind-111111111111111111111111", "response": "UNSAT"}
  ]
}
```

Responses must match the exact public trial ID roster in order; empty, extra,
duplicate or reordered rosters fail closed. A valid plan is *executed* from
the actual initial state under the provided temporal transition tables,
rather than accepted because its text claims success. `UNSAT` is scored
only after **two separate source codepaths** agree no bounded trace exists:
shortest-path BFS and complete Cartesian action-trace enumeration. These are
internal consistency oracles, **not independent human/model reviewers**.

After the candidate responses are recorded, on the examiner host:

```powershell
python -m labyrinth.blind_structural_exam score --private C:/exam/private.json --public C:/exam/public.json --precommit C:/exam/precommit.json --responses C:/exam/responses.json --expected-precommit-sha256 ORIGINAL_RETAINED_PRECOMMIT_SHA256 --out C:/exam/scored.json
```

Optional externally pinned Ed25519 attestation arguments:

```text
--examiner-attestation C:/exam/signed.json
--examiner-public-key C:/exam/examiner.raw.pub
--expected-examiner-key-sha256 RETAINED_PUBLIC_KEY_SHA256
--examiner-exam-id YOUR_EXAM_ID
```

All four signature arguments must be provided together. The attestation
contract and limitations are in `LIVE_MODEL_OBSERVATION_V1.md`. Signature
verification authenticates the signer *relative to a caller-supplied public
key pin*, not whether the signer is actually independent or whether cases
were prepared before candidate access.

For *examiner-authored tables*, supply `--worlds-file C:/exam/rules.json`
to `prepare`; the file is a JSON array of 1–16 world objects with exact
keys `state_count` (4..16), `start`, `target`, `max_moves` (1..6)
and `action_tables`. `action_tables` has exactly A/B/C keys, each containing
a pair of phase transition arrays of length `state_count`. Entries are
next-state indices or -1 blocked. State and goal must differ. The author of
that external data and its novelty are **not** authenticated.

The precommit hashes exact raw bytes of both public and private files and
carries an unkeyed digest, original exam ID and explicit
`CALLER_CONTROLLED_UNATTESTED` custody marker. The scorer requires the
separately retained digest; it independently recomputes the exact original
private witnesses, source-public alignment and both oracle existence checks.
Score reports are exclusive-create; no keys or model weights are modified.

## Falsifiers and qualification ceiling

- A default **all-UNSAT zero-update control** gets only unreachable cases,
  not the deliberately reachable half. It cannot win by public case ID.
- A **public-table symbolic solver** may get every case right with *zero
  learning*. Its success proves the evaluator's replay works, not that an
  LLM reasoned or adapted. Require actual model outputs and lineage for that.
- A malicious caller controlling both original precommit SHA and source
  files can forge a complete exam; a signature from a colluding or
  retrospective signer does not remove this threat.
- Same-OS-account private/public directories are not a trust boundary.
  Independent examiner identity, restricted case/key ACLs, challenge freeze
  timestamp, secure model-serving output capture, before/after model weight
  provenance and OOD semantic review remain **UNKNOWN** without real probes.
- No auto-admission, graduation, learning flag or protected TrainingDay
  launch results from this tool. `neural_learning_verified=false`,
  `independent_custody_verified=false`, and
  `score_evidence=SOURCE_ONLY_LOCAL_REPLAY` are permanent here.

Reproduce source-only acceptance tests:

```bash
python -m unittest discover -s tests -p 'test_blind_structural_exam.py' -q
python -O -m unittest discover -s tests -p 'test_blind_structural_exam.py' -q
python -m unittest discover -s tests -q
```
