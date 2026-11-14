# Structural-transfer evidence prescreen v1

This module is a **fail-closed source-level prescreen**, not a model graduation
authority. It complements `graduation_gate.py`; it does not replace independent
evaluator custody, physical checkpoint readback, protected private questions,
statistical admission controls, or the official Qwen TrainingDay process.

## What it prevents

The public `minotaur-exam` command runs eight fixtures across changing seeds
and wording variants. The same source mechanics can be recognized by a
zero-update catalogue, so eight correctly repeated answers **are not eight
structurally new observations**. An executed regression runs the actual public
exam at three different seed/variant combinations and obtains all eight of the
same declared mechanic identities and verdicts. Its attempt to make TRAIN,
HELDOUT and TRANSFER from those runs is rejected for split overlap.

`structural_transfer_screen.py` requires:

1. A precommitted, immutable, exact case roster, with each prompt and case ID
   unique. At least eight distinct declared mechanical families must appear
   in **each** evaluation slice. Families cannot recur in training, heldout,
   or transfer under another prompt or ID.
2. Declared SHA-256 identities for base, candidate before/after, candidate
   reloaded, zero-update, corrupted-label checkpoints, plus a bounded nonzero
   optimizer-step count and a changed candidate checkpoint. Those are
   *declarations*, not actual host measurements.
   The untrained comparator `base_sha256` need not equal the initialized
   candidate checkpoint before adaptation (for example, an adapter can have
   an initialized state); the screen does not reconstruct adapter/base lineage.
   That relationship requires an independent model-loading and update receipt.
   A candidate-after checkpoint **identical** to the declared base,
   zero-update, or corrupted-label control is rejected: signed claims of a
   performance gain cannot establish a treatment effect between identical
   checkpoint bytes. This still does not verify that reported hashes match
   real host files or runtime model state.
3. A complete evaluator report covering the exact roster. It binds to the
   manifest digest and is authenticated with a versioned-domain HMAC key.
   The key must be genuinely kept outside the trainee for authentication to
   mean anything. HMAC does not establish that the signed assertions are true.
4. Matched family-level results for candidate, base, zero-update,
   corrupted-label, and memorizer controls. All paraphrases of one mechanic
   count as **one** unit. A perfect memorizer blocks an equally perfect
   candidate: no control gain, no transfer credit. In **each** heldout and
   transfer split the candidate must pass at least 75% of declared families,
   exceed **each** matched control by at least ten percentage points, **and**
   pass a one-sided exact paired-family sign test at `p <= 0.05` against
   every control separately. Family outcomes are all-or-nothing across a
   family's paraphrases, and tied candidate/control outcomes are excluded
   from the sign test. This is a conservative conjunctive prescreen, not a
   claim of a calibrated population generalization rate.

   **Counterexample, reproduced before the correction:** a candidate that
   passes all eight families while each control passes seven of the same
   eight cleared the old `>= 10%` gain-only screen (12.5 percentage points).
   It has just one discordant matched family: an exact one-sided sign-test
   `p = 1/2`, not meaningful paired evidence. The regression now blocks it.
   A synthetic seven-out-of-eight candidate facing controls that fail every
   family still reaches only the pending-custody disposition.

## Verdict and epistemic limit

`assess()` returns `BLOCKED` for malformed, missing, public-only, contaminated,
unreloaded, non-updated, or non-improving evidence. Even a fully valid synthetic
fixture can return only `PENDING_INDEPENDENT_CUSTODY`, **never**
`GRADUATED` or `VERIFIED_LEARNING`.

The source-visible evaluator may lie about structural labels, model identity,
optimizer steps, heldout secrecy, or observed correctness. No code inside the
trainee's process can authenticate those physical claims. The exact conditional
sign-test tail assumes independent, exchangeable declared family-level
discordances under its null; it cannot establish that the *actual* mechanical
families are independent, assess an external transfer population, or account
for evaluator fabrication. Downstream independent qualification must supply
appropriate controlled model execution, semantic independence review,
precommitted study design, and statistical evidence.

The negative-control tests also show that two **isomorphic exact-cover graphs**
with renamed sites and reordered crews receive different raw provenance
fingerprints. Those digests may identify serialized source records, but they
cannot by themselves establish distinct *mechanics*. A fabricated report
using these distinct strings can still reach only `PENDING_INDEPENDENT_CUSTODY`;
an external evaluator must independently canonicalize or attest to graph
isomorphism classes before claiming structural transfer.

The focused tests use a plainly test-only HMAC key and synthetic good results
for checking the contract. They do not call a neural model or make parameter
updates. The real public-exam comparison is the contamination counterexample.

## Reproduce

```sh
python -m unittest tests.test_structural_transfer_screen -q
python -O -m unittest tests.test_structural_transfer_screen -q
```

These tests cannot authorize a production Minotaur graduation. Honest
uncertainty is a required outcome until model training, private evaluator
isolation, provenance, and genuinely unfamiliar mechanics have been verified
independently.
