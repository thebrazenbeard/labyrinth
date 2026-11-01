# R100 Minotaur-to-HF TRAIN exposure source bridge

**Status: SOURCE_ONLY_UNATTESTED — merge/deployment HOLD.** This change
connects feedback-conditioned Minotaur TRAIN chamber selection to a private
HF-compatible case catalog and an exact planned optimizer-visit roster. It
does **not** record a physical pretrained model update or authorize Q5–Q9
graduation.

## Source contracts

`collect_minotaur_hf_roster` consumes a fresh, approved arithmetic TRAIN
`TrainingCorridorRunner`, a matching bounded HF `PilotConfig`, and a trusted
operator callback receiving only a prompt-level `TraineeRoomView`. The
operator submits responses one chamber at a time, so each outcome can affect
the policy's next TRAIN selection. The trusted operator alone obtains each
approved private `Challenge`, including gold supervision. Case identities,
prompts, TRAIN splits, registered admission hashes, callback integrity and
nonduplicated staged examples are checked before returning a roster.

The result carries private `train_cases` and
`ordered_optimizer_exposures` — ordered `(case_id, SHA256(UTF-8 prompt))`
pairs for every optimizer visit, not just distinct examples. The HF source
trainer can accept this exact schedule, preflight it before encoding, require
each optimizer step's source commitment to match, and record visits only
after the step returns. Matched candidate, zero-update sham and corrupted-
label controls are held to the same pinned source visits; partial failures
retain the count of already completed steps.

The offline HF report auditor treats this as an optional addition to older
v2/v3 reports. If any arm includes `visited_exposures`, all arms must include
the exact regenerated case/prompt schedule. Malformed, reordered or forged
visits fail the source audit; legacy reports without exposure receipts remain
readable. This is a consistency check, not independent execution custody.

This is a **two-stage source seam**, not an end-to-end pretrained run:
ordinary `run_pilot` still generates its own TRAIN catalog and does not
consume a staged Minotaur roster. Source-level hash checks do not
authenticate which token IDs a real model saw, which gradients were applied,
or who physically operated the trainer. The callback and trainer share a
Python process; the exposed trusted-operator accessor is not a sandbox
against code that can access the runner.

## Cross-run evaluation firewall

The Q6 arithmetic study now accepts an explicit history of previously
exposed TRAIN prompts, snapshots it before generating new material, and
rejects any later heldout or transfer item with the same bounded arithmetic
semantics. Current TRAIN replay remains permitted. Previous EVAL histories
remain barred from all new partitions. An optional domain-separated digest
pins the caller-provided history, but cannot prove that the caller supplied
**all** past exposures; `history_custody_verified` remains false.

## Evidence and release boundary

Adversarial local tests cover unauthorized pending-case access, feedback
routing, reordered or mutated TRAIN exposures, identical control visits,
mid-run corruption, and historical TRAIN-to-EVAL collisions. The pinned
neural CPU workflow includes the new bridge source/tests in both event path
filters and requires non-skipped module-level test minima. These tests are
source-integrity checks, not pretrained GPU evidence.

**Still required before promotion:** independent exposure custody, a
verified pretrained update with matched zero-update and corrupted-label
controls, evaluator-held Q5/Q6/Q7 material, a host-owned signer and Q8
lease/receipt, the live Executor route, and explicit Q9 graduation. A draft
source PR may be reviewed while all such gates remain closed.
