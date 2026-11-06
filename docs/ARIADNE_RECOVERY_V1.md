# Ariadne recovery v1 — full-state evolutionary ancestry (source-only)

**Status:** additive local implementation, not a qualified live trainer restore.
The source anchor is Draft PR #125, commit
`11f443819061bdd528dc208c9e08e2a80ab9a13e`. This slice does not edit
`SagaSession`, invoke `train_supervised`, or install a NAS control plane.

**Platform restriction:** The journal writer/replayer currently requires a
Linux or WSL Python interpreter with POSIX `flock`, `O_NOFOLLOW`,
`O_DIRECTORY`, hard links and `fsync`. Native Windows can *import* the module,
but `AriadneJournal.create/open` immediately raises
`UnsupportedJournalPlatform` before touching a path. A best-effort native
locking fallback would not meet the same tested durability contract. VN1
native Windows must use an explicitly qualified WSL/Linux worker or await a
separately reviewed native backend. Linux test success does not qualify WSL
mounts, NAS filesystem durability or live VN1.

## What Ariadne keeps

`labyrinth.ariadne_recovery_v1` is a small append-only journal that binds:

1. Founder **W0**, including a write-once `campaign_full_state_v1` manifest
   referring to real model base files, tokenizer, config, LoRA adapter (or full
   weights), optimizer, scheduler, RNG and data cursor.
2. Versioned curriculum worlds and their full JSON world states; each world
   version names a known parent world/version.
   The recovery descriptor's `selected_world_id` follows a champion's
   subsequently recorded child world, not the pre-TRAIN branch-world ID.
   This derived pointer is rebuilt from the write-once event history on each
   readback; merely creating an unrelated proposed world cannot select it.
3. Branch-specific observations, actions, outcomes and selected navigation
   path, bound to the currently selected physical parent and generation.
4. Before-effect `TRAIN_PREPARED` with an unguessable effect token, complete
   training-state effect scope, and the expected parent state.
5. After-effect `TRAIN_COMMITTED` with its matching token, trainer-reported
   completion and a distinct physical child manifest whose immutable ancestry
   names the exact parent and W0 origin.
6. Candidate evaluation metrics/evidence, explicit rejected candidates and
   exactly one selected candidate from a resolved cohort of at least two.
   The selected child becomes the *only* physical parent for the next round.

Each event records `schema`, strict integer `event_version=1`, sequence number, random event ID, unique
idempotency key, event type, typed payload, previous event digest and its own
domain-separated SHA-256. Event filenames are contiguous zero-padded sequence
numbers. Writers take the same OS `flock`, re-read the entire journal, validate
all physical manifests and apply all transition invariants before appending.
Entries are written to a unique staged file, file-fsynced, atomically linked to
the write-once sequence filename, and directory-fsynced **before the call
returns**. The caller may invoke a trainer only after `prepare_training`
returns its one-use `TrainPermit`. Crash-abandoned staged files are fail-closed,
not silently discarded.

## Minimum call sequence

```python
from pathlib import Path
from labyrinth.ariadne_recovery_v1 import AriadneJournal, trainer_request

ariadne = AriadneJournal.create(Path("/abs/campaign/ariadne"))
ariadne.initialize(
    w0_manifest_path=Path("/abs/campaign/W0.json"),
    world_id="world_1", version=1,
    curriculum={"objective": "public development"},
    world_state={"rooms": ["entrance", "threshold"]},
    idempotency_key="w0",
)
ariadne.record_branch(
    branch_id="left", world_id="world_1",
    experience={"observations": [{"scene": "threshold"}],
                "actions": ["left", "train"],
                "outcomes": [{"resource": "changed"}]},
    selected_path=["entrance", "threshold", "left"],
    idempotency_key="left_branch",
)
permit = ariadne.prepare_training(
    candidate_id="left_child", branch_id="left",
    trainer_request=trainer_request("trusted_trainer", parameters={"steps": 1}),
    idempotency_key="left_intent",
)
# Only now may the owner call its real trainer, save ALL successor state,
# and publish a write-once child full-state manifest. The trainer must
# return a trustworthy completion receipt. This module NEVER invokes it.
ariadne.commit_training(
    permit, child_manifest_path=Path("/abs/campaign/left_child.json"),
    trainer_receipt={"trainer_id": "trusted_trainer", "effect_status": "completed",
                     "effect_token": permit.effect_token, "details": {}},
    idempotency_key="left_commit",
)
```

In a real campaign, repeat branch/prepare/train/commit/evaluate for at least
two sibling candidates, reject losers and select the single evaluated winner.
After the second generation, open the journal in a **new Python process** and
read a descriptor:

```python
reopened = AriadneJournal.open(Path("/abs/campaign/ariadne"))
descriptor = reopened.continuation_descriptor(
    expected_head_sha256="a-previously-witnessed-64-character-sha256",
)
```

Use a real 64-character digest from an independently retained previous receipt
in place of the illustrative value above. `next_generation_index == 2` means
there are two selected child generations and the descriptor names the exact
selected full-state parent for a potential third generation. It includes all
worlds, branch experiences, candidate fates, selected lineage, event head,
physical manifest path and the required external restore roles.

**There is no `RESUMED` result.** A healthy read returns
`READY_FOR_EXTERNAL_FULL_STATE_LOAD`, with `model_loaded=False`,
`optimizer_loaded=False`, `rng_loaded=False`,
`data_cursor_restored=False`, `restorable_training_state_proven=False`,
`neural_update_proven=False`, and `graduation_authorized=False`.

If a process crashes after `TRAIN_PREPARED` but before `TRAIN_COMMITTED`,
reopen returns `INDETERMINATE_TRAINER_EFFECT`. Any mutation except an original
in-process matching completion attempt is blocked. A reopened process cannot
reuse the original permit, and a repeated idempotency key cannot issue a new
training permit. The effect might already have modified weights or optimizer;
**Ariadne must never automatically retry it**. There is deliberately no
`mark_success`, `abort_and_retry`, or fictional rollback method.

## Verified and unverified boundaries

- `verify_campaign_state(..., fresh_process=True)` independently reopens and
  rehashes every constituent local file in another Python interpreter. Replays
  also check distinct child contents, origin and parent digests, generation
  increments, mutation family, unchanged LoRA base and changed adapter (or
  changed full-weight tensors with unchanged tokenizer/config).
- A malformed/truncated, rewritten, missing-middle, duplicate or replayed
  event, broken hash chain, unknown type, duplicate idempotency/event ID,
  bad W0/child bytes, unsafe path or staged orphan is rejected. Two competing
  writers to one journal cannot select two winners; `flock` serialization and
  transition rechecks prevent the second fork.
- A **deleted trailing event** or fully rewritten history cannot be detected
  from a self-contained hash chain alone. The optional
  `expected_head_sha256` comparison detects a lost/replaced tail only if the
  expected value is held by a separately trusted monotonic witness. A hostile
  local file owner could rewrite every event and all files. A shared NAS must
  provide trustworthy durability, locking, access control and anti-rollback
  before this is a real distributed production ledger.
- `TRAIN_PREPARED` is durable only if the application actually calls the
  method and waits for it **before** its effect. Existing `SagaSession` TRAIN
  entrypoints are not yet wired to this hook. The sample sequence and tests
  demonstrate the intended ordering; they cannot assert that arbitrary
  existing trainers use it.
- A caller-supplied `trainer_receipt` is not cryptographic proof of SGD,
  causal parent loading or an authorized external effect. This v1 never loads
  model weights, optimizer, scheduler, RNG or data cursor into a real trainer.
  It therefore does not establish process-death training continuation,
  behavioral generalization, independent selection fitness or graduation.
- Ariadne logs caller-supplied metrics and evidence. It **does not** validate
  Minotaur thresholds, unchanged parent/control comparisons, private exam
  results or selection policy authority. A journaled `CANDIDATE_SELECTED`
  event is not graduation or promotion evidence.
- Operator code must keep private examination items, answers, credentials and
  sensitive traces out of `curriculum`, `experience`, `evaluation` and
  `trainer_receipt`. The journal records those JSON payloads verbatim.
- Tests use tiny physical safetensors fixtures and a mocked effect receipt.
  A successful source-only fixture is not a successful live neural campaign.

Focused verification:

```bash
python -m unittest tests.test_ariadne_recovery_v1 -v
```
