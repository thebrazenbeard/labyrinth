# R57 Labyrinth–Rezon source-pointer compatibility (non-authorizing)

Status: **OBSERVED_SOURCE / PARTIAL_SOURCE_ALIGNMENT_UNATTESTED**. This is a
Labyrinth-owned source-triage helper, not a Rezon consumer, runtime integration,
physical qualification, proof of receipt authenticity, or authorization service.
It does not establish `AGI_QUALIFICATION` or `AUTHORIZED_CAPABILITY`.

## Exact sources independently inspected

| Source | Inspected exact commit | Scope |
| --- | --- | --- |
| Labyrinth Draft PR #70 | `fdfcb374cba28bfa7ec3cefc40d372f3f0a17ccf` | `labyrinth/evidence_pointer.py`; `docs/qualification/LABYRINTH_EVIDENCE_POINTER_V1.md` |
| Rezon Draft PR #144 | `7dcb8b09c61c98c255e57ac0217250d821dfdd04` | `experiments/labyrinth_minotaur_receipts/qualification.py` (`ExactPins`, `QualificationVerdict`) |
| Vera Mono `main` | `5be71f4c9171417f8d492db341a058a7072822ae` | `packages/vera_core/src/vera_core/portfolio_system_model.py`, `agi_qualification.py` |

These are reviewed repository heads, **not independent run or artifact expected
pins**, signed Rezon receipt verification, or authority grants. They must not be
substituted for run-specific `ExpectedSourcePinsV1` or `ExpectedRezonPinsV1`.

## Source mapping and deliberately unresolved differences

| Labyrinth pointer field | Rezon `ExactPins` or external field | Treatment |
| --- | --- | --- |
| `source_head` | `labyrinth_source_head` | Compare exact externally expected Git source reference |
| `source_tree` | No matching Rezon `ExactPins` field | Independently pin and check via Labyrinth pointer only |
| `rezon_source_head` | `rezon_source_head` | Compare independently pinned Rezon source reference |
| `rezon_receipt_bundle_sha256` | **Not native** to Rezon `ExactPins` | Opaque bundle reference independently supplied; **never** infer receipt validity |
| `control_matrix_sha256` | `control_matrix_sha256` | Exact cross-source comparison only |
| `development_event_sha256` | `development_event_sha256` | Exact cross-source comparison only |
| `train_exposure_sha256` | `maze_exposure_sha256` | **UNRESOLVED**: training exposure vs maze exposure; neither equality nor equivalence asserted |
| `checkpoint_manifest_sha256` | `checkpoint_sha256` | **UNRESOLVED**: checkpoint manifest vs checkpoint digest; neither equality nor equivalence asserted |
| `run_id` | `run_id` | Separate runs/episodes; each independently pinned, **not equated** |
| Labyrinth MiniLM five-arm experiment | `LABYRINTH_MINILM_FIVE_ARM`, `PROVENANCE_ONLY`, five arm names | Strict program family guard; no Qwen four-control substitution |

The helper (`labyrinth/rezon_pointer_mapping_r57.py`) calls Labyrinth's
canonical bounded pointer verifier first. It then checks exact typed Rezon
source claims against a *separately supplied* expected Rezon pin set and
compares only the five explicitly comparable cross-repository references.
It accepts distinct exposure/checkpoint hashes only as **unresolved source
objects** and returns an immutable object with all operational authority,
host, checkpoint, receipt authenticity, identity, consent, cross-model grant,
graduation, merge, deployment and AGI claims hard-coded false.

Crucial limitation: a caller can falsely label self-derived expectations
"independent." This helper has no way to prove an owner's governance or
the provenance of expectations; proper durable registries, identity,
signatures, replay prevention, authorization and host attestation remain
separate owner-controlled responsibilities. Rezon HMAC role envelopes,
lease/route/effect records and claimed validity are not authenticated here.

## Two-model owner acceptance matrix

| Owner | Acceptance needed for model adapter A | Acceptance needed for model adapter B | Forbidden cross-model inheritance |
| --- | --- | --- | --- |
| Labyrinth source producer | Independently pin source head **and tree**, Lab run, plan, train exposure, controls, development event, checkpoint manifest, scope, optional Rezon source/bundle references | Same checks for the exact source pointer B observes; a shared source observation is allowed | Source pointer grants no execution, training or graduation |
| Rezon receipt/coordination owner | Independently pin Rezon source head, separately named Rezon episode, maze exposure, checkpoint digest, control matrix, development event, fixed five arms and `PROVENANCE_ONLY` | Independently confirm the Rezon episode B observes; neither derive expectations from A's claims nor assume A's grant | Receipt source claims are **not** owner/signer/host attestation, consent or lease transfer |
| Vera Mono owner | Register discovered source as `OBSERVED_SOURCE` only in a governed *monorepo-local* consumer; keep `AGIEvaluationPacket` unmodified | Independently perform B's model-bound evaluations and registry checks; never infer `AUTHORIZED_CAPABILITY` | No adapter may inherit identity, model-scoped consent, permission or AGI graduation from the other |
| Host authorization owner | Bind verified model-A identity, consent, capabilities, replay fence, run/host lease and checkpoint bytes to a separate host-owned authorization decision | Bind model B independently against its own versioned context | A's authority, process, permission or host lease never authorizes B |

No Rezon or Vera Mono dependency/import is introduced by the Labyrinth helper;
this makes source-observation portable but does **not** close those owners'
acceptance gates. A Labyrinth-owned *read-only source producer/triage helper*
is warranted because it can safely reject substitution and name unresolved
fields. A **runtime consumer adapter** should instead be implemented in Vera
Mono only with that owner's approval and authenticated, run-specific expected
pins; no such adapter is asserted built in R57.

## Hostile test vector and verification

With valid Labyrinth/Rezon source head, Rezon bundle, control matrix and
development event matches, offer Rezon `maze_exposure_sha256=8×64` and
`checkpoint_sha256=9×64` while Labyrinth pins
`train_exposure_sha256=2×64` and `checkpoint_manifest_sha256=5×64`.
The result is **partial/unattested**, listing both unresolved pairs, with every
grant false. Alter the independent expected Rezon `maze_exposure_sha256` or the
expected Labyrinth source tree: the adapter rejects. Independently pin a
Rezon record whose control matrix, development event, source heads or bundle
disagree with Labyrinth: rejects. Substitute Qwen four-control arms or an
execution semantics label: rejects. Add identity/consent/permission/lease
fields or try to promote the tuple-backed result's authority flags: rejects.
The result is now a tuple-backed record exposing only two source run IDs as
constructor fields. Direct assignment **and** `object.__setattr__` attempts to
overwrite `can_authorize_deployment`, `qualification_state`, `disposition`,
consent, permissions, or AGI fields fail; tests exercise the former frozen-slots
dataclass bypass explicitly. Caller-controlled authorization parameters are
not accepted by the result constructor.

Local verification command:

```bash
python3 -m unittest tests.test_rezon_pointer_mapping_r57 tests.test_evidence_pointer -v
```

This suite proves only local parsing/alignment invariants. It does not prove
physical model training, checkpoint byte identity, cross-host health, signer
authenticity, live Rezon receipt acceptance, durable replay exclusion, or any
merge/deployment qualification. PRs #70 and #144 remain governed Draft sources.
