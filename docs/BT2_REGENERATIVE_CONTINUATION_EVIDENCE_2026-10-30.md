# BT2 regenerative Labyrinth continuation: evidence ledger (2026-10-30)

This ledger separates historical conversation intent from current implementation
evidence. It is **not** a qualification certificate or a runtime-currentness record.

## Conversation archive provenance and limits

- User-supplied attachment: `Deploy MeZO Research PR.zip` (saved conversation
  HTML, PDF, and associated assets). No archived code or embedded scripts were
  executed, and no embedded account/session metadata was copied into this repo.
  ZIP SHA-256: `47d1ce508fef9458566fab36f8980ae25ac1dd5eccc96ad18ec0975c81b2c13d`.
- The PDF contains 11 pages; extracted text includes the core Labyrinth metaphor,
  explicit learning-continuation requirement, proposed mastery criteria, and
  subsequent requests to implement, qualify, and use subagents.
- The saved HTML contains a richer early exchange and short entries for later
  turns, but it does **not** expose every later assistant response as full
  conversation text. Neither format is demonstrably a lossless chat export.
- The separately supplied `Pasted text(20261030-145427).txt` is the explicit
  continuation work order (SHA-256:
  `6717f7faaf1583802c2f061bf239e11f15afc7b78553128dc48cbae1682815af`).
  Historical discussion is evidence of intent, not evidence of the current
  Git or workstation state.

## Recovered user decisions (conversation evidence)

1. A learner declares a *specific* learning objective at the entrance; the
   declared scale and specificity determine the maze, challenge families,
   requirements, and budget.
2. Chambers provide training feedback and adapt to weaknesses. At the center,
   Minotaur confronts the learner with unfamiliar development challenges.
3. On failure, the maze regenerates with new tasks informed by observed
   weaknesses. **The maze resets; the learner does not.** Actual learned model
   parameters must survive into the successor cycle.
4. Improvement on development challenges is not independent graduation.
   Final qualifications require fresh, independently controlled evidence and
   a scope-specific finding; unknown is not a pass.
5. Observable empathy-related competence is a legitimate example objective,
   without treating behavioral scores as proof of subjective feeling.
6. MeZO, LoRA, AdaMeZO, LOZO+, and other methods are possible training
   mechanisms, not the definition or proof of the larger learning architecture.
7. Draft PR #62 is an authorized integration vehicle, but merging is
   conditional on correctly building and qualifying the system.

## Live source checkpoint (separate evidence class)

- On 2026-10-30, the GitHub PR connector returned
  `thebrazenbeard/labyrinth` PR #62 **OPEN / DRAFT**, base
  `39a95ac1d12cce372de980b8cad2bc41016e1705`, head
  `be660ecb1c58c3ee995e2857440c6e5e174350ee`.
- Current source and tests establish a bounded journey prototype, a tabular
  taskmaster, toy arithmetic training, and a source-level HF/PEFT adapter
  continuation seam. They do not establish pretrained-model learning or
  structural generalization through regenerated cycles.
- Separate Draft PR #60 explores MeZO; its v4 report format is not yet
  compatible with PR #62's v3-only predecessor check. Combining source branches
  without a versioned contract and tests would be unsafe.
- Currentness of any live Lantern/SQL runtime remains unknown. No historical
  WoWSQL state is used as evidence here.

## Open qualification work

| Gate | Current evidence boundary |
| --- | --- |
| Q0 source | Exact PR head and branch retrieved; source is mutable after this checkpoint |
| Q1 tests | Baseline unit/CI checks exist; new changes require exact-head reruns |
| Q2 integration | Toy journey exercises a bounded control loop; neural link is incomplete |
| Q3 neural | A genuinely pretrained model training on a real host is unverified |
| Q4 persistence | HF trainable adapter reload is source/mocked-test evidence only |
| Q5 controls | Toy same-family controls exist; real matched neural controls missing |
| Q6 transfer | Unseen program structures and retained prior skills unverified |
| Q7 independence | Private final examiner evidence not established |
| Q8 recovery | Interrupted neural optimization/restart not qualified |
| Q9 merge | HOLD Draft pending relevant gates |

This ledger should be updated with exact commit and experimental artifact
references as they become available; it must not be promoted to a live-runtime
assertion merely by changing its timestamp.
