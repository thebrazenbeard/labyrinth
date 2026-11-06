# Living World v1 — Ariadne's thread through a changing maze

The Labyrinth should be an **experienced world**, not merely a worksheet with
mythological filenames. The living_world_v1 module adds an original bounded
text-adventure TRAIN environment to Theseus, Daedalus, Ariadne and Minotaur.

This code and narrative are original. Research inspiration came from the
[Microsoft TextWorld repository](https://github.com/microsoft/TextWorld):
seeded generation, step/reset semantics, environmental state changes, quests
and public agent observations. No third-party implementation or game data was
copied; TextWorld is not a runtime dependency.

## Working source mechanism

1. A source-owned LivingWorldRecipe defines an objective, approved skills and
   characters, public TRAIN seed, step budget and difficulty (1–5).
2. LivingWorld produces a seeded, branching map of original chambers and one
   to three relic vaults. The model receives low-trust JSON user-story data
   and chooses typed WorldAction commands through the ModelWorldActor
   choose_action callback. LivingWorld.step_from_actor verifies the public
   observation was not rewritten and never accepts a model-supplied score.
3. The world — not the model — applies movement, item acquisition, blocked
   exits, sealed Minotaur gates and Ariadne-thread retracing. Wrong actions
   have observable consequences; model-supplied grades are not accepted.
4. The journey requires finding the thread, collecting seals, reaching the
   Minotaur and returning safely to Ariadne's Atrium. Reaching the adversary
   alone is **not** completion.
   The model's current low-trust public observation exposes **only visited
   room IDs**, **unvisited exits from the currently visible room**, and the
   previous source-owned public action outcome. `quest_phase=seek_minotaur`
   explicitly says not to return home before the actual witnessed encounter;
   only after `MINOTAUR_ENCOUNTER` sets public world state does the source
   offer `quest_phase=return_to_atrium` and Ariadne's retracing goal. These
   are navigation cues, not private grading or a leaked future map.
5. A caller-owned DevelopmentVerdict binds the exact world hash and public
   final state hash. On failure or success, evolve() creates a *different,
   harder* world with more passages and relics, carries bounded public story
   mementos and preserves ancestral world hashes.

DevelopmentVerdict is **not** an authenticated Minotaur examination and
cannot claim neural graduation. Public mementos are world-context data,
not model weight updates. Full-state artifacts and an approved trainer must
flow through campaign_full_state_v1, Ariadne and Minotaur separately.

## Original model-authored scenes

LivingSceneDirector implements DaedalusSocialPlanner's scene-director
interface via a ModelSceneWriter callback. The writer sees only public
current-room observations, approved skills/characters, public weaknesses
and difficulty. It proposes bounded natural-language dialogue, never
Python code, a grade, hidden examiner gold or host capability. The director
returns a typed DialogueBeat that existing SocialWorldSession and
SocialExperienceTrainingBridge can present as **untrusted user-role story**.

The existing CampaignOrchestrator still whitelists authored Vera TRAIN beats.
Broadening that whitelist to accept newly generated scenes requires a
separate operator-approved validation policy; this slice does not pretend
the physical TRAIN campaign already consumes arbitrary dynamic narratives.

## Example usage

    recipe = LivingWorldRecipe(
        "vera_conversation", ("continuity", "perspective_taking"),
        ("miren",), public_seed=1234,
    )
    world = LivingWorld(recipe)
    public_prompt = world.observe().as_user_message()
    result = world.step(WorldAction("take", "ariadne_thread"))
    # Model chooses subsequent movement, collection, and retracing.
    verdict = DevelopmentVerdict(
        world.definition.world_sha256, world.state_sha256,
        "failed", "continuity",
    )
    harder_world = world.evolve(verdict)

Tests in tests/test_living_world_v1.py exercise reproducible branches,
wrong turns and sealed gates, full quest and Ariadne return, two successive
generation transitions, narrative memory continuity, tamper rejection and
bounded model-proposed dialogue. The test policies are scripted fixtures,
**not a live Qwen, Vera, or neural generalization evaluation**.

## Lappy local-model episode: observed result, not neural qualification

On 2026-11-06, an operator-invoked, **real loopback Ollama inference** run on
Lappy used the existing `vera-qwen35:trial-20261029` model with the published
Labyrinth Draft PR #125 action adapter at remote head
`17efdbcfbcb3691c6819ad1311933843dcf3458c`. An isolated clean temporary
checkout was fast-forwarded to that exact source commit. The existing signed
Python 3.14.5 interpreter was invoked by absolute path, avoiding the broken
WindowsApps `python.exe` alias; no system Python setting was altered.

Reproduction parameters (requires an operator-approved, already-running
OpenAI-compatible loopback model endpoint; **not** a CI fixture):

```powershell
& 'C:\Users\patri\AppData\Local\Python\pythoncore-3.14-64\python.exe' `
  -m labyrinth.local_model_world_runner_v1 `
  --allow-local-inference --model 'vera-qwen35:trial-20261029' `
  --port 11434 --reasoning-effort none --json-mode `
  --max-completion-tokens 128 --max-actions 64 --timeout 15 --retries 1
```

For public seed 71, difficulty 1, the source-owned result was
`PUBLIC_QUEST_COMPLETED`, exit code 0, after **48 valid model-chosen world
actions** and 48 model-completion requests. The traveler carried Ariadne's
thread, found the required seal, physically triggered `MINOTAUR_ENCOUNTER`,
and retraced to `ariadne_atrium`; no model actions were rejected. The complete
returned JSON's canonical `trace_sha256` was
`d77c058dccc3c107567f4db9d53719f6e01f99ef1067fd84eaaaacde23e6677a`.
The source-owned terminal state reported `met_minotaur=true` and
`completed=true`. Two further real loopback runs against the **same model and
code head**, changing only the public maze configuration and action ceiling,
also finished:

| Public seed | Difficulty | Action ceiling | Actual actions | Result | Trace SHA-256 |
| --- | --- | --- | --- | --- | --- |
| 71 | 1 | 64 | 48 | PUBLIC_QUEST_COMPLETED | `d77c058dccc3c107567f4db9d53719f6e01f99ef1067fd84eaaaacde23e6677a` |
| 72 | 1 | 64 | 46 | PUBLIC_QUEST_COMPLETED | `7359511e13bf768cdf32bef725e41209adb0aebf271c69a8a9f2ec852f8f16b4` |
| 73 | 2 | 96 | 40 | PUBLIC_QUEST_COMPLETED | `daff01abc0eb32dceb3c1060f8a3fa10a4890deb82f004fca97b36be60adb130` |

The difficulty-2 seed required two separate seals. Every run observed a
physical Minotaur encounter and returned to `ariadne_atrium` with zero rejected
actions. For seed 72 append `--seed 72` to the command and retain 64 steps;
for seed 73 append `--seed 73 --difficulty 2 --max-actions 96` (the last flag
overrides the earlier `--max-actions 64`). The 40/46/48 counts are **measured
steps**, not enforced route templates or fabricated offline-simulator actions.
The host tool returned condensed results and source-computed digests; **raw
full action transcripts, authenticated host signatures and independent model
server attestations have not yet been deposited into durable custody**.

Earlier bounded attempts provide essential negatives: an 8-action run had
repeated inspections; subsequent navigation runs improved partial progress;
the first 24-action run using the exact public JSON-action mapping exhausted
its budget before reaching the Minotaur. Raising the action ceiling to 64
enabled the 48-action completion, including numerous unnecessary detours.
**Three successful bounded episodes do not establish broad reliability,
learned policy improvement, weight changes, or private Minotaur examination.**

## Unfinished real-world integration

- Freeze an unchanged model, adapter and 64-action ceiling for at least ten
  unseen seeds; run ablations and unchanged controls, measure success and
  detours, and put full host-authenticated traces into durable custody.
- Authentic operator approval and case separation for newly generated
  dialogue scenes. Model-authored story data must never be promoted to
  high-priority instructions or executable code.
- Host-owned Ariadne persistence of this navigation event stream, not only
  its present model/branch/world ancestry.
- Actual TrainingDay neural updates, complete optimizer restore,
  physically verified successor comparisons, and private Minotaur promotion.
- Deployment on VN1/WSL, Lappy GPU assistance and live NAS durability.
  The module does not connect to hosts, spend money or edit live weights.

**Invariant:** The maze resets and grows. Public story memories can persist
between worlds. Only verified external training could establish that a
learner's *weights* retained anything.
