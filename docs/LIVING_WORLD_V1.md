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

## Unfinished real-world integration

- Reliable complete live-model navigation remains to be demonstrated. An
  opt-in local inference adapter and episode runner exist; one bounded live
  pilot exposed repeated inspections, and subsequent source fixes now carry
  public actor/action history and explicit exploration/return phases. Source
  tests alone cannot prove a model will choose sound routes.
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
