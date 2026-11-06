# Optional genuine local-model action adapter (source-only)

`labyrinth/local_model_action_adapter_v1.py` implements the missing
`ModelWorldActor.choose_action` seam for a **user-managed, OpenAI-compatible
local chat endpoint**. This is only a source feature; no model is running,
installed, loaded or verified *merely by its presence in Git*. A bounded
operator-host probe has separately obtained a real model's valid public
action, but has not demonstrated a full neural-driven episode or weight update.

The actor receives **only** `LivingWorld.observe()` public story JSON (known
room, visible exits/items, public mementos, allowed action tokens). The
observation also records **previously entered** room IDs and the last
source-owned public action/outcome, but never unrevealed rooms, the full map,
or Minotaur grading. To avoid a temperature-zero model repeating the same
stateless choice forever, the actor includes at most eight of its own prior
validated public choices plus an unchanged-scene indicator; this is not a
memory of neural weights or proof of objective progress. It never
receives private Minotaur test gold, grading rubrics, checkpoint files,
optimizer state, secrets, host tools or examiner feedback. The response must
be one strict JSON object like:

```json
{"kind":"take","target":"ariadne_thread"}
```

The adapter rejects duplicate JSON keys, code fences, extra grading fields,
unknown action types, invented targets and actions absent from the exact
public observation. On malformed model output it retries at most twice (three
attempts total), including a static source-owned schema correction that never
echoes model text; truncated/non-normal completions are rejected. Failure
leaves the world unchanged. Only `LivingWorld.step`
changes public game state. The model does **not** supply an effect or grade.

## Using the source adapter

```python
from labyrinth.local_model_action_adapter_v1 import (
    LocalModelActionConfig, LocalModelWorldActor, LoopbackHTTPTransport,
)
from labyrinth.living_world_v1 import LivingWorld, LivingWorldRecipe

world = LivingWorld(LivingWorldRecipe(
    "saga_continuity", ("continuity",), ("miren",), public_seed=71))

# Explicit operator decision required for actual loopback HTTP.
actor = LocalModelWorldActor(
    LocalModelActionConfig(
        port=11434, model="local-model", retries=1,
        reasoning_effort="none", json_mode=True, max_tokens=128,
    ),
    LoopbackHTTPTransport(allow_http=True),
)

# Performs one inference request against an ALREADY-running local server.
event = world.step_from_actor(actor)
```

No server is started, no network is used by tests, and no production host is
configured automatically. A caller can substitute a fake transport for every
test. The endpoint is fixed to `http://127.0.0.1:<port>/v1/chat/completions`
(IPv6 `::1` also allowed), with literal loopback IP only, redirects denied,
environment proxies disabled, bounded 16-KiB request / 64-KiB response,
0.5–30-second timeout, and restricted retry budget. It is compatible in
principle with local servers offering OpenAI chat completions, including
KoboldCPP variants that support that API; **specific builds are not verified**.
The compatibility settings in this example are optional: `reasoning_effort`
and `response_format=json_object` are used only when a particular local
server supports them. They remain absent in the default payload to avoid
silently assuming all OpenAI-compatible servers implement Ollama controls.
Completion tokens are bounded to 32..256; values outside this range are
rejected. The [episode runner](LOCAL_MODEL_WORLD_RUNNER_V1.md) exposes the
same controls at its CLI.

```bash
python -m unittest -q tests.test_local_model_action_adapter_v1
```

> **Hostile boundary:** Source code cannot prove an arbitrary local model
> endpoint or its server is trustworthy. The model is only an action chooser;
> the host that runs its inference and its permissions remain the operator's
> concern. There is no automatic TRAIN, scoring, promotion or Minotaur access
> in this adapter. It cannot transform a deterministic fixture success into
> real model reasoning evidence.
