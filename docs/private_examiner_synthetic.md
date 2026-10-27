# Synthetic private examiner seam (not qualification evidence)

`py -3 -m labyrinth.private_examiner PRIVATE_CONFIG.json OUTPUT_PREFIX` runs the
four configured subprocess arms on the same cases, checks their plain-text
responses against examiner-side integer answers, and writes an HMAC-signed
receipt and detached signature. It never sends `answer` or a serialized case
to an adapter. Adapter contract: `command ... CHECKPOINT_PATH`, raw prompt on
stdin, answer text on stdout; nonzero exit, timeout, or missing data blocks.
The process is newly started for each prediction.

Config schema (all values are operator-owned, **not** trainee-supplied):

```
{
  "policy": {"objective_id": "...", "trainee_sha256": "...",
             "base_sha256": "...", "session_nonce": "...",
             "expected_evaluator_sha256": "...",
             "expected_conditions_sha256": "..."},
  "attempt": 1, "run_id": "...", "synthetic": true,
  "evaluator_sha256": "...", "conditions_sha256": "...",
  "arms": {"candidate": {"checkpoint": "...", "command": ["..."],
                          "training_steps": 32},
           "baseline": {"checkpoint": "...", "command": ["..."],
                          "training_steps": 0},
           "zero_update": {"checkpoint": "...", "command": ["..."],
                           "training_steps": 32},
           "corrupted": {"checkpoint": "...", "command": ["..."],
                         "training_steps": 32}},
  "training_cases": [{"id": "...", "prompt": "...", "answer": "42"}],
  "cases": {"heldout": [{"id": "...", "prompt": "...", "answer": "42"}],
            "transfer": [{"id": "...", "prompt": "...", "answer": "42"}]}
}
```

Supply `EXAMINER_HMAC_KEY_HEX` (at least 32 random bytes, encoded as hex)
through a custodian-only process environment, never in the config, repository,
logs, or trainee environment. The receiving gate must receive the *same* key
from a separate trusted channel and persist `GateState` outside trainee custody.
Reserve fresh session nonce, suite, run ID, and cases before evaluation. Protect
the private config and receipt output paths with OS ACLs and operational controls;
this code does not create an isolation boundary or manage those permissions.

**Limits:** `synthetic: true` is mandatory; all other modes fail closed. The
fixture adapter deliberately reveals its answers inside toy prompts, so it is
not a private blind test. The caller-supplied source/condition pins and step
counts are compared to policy but their provenance is **not verified** here.
Likewise checkpoint file bytes are hashed, but an external command's actual
weight loading is not introspected. A fresh process is not proof of a genuine
model reload. The gate recognizes this receipt format, not whether its
assertions were truthful. A real graduation examiner would need an independently
custodied secret set and oracle, verified adapter/weight loading, authenticated
training provenance, trusted evaluator-source/condition hashing, and operational
key separation. Never treat a synthetic gate `GRADUATED` decision as permission
for deployment or as evidence of real model skill.
