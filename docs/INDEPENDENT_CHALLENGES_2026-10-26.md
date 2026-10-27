# Independent challenge record — 2026-10-26

Scope: local prototype commit `a7b6584b5fc230fdf922694a792115c932a67746`, subsequently published with equivalent source tree on GitHub feature branch as `76527836e28875dc621dba8e33ad96e421a02797` (Draft PR #1). Reviewers inspected the **local** revision; they did not independently verify the remote commit's tree. OpenAI ChatGPT native agent delegation was used with requested overrides `gpt-6-astra`, `gpt-6.1-terra`, and `gpt-6-luna`. The actual deployed model/runtime identities were **not exposed to the reviewers**, so independent execution contexts are verified but different underlying model weights are not independently attested. These are model reviews, not external human review or empirical model-training evidence.

## Material falsifiers and dispositions

| Review lane | Material finding | Disposition in follow-up source |
| --- | --- | --- |
| Evaluation/red-team (`gpt-6.1-terra` requested) | Seed 3, tier 2, count 2 has labels `[10,10]`; the original circular shuffled labels reproduce the candidate prompt exactly. | Reproduced, then replaced with full label derangement and fail-closed invalid-control behavior; regression added. |
| Systems architecture (`gpt-6-astra` requested) | Scored training responses are recorded but never influence candidate demonstrations. Different training trajectories can yield identical operative candidate prompts. | Explicitly disclosed as **in-context exemplar evaluation**, not feedback-adaptive training. Added a separate toy numeric-parameter update path rather than relabeling that experiment. |
| Systems architecture | Three fixed prompt templates with fresh numeric seeds are not structural generalization; nominal tier order is not empirical difficulty. | Added bounded tier-4 register programs with disjoint shorter training and longer/novel-structure holdout patterns, structure fingerprints, and an independent test interpreter. No claim of general transfer. |
| Training practicality (`gpt-6-luna` requested) | Core has no real LM weight-update route; don't disguise replay/Ollama generation or a rule learner as fine-tuning. | Added CPU toy learner explicitly marked **not an LLM** and an optional future local SFT/LoRA qualification contract. The LM backend remains unimplemented. |
| Evaluation/red-team | Public deterministic test generation, same family/templates, and repeated tuning contaminate generalization claims; no paired power/statistical plan or verified model weight digest. | Created `EXPERIMENT_PROTOCOL.md` with freeze, held-out, negative-control, structural transfer and provenance requirements. No real-model gain is claimed. |

## Executed evidence at review time

The original local prototype passed 42 Python tests. The benchmark review directly reproduced the identical candidate/shuffled prompt under the seed above. Architectural review directly compared two distinct training trajectories that led to identical candidate prefixes. Reviewers did **not** establish improvement of model reasoning or training of neural weights. Follow-up test commands and exact new commits must be reported separately after integration.

The original NousResearch DeepHermes AscensionMaze model card is self-generated and explicitly warns that its claims may be false. The archived Atropos RLAIF environment is an example of an environment/feedback loop, not a validated escalating reasoning puzzle benchmark. Labyrinth borrows architectural inspiration only and uses its own deterministic oracles and controls.
