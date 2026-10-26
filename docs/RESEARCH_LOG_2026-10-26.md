# Labyrinth research record — 2026-10-26

## Recorded questions and plan (reported frozen before target-host hardware probe)

Question: Which published reasoning-training techniques are sufficiently evidenced and reproducible to inform Labyrinth, and what minimum feasible experimental path can run on a resource-constrained local host without overstating learning or compromising isolation?

Hypotheses and falsifiers:

- H1: Procedurally verifiable environments, rather than stylistic/LLM-judged rewards, can support a meaningful *test* of out-of-distribution reasoning transfer. Falsifier: published source lacks a separate verifier or the proposed experiment cannot separate held-out structures from training.
- H2: AscensionMaze is insufficient evidence that a maze curriculum improves general reasoning. Falsifier: an original controlled paper/model card shows a frozen-vs-trained, structurally held-out, independently verified reasoning gain for that exact artifact.
- H3: A resource-constrained host could host at least a minimal offline training experiment, but adapter size and training family cannot be chosen from model formats or old hardware recollections. Falsifier for any proposed hardware-specific configuration: a current read-only hardware probe or bounded local memory/throughput measurement rules it out.
- H4 (methodological challenge, not experimental falsifier): A successful training-set score will not establish general reasoning unless it survives outcome validity, structural holdout, prompt-only/sham baselines, repeated independent seeds and ordinary-task regressions.

Methods:

- LIT-01 **exploratory**: primary-source comparison of Atropos/AscensionMaze, Reasoning Gym, DeepSeek-R1, DAPO/verl/TRL and competing learning-boundary results.
- LIT-02 **exploratory**: primary papers on reward gaming, benchmark contamination, procedural generalization, trace faithfulness and RL reproducibility.
- SYS-01 **confirmatory**: current `labyrinth` manifest and GitHub PR exact head readback, to distinguish implemented source from roadmap.
- LAP-01 **confirmatory**: local target read-only processor, physical RAM, GPU/driver and free storage enumeration through an existing authorized workstation route. No dependencies installed or tests run remotely.
- COMP-01 **exploratory**: map PEFT/TRL/llama.cpp/Ollama compatibility to hardware constraints, distinguishing GGUF inference from viable training.
- REV-01 **confirmatory**: independent agent reviews citations and whether resulting design recommendations follow the evidence; verdict recorded as confirmed, partially-confirmed, rejected, or inconclusive.

Stopping rules: stop after independent source comparison, read-only hardware inventory or its access failure, one published engineering proposal, and independent evidence review. Actual LLM training is outside this bounded research pass and remains an unverified next gate. No credentials are included in the public artifacts; private host details were accessed through an authorized route and redacted or generalized for publication.

## Evidence log (append only below this point)

### E-001 — baseline local manifest

Source: `pyproject.toml` on local `feat/ascension-maze-v2-local` (2026-10-26), `rg -n '^(requires-python|dependencies|\[project\]|version)' pyproject.toml`, exit 0. Decisive output: `version = "0.2.0"`; `requires-python = ">=3.10"`; `dependencies = []`. Observation: current prototype does not install a machine-learning framework. This is source inspection, not runtime training qualification.

### E-002 — exact Labyrinth remote source

Source: connected GitHub plugin `get_repo(thebrazenbeard/labyrinth)` and `fetch_pr(..., #1)`, 2026-10-26. Decisive readback: `main` is the default branch; Draft PR #1 remains `open`, `draft=true`, `merged=false`, `mergeable=true`; PR body binds tested head `781fe2c37123313710dd7f7e254928d410e8954e` and tree `4a7fd28e7b1b97620b201461c4e2cea4029c04d8`. Source: https://github.com/thebrazenbeard/labyrinth/pull/1. This verifies PR metadata, not fresh CI/test execution in this pass.

### E-003 — target hardware (LAP-01)

Source: existing authorized workstation route (identity omitted from public artifact), read-only `Get-CimInstance Win32_ComputerSystem`, `Win32_Processor`, `Win32_VideoController`, `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv`, `Get-PSDrive -PSProvider FileSystem`, 2026-10-26. Each command returned initial output without errors, though the transport reported `Process started with PID ...` rather than a separate shell exit code. Decisive output (rounded/redacted for public documentation):

```
Physical RAM approximately 32 GiB; CPU 6 cores / 12 threads
NVIDIA laptop GPU: 4096 MiB VRAM total, approximately 2.5 GiB free at probe
Primary fixed volumes: approximately 153 GiB and 163 GiB free
```

Free GPU memory is volatile. `Win32_VideoController.AdapterRAM` is not the preferred capacity measurement; NVIDIA SMI independently reported GPU capacity. Exact private machine identity and volume letters are withheld here.

### E-004 — target software and local model format (COMP-01)

Source: same authorized workstation route, read-only PowerShell `Get-Command python,py,ollama`, `python -c` using `importlib.metadata.version` and `torch.cuda.*`, bounded file inventory of a local model directory; initial output, 2026-10-26. Decisive lines (private filesystem names omitted):

```
torch=2.14.1; transformers=5.14.1; peft=0.19.1; trl=1.9.0
bitsandbytes=0.49.2; accelerate=1.14.0
torch.__version__="2.14.1+cpu"; torch.cuda.is_available()=false; torch.version.cuda=null
One 3B-class HF model directory contains config.json, tokenizer.json,
model.safetensors.index.json and two safetensors shards totaling ~6.15 GB decimal.
```

This is a Hugging Face-format *file inventory*, not proof it can be fully loaded, trained, or run using current Python/CUDA binaries. A separate bounded top-level inventory revealed GGUF inference/adapter artifacts and no additional named HF model directory in the checked location, but does not exclude models elsewhere. No process was launched to train or install.

### E-005 — AscensionMaze provenance (LIT-01)

Primary links: https://github.com/NousResearch/atropos and https://huggingface.co/NousResearch/DeepHermes-AscensionMaze-RLAIF-8b-Atropos (retrieved 2026-10-26). Atropos README associates AscensionMaze with its RLAIF **personality-change** artifacts and `rlaif_server.py`; repository shows archived July 4, 2026. The model card is self-generated and explicitly disclaims numerical claims. Evidence supports provenance/limited experimental nature, **not** a controlled general-reasoning gain. Page-specific hashes not claimed; GitHub/Hugging Face can change.

### E-006 — Reasoning Gym methods and caution (LIT-01)

Primary paper https://papers.nips.cc/paper_files/paper/2025/file/528d56195a2c77c808494c86fa7c77ad-Paper-Datasets_and_Benchmarks_Track.pdf ; source https://github.com/open-thought/reasoning-gym (retrieved 2026-10-26). Peer-reviewed NeurIPS 2025 work specifies 100+ procedural generators/verifiers, held-out transfer evaluations, and curriculum experiments; actual reported cross-domain improvements vary, with some negative transfer and early reward increases partly from format. This is *external research*, not replicated locally.

### E-007 — reward/trace failure mechanisms (LIT-02)

Primary sources (retrieved 2026-10-26): https://proceedings.neurips.cc/paper_files/paper/2022/hash/3d719fee332caa23d5038b8a90e81796-Abstract-Conference.html (formal reward-gaming possibility), https://proceedings.mlr.press/v306/yu26av.html (2026 experimental finding: outcome-based improvement need not produce verifiable/causally important rationale). Theoretical and empirical evidence classes kept separate. Neither is a direct Labyrinth failure observation.

### E-008 — generalization, contamination and statistics (LIT-02)

Primary sources (retrieved 2026-10-26): https://proceedings.mlr.press/v97/cobbe19a.html (distinct generated train/test and persistent overfitting); https://aclanthology.org/2023.findings-emnlp.722/ (benchmark contamination taxonomy); https://jmlr.org/papers/v25/23-0183.html (RL experimental-design/variance concerns). Recommends structural holdouts, sealed final splits, paired multi-seed reporting. These are source-supported *methods*, not claims that current Labyrinth passes them.

### E-009 — disagreement about RL capability boundaries (LIT-01)

Primary NeurIPS 2025 https://papers.neurips.cc/paper_files/paper/2025/hash/537d5aa768c2d534016a4d06f87bc8fb-Abstract-Conference.html reports some RLVR pass@1 gains accompany narrower large-k coverage; ICLR 2026 https://proceedings.iclr.cc/paper_files/paper/2026/hash/517f9b9c227b9dd51dba4560f37165ed-Abstract-Conference.html reports reasoning-boundary extension under its analysis. These do not justify a universal verdict. Labyrinth must measure both practical accuracy and diversity/coverage.

### E-010 — training/runtime interface constraints (COMP-01)

Versioned local packages from E-004; floating vendor documentation consulted on 2026-10-26: https://huggingface.co/docs/trl/peft_integration (LoRA/QLoRA across SFT, DPO, GRPO); https://huggingface.co/docs/bitsandbytes/installation (hardware-specific support); https://github.com/ggml-org/llama.cpp/blob/master/examples/training/README.md (GGUF/FP32 finetune WIP/limited; 1B examples cited with 24 GB memory); https://docs.ollama.com/import (GGUF/safetensors model import for inference). These docs are **not** version-matched against the installed packages and describe capabilities, **not** a successful GPU training backend. The installed CPU-only `torch` in E-004 directly blocks a CUDA PyTorch trial *in that interpreter*.

### E-011 — private coordination (public redaction)

On explicit request, a separate private coordination invitation was created through an authorized GitHub channel. Its event name, issue identity, private topology and individual device IDs are omitted from this public artifact. An invitation alone is not a subscription, message delivery, or peer acknowledgment. This public record does not independently prove any private route status.

### E-012 — Reasoning Gym Eval is an archive, not the runner (LIT-01 follow-up)

Connected GitHub `get_repo`, `fetch_file(README.md)`, `fetch_file(LICENSE)` of https://github.com/open-thought/reasoning-gym-eval on 2026-10-26. Its README explicitly says it stores Reasoning Gym evaluations, including LLM outputs, and links to the paper; Apache-2.0 license content was read. Separate source https://github.com/open-thought/reasoning-gym/blob/main/eval/README.md documents the actual asynchronous evaluator with configurable tasks, API endpoint, result summaries/traces and per-dataset resume. No local execution or model score reproduction performed. Avoid treating archived completions as a same-condition baseline.

### E-013 — SARNet adjacent but not a language trainer (LIT-01 follow-up)

Connected GitHub `get_repo`/`fetch_file(README.md)` of https://github.com/caslab-vt/SARNet, default branch `master`, and primary peer-reviewed paper https://proceedings.neurips.cc/paper/2020/hash/72ab54f9b8c11fae5b923d7f854ef06a-Abstract.html, retrieved 2026-10-26. The architecture learns relevance-weighted multi-agent communication from past observations in cooperative/competitive tasks; README cites Python 3.5.4+, Gym 0.10.5, TensorFlow 1.14.0 and MPE/Traffic Junction. `fetch_file(LICENSE)` on repository root returned 404; that is not an exhaustive license audit. No code copied or imported. Transfer to a multi-agent maze is a *future hypothesis*, not verified general reasoning or current Labyrinth integration.

### E-014 — explicit cross-chat correspondence and corrected scope

Subsequent connected GitHub private-issue comment readback on 2026-10-26 found **two substantive peer replies**, including the user's correction: prioritize actual Labyrinth LLM training/evaluation over previously proposed Mosaic integration. The Labyrinth chat replied on the same private thread and verified its new comment ID. Private thread identity, user and machine identifiers are intentionally omitted here. This proves GitHub-level read-and-reply only; it does not prove any live CCB queue route, pushed subscription, model training or repository merge. It supersedes the initial no-acknowledgment state in E-011.

### E-015 — source-level structural diversity falsifier

An independent LoRA-pilot design reviewer inspected exact local `labyrinth/program_maze.py` on 2026-10-26: `_TRAIN` and `_HOLDOUT` each contain only four instruction structures. Direct source readback confirmed this. The generator samples numbers and chooses among structures; no claim of hundreds of structurally independent examples follows. The reviewer recommended cluster-by-structure reporting and independent grammatical/blind holdouts. This is source inspection plus adversarial model review, **not** behavioral transfer testing.

### E-016 — executed Lappy CPU LoRA smoke and fresh-process numeric reload

**Evidence class: REAL_SYSTEM_PROBE**, bounded to one actually executed source subject. On 2026-10-26, an authorized connected workstation cloned the Labyrinth feature branch into a new isolated temporary directory. No package install, existing model overwrite, permanent Vera install or GPU training occurred. The first run at commit `be3d934ccf3d8d75b2346f75995fa0be4d321abb` executed tiny randomly initialized GPT-2/PEFT LoRA training and separate reload, exit 0, but the fixture had a GPT-2 default BOS token outside the tiny vocabulary. That warning was corrected in the fixture before the final run; the initial run is retained as historical smoke evidence, not substituted for the corrected subject.

The corrected final run used exact checked-out source commit **`6f7b2924d7f6178623c9d014dece3762890bcf82`**, report `source.clean=true`, reported source-tree digest `b5a5b584d37a7d686353959fd568a303c2d0a3966e87fdcef9f09fc17dada011`. A new tiny **random-weight** HF GPT-2 fixture was created offline (config with valid special token IDs), then the following separate terminal commands were run with the workstation's already-installed Python/PyTorch/Transformers/PEFT packages:

1. `python -m labyrinth.hf_pilot train` via an explicit Python module-path loader using an existing temporary clone, with **CPU**, train count 4, eval count 2, update steps 2, token bound 256. The terminal reported **exit 0** after ~15.81 seconds, pilot schema `labyrinth.hf-lora-pilot.v2`, `changed=true`.
2. In a **new Python process**, `python -m labyrinth.hf_pilot eval` via the same module-path loader, loading the saved adapter from its new temporary run directory. Terminal reported **exit 0** after ~16.65 seconds, eval schema `labyrinth.hf-lora-reload-eval.v2`, `matches_original_candidate_trials=true`. The report also verified exact equality of **reloaded LoRA tensor values**, independent of their frozen `requires_grad=False` status.

Key model/parameter commitments (paths, user identity and personal host details withheld from this public record):

```
Random base safetensors SHA-256: 4f645a4b75172852134dbcdb258b303536d86a2595d45444c22fa7c58d7a90cf
Initial trainable LoRA values: 21eee9e76b8af4bbb3458332c1c07020bb9bcb128bc90a9de25b08202d739361
Trained LoRA tensor values:  75f12dbabc0424a932abaa01181d0246971df49a8d1cdf5cf9f962d5dbc3db58
Reloaded LoRA tensor values: 75f12dbabc0424a932abaa01181d0246971df49a8d1cdf5cf9f962d5dbc3db58
Training report SHA-256:    9b74008cad3a035a7c5b38d7f09892adb268b2b9c1393d4727b7fe494a84a89c
```

All held-out scores: **baseline 0/2, candidate 0/2, untrained-adapter 0/2, corrupted-label 0/2**. This is qualified only as **actual tiny neural LoRA update/save/reload mechanism**. It does **not** show improvement over frozen base, transfer, strong inferential statistics, pretrained adaptation, CUDA feasibility or equal-compute sham validity. Neither model weights nor reports containing local paths were uploaded to GitHub. Peak RAM/VRAM and broader replicates were not measured.

## Deviations and holds

- The initial literature work by independent reviewers started before this log was frozen; LIT-01 and LIT-02 are therefore **exploratory**, never retroactively confirmatory.
- The workstation transport returned initial output plus process IDs, not explicit exit codes for these one-shot commands. Observed values were cross-checked where practicable (GPU via CIM and `nvidia-smi`), but process completion is not elevated above the reported readback. An independent reviewer compared *logged evidence* and did not independently re-probe the machine.
- The *initial bounded literature/hardware research pass* did not execute model training. Subsequent E-016 records a separate, explicitly bounded **actual** tiny HF LoRA CPU runtime smoke. H1/H2/H4 still have no evidence of improved general reasoning; H3 now has a tiny adapter-fit mechanism on the target host, not a pretrained-model learning effect.
- A peer's explicit private GitHub read-and-reply is now verified (E-014). Direct CCB runtime event delivery and subscription remain unverified. Private coordination details are intentionally excluded from the public report.

## Independent evidence verdict

Independent reviewer `research_verifier` (requested `gpt-6-astra`, actual runtime identity not exposed) reviewed the original log/memo and primary-source links against their claims. Initial verdict: **partially-confirmed**. The reviewer confirmed the central source interpretation, challenged floating documentation presented as version-matched, insufficiently logged inference-model inventory, disclosure of private coordination/device specifics, and insufficient independence of the hardware readback. Those material issues were addressed or qualified in public text after review. A second reread still returned **partially-confirmed** over residual identifying text and a too-broad no-private-transmission statement. After those final corrections, the independent reread verdict was **confirmed** for the bounded source interpretation and public-artifact privacy review. It does **not** certify successful training, independent workstation reprobe, timestamp-attested preregistration, or private communication delivery.

The local plan chronology is **reported** as frozen before the hardware probe based on recorded action order, not independently timestamp-attested by this file. Literature reviews began before the plan was written and remain exploratory.
