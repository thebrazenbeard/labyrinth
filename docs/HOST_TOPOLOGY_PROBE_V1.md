# Labyrinth VN1 / Lappy offline topology and readiness probe v1

**Status:** SOURCE-ONLY deployment candidate; no host deployment was made.
Source inspection is not a current VN1 or Lappy runtime probe. The executable
inventory is read-only and locally scoped; its JSON is a self-report, not
authenticated evidence that a named physical machine actually ran it.

## Roles, boundaries, and architecture

| Role | Candidate | Intended responsibility | Not established by this module |
| --- | --- | --- | --- |
| Controller | **VN1** | Build bounded TRAIN jobs, keep source/rights/split/nonce control, verify worker receipts and select candidates | VN1 identity, connectivity, scheduler, key custody, deployment |
| GPU / memory worker | **Lappy** (Windows) | Load approved local pretrained model, run explicitly authorized TRAIN/inference and clean-process reload | Lappy identity, active GPU PyTorch backend, physical trainer, free model memory, actual weight update |

VN2 is **not** substituted for the coordinator. Lappy's authoritative Vera
root is **`D:\VERA`**; `C:\VERA` is obsolete and is not an alternate
fallback. This design uses an **existing separately authorized** LAN/Tailscale
or WorkBridge path if one is healthy; no new tunnel, port, remote agent, VPN,
service, credential or paid infrastructure is installed or opened by the
probe. No connection is made between VN1 and Lappy here.

The Source TrainingDay adapter is pinned to
`trainingday.lowmem_experiment.one_step_kernel` at
`model-training@2b83430dd36076c5f65624709968e372b170ea36`.
Its `invoke_trainingday_trainer` raises
`TrainingDayPhysicalExecutionHold` (`SOURCE_ONLY_HARD_HOLD`). This probe
**cannot lift that hold**; finding local packages or a GPU is not permission
to launch training. Nor does it establish that Qwen3.5-4B can fit in available
VRAM or that a candidate learned. The historical workstation report of
CPU-only PyTorch and an RTX 3050 is dated evidence, not a current measurement.

## Read-only local inventory

`labyrinth.host_topology_probe_v1` has two seams:

- `collect_local_observation(role, root, inspect_gpu=False,
  inspect_torch=False)` inventories only the **machine executing Python**.
  It records OS/architecture/Python, CPU count, physical RAM, available disk
  at an explicitly supplied existing root, installed *distribution metadata*
  for torch/transformers/peft/accelerate, and optional NVIDIA/Torch observations.
  It does not enumerate model directories, private files, account details or
  process lists. It never uses the network.
- `evaluate_observation(HostObservation)` is pure and deterministic, producing
  machine-readable `HostReadiness` checks and exact failure reasons. Tests
  inject fake Windows/Linux observations without touching real hosts.

By default it does not import PyTorch or call `nvidia-smi`. Explicit flags
permit only these extra local reads:

- `--inspect-gpu`: `nvidia-smi` memory-total/memory-free/driver CSV, bounded
  at 16 devices, with a five-second timeout and no model load.
- `--inspect-torch`: import locally installed PyTorch and call
  `torch.cuda.is_available()`. This can initialize parts of the local CUDA
  runtime; it is *opt-in* and still does not train. Its result is not a
  load/optimizer qualification.

Example only, **to run locally on the named machine after verifying its
identity and getting the operator's permission**:

```powershell
# Lappy Windows PowerShell, existing source checkout and Python environment
py -3 -m labyrinth.host_topology_probe_v1 --role lappy-worker --root 'D:\VERA' --inspect-gpu --inspect-torch
```

```bash
# VN1 controller candidate: only in its own existing local source checkout
python -m labyrinth.host_topology_probe_v1 --role vn1-controller --root .
```

Both commands emit one JSON object without hostname, raw private paths,
credentials or logs. Their exit status is **2 even if resource checks pass**:
the hard hold prohibits treating a successful inventory as deployment
approval. `--help` exits 0 normally.

### States and conservative triage

| Check | PASS means | FAIL means | UNKNOWN means |
| --- | --- | --- | --- |
| `worker_os` | Locally reports Windows | Wrong OS for targeted Lappy worker | N/A |
| `authoritative_root` | Existing exact `D:\VERA` (Lappy), or operator-chosen existing VN1 root | Wrong/missing local path | N/A |
| CPU / RAM / free disk | Meets **illustrative** floors: Lappy 2 cores / 8 GiB / 10 GiB, VN1 2 / 2 GiB / 2 GiB | Below floor | Observation unavailable |
| Required worker packages | Distribution metadata found for torch, transformers, peft, accelerate | Package metadata missing | N/A; package presence still does not prove runtime compatibility |
| GPU | One NVIDIA GPU reports at least 4 GiB total / 2 GiB free | Probe failed/absent/insufficient | Optional probe not run |
| `torch_cuda` | Local Torch reports CUDA available | CPU-only or inaccessible CUDA backend | Optional runtime query not run |
| `host_identity` | Never PASS from this module | — | No independently authenticated host attestation |
| `controller_worker_transport` | Never PASS from this module | — | No authenticated handshake or path verification |

`resource_gate` is `BLOCKED` for resource failures,
`INDETERMINATE` for uninspected/unknown resources, or
`RESOURCE_CHECKS_PASS_UNATTESTED` if all observed resource checks pass.
All three remain **`HARD_HOLD_NO_AUTHORIZED_DEPLOYMENT`** at the deployment
gate. The resource floors are initial negative filters, not sufficiency
proof for the selected model, optimizer, sequence length, adapter or batch.

## Gates before any separately authorized deployment

These requirements are **not actions performed by this source change**:

1. Independently establish actual VN1 and Lappy host identities, operating
   systems, operator approval, host time and currently reachable authorized
   transport. Do not infer identity from a `--role` argument or reported OS.
2. Verify source head, rights/corpus provenance, no private Minotaur/holdout
   material in TRAIN, and local model/tokenizer/adapter hashes. Do not send
   model paths or credentials from Lappy to public logs.
3. Reconcile on-host installed trainer source and its physical launch
   authority. The current TrainingDay adapter is **hard-held**, so this gate
   is presently **FAIL** regardless of GPU inventory. No bypass is offered.
4. Obtain an explicit bounded budget/job authorization and reserve host
   resources without clobbering existing workloads; do a current process,
   capacity and storage preflight through authorized host tooling.
5. For any later independently approved job, pin scope/split/job ID/nonce,
   expiry, source and model hashes, candidate identifier, deterministic seed,
   output path and $0 budget in an authenticated envelope. Worker receipts
   must bind actual source/process/run start, consumed TRAIN IDs, optimizer
   step counts, artifacts and fresh-process inference. VN1 must verify
   signatures, replay/counter and custody independently.
6. Keep holdout/transfer/regression and private graduation material under
   independent evaluator custody; compare W0 to fresh Wt inference before
   discussing improvement. A saved adapter or loss change does not qualify
   natural conversation or generalization.

No restart, deployment, install, model download, network service, host file
write, optimizer invocation or credentials are part of this implementation.
If a gate fails, stop that line of work; do not reinterpret source tests or
historical hardware reports as current physical readiness.

## Deterministic test receipt

```sh
python -m unittest tests.test_host_topology_probe_v1 -v
```

Fake-host tests cover resource-pass-but-deploy-denied, CPU-only Torch,
uninspected NVIDIA/CUDA, missing trainer packages, wrong OS/root/capacity,
VN1 controller without CUDA, parser corruption and forged frozen objects,
no GPU calls by default and CLI exit-2 hard hold. Tests make no remote calls
and perform no physical TRAIN.
