# Owner-controlled TrainingDay broker: developmental dry-run

The still-current `thebrazenbeard/model-training/main`
`trainingday/physical_launch_authority.py` raises an unconditional
`PhysicalLaunchAuthorityUnavailable`. That is a **missing trusted host
capability**, not a requirement to recruit an independent human reviewer.
Its deliberate hold must remain until an actual privileged WSL/Windows host
service can bind and enforce the resource, consent, data and process controls.

This source-only probe is a small, auditable next step:
`labyrinth.trainingday_host_broker_dryrun_v1.inspect_owner_broker_dryrun`.
It accepts a typed and exact one-step, $0, public-TRAIN-only intent; requires a
canonical AF_UNIX socket in an owner-protected (root-owned) directory; rejects
symlinked/writable **directory** ancestry and self-owned pseudo-brokers.
The socket itself may be writable to permit nonroot clients to connect,
but only the root-owned broker may approve an operation after checking
the client's real OS identity and owner intent. The probe asks the
kernel for `SO_PEERCRED` and compares a bounded broker response to a fresh
nonce and exact intent SHA-256. A broker may say
`OWNER_APPROVED_DEV_ONLY`, but the receipt **still sets**
`physical_training_authorized=false`,
`gpu_lease_enforced_proven=false`, `owner_approval_proven=false`.
The word 'approved' is a service claim, not an independently qualified owner
interaction or permission to call the protected trainer.

This client installs no daemon, creates no socket, stores no credential,
performs no network/cloud calls and never invokes `one_step_kernel`.
Its mocked tests demonstrate protocol rejection/acceptance *logic*, not
authenticated host broker ownership. On native Windows without Linux
`SO_PEERCRED`, it fails closed; an eventual Windows service requires a
separate named-pipe/Windows-token-specific implementation and acceptance.

## What must actually be installed next

1. A host-owned broker with a protected installation and a separate service
   identity, root-owned WSL socket and OS-enforced exclusive GPU/process lease.
2. A real owner-consent event bound to the exact pinned source/plan/model
   identity, resource ceilings, public TRAIN rights and a one-use nonce.
3. A separate physical launch route that consumes the grant while the broker
   holds the lease, not a locally supplied callback/boolean or merely a
   self-authored receipt. The launcher must kill/reconcile lost workers,
   audit exact actual file hashes, and restore optimizer/RNG/cursor state.
4. End-to-end readback from the installed broker and training worker. Only
   then modify `model-training`'s unconditional source hold to accept that
   specific qualified route, with no human-review bureaucracy for ordinary
   development.

This is deliberately an **adapter and falsifiable protocol**, not a fake
replacement for protected physical authority.
