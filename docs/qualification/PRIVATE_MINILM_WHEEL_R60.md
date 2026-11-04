# R60 proposal — package-private MiniLM source contracts

Status: **SOURCE-ONLY PACKAGE COMPATIBILITY PROPOSAL — independent review pending**.

## Demonstrated problem

The baseline `labyrinth-reasoning` wheel publishes `labyrinth*` packages but
several installed public source contracts eagerly import
`scripts.qualify_maze_conditioned_minilm_cpu`, which exists only in a checkout.
A real offline-built wheel loaded by an isolated interpreter fails with
`ModuleNotFoundError: No module named 'scripts'`.

The earlier R57 wheel-only workaround made `scripts*` an installable generic
top-level package and bundled 17 development probes. It fixes importability
but broadens public distribution and creates a global-name collision risk.

## Narrower structure

Three mutually dependent, pre-existing MiniLM source probes now live as
package-private Labyrinth implementation modules:

- `labyrinth.minilm_cpu_experiment`
- `labyrinth.maze_conditioned_minilm_cpu_experiment`
- `labyrinth.maze_multiseed_protocol_cpu_experiment`

Only the actual library consumers import those package-owned modules.
The original `scripts.qualify_*` files remain **source-checkout-only entry
points** that alias the same module objects on import or delegate `-m`
execution. They are not included in the installed wheel and do not create
new installed executable entry points.

The legacy single-model wrapper additionally exposes its imported symbols
when a source-only tool loads its file through `spec_from_file_location`, and
locates its same-checkout library path for direct CLI use outside the repo
working directory. This accommodation is for source-only compatibility,
not a permission or policy bypass.

The multi-seed child command keeps the historical `scripts.*` entrypoint in
git source checkouts, and selects the package-private module outside that
verified checkout layout. Merely installing an unrelated top-level `scripts`
module adjacent to the wheel cannot redirect the packaged child command.
This is a source layout compatibility decision, not a host authority or
physical training license. All reported experiment evidence remains public
development only.

## Test contract

`tests/test_private_wheel_minilm_contract_r60.py` builds a real offline wheel
from a source copy containing **no `scripts` directory**, inspects its archive
for the three package-private implementations and no top-level `scripts/`,
then loads seven previously affected `labyrinth` consumer modules in fresh
`python -I -B` without eager Torch/Transformers or import-time output files.
The test also plants a foreign `scripts` module in installed site-packages and
checks that the installed multi-seed runner still launches its private module.

This suite is independent of claims about training improvement, protected
private examinations, host owner acceptance, consent, graduation, or
deployment. Installing a wheel with working imports is not any such gate.

## Open acceptance items

- Independent hostile review of source `-m`/`runpy` canonical class identity,
  multi-seed source digest calculation, and installed-wheel behavior.
- Exact GitHub CI no-skip pinning for new wheel test and its changed library
  modules before any production merge.
- The existing repository license metadata remains a separate product-policy
  item; this proposal does not select a license or release distribution.
