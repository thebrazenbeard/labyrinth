# R34 public toy metamorphic challenge pairs — SOURCE ONLY / HOLD

This module extends the reviewed R31/R32 **public synthetic** textual challenge
generator (Labyrinth Draft #68) with a second, deliberately different wording
for each already-distinct arithmetic, source-priority, and program-operation toy
case. The toy expected answer is mathematically identical for each pair; there
is no claimed answer consistency for a real neural model until physically run.

The variant deliberately changes source mention order (higher-priority vs
lower-priority), switches prose arithmetic to ordered steps, and switches
program-operation prose to simple pseudocode statements. The case and semantic
commitments remain bound to the exact underlying R29 receipt. A pair digest
commits both exact prompt hashes, semantic hash, split, skill, and case ID.

The public development partition contains the already-public expected answers;
the public challenge partition exports exactly the two prompts and metadata,
without answer/expected fields. Neither partition is a protected independent
evaluation bank. This code does not generate, read, derive, or unseal the private
OOD examination data; a user reading this public source can reconstruct the toy
answers. Cross-split semantic uniqueness is inherited and checked again.

Input type checks copy exact per-instance fields into bounded builtin snapshots
before comparison against independently regenerated public content; generated
frozen-dataclass equality is not invoked on live untrusted objects. In
particular, deleting any of the four default-false authority flags can make
`getattr` fall back to a class default: all four are now explicitly required
in the instance's exact schema, as are every other expected field, with no
surplus metadata admitted. Missing/deleted or unexpected fields fail closed
with a source-only invalid verdict, and the serializer converts the same
malformed bundle to `ValueError` before export. Regression tests cover
deletion of `variant_prompt`, `commitment_sha256`, each default authority flag,
and introduction of an unknown field on frozen instances. Export regenerates
from the receipt after verification, avoiding
mutation of caller-owned data in its output. Oversized public fields and
excessive pair counts are rejected.

All effect authority fields are always false. A consistent pair does **not**
authorize training, graduation, model promotion, deployment, a neural weight
improvement claim, or an equal-time optimizer comparison. R18 MeZO historical
elapsed overrun remains invalid. Real Qwen3.5-4B fresh-process inference and
independent held-out behavioral evaluation are still necessary.

Testing is deterministic, CPU/stdlib-only:

`python -m unittest tests.test_r34_public_metamorphic_challenges -v`

`python -m unittest discover -s tests -p 'test_*.py' -q`

The source may be published as a stacked Draft for further exact-head hostile
review. No merge or protected effect follows from passing these tests.
