# Main-source Minotaur neural-to-public-evaluator integration

The compact algebraic learner, the oracle-free transition inference path,
the independent V5 public arithmetic evaluator, and inactive guarded
graduation state are already in Labyrinth main. This bridge connects the
learned coefficient head to the independently implemented public evaluator
for **read-only research diagnostics only**, not for model deployment or
qualification.

Call labyrinth.minotaur_public_evidence.evaluate_compact_public with an
in-memory trained head, explicit public AtomicEvaluationCase and
ProgramRolloutCase inputs, training atom identities and Torch. The bridge
rejects direct training atom overlap, duplicate training identities, forged
public oracle labels, invalid program observations and false first-error
claims. Both public evaluators preflight case validity before the bridge
calls the neural model. The neural prediction callback receives only the
opcode and the two operand values; it cannot inspect expected answers or
observed program steps. The model's incoming train/eval state is restored.

The result always states qualification NOT_ESTABLISHED and
graduation_authority none, even if its public test accuracy is perfect.
This bridge does not open private data custody, provide a trusted separate
evaluator process, attest actual model runtime execution, create signed
graduation receipts, or activate models in Vera. Gate state is not touched.

The public tests run actual 12-coefficient trained inference against
source-disjoint atomic combinations and mixed clean/corrupted programs,
verify false claims are refused, and establish that malformed program
inputs cannot trigger model inference before being rejected. The historical
large Minotaur draft is deliberately not merged wholesale into main.
