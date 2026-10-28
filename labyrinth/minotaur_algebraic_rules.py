"""Tiny learned modular algebraic operations; never a memorized atomic table.

For three operation tokens, learn four coefficients each over a shared
[1, x/8, y/8, xy/64] basis. The basis supplies multiplication and modular
reduction as explicit inductive biases; supervised optimization LEARNS which
combination represents each operation. This is bounded modulo-97 transfer,
not discovery of multiplication from unconstrained raw pixels, novel opcodes,
or an independent Minotaur graduation system.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

MOD = 97
OPS = 3
TRAIN_MAX = 8


def incomplete_training_atoms(seed: int, per_operation: int = 12) -> list[tuple[int,int,int,int]]:
    """Select only non-wrapping small operands (never all 28,227 atoms)."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be a 31-bit integer")
    if type(per_operation) is not int or not 5 <= per_operation <= 24:
        raise ValueError("per_operation must be in [5, 24]")
    rng = random.Random(seed)
    result = []
    for op in range(OPS):
        pairs = [(a,b) for a in range(TRAIN_MAX+1) for b in range(TRAIN_MAX+1)
                 if op != 1 or a >= b]
        rng.shuffle(pairs)
        for a,b in pairs[:per_operation]:
            # This integer arithmetic exists exclusively in data creation.
            y = a+b if op == 0 else a-b if op == 1 else a*b
            assert 0 <= y < MOD
            result.append((op,a,b,y))
    return result


def corrupt_labels_within_op(
    examples: list[tuple[int,int,int,int]], seed: int
) -> list[tuple[int,int,int,int]]:
    """Preserve each operation's label multiset, prohibit unchanged labels."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("invalid corruption seed")
    rng = random.Random(seed)
    output = list(examples)
    for op in range(OPS):
        positions = [i for i,x in enumerate(examples) if x[0] == op]
        labels = [examples[i][3] for i in positions]
        if len(positions) < 2:
            raise ValueError("each operation requires at least two examples")
        shuffled = None
        for _ in range(5000):
            trial = labels[:]
            rng.shuffle(trial)
            if all(a != b for a,b in zip(labels,trial)):
                shuffled=trial
                break
        if shuffled is None:
            raise ValueError("cannot derange operation labels without collisions")
        for i,label in zip(positions,shuffled):
            old=examples[i]
            output[i]=(old[0],old[1],old[2],label)
    return output


def build_algebraic_head(torch: Any, *, seed: int = 20261028) -> Any:
    """Create a 12-parameter, float64 neural operation head."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("invalid neural seed")
    nn=torch.nn
    generator=torch.Generator(device="cpu")
    generator.manual_seed(seed)

    class AlgebraicRuleHead(nn.Module):
        def __init__(self):
            super().__init__()
            initial=torch.randn((OPS,4),generator=generator,dtype=torch.float64)*0.01
            self.coefficients=nn.Parameter(initial)

        def raw(self, op, x, y):
            """Differentiable polynomial output before modular reduction."""
            op=op.to(dtype=torch.long)
            if torch.any((op < 0) | (op >= OPS)):
                raise ValueError("unknown operation index")
            x=x.to(dtype=torch.float64)
            y=y.to(dtype=torch.float64)
            features=torch.stack((torch.ones_like(x),x/8.0,y/8.0,x*y/64.0),dim=-1)
            return (features*self.coefficients[op]).sum(dim=-1)*64.0

        def predict(self,op,x,y):
            # Modular projection is an explicit architectural prior.
            return torch.remainder(torch.round(self.raw(op,x,y)).long(),MOD)

        def logits(self,op,x,y):
            raw=self.raw(op,x,y)
            residues=torch.arange(MOD,dtype=raw.dtype,device=raw.device)
            delta=torch.remainder(raw[...,None]-residues+MOD/2,MOD)-MOD/2
            return -delta.square()

    return AlgebraicRuleHead()


def train_rule_head(
    head: Any, examples: list[tuple[int,int,int,int]], torch: Any,
    *, steps: int = 500, learning_rate: float = 0.06,
) -> int:
    """Actual gradient updates, fixed steps and batch for matched controls."""
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError("bounded steps required")
    if type(learning_rate) is not float or not 0.0 <= learning_rate <= 0.1:
        raise ValueError("bounded nonnegative learning rate required")
    if type(examples) is not list or not examples or len(examples) > 3*24:
        raise ValueError("bounded nonempty supervised examples required")
    for row in examples:
        if (type(row) is not tuple or len(row)!=4 or any(type(v) is not int for v in row)):
            raise ValueError("invalid supervised tuple")
        op,x,y,label=row
        if (op not in (0,1,2) or not 0 <= x <= TRAIN_MAX or
                not 0 <= y <= TRAIN_MAX or (op == 1 and x < y)
                or not 0 <= label < MOD):
            raise ValueError("supervision outside controlled incomplete domain")
    device=head.coefficients.device
    ops=torch.tensor([x[0] for x in examples],dtype=torch.long,device=device)
    x=torch.tensor([row[1] for row in examples],dtype=torch.float64,device=device)
    y=torch.tensor([row[2] for row in examples],dtype=torch.float64,device=device)
    labels=torch.tensor([row[3] for row in examples],dtype=torch.float64,device=device)/64.0
    optimizer=torch.optim.AdamW(head.parameters(),lr=learning_rate,weight_decay=0.0)
    head.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predictions=head.raw(ops,x,y)/64.0
        loss=(predictions-labels).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite algebraic loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
    head.eval()
    return steps


def all_atom_accuracy(head: Any, torch: Any) -> dict[str,int]:
    """Evaluate all 28,227 possible tuples, including missing train atoms."""
    residues=torch.arange(MOD,dtype=torch.long,device=head.coefficients.device)
    a=residues.repeat_interleave(MOD).repeat(OPS)
    b=residues.repeat(MOD).repeat(OPS)
    op=torch.arange(OPS,dtype=torch.long,device=residues.device).repeat_interleave(MOD*MOD)
    with torch.no_grad():
        prediction=head.predict(op,a,b)
        oracle=torch.where(op==0,a+b,torch.where(op==1,a-b,a*b)).remainder(MOD)
        correct=int((prediction==oracle).sum().item())
        prediction_sha256=hashlib.sha256(
            prediction.to(dtype=torch.uint8).cpu().contiguous().numpy().tobytes()
        ).hexdigest()
    return {"correct":correct,"total":OPS*MOD*MOD,
            "prediction_sha256":prediction_sha256}
