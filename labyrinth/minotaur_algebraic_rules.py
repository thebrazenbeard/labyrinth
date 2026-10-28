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
from collections import Counter
from typing import Any

MOD = 97
OPS = 3
TRAIN_MAX = 8


class ClosureBudgetExhausted(RuntimeError):
    """The optimizer requested another gradient closure past a hard cap."""

    def __init__(self, closure_evaluations: int):
        super().__init__("L-BFGS closure budget exhausted")
        self.closure_evaluations=closure_evaluations


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
    """Deterministically derange each operation's unchanged label multiset.

    A majority label makes this control impossible and is rejected without
    changing the caller's training examples.
    """
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("invalid corruption seed")
    rng = random.Random(seed)
    output = list(examples)
    for op in range(OPS):
        positions = [i for i,x in enumerate(examples) if x[0] == op]
        if len(positions) < 2:
            raise ValueError("each operation requires at least two examples")
        rng.shuffle(positions)
        positions.sort(key=lambda i: examples[i][3])
        labels = [examples[i][3] for i in positions]
        largest_group = max(Counter(labels).values())
        if 2 * largest_group > len(labels):
            raise ValueError("cannot derange operation labels: majority label")
        # With equal labels sorted into blocks of length <= largest_group,
        # every shift in [largest_group, n-largest_group] avoids fixed labels.
        # Draw among valid shifts so the control actually depends on its seed.
        shift = rng.randint(largest_group, len(labels)-largest_group)
        rotated = labels[shift:] + labels[:shift]
        for i,label in zip(positions,rotated):
            old=examples[i]
            output[i]=(old[0],old[1],old[2],label)
    return output


def build_algebraic_head(
    torch: Any, *, seed: int = 20261028, basis: str = "bilinear",
) -> Any:
    """Create a 12-parameter, float64 neural head with an explicit basis."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("invalid neural seed")
    if type(basis) is not str or basis not in ("bilinear", "no_xy"):
        raise ValueError("unknown algebraic feature basis")
    nn=torch.nn
    generator=torch.Generator(device="cpu")
    generator.manual_seed(seed)

    class AlgebraicRuleHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.basis=basis
            initial=torch.randn((OPS,4),generator=generator,dtype=torch.float64)*0.01
            self.coefficients=nn.Parameter(initial)

        def raw(self, op, x, y):
            """Differentiable polynomial output before modular reduction."""
            op=op.to(dtype=torch.long)
            if torch.any((op < 0) | (op >= OPS)):
                raise ValueError("unknown operation index")
            x=x.to(dtype=torch.float64)
            y=y.to(dtype=torch.float64)
            cross=x*y if self.basis=="bilinear" else x*x
            features=torch.stack((torch.ones_like(x),x/8.0,y/8.0,cross/64.0),dim=-1)
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


def _checked_supervision_tensors(head: Any, examples: list[tuple[int,int,int,int]], torch: Any):
    """Validate one small supervised batch for either optimizer, without oracle access."""
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
    return ops,x,y,labels


def train_rule_head(
    head: Any, examples: list[tuple[int,int,int,int]], torch: Any,
    *, steps: int = 500, learning_rate: float = 0.06,
) -> int:
    """Actual AdamW gradient updates, fixed steps and batch for matched controls."""
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError("bounded steps required")
    if type(learning_rate) is not float or not 0.0 <= learning_rate <= 0.1:
        raise ValueError("bounded nonnegative learning rate required")
    ops,x,y,labels=_checked_supervision_tensors(head,examples,torch)
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


def train_rule_head_lbfgs(
    head: Any, examples: list[tuple[int,int,int,int]], torch: Any,
    *, max_iter: int = 200, max_eval: int | None = None,
    learning_rate: float = 1.0,
    require_fit: bool = True, closure_budget: int | None = None,
) -> dict[str,Any]:
    """Opt-in float64 L-BFGS coefficient study; never a graduation gate.

    No table, arithmetic oracle, teacher state, or closed-form coefficient
    solution enters training. A fit means low supervised MSE only; exact
    withheld generalization must be measured separately. Strong-Wolfe may
    call the closure more times than max_iter or the requested max_eval, so
    actual counts are returned. Sparse samples may be rank-deficient even
    when they admit a zero-loss training fit.
    """
    if type(max_iter) is not int or not 1 <= max_iter <= 500:
        raise ValueError("bounded L-BFGS iteration budget required")
    if max_eval is not None and (type(max_eval) is not int or not 1 <= max_eval <= 1000):
        raise ValueError("bounded L-BFGS function-evaluation budget required")
    if closure_budget is not None and (type(closure_budget) is not int
                                       or not 1 <= closure_budget <= 1000):
        raise ValueError("bounded hard closure budget required")
    if type(learning_rate) is not float or not 0.0 < learning_rate <= 1.0:
        raise ValueError("bounded positive L-BFGS learning rate required")
    if type(require_fit) is not bool:
        raise ValueError("require_fit must be bool")
    if (head.coefficients.dtype != torch.float64 or
            head.coefficients.device.type != "cpu" or
            tuple(head.coefficients.shape) != (OPS,4) or
            sum(p.numel() for p in head.parameters()) != OPS*4):
        raise ValueError("L-BFGS research requires 12 CPU float64 coefficients")
    ops,x,y,labels=_checked_supervision_tensors(head,examples,torch)
    selected_basis=getattr(head,"basis","bilinear")
    if selected_basis not in ("bilinear","no_xy"):
        raise ValueError("unsupported neural feature basis")
    with torch.no_grad():
        initial=head.coefficients.detach().clone()
        initial_loss=float(((head.raw(ops,x,y)/64.0-labels)**2).mean().item())
        cross=x*y if selected_basis=="bilinear" else x*x
        design=torch.stack((torch.ones_like(x),x/8.0,y/8.0,cross/64.0),dim=-1)
        ranks=[int(torch.linalg.matrix_rank(design[ops==op]).item()) for op in range(OPS)]
    optimizer=torch.optim.LBFGS(
        head.parameters(),lr=learning_rate,max_iter=max_iter,history_size=100,
        max_eval=max_eval,
        tolerance_grad=1e-15,tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )
    calls=0

    def closure():
        nonlocal calls
        # PyTorch's max_eval is a soft bound during strong-Wolfe line search.
        # A hard cap must reject *before* the next forward/backward attempt;
        # the outer exception handler restores parameters after an abort.
        if closure_budget is not None and calls >= closure_budget:
            raise ClosureBudgetExhausted(calls)
        calls+=1
        optimizer.zero_grad(set_to_none=True)
        prediction=head.raw(ops,x,y)/64.0
        loss=(prediction-labels).square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite L-BFGS objective")
        loss.backward()
        if head.coefficients.grad is None or not bool(torch.isfinite(head.coefficients.grad).all()):
            raise RuntimeError("nonfinite L-BFGS gradient")
        # Never clip L-BFGS gradients: Wolfe requires the actual derivative.
        return loss

    head.train()
    try:
        optimizer.step(closure)
        with torch.no_grad():
            final_loss=float(((head.raw(ops,x,y)/64.0-labels)**2).mean().item())
            parameter_delta_l2=float(torch.linalg.vector_norm(
                head.coefficients-initial).item())
        if not torch.isfinite(torch.tensor(final_loss)) or not torch.isfinite(torch.tensor(parameter_delta_l2)):
            raise RuntimeError("nonfinite L-BFGS result")
        state=optimizer.state.get(head.coefficients,{})
        iterations=int(state.get("n_iter",0))
        fitted=final_loss <= 1e-12
        result={"optimizer":"LBFGS_strong_wolfe",
                "max_iter":max_iter,
                "max_eval":int(optimizer.param_groups[0]["max_eval"]),
                "closure_budget":closure_budget,
                "learning_rate":learning_rate,
                "optimizer_iterations":iterations,"closure_evaluations":calls,
                "initial_loss":initial_loss,"final_loss":final_loss,
                "parameter_delta_l2":parameter_delta_l2,
                "design_rank_by_operation":ranks,
                "identifiable_all_operations":all(rank==4 for rank in ranks),
                "training_fit":fitted,"qualification":"NOT_ESTABLISHED"}
        if require_fit and not fitted:
            raise RuntimeError(f"L-BFGS did not fit supervised examples: mse={final_loss:.6g}")
        return result
    except Exception:
        # A failed fit or optimizer error must not leave a partially updated
        # research head available for accidental inference.
        with torch.no_grad():
            head.coefficients.copy_(initial)
            head.coefficients.grad=None
        raise
    finally:
        head.eval()


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
