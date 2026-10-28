"""Neuro-symbolic composition from an incomplete-data learned ADD primitive.

This is intentionally NOT a model that learned SUB/MUL algorithms. Their
control flow is given. All value updates come from neural ADD predictions;
there is no exhaustive pair lookup or arithmetic oracle inside the predictor.
"""
from __future__ import annotations


def predict_composed_operators(model, op, x, y, *, semantic_ids=(0, 1, 2)):
    """Vectorized 7-bit double-and-add, unary inverse and direct learned ADD.

    The vector returned is a *neural-guided* result, not a verified verdict.
    Semantic operation identities and the modulo-97 inverse operation are
    structural priors. The caller must independently check before approval.
    """
    import torch
    if (type(semantic_ids) is not tuple or len(semantic_ids) != 3
            or any(type(k) is not int or not 0 <= k < 3 for k in semantic_ids)
            or len(set(semantic_ids)) != 3):
        raise ValueError("semantic_ids must be permutation of three op IDs")
    if (not isinstance(op,torch.Tensor) or not isinstance(x,torch.Tensor)
            or not isinstance(y,torch.Tensor) or
            op.dtype != torch.long or x.dtype != torch.long or y.dtype != torch.long
            or op.ndim != 1 or op.shape != x.shape or op.shape != y.shape or
            bool(((x < 0) | (x >= 97) | (y < 0) | (y >= 97)).any()) or
            bool((~torch.isin(op,torch.tensor(semantic_ids,device=op.device))).any())):
        raise ValueError("bounded long input vectors required")

    add_id,sub_id,mul_id=semantic_ids
    model.eval()
    def learned_add(a,b):
        logits=model(torch.full_like(op,add_id),a,b)
        if logits.shape != (len(op),97) or not torch.isfinite(logits).all():
            raise ValueError("invalid learned addition logits")
        return logits.argmax(dim=1)

    with torch.no_grad():
        add=learned_add(x,y)
        # Explicit algebraic prior: subtraction is addition of inverse.
        neg_y=torch.remainder(-y,97)
        sub=learned_add(x,neg_y)

        # Explicit algebraic prior: multiplication as at most seven
        # learned-addition doublings/accumulations, never x*y.
        bits=y.clone()
        accumulator=torch.zeros_like(x)
        doubled=x.clone()
        for _ in range(7):
            summed=learned_add(accumulator,doubled)
            accumulator=torch.where((bits & 1)!=0,summed,accumulator)
            bits=torch.bitwise_right_shift(bits,1)
            doubled=learned_add(doubled,doubled)
        result=torch.where(op==add_id,add,
                           torch.where(op==sub_id,sub,accumulator))
    return result
