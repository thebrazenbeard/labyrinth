"""Read-only public evaluator bridge for real compact Minotaur inference.

A genuine learned 12-coefficient head supplies only op/x/y predictions.
Independent V5 checks the supplied public challenge claims and independently
replays programs. This diagnostic has *no* private oracle, evaluator key,
campaign state writes, graduation authority, or Vera activation.
"""
from __future__ import annotations

from typing import Any, Sequence

from .minotaur_algebraic_eval_v5 import (
    AtomicEvaluationCase, ProgramRolloutCase,
    evaluate_atomic_predictions, evaluate_program_rollouts,
)

_OPS={"ADD":0,"SUB":1,"MUL":2}


def _refusal(reason: str) -> dict[str, Any]:
    return {
        "disposition":"REFUSED", "reason":reason,
        "qualification":"NOT_ESTABLISHED", "graduation_authority":"none",
    }


def evaluate_compact_public(
    head: Any,
    atomic_cases: Sequence[AtomicEvaluationCase],
    program_cases: Sequence[ProgramRolloutCase],
    training_examples: Sequence[tuple[int,int,int,int]],
    torch: Any,
) -> dict[str, Any]:
    """Validate public unseen-atom eligibility, then run neural+independent V5.

    Training examples are supplied only as identity triples for overlap
    detection, never to the model predictor. Claimed output values are
    interpreted solely by the independent evaluator, never by the predictor.
    """
    if (not isinstance(atomic_cases,(list,tuple))
            or not 1 <= len(atomic_cases) <= 4096
            or any(type(c) is not AtomicEvaluationCase for c in atomic_cases)
            or not isinstance(program_cases,(list,tuple))
            or len(program_cases)>4096
            or any(type(c) is not ProgramRolloutCase for c in program_cases)
            or not isinstance(training_examples,(list,tuple))
            or not 1 <= len(training_examples) <= 72):
        return _refusal("invalid_public_evaluation_container")
    training=[]
    for example in training_examples:
        if (type(example) is not tuple or len(example)!=4
                or any(type(v) is not int for v in example)
                or example[0] not in range(3)
                or not 0 <= example[1] < 97
                or not 0 <= example[2] < 97
                or not 0 <= example[3] < 97):
            return _refusal("invalid_training_atom_identity")
        training.append((tuple(_OPS)[example[0]],example[1],example[2]))
    if len(training)!=len(set(training)):
        return _refusal("repeated_training_atom")
    trained=set(training)
    if any((case.op,case.x,case.y) in trained
           for case in atomic_cases
           if type(case.op) is str and type(case.x) is int and type(case.y) is int):
        return _refusal("training_atom_overlap")
    if program_cases:
        # Reject forged observed states and invalid claimed first errors
        # before any model prediction or other user-supplied callback occurs.
        preview_program=evaluate_program_rollouts(
            program_cases,lambda _op,_x,_y:None,
        )
        if preview_program["disposition"]!="SCORED":
            return _refusal("independent_evaluator_refused")
    try:
        predictor = getattr(head,"predict")
        enter_eval = getattr(head,"eval")
        restore_mode = getattr(head,"train")
        was_training = getattr(head,"training")
    except Exception:
        return _refusal("unsupported_neural_head")
    if (not callable(predictor) or not callable(enter_eval)
            or not callable(restore_mode) or type(was_training) is not bool):
        return _refusal("unsupported_neural_head")
    try:
        # torch.nn.Module.train() recursively changes every descendant, even
        # when some children were deliberately in a different mode. Preserve
        # the complete pre-evaluation mode map rather than just the root flag.
        module_iter = getattr(head, "modules", None)
        if module_iter is None:
            modes = [(head, was_training)]
        else:
            if not callable(module_iter):
                return _refusal("unsupported_neural_head")
            modes = []
            seen = set()
            for module in module_iter():
                identifier = id(module)
                if identifier in seen or len(modes) >= 4096:
                    return _refusal("unsupported_neural_head")
                seen.add(identifier)
                mode = getattr(module, "training")
                if type(mode) is not bool or not callable(getattr(module, "train")):
                    return _refusal("unsupported_neural_head")
                modes.append((module, mode))
            if not modes or modes[0][0] is not head or modes[0][1] is not was_training:
                return _refusal("unsupported_neural_head")
    except Exception:
        return _refusal("unsupported_neural_head")

    def neural_predict(op: str,x: int,y: int) -> int:
        if (type(op) is not str or op not in _OPS
                or type(x) is not int or not 0 <= x < 97
                or type(y) is not int or not 0 <= y < 97):
            raise ValueError("invalid_neural_input")
        result=predictor(
            torch.tensor(_OPS[op],dtype=torch.long),
            torch.tensor(x,dtype=torch.long),
            torch.tensor(y,dtype=torch.long),
        )
        if result.ndim!=0:
            raise ValueError("expected_scalar_neural_output")
        return int(result.item())

    try:
        enter_eval()
        with torch.no_grad():
            # Recheck public oracle claims before invoking predictions.
            # The independent evaluator rejects forged labels or malformed IDs.
            preview=evaluate_atomic_predictions(
                atomic_cases,{case.case_id:None for case in atomic_cases},
                training_atoms=training,
            )
            if preview["disposition"]!="SCORED":
                return _refusal("independent_evaluator_refused")
            predictions={
                case.case_id:neural_predict(case.op,case.x,case.y)
                for case in atomic_cases
            }
            atomic=evaluate_atomic_predictions(
                atomic_cases,predictions,training_atoms=training,
            )
            if atomic["disposition"]!="SCORED":
                return _refusal("independent_evaluator_refused")
            if program_cases:
                program=evaluate_program_rollouts(program_cases,neural_predict)
                if program["disposition"]!="SCORED":
                    return _refusal("independent_evaluator_refused")
            else:
                program={"disposition":"NOT_EVALUATED",
                         "qualification":"NOT_ESTABLISHED",
                         "graduation_authority":"none"}
    except Exception:
        # A failed neural callback must never become a scored public result.
        # A normal refusal carries no raw untrusted exception data.
        return _refusal("neural_inference_failed")
    finally:
        try:
            # PyTorch's modules() visits parents before children. Restore in
            # that order so a parent's recursive train() cannot overwrite the
            # original mixed training/evaluation state of a descendant.
            for module, mode in modes:
                module.train(mode)
            if module_iter is not None:
                current_iter = getattr(head, "modules")
                if not callable(current_iter):
                    return _refusal("neural_state_restore_failed")
                current = []
                for module in current_iter():
                    if len(current) >= len(modes):
                        return _refusal("neural_state_restore_failed")
                    current.append(module)
                if (len(current) != len(modes)
                        or any(actual is not expected for actual, (expected, _) in zip(current, modes))):
                    return _refusal("neural_state_restore_failed")
            if any(getattr(module, "training") is not mode for module, mode in modes):
                return _refusal("neural_state_restore_failed")
        except Exception:
            # An unreadable model state invalidates the whole diagnostic,
            # including a score that was otherwise computed successfully.
            return _refusal("neural_state_restore_failed")

    return {
        "disposition":"PUBLIC_DIAGNOSTIC",
        "atomic":atomic,"program":program,
        "qualification":"NOT_ESTABLISHED",
        "graduation_authority":"none",
        "model_source":"caller_supplied_learned_head",
        "independent_private_oracle":False,
        "limits":["Public challenge labels are independently recomputed",
                  "There is no independent model execution attestation",
                  "No evaluator signing key or promotion authority"],
    }
