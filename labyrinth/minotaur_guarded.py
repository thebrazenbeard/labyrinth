"""Independent, bounded arithmetic checker around scratch Minotaur neural advice.

The checker deliberately does NOT use Labyrinth's synthetic generator oracle.
Supported grammar is intentionally narrow: a two-register modulo-97 program.
Unrecognized or adversarial text ABSTAINS. Model responses do not authorize
graduation and cannot overrule the independent trace checker.
"""
from __future__ import annotations

import re
from typing import Any

_DOMAIN = "bounded_mod97_register_trace"
_HEADER = (
    "Execute this register program in order. ADD, SUB, and MUL update the named "
    "target register using its current value and the operand. An operand is "
    "either a literal integer or the current value of the named register. "
    "After each instruction, replace the updated value with its nonnegative "
    "remainder modulo 97. Other registers do not change.\n"
)
_FOOTER = "What is the final value of A? Respond with exactly one line: ANSWER: <integer>."
_INITIAL = re.compile(r"Initial registers: A=(\d{1,2}), B=(\d{1,2})")
_COMMAND = re.compile(r"([1-6])\. (ADD|SUB|MUL) ([AB]) ([AB]|\d{1,2})")
_OBSERVED = re.compile(r"Step ([1-6]): A=(\d{1,2}), B=(\d{1,2})")


def _result(status: str, step: int | None, reason: str) -> dict[str, Any]:
    return {
        "disposition": status, "first_error_step": step,
        "reason": reason, "task_domain": _DOMAIN,
        "qualification": "NOT_ESTABLISHED",
        "graduation_authority": "none",
    }


def _parse(question: str, steps: tuple[str, ...]):
    if (type(question) is not str or len(question) > 4096 or type(steps) is not tuple
            or not 2 <= len(steps) <= 6 or
            any(type(step) is not str or len(step) > 80 for step in steps)):
        raise ValueError("unsupported input type, size or trace length")
    if not question.startswith(_HEADER) or not question.endswith(_FOOTER):
        raise ValueError("unsupported program statement header/footer")
    body = question[len(_HEADER):-len(_FOOTER)]
    lines = body.splitlines()
    if len(lines) != 1 + len(steps):
        raise ValueError("unexpected text inside program")
    match = _INITIAL.fullmatch(lines[0])
    if match is None:
        raise ValueError("invalid register initialization")
    reg = {"A": int(match[1]), "B": int(match[2])}
    if any(not 0 <= n < 97 for n in reg.values()):
        raise ValueError("initial state outside modular domain")
    cmds = []
    for i, line in enumerate(lines[1:], 1):
        m = _COMMAND.fullmatch(line)
        if m is None or int(m[1]) != i:
            raise ValueError("invalid instruction or sequence")
        operand = m[4]
        if operand not in reg and not 0 <= int(operand) < 97:
            raise ValueError("operand outside modular domain")
        cmds.append((m[2], m[3], operand))
    observed = []
    for i, row in enumerate(steps, 1):
        m = _OBSERVED.fullmatch(row)
        if m is None or int(m[1]) != i:
            raise ValueError("malformed trace step")
        values = (int(m[2]), int(m[3]))
        if any(not 0 <= n < 97 for n in values):
            raise ValueError("trace state outside modular domain")
        observed.append(values)
    return reg, cmds, observed


def check_register_trace(question: str, steps: tuple[str, ...]) -> dict[str, Any]:
    """Re-execute each instruction independently; fail closed on unknown grammar."""
    try:
        reg, commands, observed = _parse(question, steps)
    except (TypeError, ValueError, OverflowError):
        return _result("ABSTAIN", None, "unsupported_or_malformed_input")

    for i, ((operation, destination, operand), state) in enumerate(
            zip(commands, observed), 1):
        value = reg[operand] if operand in reg else int(operand)
        previous = reg[destination]
        if operation == "ADD":
            updated = previous + value
        elif operation == "SUB":
            updated = previous - value
        else:
            updated = previous * value
        reg[destination] = updated % 97
        if (reg["A"], reg["B"]) != state:
            return _result("ERROR_STEP", i, "computed_trace_mismatch")
    return _result("ALL_CORRECT", None, "all_steps_independently_verified")


def reconcile_guardian(question: str, steps: tuple[str, ...],
                       neural_first_error_step: int | None) -> dict[str, Any]:
    """Never allow a neural approval to override the deterministic checker.

    A neural disagreement ABSTAINS. A matching answer remains domain-only
    evidence; this is never a decision to release a Labyrinth trainee.
    """
    independent = check_register_trace(question, steps)
    if independent["disposition"] == "ABSTAIN":
        return independent
    if (type(neural_first_error_step) is not int or
            not 0 <= neural_first_error_step <= len(steps)):
        return _result("ABSTAIN", None, "invalid_or_missing_neural_prediction")
    independent_label = independent["first_error_step"] or 0
    if neural_first_error_step != independent_label:
        return _result("ABSTAIN", None, "independent_checker_and_neural_disagree")
    return independent


def evaluate_guarded_scratch(model_dir: str, cases: list[Any]) -> dict[str, Any]:
    """Run the ACTUAL scratch checkpoint + independent checker on public traces.

    This checks a narrow, fully specified register language. The independent
    checker is responsible for zero wrong approvals; do NOT attribute its
    reliability to the scratch neural model.
    """
    from pathlib import Path
    from .guardian_benchmark import ProcessGuardianCase
    from .minotaur_scratch import (
        ScratchConfig, _features, _model, _parameter_digest, _torch,
        reload_scratch, tokenize_process_case,
    )
    import json

    if (type(cases) is not list or not 1 <= len(cases) <= 4096 or
            any(type(case) is not ProcessGuardianCase for case in cases)):
        raise ValueError("require a bounded set of public process cases")
    root = Path(model_dir)
    # Validates weight digest, strict state_dict, exact historical predictions,
    # and the saved training report. No user-supplied weights are ever executed
    # as code; only the Safetensors format is read.
    reload_scratch(root)
    stored = json.loads((root / "report.json").read_text(encoding="utf-8"))
    config = ScratchConfig(**stored["config"]).validate()
    torch, load_file, _ = _torch()
    model = _model(config, torch)
    model.load_state_dict(load_file(str(root / "minotaur.safetensors"), device="cpu"),
                          strict=True)
    if _parameter_digest(model) != stored["trained_weight_sha256"]:
        raise ValueError("trained parameter values changed")
    model.eval()

    checkers = [check_register_trace(case.question, case.steps) for case in cases]
    neural_predictions: dict[int, int] = {}
    valid = []
    valid_index = []
    for i, (case, checker) in enumerate(zip(cases, checkers)):
        if checker["disposition"] != "ABSTAIN":
            try:
                tokens = tokenize_process_case(case.question, case.steps)
            except ValueError:
                continue
            valid.append({"tokens": tokens, "label": 0})
            valid_index.append(i)
    if valid:
        x, _ = _features(valid, config.max_tokens, torch)
        with torch.no_grad():
            for start in range(0, len(valid), 32):
                pred = model(x[start:start + 32]).argmax(-1).tolist()
                for index, value in zip(valid_index[start:start + 32], pred):
                    neural_predictions[index] = int(value)

    decisions = []
    false_approvals = false_rejections = accepted = correct = abstentions = 0
    checker_disagreement = 0
    wrong = sum(case.first_error_step is not None for case in cases)
    good = len(cases) - wrong
    for i, (case, checker) in enumerate(zip(cases, checkers)):
        gold = case.first_error_step or 0
        independent = checker["first_error_step"] or 0
        if checker["disposition"] == "ABSTAIN" or independent != gold:
            checker_disagreement += 1
            decisions.append(_result("ABSTAIN", None, "checker_cannot_attest_gold"))
        else:
            decisions.append(reconcile_guardian(
                case.question, case.steps, neural_predictions.get(i)))
        disposition = decisions[-1]["disposition"]
        if disposition == "ABSTAIN":
            abstentions += 1
        else:
            accepted += 1
            if ((disposition == "ALL_CORRECT" and gold == 0)
                    or (disposition == "ERROR_STEP"
                        and decisions[-1]["first_error_step"] == gold)):
                correct += 1
        false_approvals += int(disposition == "ALL_CORRECT" and gold != 0)
        false_rejections += int(disposition == "ERROR_STEP" and gold == 0)
    return {
        "total": len(cases), "wrong_cases": wrong, "correct_cases": good,
        "decisions": accepted, "abstentions": abstentions,
        "decision_coverage": accepted / len(cases),
        "correct_decisions": correct,
        "selective_accuracy": correct / accepted if accepted else None,
        "false_approvals": false_approvals, "false_rejections": false_rejections,
        "false_approval_rate": false_approvals / wrong if wrong else None,
        "false_rejection_rate": false_rejections / good if good else None,
        "checker_disagreement_on_gold": checker_disagreement,
        "task_domain": _DOMAIN, "graduation_authority": "none",
        "qualification": "NOT_ESTABLISHED",
        "warning": "False-approval prevention is from independent symbolic checker, not neural competence.",
    }


def predict_guarded_trace(model_dir: str, question: str,
                          steps: tuple[str, ...]) -> dict[str, Any]:
    """Advisory inference from the REAL saved scratch checkpoint with a fail-closed guard.

    This has NO graduation authority. All supported tests require an
    independently computed trace verdict agreeing with the neural prediction.
    """
    from pathlib import Path
    import json
    from .minotaur_scratch import (
        ScratchConfig, _features, _model, _parameter_digest, _torch,
        reload_scratch, tokenize_process_case,
    )

    checker = check_register_trace(question, steps)
    if checker["disposition"] == "ABSTAIN":
        return {**checker, "neural_first_error_step": None}
    root = Path(model_dir)
    reload_scratch(root)  # validates reported corpus and exact checkpoint
    config = ScratchConfig(**json.loads((root / "report.json").read_text())["config"])
    torch, load_file, _ = _torch()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = _model(config, torch)
    model.load_state_dict(load_file(str(root / "minotaur.safetensors"), device="cpu"),
                          strict=True)
    meta = json.loads((root / "report.json").read_text())
    if _parameter_digest(model) != meta["trained_weight_sha256"]:
        raise ValueError("model changed since sealed scratch report")
    entry = {"tokens": tokenize_process_case(question, steps), "label": 0}
    features, _ = _features([entry], config.max_tokens, torch)
    model.eval()
    with torch.no_grad():
        neural = int(model(features).argmax(-1)[0].item())
    decision = reconcile_guardian(question, steps, neural)
    return {**decision, "neural_first_error_step": neural,
            "verified_checkpoint_sha256": meta["safetensors_sha256"]}


def main(argv: list[str] | None = None) -> int:
    """Local, opt-in advisory CLI. No network, execution of model text or releases."""
    import argparse
    import json
    from pathlib import Path
    import sys

    parser = argparse.ArgumentParser(description="Scratch Minotaur guarded trace advice")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-json", required=True)
    args = parser.parse_args(argv)
    path = Path(args.input_json)
    try:
        if not path.is_file() or path.is_symlink():
            raise ValueError("input must be a regular non-symlink JSON file")
        with path.open("rb") as handle:
            payload = handle.read(16385)
        if len(payload) > 16384:
            raise ValueError("input JSON exceeds 16 KiB")

        def unique_keys(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError("duplicate JSON key")
                obj[key] = value
            return obj

        source = json.loads(payload, object_pairs_hook=unique_keys)
        if (type(source) is not dict or set(source) != {"question", "steps"}
                or type(source["question"]) is not str
                or type(source["steps"]) is not list
                or any(type(s) is not str for s in source["steps"])):
            raise ValueError("input JSON requires question string and steps list")
        verdict = predict_guarded_trace(args.checkpoint, source["question"],
                                       tuple(source["steps"]))
        print(json.dumps(verdict, sort_keys=True))
        return 0
    except (ValueError, TypeError, OSError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Minotaur input/checkpoint refused: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
