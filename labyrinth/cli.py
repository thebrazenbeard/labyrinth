"""Command-line interface for offline mazes and controlled prompt experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters import AdapterError, load_adapter
from .runner import (
    build_training_artifact,
    load_json,
    run_experiment,
    run_maze,
    save_json_new,
)
from .maze import generate_maze
from .toy_backend import ToyLinearAdapter, evaluate_toy_model, run_toy_experiment
from .report_audit import audit_report
from .hf_report_audit import audit_hf_report


def _tiers(raw: str) -> tuple[int, ...]:
    try:
        tiers = tuple(int(token.strip()) for token in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tiers must be comma-separated integers") from exc
    if not tiers or len(tiers) != len(set(tiers)) or any(t not in (1, 2, 3, 4) for t in tiers):
        raise argparse.ArgumentTypeError("tiers must be unique values drawn from 1,2,3,4")
    return tiers


def _positive(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="labyrinth", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    def shared(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--seed", type=int, default=2026)
        subparser.add_argument("--tiers", type=_tiers, default=(1, 2, 3))
        subparser.add_argument("--adapter", choices=("replay", "ollama"), default="replay")
        subparser.add_argument("--replay-file", metavar="JSON", help="Exact prompt-to-answer JSON mapping")
        subparser.add_argument("--model", help="Required Ollama model identifier")
        subparser.add_argument("--base-url", default="http://127.0.0.1:11434")
        subparser.add_argument("--timeout-seconds", type=float, default=30.0)
        subparser.add_argument("--allow-remote-ollama", action="store_true")
        subparser.add_argument("--report", metavar="JSON", help="New report path; existing files are never overwritten")

    run = subcommands.add_parser("run", help="Generate and score a deterministic maze")
    shared(run)
    run.add_argument("--count", type=_positive, default=3, help="Tasks PER tier (default: 3)")
    run.add_argument("--split", default="eval", help="Short split identifier (default: eval)")

    experiment = subcommands.add_parser("experiment", help="Compare four held-out in-context conditions")
    shared(experiment)
    experiment.add_argument("--train-count", type=_positive, default=3, help="Training examples PER tier")
    experiment.add_argument("--eval-count", type=_positive, default=3, help="Held-out tasks PER tier")
    artifacts = experiment.add_mutually_exclusive_group()
    artifacts.add_argument("--artifact-in", metavar="JSON", help="Reuse a saved demonstrations artifact")
    artifacts.add_argument("--artifact-out", metavar="JSON", help="Persist a new demonstrations artifact")

    toy = subcommands.add_parser("train-toy", help="Run a CPU-only toy parameter-learning experiment (NOT an LLM)")
    toy.add_argument("--seed", type=int, default=0)
    toy.add_argument("--train-count", type=_positive, default=64)
    toy.add_argument("--eval-count", type=_positive, default=64)
    toy.add_argument("--epochs", type=_positive, default=80)
    toy.add_argument("--learning-rate", type=float, default=0.5)
    toy.add_argument("--checkpoint-out", metavar="JSON", help="Exclusively create the trained toy checkpoint")
    toy.add_argument("--report", metavar="JSON", help="Exclusively create a JSON comparison report")

    toy_eval = subcommands.add_parser("toy-eval", help="Evaluate a saved toy checkpoint in a separate process")
    toy_eval.add_argument("--checkpoint", metavar="JSON", required=True)
    toy_eval.add_argument("--seed", type=int, default=0)
    toy_eval.add_argument("--eval-count", type=_positive, default=64)
    toy_eval.add_argument("--report", metavar="JSON", help="Exclusively create an evaluation report")
    audit = subcommands.add_parser("audit-report", help="Recheck saved report's internal consistency offline")
    audit.add_argument("--input", metavar="JSON", required=True, help="Saved maze or four-arm report")
    audit.add_argument("--artifact", metavar="JSON", help="Training artifact needed for full four-arm prompt audit")
    audit.add_argument("--report", metavar="JSON", help="Exclusively create an audit result file")
    hf_audit = subcommands.add_parser("audit-hf-report", help="Offline internal consistency audit of a saved HF LoRA pilot")
    hf_audit.add_argument("--input", metavar="JSON", required=True, help="Saved HF pilot report")
    hf_audit.add_argument("--report", metavar="JSON", help="Exclusively create an audit result file")
    rg = subcommands.add_parser("benchmark-rg", help="Optional Reasoning Gym development-only scoring; not sealed evaluation")
    rg.add_argument("--dataset", required=True, help="Registered reasoning-gym task identifier")
    rg.add_argument("--size", type=_positive, default=16, help="Number of development cases (max 128)")
    rg.add_argument("--seed", type=int, default=2026)
    rg.add_argument("--adapter", choices=("replay", "ollama"), default="replay")
    rg.add_argument("--replay-file", metavar="JSON", help="Exact prompt-to-answer mapping")
    rg.add_argument("--model", help="Required Ollama model identifier")
    rg.add_argument("--base-url", default="http://127.0.0.1:11434")
    rg.add_argument("--timeout-seconds", type=float, default=30.0)
    rg.add_argument("--allow-remote-ollama", action="store_true")
    rg.add_argument("--report", metavar="JSON", help="Exclusively create a JSON report")
    subcommands.add_parser("train-hf", help="Opt-in bounded LoRA update of an existing local HF checkpoint")
    subcommands.add_parser("eval-hf", help="Reload and evaluate a saved HF LoRA pilot in a new process")
    journey_plan = subcommands.add_parser("journey-plan", help="Preview a declared, regenerative training maze (no training)")
    journey_declaration = journey_plan.add_mutually_exclusive_group(required=True)
    journey_declaration.add_argument("--scope", metavar="JSON", help="Predeclared immutable training-scope manifest")
    journey_declaration.add_argument("--objective-contract", metavar="JSON",
                                     help="Scope plus observable skill objectives/rubric declarations (not implemented graders)")
    journey_plan.add_argument("--cycle", type=int, default=0, help="Bounded zero-based regeneration cycle")
    journey_plan.add_argument("--weak-skills", default="", help="Comma-separated DEVELOPMENT weaknesses only")
    journey_plan.add_argument("--materialize-training", action="store_true",
                              help="Also generate trainee-visible training prompts using bounded Minotaur routing (arithmetic only)")
    journey_plan.add_argument("--report", metavar="JSON", help="Exclusively create a JSON plan")
    journey_toy = subcommands.add_parser("journey-toy", help="Real toy-weight adaptive journey; not an LLM or graduation")
    journey_toy.add_argument("--scope", required=True, metavar="JSON", help="Predeclared immutable training-scope manifest")
    journey_toy.add_argument("--learner-seed", type=int, default=0)
    journey_toy.add_argument("--epochs-by-cycle", help="One positive toy epoch count per authorized cycle (default: 4 each)")
    journey_toy.add_argument("--learning-rate", type=float, default=0.5)
    journey_toy.add_argument("--report", metavar="JSON", help="Exclusively create JSON receipts")
    journey_controls = subcommands.add_parser("journey-toy-controls", help="Matched arithmetic toy learning controls; not formal qualification")
    journey_controls.add_argument("--scope", required=True, metavar="JSON")
    journey_controls.add_argument("--learner-seed", type=int, default=0)
    journey_controls.add_argument("--epochs-by-cycle", help="Comma-separated positive epochs, one per cycle (default: 4 each)")
    journey_controls.add_argument("--learning-rate", type=float, default=0.5)
    journey_controls.add_argument("--report", metavar="JSON", help="Exclusively create JSON control receipts")
    return parser


def _read_training_scope(path: str):
    """Bound scope bytes before parsing; a declaration is never executed."""
    from .training_scope import MAX_SCOPE_JSON_BYTES, parse_scope_json

    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_SCOPE_JSON_BYTES + 1)
    return parse_scope_json(raw)


def _epochs_by_cycle(raw: str | None, cycles: int) -> tuple[int, ...] | None:
    """Validate a declaration-bounded schedule before any model work."""
    if raw is None:
        if cycles * 4 > 256:
            raise ValueError("default total toy epoch budget exceeded")
        return None
    try:
        epochs = tuple(int(token.strip()) for token in raw.split(","))
    except ValueError as exc:
        raise ValueError("epochs-by-cycle must be comma-separated integers") from exc
    if (not epochs or len(epochs) != cycles
            or any(type(n) is not int or not 1 <= n <= 128 for n in epochs)
            or sum(epochs) > 256):
        raise ValueError("epochs-by-cycle must supply bounded positive epochs for every cycle")
    return epochs


def _load_model(args: argparse.Namespace):
    if args.adapter == "ollama":
        if not args.model:
            raise ValueError("--model is required for the Ollama adapter")
        if args.replay_file:
            raise ValueError("--replay-file cannot be used with Ollama")
        return load_adapter(
            "ollama", model=args.model, base_url=args.base_url,
            timeout_seconds=args.timeout_seconds, allow_remote=args.allow_remote_ollama,
        )
    if args.model or args.allow_remote_ollama:
        raise ValueError("Ollama flags require --adapter ollama")
    if args.replay_file:
        responses = load_json(args.replay_file)
        if not isinstance(responses, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in responses.items()
        ):
            raise ValueError("replay file must contain a JSON string-to-string object")
        return load_adapter("replay", responses=responses)
    # Reproducible negative-control model: always responds with the same value.
    # It is explicitly not a reasoning model or a trainer.
    return load_adapter("replay", responses={}, default="ANSWER: 0", model_id="replay:constant-zero")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in ("train-hf", "eval-hf"):
        # Separate parser owns optional HF flags. Importing this module does
        # not import torch, transformers or PEFT; the normal CLI stays offline.
        from .hf_pilot import main as hf_main

        return hf_main(["train" if arguments[0] == "train-hf" else "eval", *arguments[1:]])
    parser = _parser()
    args = parser.parse_args(arguments)
    try:
        if args.command == "audit-report":
            report = audit_report(load_json(args.input, strict=True),
                                  artifact=load_json(args.artifact, strict=True) if args.artifact else None)
        elif args.command == "audit-hf-report":
            report = audit_hf_report(load_json(args.input, strict=True))
        elif args.command == "train-toy":
            report = run_toy_experiment(
                seed=args.seed, train_count=args.train_count, eval_count=args.eval_count,
                epochs=args.epochs, learning_rate=args.learning_rate,
            )
            if args.checkpoint_out:
                candidate = ToyLinearAdapter(seed=args.seed)
                training_cases = generate_maze(seed=args.seed, tiers=(1,), count=args.train_count, split="train")
                candidate.train(training_cases, epochs=args.epochs, learning_rate=args.learning_rate)
                if candidate.model_hash != report["arms"]["trained"]["model_hash"]:
                    raise ValueError("trained model hash differs from controlled experiment")
                report["checkpoint_integrity_sha256"] = candidate.save_checkpoint(args.checkpoint_out)
        elif args.command == "toy-eval":
            candidate = ToyLinearAdapter.load_checkpoint(args.checkpoint)
            report = evaluate_toy_model(candidate, seed=args.seed, eval_count=args.eval_count)
        elif args.command == "journey-plan":
            from dataclasses import asdict
            from .regenerative_maze import generate_maze_plan

            contract = None
            if args.objective_contract:
                from .objective_contract import MAX_OBJECTIVE_JSON_BYTES, parse_objective_json

                with Path(args.objective_contract).open("rb") as stream:
                    contract = parse_objective_json(stream.read(MAX_OBJECTIVE_JSON_BYTES + 1))
                scope = contract.training_scope
            else:
                scope = _read_training_scope(args.scope)
            weak = tuple(token.strip() for token in args.weak_skills.split(",") if token.strip())
            plan = generate_maze_plan(scope, args.cycle, weak_skills=weak)
            report = {"schema": "labyrinth.journey-plan.v1", "scope": scope.to_dict(),
                      "scope_fingerprint": scope.fingerprint, "plan": asdict(plan),
                      "limitations": ["Room descriptors are not actual learning or private exams."]}
            if contract is not None:
                report["objective_contract"] = contract.to_dict()
                report["objective_fingerprint"] = contract.fingerprint
                report["selected_focus"] = {
                    skill: list(contract.selected_focus(skill)) for skill in scope.skills
                }
                report["limitations"].append(
                    "Objective text/rubric thresholds express operator intent, not generated tests or trained behavior."
                )
            if args.materialize_training:
                if scope.skills == ("arithmetic",):
                    from .taskmaster_routing import route_training_rooms

                    routed = route_training_rooms(scope, plan)
                    report["training_provider"] = "bounded_minotaur_arithmetic_routing"
                    report["training_views"] = [
                        {**asdict(entry.trainee_view), "difficulty": entry.suggestion.difficulty,
                         "tactic": entry.suggestion.tactic, "planner_source": entry.suggestion.source}
                        for entry in routed
                    ]
                else:
                    from .journey_challenges import prepare_training_rooms

                    prepared = prepare_training_rooms(plan, semantic_scope=scope)
                    report["training_provider"] = "approved_skill_providers_with_program_focus"
                    report["training_views"] = [
                        {**asdict(entry.trainee_view),
                         "semantic_focus": dict(entry.challenge.metadata).get("semantic_focus")}
                        for entry in prepared
                    ]
                report["limitations"].append(
                    "Generated training-only prompts are not model updates; gold stays evaluator-side."
                )
        elif args.command == "journey-toy":
            from .journey_toy import run_toy_journey

            scope = _read_training_scope(args.scope)
            epochs = _epochs_by_cycle(args.epochs_by_cycle, scope.max_cycles)
            report = run_toy_journey(scope, learner_seed=args.learner_seed,
                                     epochs_by_cycle=epochs, learning_rate=args.learning_rate)
        elif args.command == "journey-toy-controls":
            from .journey_controls import run_matched_toy_controls

            scope = _read_training_scope(args.scope)
            epochs = _epochs_by_cycle(args.epochs_by_cycle, scope.max_cycles)
            report = run_matched_toy_controls(scope, learner_seed=args.learner_seed,
                                              epochs_by_cycle=epochs, learning_rate=args.learning_rate)
        elif args.command == "benchmark-rg":
            from .reasoning_gym_bridge import benchmark_reasoning_gym

            try:
                report = benchmark_reasoning_gym(
                    _load_model(args), args.dataset, size=args.size, seed=args.seed,
                )
            except RuntimeError as exc:
                raise ValueError(str(exc)) from exc
        else:
            adapter = _load_model(args)
            if args.command == "run":
                report = run_maze(adapter, seed=args.seed, count=args.count, tiers=args.tiers, split=args.split)
            else:
                artifact = (
                    load_json(args.artifact_in) if args.artifact_in else
                    build_training_artifact(
                        seed=args.seed, count=args.train_count, tiers=args.tiers,
                        adapter=adapter,
                    )
                )
                report = run_experiment(
                    lambda: _load_model(args), seed=args.seed, eval_count=args.eval_count,
                    tiers=args.tiers, artifact=artifact,
                )
                if args.artifact_out:
                    save_json_new(args.artifact_out, artifact)
        if args.report:
            save_json_new(args.report, report)
            print(f"Wrote reproducible report to {args.report}", file=sys.stderr)
        else:
            print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0 if args.command not in ("audit-report", "audit-hf-report") or report["status"] == "PASS" else 2
    except (ValueError, TypeError, OSError, json.JSONDecodeError, AdapterError) as exc:
        print(f"labyrinth: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
