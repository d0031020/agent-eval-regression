"""CLI exit codes: 0 success, 1 gate/run failure, 2 configuration/input error."""

import argparse
import sys

import yaml

from .agent import run_agent
from .evals import load_cases, print_report, run_suite
from .llm import AnthropicClient, FakeLLM
from .promotion import promote_trace


def _pass_rate(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("min-pass-rate must be between 0 and 1")
    return number


def _steps(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 100:
        raise argparse.ArgumentTypeError("max-steps must be between 1 and 100")
    return number


def _model_arguments(parser):
    parser.add_argument("--backend", choices=["fake", "anthropic"], default="fake")
    parser.add_argument("--scenario", choices=["fixed", "buggy"], default="fixed")
    parser.add_argument("--model", help="explicit model ID for the optional Anthropic adapter")
    parser.add_argument("--max-steps", type=_steps, default=6)
    parser.add_argument("--trace-dir", default="traces")


def _client_factory(args):
    if args.backend == "fake":
        if args.model:
            raise ValueError("--model requires --backend anthropic")
        return lambda: FakeLLM(args.scenario)
    if not args.model:
        raise ValueError("--backend anthropic requires --model")
    if args.scenario != "fixed":
        raise ValueError("--scenario buggy is only available with the fake backend")
    return lambda: AnthropicClient(args.model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trace, evaluate, and regress a local tool agent")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one prompt and save its trace")
    run.add_argument("prompt")
    run.add_argument("--run-id", help="optional unique ID; existing traces are never overwritten")
    _model_arguments(run)
    evaluate = commands.add_parser("eval", help="run all YAML cases")
    evaluate.add_argument("--cases-dir", default="evals/cases")
    evaluate.add_argument("--min-pass-rate", type=_pass_rate, default=1.0)
    _model_arguments(evaluate)
    promote = commands.add_parser("promote-trace", help="write assertions for a captured wrong answer")
    promote.add_argument("trace")
    promote.add_argument("--expect-contains", action="append", default=[])
    promote.add_argument("--must-call-tool", action="append", default=[])
    promote.add_argument("--cases-dir", default="evals/cases")
    args = parser.parse_args(argv)
    try:
        if args.command == "promote-trace":
            path = promote_trace(args.trace, expect_contains=args.expect_contains,
                                 must_call_tool=args.must_call_tool, cases_dir=args.cases_dir)
            print(f"Wrote {path}")
            return 0
        factory = _client_factory(args)
        if args.command == "run":
            result = run_agent(args.prompt, factory(), max_steps=args.max_steps,
                               trace_dir=args.trace_dir, run_id=args.run_id)
            print(result.answer or f"Run failed: {result.error}")
            print(f"Trace: {result.trace_path}")
            return 1 if result.error else 0
        cases = load_cases(args.cases_dir)
        outcomes = run_suite(cases, factory, trace_dir=args.trace_dir, max_steps=args.max_steps)
        rate = print_report(outcomes)
        return 0 if rate >= args.min_pass_rate else 1
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
