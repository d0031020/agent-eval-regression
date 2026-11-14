"""Strict YAML cases and transparent assertions over an agent run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from .agent import RunResult, run_agent
from .llm import FakeLLM, LLMClient

CHECK_NAMES = {"must_call_tool", "answer_contains", "answer_not_contains",
               "max_steps", "no_tool_errors"}


def validate_checks(checks: dict) -> None:
    if not isinstance(checks, dict) or not checks:
        raise ValueError("checks must be a nonempty mapping")
    if set(checks) - CHECK_NAMES:
        raise ValueError("unknown eval check")
    for name, value in checks.items():
        if name in {"must_call_tool", "answer_contains", "answer_not_contains"}:
            if (not isinstance(value, list) or not value
                    or any(not isinstance(v, str) or not v.strip() for v in value)):
                raise ValueError(f"{name} must be a nonempty list of nonempty strings")
        elif name == "max_steps":
            if type(value) is not int or not 1 <= value <= 100:
                raise ValueError("max_steps check must be an integer between 1 and 100")
        elif type(value) is not bool:
            raise ValueError("no_tool_errors must be a boolean")


@dataclass
class EvalCase:
    id: str
    prompt: str
    checks: dict
    source: dict | None = None

    @classmethod
    def from_dict(cls, data: dict) -> EvalCase:
        if not isinstance(data, dict) or set(data) - {"id", "prompt", "checks", "source"}:
            raise ValueError("case must be a mapping with id, prompt, checks and optional source")
        for field in ("id", "prompt"):
            if not isinstance(data.get(field), str) or not data[field].strip():
                raise ValueError(f"case {field} must be a nonempty string")
        validate_checks(data.get("checks"))
        if data.get("source") is not None and not isinstance(data["source"], dict):
            raise ValueError("case source must be a mapping")
        return cls(data["id"], data["prompt"], data["checks"], data.get("source"))


def load_cases(directory: Path | str) -> list[EvalCase]:
    paths = sorted(Path(directory).glob("*.yaml"))
    if not paths:
        raise ValueError("no YAML eval cases found")
    cases, seen = [], set()
    for path in paths:
        try:
            case = EvalCase.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
        except (ValueError, yaml.YAMLError) as exc:
            raise ValueError(f"invalid case {path.name}: {exc}") from exc
        if case.id in seen:
            raise ValueError(f"duplicate case id: {case.id}")
        cases.append(case)
        seen.add(case.id)
    return cases


def check_result(result: RunResult, checks: dict) -> list[str]:
    """Strings are case-insensitive substrings; must_call_tool counts attempts.

    Pair must_call_tool with no_tool_errors to require successful tool execution.
    A runtime failure always fails the case, independently of requested checks.
    """
    validate_checks(checks)
    failures = [f"run_error: {result.error}"] if result.error else []
    called = {s["name"] for s in result.spans if s["type"] == "tool"}
    for tool in checks.get("must_call_tool", []):
        if tool not in called:
            failures.append(f"must_call_tool: missing {tool}")
    answer = result.answer.casefold()
    for text in checks.get("answer_contains", []):
        if text.casefold() not in answer:
            failures.append(f"answer_contains: missing {text!r}")
    for text in checks.get("answer_not_contains", []):
        if text.casefold() in answer:
            failures.append(f"answer_not_contains: found {text!r}")
    if "max_steps" in checks and result.steps > checks["max_steps"]:
        failures.append(f"max_steps: {result.steps} > {checks['max_steps']}")
    if checks.get("no_tool_errors"):
        errors = [s["name"] for s in result.spans if s["type"] == "tool" and s["error"]]
        if errors:
            failures.append(f"no_tool_errors: {', '.join(errors)}")
    return failures


@dataclass
class EvalOutcome:
    case: EvalCase
    result: RunResult
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def run_suite(
    cases: list[EvalCase], client_factory: Callable[[], LLMClient] = FakeLLM, *,
    trace_dir: Path | str = "traces", max_steps: int = 6,
) -> list[EvalOutcome]:
    if not cases:
        raise ValueError("cannot run an empty eval suite")
    outcomes = []
    for case in cases:
        result = run_agent(case.prompt, client_factory(), trace_dir=trace_dir, max_steps=max_steps)
        outcomes.append(EvalOutcome(case, result, check_result(result, case.checks)))
    return outcomes


def print_report(outcomes: list[EvalOutcome]) -> float:
    if not outcomes:
        raise ValueError("cannot report an empty eval suite")
    width = max(len("CASE"), *(len(o.case.id) for o in outcomes))
    print(f"{'CASE':<{width}}  RESULT  STEPS  DETAILS")
    print(f"{'-' * width}  ------  -----  -------")
    for outcome in outcomes:
        status = "PASS" if outcome.passed else "FAIL"
        detail = "; ".join(outcome.failures) or "all checks passed"
        print(f"{outcome.case.id:<{width}}  {status:<6}  {outcome.result.steps:>5}  {detail}")
    passed = sum(o.passed for o in outcomes)
    rate = passed / len(outcomes)
    print(f"Pass rate: {passed}/{len(outcomes)} ({rate:.1%})")
    return rate
