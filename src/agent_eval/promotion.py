"""Convert a completed wrong-answer trace into a model-independent eval case."""

from dataclasses import asdict
from pathlib import Path

import yaml

from .agent import RunResult
from .evals import EvalCase, check_result, validate_checks
from .tracing import read_trace


def result_from_trace(path: Path | str) -> RunResult:
    spans = read_trace(path)
    llm_spans = [s for s in spans if s["type"] == "llm"]
    if not llm_spans or spans[0]["type"] != "llm":
        raise ValueError("trace must start with an LLM span")
    if [s["step"] for s in llm_spans] != list(range(1, len(llm_spans) + 1)):
        raise ValueError("trace LLM steps must be consecutive starting at 1")
    step = 0
    for span in spans:
        if span["type"] == "llm":
            step += 1
        if span["step"] != step:
            raise ValueError("trace spans are out of step order")
    messages = llm_spans[0]["input"].get("messages")
    if (not isinstance(messages, list) or not messages
            or not isinstance(messages[0], dict) or messages[0].get("role") != "user"
            or not isinstance(messages[0].get("content"), str)
            or not messages[0]["content"].strip()):
        raise ValueError("trace must contain the original user prompt")
    final = spans[-1]
    output = final["output"]
    if (final["type"] != "llm" or final["error"] is not None
            or not isinstance(output, dict) or output.get("tool_calls") != []
            or not isinstance(output.get("answer"), str) or not output["answer"].strip()):
        raise ValueError("trace must end with a completed final answer")
    return RunResult(spans[0]["run_id"], messages[0]["content"], output["answer"],
                     len(llm_spans), spans, Path(path))


def promote_trace(
    path: Path | str, *, expect_contains: list[str] | None = None,
    must_call_tool: list[str] | None = None, cases_dir: Path | str = "evals/cases",
) -> Path:
    if not expect_contains and not must_call_tool:
        raise ValueError("provide --expect-contains or --must-call-tool")
    checks = {"no_tool_errors": True}
    if expect_contains:
        checks["answer_contains"] = expect_contains
    if must_call_tool:
        checks["must_call_tool"] = must_call_tool
    validate_checks(checks)
    result = result_from_trace(path)
    failures = check_result(result, checks)
    if not failures:
        raise ValueError("trace already passes the supplied checks; nothing to regress")
    case = EvalCase(
        id=f"regression_{result.run_id}", prompt=result.prompt, checks=checks,
        source={"trace": Path(path).name, "observed_answer": result.answer,
                "failed_checks": failures},
    )
    directory = Path(cases_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{case.id}.yaml"
    with destination.open("x", encoding="utf-8") as file:
        yaml.safe_dump(asdict(case), file, sort_keys=False, allow_unicode=True)
    return destination
