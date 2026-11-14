import copy
import json
from pathlib import Path

import pytest
import yaml

from agent_eval.agent import run_agent
from agent_eval.evals import check_result, load_cases, run_suite
from agent_eval.llm import FakeLLM
from agent_eval.promotion import promote_trace, result_from_trace

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "traces/example_buggy_order.jsonl"


def test_promote_captured_failure_then_gate_fixed_and_buggy_models(tmp_path):
    destination = promote_trace(EXAMPLE, expect_contains=["processing"],
                                must_call_tool=["lookup_order"], cases_dir=tmp_path / "cases")
    assert destination.name == "regression_example_buggy_order.yaml"
    data = yaml.safe_load(destination.read_text())
    assert data["prompt"] == "What is the status of order ORD-1001?"
    assert data["checks"] == {"no_tool_errors": True, "answer_contains": ["processing"],
                              "must_call_tool": ["lookup_order"]}
    assert data["source"]["trace"] == EXAMPLE.name
    assert data["source"]["observed_answer"] == "Order ORD-1001 was delivered yesterday."
    assert len(data["source"]["failed_checks"]) == 2
    assert "scenario" not in data
    cases = load_cases(destination.parent)
    assert run_suite(cases, trace_dir=tmp_path / "traces")[0].passed
    assert not run_suite(cases, lambda: FakeLLM("buggy"), trace_dir=tmp_path / "traces")[0].passed
    assert check_result(result_from_trace(EXAMPLE), data["checks"]) == data["source"]["failed_checks"]
    checked_in = yaml.safe_load((ROOT / "evals/cases" / destination.name).read_text())
    assert data == checked_in


def test_promote_refuses_to_overwrite(tmp_path):
    destination = promote_trace(EXAMPLE, expect_contains=["processing"], cases_dir=tmp_path)
    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        promote_trace(EXAMPLE, expect_contains=["processing"], cases_dir=tmp_path)
    assert destination.read_bytes() == original


def test_promote_requires_assertions_and_actual_failure(tmp_path):
    with pytest.raises(ValueError, match="provide"):
        promote_trace(EXAMPLE, cases_dir=tmp_path)
    with pytest.raises(ValueError, match="already passes"):
        promote_trace(EXAMPLE, expect_contains=["delivered"], cases_dir=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("assertions", [
    {"expect_contains": ["processing"]}, {"must_call_tool": ["lookup_order"]},
])
def test_either_assertion_flag_can_be_used_alone(tmp_path, assertions):
    assert promote_trace(EXAMPLE, cases_dir=tmp_path, **assertions).is_file()


def test_trace_reconstruction_counts_llm_steps_not_spans(tmp_path):
    run = run_agent("Status of ORD-1001?", trace_dir=tmp_path)
    reconstructed = result_from_trace(run.trace_path)
    assert reconstructed.prompt == run.prompt
    assert reconstructed.answer == run.answer
    assert reconstructed.steps == 2


@pytest.mark.parametrize("mutation", [
    "mixed_runs", "unsafe_id", "missing_prompt", "empty_prompt", "missing_final",
    "final_error", "tool_end", "bad_step", "out_of_order", "no_llm", "no_messages",
])
def test_invalid_or_incomplete_traces_cannot_be_promoted(tmp_path, mutation):
    spans = [json.loads(line) for line in EXAMPLE.read_text().splitlines()]
    if mutation == "mixed_runs":
        other = copy.deepcopy(spans[0])
        other["run_id"] = "different"
        spans.append(other)
    elif mutation == "unsafe_id":
        spans[0]["run_id"] = "../escape"
    elif mutation == "missing_prompt":
        spans[0]["input"]["messages"][0]["role"] = "assistant"
    elif mutation == "empty_prompt":
        spans[0]["input"]["messages"][0]["content"] = " "
    elif mutation == "missing_final":
        spans[0]["output"] = {"answer": None, "tool_calls": []}
    elif mutation == "final_error":
        spans[0]["error"] = "interrupted"
    elif mutation == "tool_end":
        tool = copy.deepcopy(spans[0])
        tool["type"] = "tool"
        spans.append(tool)
    elif mutation == "bad_step":
        spans[0]["step"] = 2
    elif mutation == "out_of_order":
        tool = copy.deepcopy(spans[0])
        tool.update(type="tool", step=2)
        spans.append(tool)
    elif mutation == "no_llm":
        spans[0]["type"] = "tool"
    elif mutation == "no_messages":
        spans[0]["input"]["messages"] = None
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in spans) + "\n")
    with pytest.raises(ValueError):
        promote_trace(path, expect_contains=["processing"], cases_dir=tmp_path / "cases")
    assert not (tmp_path / "cases").exists()
