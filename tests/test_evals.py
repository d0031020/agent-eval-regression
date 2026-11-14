from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from agent_eval.agent import run_agent
from agent_eval.evals import (
    EvalCase, check_result, load_cases, print_report, run_suite, validate_checks,
)
from agent_eval.llm import FakeLLM

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def order_run(tmp_path):
    return run_agent("Status of ORD-1001?", trace_dir=tmp_path)


@pytest.mark.parametrize("checks", [
    {"must_call_tool": ["lookup_order"]},
    {"answer_contains": ["PROCESSING", "2026-11-20"]},
    {"answer_not_contains": ["delivered", "canceled"]},
    {"max_steps": 2}, {"no_tool_errors": True}, {"no_tool_errors": False},
])
def test_every_check_passes(order_run, checks):
    assert check_result(order_run, checks) == []


@pytest.mark.parametrize("checks,expected", [
    ({"must_call_tool": ["lookup_order", "calculator"]}, "must_call_tool"),
    ({"answer_contains": ["processing", "DELIVERED"]}, "answer_contains"),
    ({"answer_not_contains": ["PROCESSING"]}, "answer_not_contains"),
    ({"max_steps": 1}, "max_steps"),
])
def test_each_assertion_can_fail(order_run, checks, expected):
    failures = check_result(order_run, checks)
    assert len(failures) == 1
    assert failures[0].startswith(expected)


def test_tool_attempt_and_tool_success_are_separate_checks(tmp_path):
    result = run_agent("Status of ORD-9999?", trace_dir=tmp_path)
    assert check_result(result, {"must_call_tool": ["lookup_order"]}) == []
    assert check_result(result, {"no_tool_errors": True}) == ["no_tool_errors: lookup_order"]
    assert check_result(result, {"no_tool_errors": False}) == []


def test_runtime_errors_cannot_silently_pass(order_run):
    failed = replace(order_run, error="exhausted")
    assert check_result(failed, {"no_tool_errors": True}) == ["run_error: exhausted"]


@pytest.mark.parametrize("checks", [
    None, {}, [], {"misspelled": True}, {"answer_contains": "processing"},
    {"answer_contains": []}, {"answer_contains": [""]}, {"must_call_tool": [1]},
    {"answer_not_contains": [None]}, {"max_steps": True}, {"max_steps": 0},
    {"max_steps": 101}, {"no_tool_errors": "true"},
])
def test_invalid_checks_fail_closed(checks):
    with pytest.raises(ValueError):
        validate_checks(checks)


def test_checked_in_suite_passes_fixed_and_rejects_buggy(tmp_path, capsys):
    cases = load_cases(ROOT / "evals/cases")
    assert len(cases) == 3
    fixed = run_suite(cases, trace_dir=tmp_path)
    assert all(o.passed for o in fixed)
    assert print_report(fixed) == 1.0
    report = capsys.readouterr().out
    assert "Pass rate: 3/3 (100.0%)" in report
    assert "CASE" in report and "STEPS" in report
    buggy = run_suite(cases, lambda: FakeLLM("buggy"), trace_dir=tmp_path)
    assert sum(o.passed for o in buggy) == 1
    assert print_report(buggy) == 1 / 3
    assert "must_call_tool: missing lookup_order" in capsys.readouterr().out


def test_suite_uses_fresh_client_per_case(tmp_path):
    calls = []

    def factory():
        calls.append(True)
        return FakeLLM()

    cases = load_cases(ROOT / "evals/cases")
    run_suite(cases, factory, trace_dir=tmp_path)
    assert len(calls) == len(cases)


def test_empty_suites_are_errors(tmp_path):
    with pytest.raises(ValueError, match="no YAML"):
        load_cases(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        run_suite([])
    with pytest.raises(ValueError, match="empty"):
        print_report([])


@pytest.mark.parametrize("content", [
    "- a list", "id: case\nprompt: test\nchecks: [invalid]", "id: missing_fields",
    "id: case\nprompt: test\nchecks: {max_steps: 2}\nextra: true", "[invalid yaml",
    "id: case\nprompt: test\nchecks: {max_steps: 2}\nsource: []",
])
def test_bad_case_files_are_errors(tmp_path, content):
    (tmp_path / "bad.yaml").write_text(content)
    with pytest.raises(ValueError, match="invalid case bad.yaml"):
        load_cases(tmp_path)


def test_duplicate_case_ids_are_errors(tmp_path):
    data = {"id": "duplicate", "prompt": "Hello", "checks": {"max_steps": 2}}
    for name in ["first", "second"]:
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(tmp_path)


def test_case_requires_nonempty_identity_and_prompt():
    for field in ["id", "prompt"]:
        case = {"id": "case", "prompt": "Hello", "checks": {"max_steps": 2}}
        case[field] = " "
        with pytest.raises(ValueError):
            EvalCase.from_dict(case)
