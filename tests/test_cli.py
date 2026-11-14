import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def invoke(tmp_path, *arguments):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run([sys.executable, "-m", "agent_eval", *map(str, arguments)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)


def test_run_cli_saves_trace_and_answers(tmp_path):
    result = invoke(tmp_path, "run", "Status of ORD-1001?", "--run-id", "cli_order")
    assert result.returncode == 0, result.stderr
    assert "processing" in result.stdout
    assert "Trace: traces/cli_order.jsonl" in result.stdout
    assert (tmp_path / "traces/cli_order.jsonl").is_file()


def test_eval_cli_success_and_failed_gate(tmp_path):
    args = ("eval", "--cases-dir", ROOT / "evals/cases", "--min-pass-rate", "1.0")
    passed = invoke(tmp_path, *args)
    assert passed.returncode == 0, passed.stderr
    assert "Pass rate: 3/3 (100.0%)" in passed.stdout
    failed = invoke(tmp_path, *args, "--scenario", "buggy")
    assert failed.returncode == 1
    assert "Pass rate: 1/3 (33.3%)" in failed.stdout
    assert "regression_example_buggy_order" in failed.stdout


@pytest.mark.parametrize("threshold,code", [("0", 0), ("0.3333333333333333", 0),
                                           ("0.3334", 1), ("1", 1)])
def test_gate_compares_actual_rate_not_rounded_display(tmp_path, threshold, code):
    result = invoke(tmp_path, "eval", "--cases-dir", ROOT / "evals/cases", "--scenario", "buggy",
                    "--min-pass-rate", threshold)
    assert result.returncode == code, result.stderr


@pytest.mark.parametrize("value", ["-0.01", "1.01", "nan", "inf", "invalid"])
def test_invalid_threshold_is_usage_error(tmp_path, value):
    result = invoke(tmp_path, "eval", "--min-pass-rate", value)
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_eval_empty_or_invalid_suite_is_configuration_error(tmp_path):
    empty = invoke(tmp_path, "eval", "--cases-dir", tmp_path, "--min-pass-rate", "0")
    assert empty.returncode == 2
    assert "no YAML" in empty.stderr
    (tmp_path / "invalid.yaml").write_text("invalid: [yaml")
    invalid = invoke(tmp_path, "eval", "--cases-dir", tmp_path)
    assert invalid.returncode == 2
    assert "invalid case" in invalid.stderr


def test_run_step_exhaustion_and_eval_runtime_failure(tmp_path):
    run = invoke(tmp_path, "run", "Status of ORD-1001?", "--max-steps", "1")
    assert run.returncode == 1
    assert "max_steps exhausted" in run.stdout
    evaluation = invoke(tmp_path, "eval", "--cases-dir", ROOT / "evals/cases", "--max-steps", "1")
    assert evaluation.returncode == 1
    assert "run_error" in evaluation.stdout


def test_promote_cli_writes_case_and_rejects_overwrite(tmp_path):
    args = ("promote-trace", ROOT / "traces/example_buggy_order.jsonl",
            "--expect-contains", "processing", "--expect-contains", "2026-11-20",
            "--must-call-tool", "lookup_order")
    result = invoke(tmp_path, *args)
    assert result.returncode == 0, result.stderr
    assert "Wrote evals/cases/regression_example_buggy_order.yaml" in result.stdout
    assert (tmp_path / "evals/cases/regression_example_buggy_order.yaml").is_file()
    repeated = invoke(tmp_path, *args)
    assert repeated.returncode == 2
    assert "File exists" in repeated.stderr


@pytest.mark.parametrize("args", [
    (), ("run",), ("run", ""), ("run", "Hello", "--run-id", "../unsafe"),
    ("run", "Hello", "--max-steps", "0"),
    ("run", "Hello", "--model", "unused-model"),
    ("run", "Hello", "--backend", "anthropic"),
    ("run", "Hello", "--backend", "anthropic", "--model", "unused-model", "--scenario", "buggy"),
    ("promote-trace", "missing.jsonl", "--expect-contains", "processing"),
    ("promote-trace", ROOT / "traces/example_buggy_order.jsonl"),
])
def test_cli_usage_and_input_errors(tmp_path, args):
    result = invoke(tmp_path, *args)
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_help_works_without_optional_sdk_or_credentials(tmp_path):
    result = invoke(tmp_path, "--help")
    assert result.returncode == 0
    assert "promote-trace" in result.stdout


def test_malformed_trace_type_returns_input_error_without_traceback(tmp_path):
    source = ROOT / "traces/example_buggy_order.jsonl"
    span = json.loads(source.read_text())
    span["type"] = []
    path = tmp_path / "malformed.jsonl"
    path.write_text(json.dumps(span) + "\n")
    result = invoke(tmp_path, "promote-trace", path, "--expect-contains", "processing")
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
