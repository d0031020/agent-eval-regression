import json

import pytest

from agent_eval.agent import run_agent
from agent_eval.llm import FakeLLM, LLMResponse, ToolCall
from agent_eval.tools import calculator, lookup_order


def test_fixed_order_uses_fixture_and_records_complete_exchange(tmp_path):
    result = run_agent("Status of ORD-1001?", trace_dir=tmp_path, run_id="order")
    assert result.error is None
    assert result.answer == (
        "Order ORD-1001 is processing. Estimated delivery: 2026-11-20."
    )
    assert result.steps == 2
    assert [s["type"] for s in result.spans] == ["llm", "tool", "llm"]
    assert [s["step"] for s in result.spans] == [1, 1, 2]
    assert result.spans[1]["name"] == "lookup_order"
    assert len(result.spans[0]["input"]["messages"]) == 1
    assert result.spans[2]["input"]["messages"][-1]["role"] == "tool"
    assert result.spans[2]["input"]["messages"][-1]["content"]["status"] == "processing"
    assert [json.loads(line) for line in result.trace_path.read_text().splitlines()] == result.spans


def test_buggy_scenario_reproduces_wrong_answer_without_tool(tmp_path):
    result = run_agent("Status of ORD-1001?", FakeLLM("buggy"), trace_dir=tmp_path)
    assert result.error is None  # Completion alone does not establish correctness.
    assert result.answer == "Order ORD-1001 was delivered yesterday."
    assert result.steps == 1
    assert all(s["type"] == "llm" for s in result.spans)


def test_calculator_and_plain_final_answer(tmp_path):
    calculated = run_agent("Calculate 19 * 3 + 2", trace_dir=tmp_path)
    assert calculated.answer == "The result is 59."
    plain = run_agent("Hello", trace_dir=tmp_path)
    assert plain.steps == 1
    assert plain.answer


def test_multiple_tool_calls_in_one_step(tmp_path):
    script = [LLMResponse(tool_calls=[
        ToolCall("a", "calculator", {"expression": "2 + 3"}),
        ToolCall("b", "lookup_order", {"order_id": "ORD-1002"}),
    ]), LLMResponse(answer="Done")]
    result = run_agent("two tools", FakeLLM(script=script), trace_dir=tmp_path)
    assert result.answer == "Done"
    assert result.steps == 2
    assert [s["type"] for s in result.spans] == ["llm", "tool", "tool", "llm"]


@pytest.mark.parametrize("name,arguments", [
    ("missing_tool", {}), ("calculator", {}), ("calculator", {"expression": "1 / 0"}),
    ("lookup_order", {"order_id": "ORD-9999"}),
])
def test_tool_failures_are_traced_and_returned_to_model(tmp_path, name, arguments):
    class RecoveringModel:
        def complete(self, messages, tools):
            if messages[-1]["role"] == "tool":
                assert messages[-1]["content"]["error"]
                return LLMResponse(answer="Unable to retrieve that fact.")
            return LLMResponse(tool_calls=[ToolCall("a", name, arguments)])

    result = run_agent("test", RecoveringModel(), trace_dir=tmp_path)
    assert result.error is None
    assert result.spans[1]["error"]
    assert result.spans[1]["output"] is None
    assert result.answer == "Unable to retrieve that fact."


def test_max_steps_is_llm_iterations_and_exhaustion_is_failure(tmp_path):
    call = LLMResponse(tool_calls=[ToolCall("a", "calculator", {"expression": "1 + 1"})])
    result = run_agent("test", FakeLLM(script=[call, call]), max_steps=2, trace_dir=tmp_path)
    assert result.steps == 2
    assert len(result.spans) == 4
    assert result.answer == ""
    assert "max_steps exhausted" in result.error


def test_llm_exception_leaves_a_trace(tmp_path):
    result = run_agent("test", FakeLLM(script=[]), trace_dir=tmp_path)
    assert "script exhausted" in result.error
    assert result.steps == 1
    assert result.spans[0]["error"] == result.error
    assert result.spans[0]["output"] is None
    assert result.trace_path.is_file()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {"not", "json"}])
def test_non_json_model_arguments_fail_with_a_valid_error_span(tmp_path, value):
    response = LLMResponse(tool_calls=[ToolCall("a", "calculator", {"expression": value})])
    result = run_agent("test", FakeLLM(script=[response]), trace_dir=tmp_path)
    assert result.error
    span = json.loads(result.trace_path.read_text())
    assert span["error"] == result.error
    assert span["output"] is None
    assert result.answer == ""


@pytest.mark.parametrize("response", [
    None, LLMResponse(), LLMResponse(answer=""), LLMResponse(answer="   "),
    LLMResponse(answer="both", tool_calls=[ToolCall("a", "calculator", {})]),
    LLMResponse(tool_calls=[ToolCall("a", "calculator", [] )]),
    LLMResponse(tool_calls=[ToolCall("a", "x", {}), ToolCall("a", "x", {})]),
    LLMResponse(tool_calls=[ToolCall(str(i), "x", {}) for i in range(9)]),
])
def test_invalid_model_output_is_a_traced_failure(tmp_path, response):
    result = run_agent("test", FakeLLM(script=[response]), trace_dir=tmp_path)
    assert result.error
    assert result.spans[0]["error"]


@pytest.mark.parametrize("max_steps", [0, -1, 101, True, 1.5])
def test_reject_invalid_step_limit(tmp_path, max_steps):
    with pytest.raises(ValueError):
        run_agent("test", max_steps=max_steps, trace_dir=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("prompt", ["", "   ", None])
def test_reject_empty_prompt(tmp_path, prompt):
    with pytest.raises(ValueError):
        run_agent(prompt, trace_dir=tmp_path)


@pytest.mark.parametrize("expression,expected", [
    ("(2 + 3) * 4", 20), ("-9 / +2", -4.5), ("2.5 + 0.5", 3),
])
def test_calculator_supported_arithmetic(expression, expected):
    assert calculator(expression) == {"result": expected}


@pytest.mark.parametrize("expression", [
    "True", "value", "f()", "(1).__class__", "[1]", "'1'", "2 ** 1000",
    "1e309", "1000000000001", "1 / 0", "x" * 201, "+".join(["1"] * 40),
    "", None,
])
def test_calculator_rejects_unsafe_or_unbounded_inputs(expression):
    with pytest.raises((ValueError, SyntaxError, ZeroDivisionError)):
        calculator(expression)


def test_lookup_validates_input_and_unknown_orders():
    assert lookup_order("ORD-1002")["status"] == "shipped"
    with pytest.raises(ValueError, match="string"):
        lookup_order(1001)
    with pytest.raises(ValueError, match="unknown order"):
        lookup_order("ORD-9999")
