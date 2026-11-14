import sys
from types import SimpleNamespace

import pytest

from agent_eval.llm import AnthropicClient, FakeLLM, LLMResponse, ToolCall
from agent_eval.tools import TOOL_SCHEMAS


def test_fake_is_deterministic_and_scenarios_are_validated():
    messages = [{"role": "user", "content": "Status of ORD-1001?"}]
    assert FakeLLM().complete(messages, []) == FakeLLM().complete(messages, [])
    with pytest.raises(ValueError, match="scenario"):
        FakeLLM("typo")


def test_fake_script_has_independent_state():
    script = [LLMResponse(answer="done")]
    one, two = FakeLLM(script=script), FakeLLM(script=script)
    assert one.complete([], []).answer == "done"
    assert two.complete([], []).answer == "done"
    with pytest.raises(ValueError, match="exhausted"):
        one.complete([], [])


def test_adapter_requires_model_and_environment_credential(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="explicit model"):
        AnthropicClient("")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicClient("unused-model")


def test_missing_optional_sdk_has_clear_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-test-placeholder")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ValueError, match="optional anthropic SDK"):
        AnthropicClient("unused-model")


def adapter_with_response(response):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return response

    # Exercise serialization offline without constructing an SDK client.
    adapter = object.__new__(AnthropicClient)
    adapter.model = "unused-model"
    adapter._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return adapter, requests


def test_adapter_converts_tool_calls_and_results_without_network():
    response = SimpleNamespace(content=[SimpleNamespace(type="tool_use", id="c", name="calculator",
                                                        input={"expression": "2 + 3"})])
    adapter, requests = adapter_with_response(response)
    messages = [{"role": "user", "content": "Calculate 2 + 3"}]
    first = adapter.complete(messages, TOOL_SCHEMAS)
    assert first == LLMResponse(tool_calls=[ToolCall("c", "calculator", {"expression": "2 + 3"})])
    assert requests[0]["tools"] == TOOL_SCHEMAS
    assert requests[0]["tool_choice"]["disable_parallel_tool_use"] is True
    messages.extend([
        {"role": "assistant", "content": first.to_dict()},
        {"role": "tool", "name": "calculator", "tool_call_id": "c", "content": {"result": 5}},
    ])
    adapter._client.messages.create = lambda **kwargs: (
        requests.append(kwargs) or SimpleNamespace(
            content=[SimpleNamespace(type="text", text="The result is 5.")], stop_reason="end_turn")
    )
    assert adapter.complete(messages, TOOL_SCHEMAS).answer == "The result is 5."
    converted = requests[-1]["messages"]
    assert converted[1]["content"][0]["type"] == "tool_use"
    assert converted[2]["content"][0]["tool_use_id"] == "c"
    assert converted[2]["content"][0]["is_error"] is False


def test_adapter_groups_multiple_tool_results_and_marks_errors():
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="Unable to calculate.")],
                               stop_reason="end_turn")
    adapter, requests = adapter_with_response(response)
    calls = LLMResponse(tool_calls=[ToolCall("a", "calculator", {}), ToolCall("b", "calculator", {})])
    messages = [
        {"role": "user", "content": "Test"},
        {"role": "assistant", "content": calls.to_dict()},
        {"role": "tool", "name": "calculator", "tool_call_id": "a", "content": {"error": "bad"}},
        {"role": "tool", "name": "calculator", "tool_call_id": "b", "content": {"result": 5}},
    ]
    assert adapter.complete(messages, TOOL_SCHEMAS).answer
    blocks = requests[0]["messages"][-1]["content"]
    assert len(blocks) == 2
    assert blocks[0]["is_error"] is True


@pytest.mark.parametrize("text,stop", [("partial", "max_tokens"), ("", "end_turn")])
def test_adapter_rejects_incomplete_model_answers(text, stop):
    adapter, _ = adapter_with_response(SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason=stop))
    with pytest.raises(ValueError, match="complete final answer"):
        adapter.complete([{"role": "user", "content": "Hello"}], TOOL_SCHEMAS)
