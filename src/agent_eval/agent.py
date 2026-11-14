"""A bounded synchronous loop; each step is one LLM request."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

from .llm import FakeLLM, LLMClient, LLMResponse, ToolCall
from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS
from .tracing import TraceWriter


@dataclass
class RunResult:
    run_id: str
    prompt: str
    answer: str
    steps: int
    spans: list[dict]
    trace_path: Path
    error: str | None = None


def _validate_response(response: LLMResponse) -> LLMResponse:
    if not isinstance(response, LLMResponse):
        raise ValueError("model must return an LLMResponse")
    if not isinstance(response.tool_calls, list) or len(response.tool_calls) > 8:
        raise ValueError("model may request at most 8 tools per step")
    has_answer = isinstance(response.answer, str) and bool(response.answer.strip())
    if (response.answer is not None and not has_answer) or has_answer == bool(response.tool_calls):
        raise ValueError("model must return either an answer or tool calls")
    ids = set()
    for call in response.tool_calls:
        if (not isinstance(call, ToolCall) or not isinstance(call.id, str) or not call.id
                or call.id in ids or not isinstance(call.name, str)
                or not isinstance(call.arguments, dict)):
            raise ValueError("invalid tool call")
        ids.add(call.id)
    return response


def run_agent(
    prompt: str, client: LLMClient | None = None, *, max_steps: int = 6,
    trace_dir: Path | str = "traces", run_id: str | None = None,
) -> RunResult:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must not be empty")
    if type(max_steps) is not int or not 1 <= max_steps <= 100:
        raise ValueError("max_steps must be an integer between 1 and 100")
    client = client if client is not None else FakeLLM()
    trace = TraceWriter(trace_dir, run_id)
    messages = [{"role": "user", "content": prompt}]
    answer, error, steps = "", None, 0
    try:
        for steps in range(1, max_steps + 1):
            response = trace.capture(
                steps, "llm", type(client).__name__,
                {"messages": messages, "tools": TOOL_SCHEMAS},
                lambda: _validate_response(client.complete(copy.deepcopy(messages), copy.deepcopy(TOOL_SCHEMAS))),
            )
            messages.append({"role": "assistant", "content": response.to_dict()})
            if response.answer is not None:
                answer = response.answer
                break
            for call in response.tool_calls:
                try:
                    output = trace.capture(
                        steps, "tool", call.name, call.arguments,
                        lambda call=call: TOOL_FUNCTIONS[call.name](**call.arguments),
                    )
                except Exception:
                    output = {"error": trace.spans[-1]["error"]}
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "name": call.name, "content": output})
        else:
            error = f"max_steps exhausted ({max_steps}) without a final answer"
    except Exception as exc:
        recorded_error = trace.spans[-1]["error"] if trace.spans else None
        error = recorded_error or f"{type(exc).__name__}: {exc}"
    finally:
        trace.close()
    return RunResult(trace.run_id, prompt, answer, steps, trace.spans, trace.path, error)
