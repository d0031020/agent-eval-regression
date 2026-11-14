"""Model boundary: scripted offline default, plus an optional SDK adapter."""

from __future__ import annotations

import copy
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    answer: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMClient(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        """Return either a final answer or tool calls, never both."""
        ...


class FakeLLM:
    """Small deterministic scripts, not a general language model.

    The fix is the order branch in the default scenario: request lookup_order,
    then answer from its result. 'buggy' reproduces the historical hallucination.
    A custom script is useful for testing failures and loop limits.
    """

    def __init__(
        self, scenario: str = "fixed", script: Sequence[LLMResponse] | None = None
    ) -> None:
        if scenario not in {"fixed", "buggy"}:
            raise ValueError("scenario must be fixed or buggy")
        self.scenario = scenario
        self._script = iter(copy.deepcopy(script)) if script is not None else None

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        if self._script is not None:
            try:
                return next(self._script)
            except StopIteration as exc:
                raise ValueError("FakeLLM script exhausted") from exc
        last = messages[-1]
        if last["role"] == "tool":
            data = last["content"]
            if "error" in data:
                return LLMResponse(answer=f"Tool failed: {data['error']}")
            if last["name"] == "lookup_order":
                return LLMResponse(
                    answer=f"Order {data['order_id']} is {data['status']}. "
                    f"Estimated delivery: {data['estimated_delivery']}."
                )
            return LLMResponse(answer=f"The result is {data['result']}.")
        prompt = next(m["content"] for m in messages if m["role"] == "user")
        order = re.search(r"\bORD-\d+\b", prompt, re.IGNORECASE)
        if order:
            if self.scenario == "buggy":
                return LLMResponse(answer=f"Order {order[0].upper()} was delivered yesterday.")
            return LLMResponse(tool_calls=[
                ToolCall("call_1", "lookup_order", {"order_id": order[0].upper()})
            ])
        if prompt.lower().startswith("calculate "):
            expression = prompt[len("calculate "):].strip()
            return LLMResponse(tool_calls=[
                ToolCall("call_1", "calculator", {"expression": expression})
            ])
        return LLMResponse(answer="Try an order ID such as ORD-1001, or Calculate 2 + 3.")


class AnthropicClient:
    """Optional adapter. SDK import and credentials are needed only at creation."""

    def __init__(self, model: str) -> None:
        if not model.strip():
            raise ValueError("an explicit model is required")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the Anthropic adapter")
        try:
            import anthropic
        except ImportError as exc:
            raise ValueError("the optional anthropic SDK is not installed") from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        converted: list[dict] = []
        for message in messages:
            role, content = message["role"], message["content"]
            if role == "assistant":
                blocks = []
                if content["answer"] is not None:
                    blocks.append({"type": "text", "text": content["answer"]})
                blocks.extend(
                    {"type": "tool_use", "id": c["id"], "name": c["name"],
                     "input": c["arguments"]}
                    for c in content["tool_calls"]
                )
                converted.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                import json

                block = {"type": "tool_result", "tool_use_id": message["tool_call_id"],
                         "content": json.dumps(content), "is_error": "error" in content}
                if converted and converted[-1]["role"] == "user":
                    converted[-1]["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})
            else:
                converted.append({"role": "user", "content": content})
        response = self._client.messages.create(
            model=self.model, max_tokens=512,
            system="Use tools for order facts and arithmetic. Answer only from tool results.",
            messages=converted, tools=tools,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )
        calls = [ToolCall(b.id, b.name, b.input) for b in response.content
                 if b.type == "tool_use"]
        if calls:
            return LLMResponse(tool_calls=calls)
        answer = "\n".join(b.text for b in response.content if b.type == "text")
        if not answer or response.stop_reason != "end_turn":
            raise ValueError("model did not produce a complete final answer")
        return LLMResponse(answer=answer)
