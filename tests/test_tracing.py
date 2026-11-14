import json

import pytest

from agent_eval.agent import run_agent
from agent_eval.tracing import TraceWriter, read_trace


def test_span_shape_latency_and_unique_runs(tmp_path):
    first = run_agent("Calculate 2 + 3", trace_dir=tmp_path)
    second = run_agent("Calculate 2 + 3", trace_dir=tmp_path)
    assert first.run_id != second.run_id
    for span in read_trace(first.trace_path):
        assert set(span) == {"run_id", "step", "type", "name", "input", "output",
                             "latency_ms", "error"}
        assert span["run_id"] == first.run_id
        assert span["latency_ms"] >= 0
        assert span["error"] is None


def test_trace_flushes_before_close_and_captures_exceptions(tmp_path):
    writer = TraceWriter(tmp_path, "failure")

    def fail():
        raise ValueError("demonstration failure")

    try:
        with pytest.raises(ValueError, match="demonstration failure"):
            writer.capture(1, "tool", "example", {}, fail)
        span = read_trace(writer.path)[0]
        assert span["error"] == "ValueError: demonstration failure"
        assert span["output"] is None
    finally:
        writer.close()


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "", ".", "x" * 81, 42])
def test_unsafe_ids_cannot_escape_trace_directory(tmp_path, run_id):
    with pytest.raises(ValueError, match="run_id"):
        TraceWriter(tmp_path, run_id)


def test_trace_does_not_overwrite(tmp_path):
    result = run_agent("Hello", trace_dir=tmp_path, run_id="same")
    original = result.trace_path.read_bytes()
    with pytest.raises(FileExistsError):
        run_agent("Hello again", trace_dir=tmp_path, run_id="same")
    assert result.trace_path.read_bytes() == original


@pytest.mark.parametrize("content", ["", "\n", "not json", "{}", "null", "[]"])
def test_read_trace_rejects_empty_or_invalid_spans(tmp_path, content):
    path = tmp_path / "bad.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_trace(path)


@pytest.mark.parametrize("field,value", [
    ("step", True), ("step", 0), ("type", "other"), ("type", []), ("latency_ms", -1),
    ("latency_ms", float("nan")), ("input", []), ("error", {}), ("name", None),
])
def test_trace_span_fields_are_validated(tmp_path, field, value):
    result = run_agent("Hello", trace_dir=tmp_path)
    span = result.spans[0]
    span[field] = value
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(span) + "\n")
    with pytest.raises(ValueError):
        read_trace(path)


def test_unserializable_input_still_leaves_an_error_span(tmp_path):
    writer = TraceWriter(tmp_path, "invalid_input")
    try:
        with pytest.raises(TypeError):
            writer.capture(1, "tool", "example", {"bad": object()}, lambda: {})
        span = read_trace(writer.path)[0]
        assert span["error"]
        assert span["output"] is None
    finally:
        writer.close()
