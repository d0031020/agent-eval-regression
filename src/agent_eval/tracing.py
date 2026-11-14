"""One exclusive JSONL file per run; spans survive model and tool exceptions."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run_id must be 1 to 80 letters, digits, underscores or hyphens")
    return run_id


class TraceWriter:
    def __init__(self, directory: Path | str = "traces", run_id: str | None = None):
        self.run_id = validate_run_id(run_id if run_id is not None else uuid.uuid4().hex)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{self.run_id}.jsonl"
        self._file = self.path.open("x", encoding="utf-8")
        self.spans: list[dict[str, Any]] = []

    def capture(self, step: int, kind: str, name: str, inputs: dict, action: Callable):
        span = {"run_id": self.run_id, "step": step, "type": kind, "name": name,
                "input": {}, "output": None, "latency_ms": 0.0,
                "error": None}
        start = time.perf_counter()
        try:
            # Snapshot and validate inside the exception boundary so invalid model
            # payloads produce valid error spans instead of false successes.
            span["input"] = json.loads(json.dumps(inputs, allow_nan=False))
            result = action()
            output = result.to_dict() if hasattr(result, "to_dict") else result
            span["output"] = json.loads(json.dumps(output, allow_nan=False))
            return result
        except Exception as exc:
            span["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
            self.spans.append(span)
            self._file.write(json.dumps(span, ensure_ascii=False, allow_nan=False) + "\n")
            self._file.flush()

    def close(self):
        self._file.close()


def read_trace(path: Path | str) -> list[dict]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        spans = [json.loads(line) for line in lines if line.strip()]
    except (ValueError, UnicodeError) as exc:
        raise ValueError("trace must contain valid JSONL") from exc
    if not spans:
        raise ValueError("trace is empty")
    run_id = None
    for span in spans:
        required = {"run_id", "step", "type", "name", "input", "output", "latency_ms", "error"}
        if not isinstance(span, dict) or not required <= span.keys():
            raise ValueError("trace contains an invalid span")
        validate_run_id(span["run_id"])
        run_id = run_id or span["run_id"]
        if span["run_id"] != run_id:
            raise ValueError("trace contains multiple runs")
        if (type(span["step"]) is not int or span["step"] < 1
                or not isinstance(span["type"], str) or span["type"] not in {"llm", "tool"}
                or not isinstance(span["input"], dict)
                or not isinstance(span["name"], str)
                or type(span["latency_ms"]) not in {int, float}
                or not 0 <= span["latency_ms"] < float("inf")
                or (span["error"] is not None and not isinstance(span["error"], str))):
            raise ValueError("trace contains an invalid span")
    return spans
