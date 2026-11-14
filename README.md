# agent-eval-regression

## What this is

A small Python demo of tracing a tool-using agent, evaluating its behavior, and
turning a wrong answer into a permanent CI regression test.

**The default model is a deterministic, scripted `FakeLLM`.** It recognizes a few
example prompts; it does not perform language understanding. No API account,
model download, network access, or tracing service is needed to run the demo.
The optional Anthropic adapter is separate from the offline test suite.

Order lookup uses fixture data in `src/agent_eval/fixtures/orders.json`; that
order data is synthetic (made up for the demo).

```text
 prompt / YAML case
        |
        v
   bounded agent loop <----> LLMClient
        |                   |-- FakeLLM (default: fixed; buggy for replay)
        |                   `-- AnthropicClient (optional)
        v
   local tool dispatch
        |-- lookup_order --> fixture JSON
        `-- calculator ----> bounded arithmetic AST
        |
        v
   traces/<run_id>.jsonl --> promote-trace + human-written expectations
                                      |
                                      v
                               regression YAML
                                      |
                              eval runner --> CI gate
```

## Why

An agent can finish successfully and still be wrong. This example captures that
distinction: a buggy script invents an order status, while evals check both its
answer and whether it consulted the order tool.

## Install

Python 3.10 or newer. From the repository root, in an environment with package
installation available:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

For an offline checkout whose virtual environment already contains PyYAML and
pytest, skip installation and use:

```sh
export PYTHONPATH=src
.venv/bin/python -m agent_eval eval --min-pass-rate 1.0
.venv/bin/python -m pytest -q
```

After installation, `agent-eval` is also available as a console script.

## Usage

### Quickstart

```sh
python -m agent_eval run 'What is the status of order ORD-1001?'
python -m agent_eval run 'Calculate 19 * 3 + 2'
python -m agent_eval eval --min-pass-rate 1.0
python -m pytest -q
```

The suite contains three cases: arithmetic, an order lookup, and the promoted
regression. Each eval run creates fresh traces. The CLI prints a table with
per-case failures and a pass rate.

### Walkthrough: wrong answer to a CI test

1. **Capture the mistake.** Reproduce the historical buggy script:

   ```sh
   python -m agent_eval run 'What is the status of order ORD-1001?' \
     --scenario buggy --run-id walkthrough_buggy_order
   ```

   It answers `Order ORD-1001 was delivered yesterday.` The fixture says
   `processing`. Open `traces/walkthrough_buggy_order.jsonl`: it has one LLM span
   and no tool span. The `run` command returns zero because it completed; this
   command has no correctness assertions. The original captured example is
   checked in as [example_buggy_order.jsonl](traces/example_buggy_order.jsonl).

2. **Supply the expected behavior and promote the trace.**

   ```sh
   python -m agent_eval promote-trace traces/walkthrough_buggy_order.jsonl \
     --expect-contains processing --must-call-tool lookup_order
   ```

   This writes `evals/cases/regression_walkthrough_buggy_order.yaml`, copying
   the prompt and recording the observed wrong answer and failed assertions.
   The generator checks that the captured trace actually fails your expectations.
   It does not infer truth from the wrong answer. Flags can be repeated to add
   more expected substrings or tools. Trace IDs must be safe filename components;
   existing traces and cases are never overwritten. Use a fresh run ID when
   repeating this walkthrough.

   The already-promoted example is
   [regression_example_buggy_order.yaml](evals/cases/regression_example_buggy_order.yaml):

   ```yaml
   checks:
     no_tool_errors: true
     answer_contains: [processing]
     must_call_tool: [lookup_order]
   ```

3. **Prove the regression catches the bug.**

   ```sh
   python -m agent_eval eval --scenario buggy --min-pass-rate 1.0
   ```

   This intentionally exits **1**. Both order cases fail for missing tool calls
   and wrong answer text. Before adding another walkthrough case, the checked-in
   suite reports `1/3 (33.3%)`. After step 2, it reports `1/4 (25.0%)`.

4. **Apply the fix and replay the same case.** The fix is already present in
   [llm.py](src/agent_eval/llm.py): the default `fixed` script requests
   `lookup_order`, waits for its result, and builds its answer from that result.
   The generic loop is unchanged between scenarios.

   ```sh
   python -m agent_eval eval --min-pass-rate 1.0
   ```

   All cases pass and the command exits **0**. The regression case stores no
   model scenario, so future model implementations face the same assertions.
   The tests also verify that the captured old trace fails these assertions and
   that a new fixed run passes them.

5. **Keep the case in CI.** The workflow at
   [.github/workflows/evals.yml](.github/workflows/evals.yml) runs pytest and
   `python -m agent_eval eval --min-pass-rate 1.0`. It is a workflow file only;
   no repository creation or CI execution is part of this demo.

### Trace and eval semantics

Each run writes an exclusive JSONL file under `traces/`. Each line is a span with
`run_id`, `step`, `type` (`llm` or `tool`), `name`, `input`, `output`,
`latency_ms`, and `error`. Successful spans have `error: null`; failed spans have
`output: null` and an error string. Inputs are snapshots of the actual exchange.
Spans flush after each call, including exceptions. This is an OpenTelemetry-like
local shape, not an OpenTelemetry exporter. Run IDs and measured latencies vary;
the fake model's responses are deterministic.

`step` counts LLM requests, starting at 1. A request may produce multiple tool
spans with the same step number. `--max-steps` bounds LLM requests (default 6,
maximum 100); exhaustion fails the run. Tool exceptions are traced and returned
to the model for a possible recovery. Model errors end the run and fail evals.
The calculator supports parentheses, unary signs, `+`, `-`, `*`, and `/`, with
length, AST size, and numeric magnitude limits. It never executes Python code.

YAML cases live in `evals/cases/*.yaml`, with `id`, `prompt`, and `checks`:

| Check | Value | Meaning |
| --- | --- | --- |
| `must_call_tool` | List of tool names | Every named tool was attempted |
| `answer_contains` | List of strings | Every substring appears, ignoring case |
| `answer_not_contains` | List of strings | None of the substrings appear, ignoring case |
| `max_steps` | Positive integer | LLM request count is at most this value |
| `no_tool_errors` | Boolean | If true, every attempted tool succeeded |

Pair `must_call_tool` with `no_tool_errors: true` when success matters. A case's
`max_steps` assertion measures behavior; the CLI's `--max-steps` is the execution
limit. Unknown fields/checks, invalid types, duplicate case IDs, and empty suites
are errors. Runtime failure always fails a case. Exit codes are **0** for a met
threshold, **1** for a failed threshold or run, and **2** for invalid input or
configuration. Thresholds are fractions in `[0, 1]`, compared before display rounding.

These simple substring checks illustrate a regression workflow; they do not
establish general factual accuracy or production readiness. Traces contain full
prompts, model outputs, tool arguments, results, and error messages. Inspect and
redact real data before sharing. Generated traces are ignored by git except for
the synthetic checked-in example.

### Plug in a real model

Implement `LLMClient.complete(messages, tools) -> LLMResponse`, returning either
a nonempty `answer` or `tool_calls`. Each `ToolCall` contains an ID, a tool name,
and an argument mapping. The loop handles dispatch, feedback, limits, and tracing.

`AnthropicClient` provides an optional adapter that lazily imports the
`anthropic` SDK and reads `ANTHROPIC_API_KEY` from the environment. The SDK is
not a project dependency and is not installed by this demo. In an environment
where it is separately available and network use is authorized, set
`ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` outside the repository, then run:

```sh
python -m agent_eval run 'What is the status of order ORD-1001?' \
  --backend anthropic --model "$ANTHROPIC_MODEL"
python -m agent_eval eval --backend anthropic --model "$ANTHROPIC_MODEL" \
  --min-pass-rate 1.0
```

Use a model ID available to your account. Live calls may incur API charges and
produce different wording. Adapter request/response conversion is tested with
offline stubs; no live API compatibility or model-quality result is claimed.

## Testing

The test suite runs fully offline (no network, no live LLM APIs). It covers:

- **Agent loop** (`tests/test_agent.py`): fixed and buggy scenarios, calculator
  and order tools, step limits, tool failures, invalid model output, input
  validation.
- **Tracing** (`tests/test_tracing.py`): span shape, flush semantics, path safety,
  validation, unserializable inputs.
- **Eval runner** (`tests/test_evals.py`): YAML check types, suite loading, pass
  rate gating, checked-in cases with fixed vs buggy models, duplicate IDs and
  invalid cases.
- **Promotion** (`tests/test_promotion.py`): trace-to-YAML promotion, overwrite
  guards, assertion requirements.
- **LLM clients** (`tests/test_llm.py`): deterministic FakeLLM, Anthropic adapter
  conversion with offline stubs (not live API calls).
- **CLI** (`tests/test_cli.py`): `run`, `eval`, `promote-trace`, exit codes,
  thresholds, usage and configuration errors.

Run:

```sh
python -m pytest -q
```

Or, with an existing venv and `PYTHONPATH=src`:

```sh
.venv/bin/python -m pytest -q
```

## Known limitations

- The default `FakeLLM` is scripted and only recognizes a few example prompts —
  it is not language understanding.
- The optional Anthropic adapter is not covered by tests against the live API.
- Eval checks are rule-based (tool calls, substrings, step counts), not semantic
  grading.
- CI passes on Python 3.12 (GitHub Actions, `ubuntu-latest`), and the suite has
  been run locally on Python 3.15; `requires-python >= 3.10` is declared but
  other versions are untested.

## License

MIT — see LICENSE.

This project was built with AI coding agents for implementation, documentation and a separate review step; the results were checked and tested before publishing.
