"""
backend/chart_tool_schema.py
==============================
Promoted from diagnostics/chart_tool_schema.py + diagnostics/
chart_tool_schema_fewshot.py — this is the production copy
MCPToolDispatcher._run_chart() imports (see mcp_tool_dispatcher.py's
_run_chart() docstring). The diagnostics/ copies are left in place for
reproducibility of the measured numbers (see claude/chart-mcp-tool-scoping.md
in the project) but are not imported from here or from any other production
path — diagnostics/ is read-only live-verification tooling, never a
production dependency (see CLAUDE.md).

CHART_TOOL_SCHEMA / validate_chart_arguments() are unchanged from the
diagnostic version — already measured against real model output. This is
the schema + prompt contract that produced the 66.7% MATCH rate on
chart-expected instructions (22/33) reported in
diagnostics/diag_shadow_chart_toolcall_v4_full.py's last run, with the
few-shot prompt (SYSTEM_PROMPT_FEWSHOT) + bracket-balanced envelope repair
(backend/json_envelope_repair.py) + one temperature-bumped retry on
malformed output.
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Tool schema — deliberately flat: chart_type (enum of 3), title, labels
# (string array), datasets (array of {label, data}). No colors, no options,
# no stacking, no multi-axis. See diagnostics/chart_tool_schema.py's
# original docstring for the full narrow-scope rationale.
# ---------------------------------------------------------------------------

CHART_TOOL_SCHEMA: dict = {
    "name": "generate_chart",
    "description": (
        "Render a chart from structured data. Use this when the user asks "
        "to chart, plot, graph, or visualize data they have provided or "
        "referenced in the conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "enum": ["bar", "line", "pie"],
            },
            "title": {
                "type": "string",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Category labels — x-axis ticks for bar/line, slice names for pie.",
            },
            "datasets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "data": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                    },
                    "required": ["label", "data"],
                },
                "description": (
                    "One or more series. Each dataset's `data` array must be "
                    "the same length as `labels`. Pie charts should use "
                    "exactly one dataset."
                ),
            },
        },
        "required": ["chart_type", "labels", "datasets"],
    },
}

KNOWN_TOOL_NAMES: frozenset[str] = frozenset({CHART_TOOL_SCHEMA["name"]})


def validate_chart_arguments(arguments: dict) -> list[str]:
    """
    Validate `arguments` against CHART_TOOL_SCHEMA's shape and internal
    consistency rules that a JSON-Schema validator alone won't catch
    (labels/data length agreement per dataset).

    Returns a list of human-readable problem strings. Empty list = valid.
    Never raises — a malformed/wrong-typed `arguments` value (e.g. a
    string instead of a dict) is reported as a problem, not an exception.
    """
    problems: list[str] = []

    if not isinstance(arguments, dict):
        return [f"arguments is not an object (got {type(arguments).__name__})"]

    chart_type = arguments.get("chart_type")
    if chart_type not in ("bar", "line", "pie"):
        problems.append(f"chart_type invalid or missing: {chart_type!r}")

    labels = arguments.get("labels")
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        problems.append("labels missing, not an array, or contains non-strings")
        labels = None
    elif len(labels) == 0:
        problems.append("labels is an empty array")

    datasets = arguments.get("datasets")
    if not isinstance(datasets, list) or len(datasets) == 0:
        problems.append("datasets missing, not an array, or empty")
        datasets = []

    for i, ds in enumerate(datasets):
        if not isinstance(ds, dict):
            problems.append(f"datasets[{i}] is not an object")
            continue
        if not isinstance(ds.get("label"), str):
            problems.append(f"datasets[{i}].label missing or not a string")
        data = ds.get("data")
        if not isinstance(data, list) or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in data
        ):
            problems.append(f"datasets[{i}].data missing, not an array, or contains non-numbers")
        elif labels is not None and len(data) != len(labels):
            problems.append(
                f"datasets[{i}].data length ({len(data)}) != labels length ({len(labels)})"
            )

    if chart_type == "pie" and len(datasets) > 1:
        problems.append("pie chart_type should have exactly one dataset, got more than one")

    return problems


# ---------------------------------------------------------------------------
# Few-shot system prompt — ported verbatim from
# diagnostics/chart_tool_schema_fewshot.py. Targets the "prose instead of
# null envelope" failure mode observed in the first diagnostic run: on
# negative-control instructions with nothing concrete to chart, the model
# answered in plain prose instead of emitting {"tool_call": null}. The third
# worked example below is built in that exact shape to show the model
# directly that the null envelope is still required even when an
# explanation would otherwise be the natural response.
# ---------------------------------------------------------------------------

_EXAMPLES = [
    (
        "Chart this: cats 12, dogs 20, birds 5",
        {
            "tool_call": {
                "name": "generate_chart",
                "arguments": {
                    "chart_type": "bar",
                    "title": "Pet Counts",
                    "labels": ["cats", "dogs", "birds"],
                    "datasets": [{"label": "Count", "data": [12, 20, 5]}],
                },
            }
        },
    ),
    (
        "Break this down as a pie chart: rent 50%, food 30%, other 20%",
        {
            "tool_call": {
                "name": "generate_chart",
                "arguments": {
                    "chart_type": "pie",
                    "title": "Budget Breakdown",
                    "labels": ["rent", "food", "other"],
                    "datasets": [{"label": "Share", "data": [50, 30, 20]}],
                },
            }
        },
    ),
    (
        "Explain what a pie chart is.",
        {"tool_call": None},
    ),
]

_EXAMPLES_TEXT = "\n\n".join(
    f"Instruction: {instr}\nResponse: {json.dumps(resp)}"
    for instr, resp in _EXAMPLES
)

SYSTEM_PROMPT_FEWSHOT = f"""You have access to the following tool:

{json.dumps(CHART_TOOL_SCHEMA, indent=2)}

When the user's instruction asks you to chart, plot, graph, or visualize
data, respond with ONLY a single JSON object of the form:
  {{"tool_call": {{"name": "generate_chart", "arguments": {{...}}}}}}

The "arguments" object must exactly match the parameters schema above:
- "chart_type" must be one of "bar", "line", "pie".
- "labels" must be an array of strings.
- "datasets" must be an array of objects, each with "label" (string) and
  "data" (array of numbers) whose length equals the length of "labels".

When no chart is requested — including when you would normally explain
something, ask a clarifying question, or say you need more information —
respond with ONLY:
  {{"tool_call": null}}

Never answer in prose under any circumstances, even when no chart is
warranted. The null envelope IS the correct response in that case, not a
fallback for when you're unsure what else to say.

Examples:

{_EXAMPLES_TEXT}

Rules:
- Output exactly one JSON object and nothing else.
- No prose, no explanation, no markdown code fences.
- Do not invent data the user did not provide or clearly reference."""
