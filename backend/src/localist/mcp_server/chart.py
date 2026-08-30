"""
Localist MCP Server — generate_chart tool implementation
===========================================================
Renders a chart (bar/line/pie) from structured data as a server-side PNG via
matplotlib, sandboxed under the same project_root convention as file_ops.py.

Argument validation is ported verbatim from
diagnostics/chart_tool_schema.py's validate_chart_arguments() — already
measured against real model output during the diagnostic phase (see
claude/chart-mcp-tool-scoping.md in the project). Kept as a local copy
rather than an import across the diagnostics/production boundary: this
codebase's diagnostics/ scripts are read-only live-verification tooling,
never a dependency of production code (see CLAUDE.md) — same reasoning
file_ops.py gives for duplicating _MAX_FILE_READ_CHARS instead of
importing it from the backend's agent stack.

PNGs are saved under project_root/charts/<uuid>.png, reusing
file_ops._sandbox_resolve() for the path-escape guard rather than
re-implementing a second sandbox check.
"""

from __future__ import annotations

import uuid

from . import file_ops

# Categorical palette (fixed order — see dataviz skill's color-formula.md /
# palette.md). Assigned by slot position, never cycled or re-derived per chart.
_CATEGORICAL_PALETTE: list[str] = [
    "#2a78d6",  # 1 blue
    "#008300",  # 2 green
    "#e87ba4",  # 3 magenta
    "#eda100",  # 4 yellow
    "#1baf7a",  # 5 aqua
    "#eb6834",  # 6 orange
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

_CHART_SURFACE   = "#fcfcfb"
_PRIMARY_INK     = "#0b0b0b"
_SECONDARY_INK   = "#52514e"
_MUTED_INK       = "#898781"
_GRIDLINE        = "#e1e0d9"
_BASELINE        = "#c3c2b7"


def validate_chart_arguments(arguments: dict) -> list[str]:
    """
    Validate `arguments` against the generate_chart schema's shape and
    internal consistency rules (labels/data length agreement per dataset,
    pie-chart single-dataset constraint). Returns a list of human-readable
    problem strings; empty list = valid. Never raises.

    Ported verbatim from diagnostics/chart_tool_schema.py.
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


def _style_axes(ax) -> None:
    ax.set_facecolor(_CHART_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_BASELINE)
    ax.tick_params(colors=_MUTED_INK, labelcolor=_SECONDARY_INK)
    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)


def _render_bar(ax, labels: list[str], datasets: list[dict]) -> None:
    n_datasets = len(datasets)
    n_labels = len(labels)
    x = range(n_labels)
    group_width = 0.8
    bar_width = group_width / n_datasets
    for i, ds in enumerate(datasets):
        color = _CATEGORICAL_PALETTE[i % len(_CATEGORICAL_PALETTE)]
        offsets = [xi - group_width / 2 + bar_width * i + bar_width / 2 for xi in x]
        ax.bar(offsets, ds["data"], width=bar_width, color=color, label=ds["label"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, color=_SECONDARY_INK)


def _render_line(ax, labels: list[str], datasets: list[dict]) -> None:
    x = range(len(labels))
    for i, ds in enumerate(datasets):
        color = _CATEGORICAL_PALETTE[i % len(_CATEGORICAL_PALETTE)]
        ax.plot(x, ds["data"], color=color, linewidth=2, marker="o", markersize=5, label=ds["label"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, color=_SECONDARY_INK)


def _render_pie(ax, labels: list[str], datasets: list[dict]) -> None:
    data = datasets[0]["data"]
    colors = [_CATEGORICAL_PALETTE[i % len(_CATEGORICAL_PALETTE)] for i in range(len(labels))]
    ax.pie(
        data,
        labels=labels,
        colors=colors,
        textprops={"color": _SECONDARY_INK},
        wedgeprops={"linewidth": 2, "edgecolor": _CHART_SURFACE},
    )
    ax.set_facecolor(_CHART_SURFACE)


def generate_chart(chart_type: str, labels: list[str], datasets: list[dict], title: str = "") -> dict:
    """
    Render a chart from structured data and save it as a PNG.

    Raises ValueError("ERROR: ...") on any validation failure — same
    convention as file_ops.py / url_fetch.py / web_search.py, so
    MCPToolDispatcher's error-normalization/stripping continues to work
    unchanged.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError(
            "ERROR: matplotlib is not installed — install the 'chart' "
            "extra (pip install localist[chart]) to use generate_chart."
        ) from exc

    arguments = {"chart_type": chart_type, "labels": labels, "datasets": datasets}
    problems = validate_chart_arguments(arguments)
    if problems:
        raise ValueError("ERROR: " + "; ".join(problems))

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=_CHART_SURFACE)
    if chart_type == "bar":
        _render_bar(ax, labels, datasets)
    elif chart_type == "line":
        _render_line(ax, labels, datasets)
    else:  # pie — validated above to have exactly one dataset
        _render_pie(ax, labels, datasets)

    if chart_type != "pie":
        _style_axes(ax)
        if len(datasets) > 1:
            ax.legend(frameon=False, labelcolor=_SECONDARY_INK)

    if title:
        ax.set_title(title, color=_PRIMARY_INK)

    fig.tight_layout()

    rel_path = f"charts/{uuid.uuid4().hex}.png"
    resolved = file_ops._sandbox_resolve(rel_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved, facecolor=_CHART_SURFACE)
    plt.close(fig)

    if title:
        summary = f"Generated {chart_type} chart: {title}"
    else:
        preview_labels = ", ".join(labels[:3])
        if len(labels) > 3:
            preview_labels += ", …"
        summary = f"Generated {chart_type} chart: {preview_labels}"

    # summary is the only field safe to put in the model-facing prompt
    # (Slot 5's 500-token ceiling) — png_path/chart_config must never be
    # appended to prompt-facing text by callers of this tool.
    return {
        "summary": summary,
        "png_path": rel_path,
        "chart_config": {
            "chart_type": chart_type,
            "title": title,
            "labels": labels,
            "datasets": datasets,
        },
    }
