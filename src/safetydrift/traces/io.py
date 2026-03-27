"""Trace I/O: save and load traces as JSON files."""

from __future__ import annotations

from pathlib import Path

from safetydrift.traces.models import Trace


def save_trace(trace: Trace, path: str | Path) -> None:
    """Save a single trace to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(trace.model_dump_json(indent=2))


def load_trace(path: str | Path) -> Trace:
    """Load a single trace from a JSON file."""
    path = Path(path)
    return Trace.model_validate_json(path.read_text())


def save_traces(traces: list[Trace], directory: str | Path) -> list[Path]:
    """Save multiple traces to a directory, one file per trace.

    Files are named {scenario_id}_{run_id}.json.
    Returns the list of paths written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for trace in traces:
        filename = f"{trace.metadata.scenario_id}_{trace.metadata.run_id}.json"
        path = directory / filename
        save_trace(trace, path)
        paths.append(path)
    return paths


def load_traces(directory: str | Path) -> list[Trace]:
    """Load all traces from a directory."""
    directory = Path(directory)
    traces = []
    for path in sorted(directory.glob("*.json")):
        traces.append(load_trace(path))
    return traces


def load_traces_by_category(directory: str | Path, category: str) -> list[Trace]:
    """Load traces filtered by scenario category."""
    return [t for t in load_traces(directory) if t.metadata.scenario_category == category]
