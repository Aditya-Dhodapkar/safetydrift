"""Tests for trace I/O (save/load)."""

import pytest

from safetydrift.traces.io import (
    load_trace,
    load_traces,
    load_traces_by_category,
    save_trace,
    save_traces,
)
from safetydrift.traces.models import Trace


class TestSingleTraceIO:
    def test_save_and_load(self, safe_trace: Trace, tmp_path):
        path = tmp_path / "trace.json"
        save_trace(safe_trace, path)
        loaded = load_trace(path)
        assert loaded.metadata.scenario_id == safe_trace.metadata.scenario_id
        assert len(loaded.steps) == len(safe_trace.steps)

    def test_creates_parent_dirs(self, safe_trace: Trace, tmp_path):
        path = tmp_path / "nested" / "dir" / "trace.json"
        save_trace(safe_trace, path)
        assert path.exists()


class TestBulkTraceIO:
    def test_save_and_load_multiple(self, safe_trace: Trace, violating_trace: Trace, tmp_path):
        save_traces([safe_trace, violating_trace], tmp_path)
        loaded = load_traces(tmp_path)
        assert len(loaded) == 2

    def test_load_by_category(self, safe_trace: Trace, violating_trace: Trace, tmp_path):
        save_traces([safe_trace, violating_trace], tmp_path)
        dh = load_traces_by_category(tmp_path, "data_handling")
        assert len(dh) == 2
        other = load_traces_by_category(tmp_path, "sysadmin")
        assert len(other) == 0

    def test_empty_directory(self, tmp_path):
        loaded = load_traces(tmp_path)
        assert loaded == []
