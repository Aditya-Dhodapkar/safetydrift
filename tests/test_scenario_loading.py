"""Tests for scenario loading and validation."""

import pytest

from safetydrift.scenarios.loader import (
    CATEGORIES,
    load_all_scenarios,
    load_scenarios_by_category,
    validate_all_scenarios,
)

SCENARIOS_DIR = "data/scenarios"


class TestLoadAllScenarios:
    def test_total_count(self):
        scenarios = load_all_scenarios(SCENARIOS_DIR)
        assert len(scenarios) == 40

    def test_all_have_required_fields(self):
        scenarios = load_all_scenarios(SCENARIOS_DIR)
        for s in scenarios:
            assert s.id, f"Missing id in scenario"
            assert s.category in CATEGORIES, f"{s.id}: invalid category '{s.category}'"
            assert s.name, f"{s.id}: missing name"
            assert s.task_prompt, f"{s.id}: missing task_prompt"
            assert len(s.tools) >= 1, f"{s.id}: needs at least 1 tool"
            assert len(s.drift_opportunities) >= 1, f"{s.id}: needs at least 1 drift opportunity"

    def test_unique_ids(self):
        scenarios = load_all_scenarios(SCENARIOS_DIR)
        ids = [s.id for s in scenarios]
        assert len(set(ids)) == len(ids), f"Duplicate IDs found: {[x for x in ids if ids.count(x) > 1]}"


class TestLoadByCategory:
    @pytest.mark.parametrize("category", CATEGORIES)
    def test_category_count(self, category):
        scenarios = load_scenarios_by_category(SCENARIOS_DIR, category)
        assert len(scenarios) == 10, f"{category}: expected 10 scenarios, got {len(scenarios)}"

    @pytest.mark.parametrize("category", CATEGORIES)
    def test_category_matches_dir(self, category):
        scenarios = load_scenarios_by_category(SCENARIOS_DIR, category)
        for s in scenarios:
            assert s.category == category, f"{s.id}: category '{s.category}' != dir '{category}'"


class TestValidation:
    def test_validate_all(self):
        result = validate_all_scenarios(SCENARIOS_DIR)
        total = sum(len(ids) for ids in result.values())
        assert total == 40
        for cat in CATEGORIES:
            assert len(result[cat]) == 10
