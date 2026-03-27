"""Scenario loading and validation from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from safetydrift.scenarios.schema import ScenarioConfig

CATEGORIES = ("data_handling", "sysadmin", "research_comms", "code_debugging")


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load a single scenario from a YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return ScenarioConfig.model_validate(data)


def load_all_scenarios(base_dir: str | Path) -> list[ScenarioConfig]:
    """Load all scenarios from the base directory (expects category subdirs)."""
    base_dir = Path(base_dir)
    scenarios = []
    for category in CATEGORIES:
        cat_dir = base_dir / category
        if cat_dir.is_dir():
            scenarios.extend(load_scenarios_by_category(base_dir, category))
    return scenarios


def load_scenarios_by_category(base_dir: str | Path, category: str) -> list[ScenarioConfig]:
    """Load all scenarios in a given category subdirectory."""
    cat_dir = Path(base_dir) / category
    scenarios = []
    for path in sorted(cat_dir.glob("*.yaml")):
        scenarios.append(load_scenario(path))
    return scenarios


def validate_all_scenarios(base_dir: str | Path) -> dict[str, list[str]]:
    """Validate all scenarios. Returns dict of category -> list of scenario IDs.

    Raises on first invalid scenario.
    """
    base_dir = Path(base_dir)
    result: dict[str, list[str]] = {}
    for category in CATEGORIES:
        cat_dir = base_dir / category
        if not cat_dir.is_dir():
            result[category] = []
            continue
        ids = []
        for path in sorted(cat_dir.glob("*.yaml")):
            scenario = load_scenario(path)
            assert scenario.category == category, (
                f"{path}: category '{scenario.category}' doesn't match dir '{category}'"
            )
            ids.append(scenario.id)
        result[category] = ids
    return result
