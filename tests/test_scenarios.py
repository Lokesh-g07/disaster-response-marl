"""Unit tests for scenario loading and validation."""

import pytest
from pathlib import Path
from simulation.scenarios.loader import load_scenario, validate_scenario


def test_load_valid_example_scenarios():
    """Verify that all default scenario example files load and validate properly."""
    examples_dir = Path("simulation/scenarios/examples")
    scenario_files = list(examples_dir.glob("*.json"))
    assert len(scenario_files) >= 2, "Expected at least 2 example scenario files"

    for path in scenario_files:
        scenario = load_scenario(str(path))
        assert "name" in scenario
        assert scenario["width"] > 0
        assert scenario["height"] > 0
        assert len(scenario["agents"]) > 0
        assert len(scenario["exits"]) > 0


def test_missing_required_fields():
    """Verify ValueError is raised when required keys are missing."""
    incomplete_scenario = {
        "name": "bad_scenario",
        "width": 10,
        "height": 10,
        # missing fire, exits, survivors, agents, walls
    }
    with pytest.raises(ValueError, match="Missing required scenario field"):
        validate_scenario(incomplete_scenario)


def test_invalid_dimensions():
    """Verify dimensions must be positive integers."""
    bad_dim_scenario = {
        "name": "bad_dim",
        "width": -5,
        "height": 10,
        "fire": [],
        "exits": [[0, 0]],
        "survivors": [],
        "agents": [{"id": "rescue_0", "position": [1, 1]}],
        "walls": [],
    }
    with pytest.raises(ValueError, match="Grid dimensions must be positive integers"):
        validate_scenario(bad_dim_scenario)


def test_out_of_bounds_coordinates():
    """Verify coordinates outside the grid bounds are rejected."""
    oob_scenario = {
        "name": "oob_test",
        "width": 10,
        "height": 10,
        "fire": [[15, 15]],  # Out of bounds
        "exits": [[0, 0]],
        "survivors": [],
        "agents": [{"id": "rescue_0", "position": [1, 1]}],
        "walls": [],
    }
    with pytest.raises(ValueError, match="out of bounds"):
        validate_scenario(oob_scenario)


def test_duplicate_agent_ids():
    """Verify duplicate agent IDs are rejected."""
    dup_agent_scenario = {
        "name": "dup_agent",
        "width": 10,
        "height": 10,
        "fire": [],
        "exits": [[0, 0]],
        "survivors": [],
        "agents": [
            {"id": "rescue_0", "position": [1, 1]},
            {"id": "rescue_0", "position": [2, 2]},
        ],
        "walls": [],
    }
    with pytest.raises(ValueError, match="Duplicate agent ID detected"):
        validate_scenario(dup_agent_scenario)


def test_nonexistent_file_raises_error():
    """Verify FileNotFoundError is raised for non-existent paths."""
    with pytest.raises(FileNotFoundError):
        load_scenario("non_existent_file_path.json")
