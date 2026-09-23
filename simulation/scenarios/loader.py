"""Scenario loader and validation utilities for CrisisRL."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


def load_scenario(source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Load and validate a disaster scenario from a JSON file path or dictionary.

    Args:
        source: File path string, Path object, or already-loaded dictionary.

    Returns:
        Validated scenario dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required fields are missing or configuration is invalid.
    """
    if isinstance(source, dict):
        scenario = json.loads(json.dumps(source))  # deep copy
    else:
        scenario_path = Path(source)
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

        with open(scenario_path, "r", encoding="utf-8") as file:
            try:
                scenario = json.load(file)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON format in scenario: {exc}") from exc

    validate_scenario(scenario)
    return scenario


def validate_scenario(scenario: Dict[str, Any]) -> None:
    """
    Validate the structure and integrity of a scenario dictionary.

    Args:
        scenario: Dictionary containing scenario definition.

    Raises:
        ValueError: If validation fails.
    """
    required_fields = [
        "name",
        "width",
        "height",
        "fire",
        "exits",
        "survivors",
        "agents",
        "walls",
    ]

    for field in required_fields:
        if field not in scenario:
            raise ValueError(f"Missing required scenario field: {field}")

    width = scenario["width"]
    height = scenario["height"]

    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("Grid dimensions must be positive integers.")

    if not isinstance(scenario["agents"], list) or len(scenario["agents"]) == 0:
        raise ValueError("At least one rescue agent is required.")

    if not isinstance(scenario["exits"], list) or len(scenario["exits"]) == 0:
        raise ValueError("At least one exit is required.")

    # Validate coordinate bounds
    def _check_coord(coord: Any, name: str) -> Tuple[int, int]:
        if not isinstance(coord, (list, tuple)) or len(coord) != 2:
            raise ValueError(f"Coordinate in '{name}' must be a 2-element [row, col] pair: {coord}")
        r, c = coord
        if not isinstance(r, int) or not isinstance(c, int):
            raise ValueError(f"Coordinates in '{name}' must be integers: {coord}")
        if not (0 <= r < height and 0 <= c < width):
            raise ValueError(
                f"Coordinate {coord} in '{name}' is out of bounds for grid {height}x{width}."
            )
        return (r, c)

    for item in scenario["fire"]:
        _check_coord(item, "fire")

    for item in scenario["exits"]:
        _check_coord(item, "exits")

    for item in scenario["survivors"]:
        _check_coord(item, "survivors")

    for item in scenario["walls"]:
        _check_coord(item, "walls")

    agent_ids = set()
    for idx, agent in enumerate(scenario["agents"]):
        if not isinstance(agent, dict):
            raise ValueError(f"Agent #{idx} must be a dictionary.")
        if "id" not in agent or not isinstance(agent["id"], str) or not agent["id"].strip():
            raise ValueError(f"Agent #{idx} missing a valid string 'id'.")
        if agent["id"] in agent_ids:
            raise ValueError(f"Duplicate agent ID detected: {agent['id']}")
        agent_ids.add(agent["id"])

        if "position" not in agent:
            raise ValueError(f"Agent '{agent['id']}' missing 'position'.")
        _check_coord(agent["position"], f"agent '{agent['id']}'")

    # Set and validate hazard metadata defaults if missing
    if "hazard" not in scenario or not isinstance(scenario["hazard"], dict):
        scenario["hazard"] = {}

    hazard = scenario["hazard"]
    hazard.setdefault("type", "fire")
    hazard.setdefault("spread_probability", 0.30)
    hazard.setdefault("max_steps", 100)

    if not (0.0 <= hazard["spread_probability"] <= 1.0):
        raise ValueError("Hazard spread_probability must be between 0.0 and 1.0.")

    if not isinstance(hazard["max_steps"], int) or hazard["max_steps"] <= 0:
        raise ValueError("Hazard max_steps must be a positive integer.")
