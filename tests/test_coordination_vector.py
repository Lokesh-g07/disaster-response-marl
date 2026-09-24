"""Unit tests for the MAPPO coordination vector generation in DisasterEnv."""

import numpy as np
import pytest
from simulation.envs.disaster_env import DisasterEnv
from simulation.engine.grid import SURVIVOR, EMPTY


@pytest.fixture
def base_scenario():
    return {
        "name": "test",
        "width": 5,
        "height": 5,
        "fire": [],
        "exits": [[0, 4]],
        "survivors": [[1, 2], [3, 2]],
        "agents": [
            {"id": "rescue_0", "position": [1, 1]},
            {"id": "rescue_1", "position": [3, 3]}
        ],
        "walls": [],
        "hazard": {"spread_probability": 0.0, "max_steps": 20},
    }


def test_coordination_vector_shape_and_range(base_scenario):
    """Test vector dimensionality and value bounds."""
    env = DisasterEnv(base_scenario, max_agents=2)
    obs, _ = env.reset(seed=42)

    vec = obs["rescue_0"]["vector"]
    assert vec.shape == (7,)  # 2 + 2*(2-1) + 3
    assert np.all(vec >= -1.0)
    assert np.all(vec <= 1.0)


def test_own_position_normalization(base_scenario):
    """Test agent's own position is correctly normalized to [0, 1]."""
    env = DisasterEnv(base_scenario, max_agents=2)
    env.reset()
    vec = env._get_coordination_vector("rescue_0")

    # Agent 0 is at (1, 1). Grid is 5x5. Normalization is by max(H-1, 1) = 4.
    assert np.isclose(vec[0], 1.0 / 4.0)
    assert np.isclose(vec[1], 1.0 / 4.0)


def test_teammate_relative_position(base_scenario):
    """Test teammate dx, dy is calculated and normalized properly."""
    env = DisasterEnv(base_scenario, max_agents=2)
    env.reset()

    # Agent 0 at (1,1), Agent 1 at (3,3)
    vec0 = env._get_coordination_vector("rescue_0")
    # Relative r: (3 - 1)/5 = 2/5 = 0.4
    # Relative c: (3 - 1)/5 = 2/5 = 0.4
    assert np.isclose(vec0[2], 0.4)
    assert np.isclose(vec0[3], 0.4)

    vec1 = env._get_coordination_vector("rescue_1")
    # Relative r: (1 - 3)/5 = -0.4
    # Relative c: (1 - 3)/5 = -0.4
    assert np.isclose(vec1[2], -0.4)
    assert np.isclose(vec1[3], -0.4)


def test_nearest_survivor_encoding(base_scenario):
    """Test nearest survivor relative position and distance."""
    env = DisasterEnv(base_scenario, max_agents=2)
    env.reset()

    # Agent 0 at (1,1). Survivors at (1,2) and (3,2).
    # Nearest to agent 0 is (1,2). dist = 1.
    vec0 = env._get_coordination_vector("rescue_0")
    dr = (1 - 1) / 5.0
    dc = (2 - 1) / 5.0
    dist = 1.0 / 10.0  # H+W = 10
    
    assert np.isclose(vec0[-3], dr)
    assert np.isclose(vec0[-2], dc)
    assert np.isclose(vec0[-1], dist)


def test_no_survivors_case(base_scenario):
    """Test target encoding when there are no survivors left."""
    scenario = base_scenario.copy()
    scenario["survivors"] = []
    
    env = DisasterEnv(scenario, max_agents=2)
    env.reset()
    
    vec = env._get_coordination_vector("rescue_0")
    # Last 3 elements should be 0.0
    assert np.isclose(vec[-3], 0.0)
    assert np.isclose(vec[-2], 0.0)
    assert np.isclose(vec[-1], 0.0)


def test_variable_agent_padding(base_scenario):
    """Test that max_agents config correctly pads with zeros."""
    env = DisasterEnv(base_scenario, max_agents=4)
    env.reset()
    
    vec = env._get_coordination_vector("rescue_0")
    # 2 + 2*(4-1) + 3 = 11
    assert vec.shape == (11,)
    
    # Teammate 1 is present at [2:4]
    assert not np.isclose(vec[2], 0.0)
    
    # Teammates 2 and 3 are missing, should be padded with 0
    assert np.isclose(vec[4], 0.0)
    assert np.isclose(vec[5], 0.0)
    assert np.isclose(vec[6], 0.0)
    assert np.isclose(vec[7], 0.0)

