"""Unit tests for DisasterEnv PettingZoo environment."""

import numpy as np
import pytest
from simulation.envs.disaster_env import DisasterEnv
from simulation.scenarios.loader import load_scenario


@pytest.fixture
def sample_scenario():
    return load_scenario("simulation/scenarios/examples/fire_small.json")


def test_environment_initialization(sample_scenario):
    """Verify initialization and properties of DisasterEnv."""
    env = DisasterEnv(sample_scenario)
    assert len(env.possible_agents) == 2
    assert "rescue_0" in env.possible_agents
    assert "rescue_1" in env.possible_agents

    for agent in env.possible_agents:
        obs_space = env.observation_space(agent)
        act_space = env.action_space(agent)
        assert obs_space.shape == (5, 5, 4)
        assert act_space.n == 5


def test_environment_reset(sample_scenario):
    """Verify reset returns valid observations and infos."""
    env = DisasterEnv(sample_scenario)
    observations, infos = env.reset(seed=42)

    assert len(observations) == 2
    assert len(infos) == 2

    for agent in env.agents:
        assert observations[agent].shape == (5, 5, 4)
        assert env.observation_space(agent).contains(observations[agent])
        assert "survivors_remaining" in infos[agent]
        assert infos[agent]["survivors_remaining"] == 5
        assert infos[agent]["step"] == 0


def test_environment_step_random_actions(sample_scenario):
    """Verify stepping with random valid actions."""
    env = DisasterEnv(sample_scenario)
    obs, infos = env.reset(seed=42)

    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    next_obs, rewards, terminations, truncations, next_infos = env.step(actions)

    assert len(next_obs) == 2
    assert len(rewards) == 2
    assert len(terminations) == 2
    assert len(truncations) == 2
    assert len(next_infos) == 2


def test_survivor_rescue_reward():
    """Verify agent receives positive reward and rescues survivor when stepping onto survivor cell."""
    scenario = {
        "name": "rescue_test",
        "width": 5,
        "height": 5,
        "fire": [],
        "exits": [[0, 4]],
        "survivors": [[1, 2]],  # Survivor directly to the right of agent
        "agents": [{"id": "rescue_0", "position": [1, 1]}],
        "walls": [],
        "hazard": {"spread_probability": 0.0, "max_steps": 10},
    }
    env = DisasterEnv(scenario)
    obs, infos = env.reset()

    # Move Right (action 4) to step onto (1, 2)
    actions = {"rescue_0": 4}
    next_obs, rewards, terminations, truncations, next_infos = env.step(actions)

    # Should receive positive reward (> 5.0)
    assert rewards["rescue_0"] > 5.0
    assert next_infos["rescue_0"]["survivors_rescued_total"] == 1
    assert next_infos["rescue_0"]["survivors_remaining"] == 0
    assert terminations["rescue_0"] is True  # All survivors rescued


def test_hazard_penalty():
    """Verify agent receives fire penalty when on a fire cell."""
    scenario = {
        "name": "fire_penalty_test",
        "width": 5,
        "height": 5,
        "fire": [[1, 2]],  # Fire directly to the right of agent
        "exits": [[0, 4]],
        "survivors": [[4, 4]],
        "agents": [{"id": "rescue_0", "position": [1, 1]}],
        "walls": [],
        "hazard": {"spread_probability": 0.0, "max_steps": 10},
    }
    env = DisasterEnv(scenario)
    obs, infos = env.reset()

    # Move Right (action 4) into the fire cell (1, 2)
    actions = {"rescue_0": 4}
    next_obs, rewards, terminations, truncations, next_infos = env.step(actions)

    # Should receive negative fire penalty (-5.0 + step penalty)
    assert rewards["rescue_0"] <= -5.0


def test_centralized_global_state(sample_scenario):
    """Verify state() method produces full CTDE global tensor."""
    env = DisasterEnv(sample_scenario)
    env.reset()
    global_state = env.state()

    assert isinstance(global_state, np.ndarray)
    assert global_state.shape == (10, 10, 5)
    # Check that wall channel has at least one wall marked
    assert np.sum(global_state[:, :, 0]) == len(sample_scenario["walls"])
    # Check that fire channel has initial fires
    assert np.sum(global_state[:, :, 1]) == len(sample_scenario["fire"])


def test_ascii_render(sample_scenario):
    """Verify ASCII rendering outputs expected string."""
    env = DisasterEnv(sample_scenario, render_mode="ansi")
    env.reset()
    rendering = env.render()
    assert isinstance(rendering, str)
    assert "A" in rendering  # Agent representation
    assert "F" in rendering  # Fire representation
    assert "S" in rendering  # Survivor representation
