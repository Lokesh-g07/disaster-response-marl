"""
Unit tests for rule-based baseline policies.

Coverage:
  - manhattan() helper
  - best_step_toward() helper including obstacle avoidance
  - GreedyNearestPolicy: target selection, action generation, obstacle handling,
    no-survivor situations, multi-agent situations
  - GreedyLargestZonePolicy: zone building, zone centre computation, target
    selection, action generation, tie-breaking, no-survivor situations
  - BaselinePolicy callable interface
  - Integration with Evaluator (5-episode smoke test per baseline)
"""

from __future__ import annotations

import pytest
import numpy as np

from baselines.rule_based import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    ACTION_UP,
    GreedyLargestZonePolicy,
    GreedyNearestPolicy,
    best_step_toward,
    manhattan,
)
from simulation.envs.disaster_env import DisasterEnv


# ─────────────────────────────────────────────────────────────────────────────
# Minimal scenario builder
# ─────────────────────────────────────────────────────────────────────────────

def make_env(
    width: int = 7,
    height: int = 7,
    agent_positions=None,
    survivor_positions=None,
    wall_positions=None,
    fire_positions=None,
    max_steps: int = 50,
) -> DisasterEnv:
    """Build a minimal DisasterEnv for testing."""
    if agent_positions is None:
        agent_positions = [{"id": "rescue_0", "position": [0, 0]}]
    if survivor_positions is None:
        survivor_positions = []
    if wall_positions is None:
        wall_positions = []
    if fire_positions is None:
        fire_positions = []

    scenario = {
        "name": "test",
        "width": width,
        "height": height,
        "fire": fire_positions,
        "exits": [[0, width - 1]],
        "survivors": survivor_positions,
        "agents": agent_positions,
        "walls": wall_positions,
        "hazard": {"spread_probability": 0.0, "max_steps": max_steps},
    }
    env = DisasterEnv(scenario)
    env.reset(seed=0)
    return env


# ─────────────────────────────────────────────────────────────────────────────
# 1. Helper: manhattan()
# ─────────────────────────────────────────────────────────────────────────────

class TestManhattan:
    def test_same_cell(self):
        assert manhattan((3, 3), (3, 3)) == 0

    def test_adjacent_row(self):
        assert manhattan((0, 0), (1, 0)) == 1

    def test_adjacent_col(self):
        assert manhattan((0, 0), (0, 1)) == 1

    def test_diagonal(self):
        # (0,0) -> (3,4): |3|+|4| = 7
        assert manhattan((0, 0), (3, 4)) == 7

    def test_symmetric(self):
        assert manhattan((2, 5), (6, 1)) == manhattan((6, 1), (2, 5))

    def test_negative_delta(self):
        assert manhattan((5, 5), (2, 3)) == 5


# ─────────────────────────────────────────────────────────────────────────────
# 2. Helper: best_step_toward()
# ─────────────────────────────────────────────────────────────────────────────

class TestBestStepToward:
    def test_move_right(self):
        env = make_env(survivor_positions=[[0, 5]])
        # agent at (0,0), target at (0,5) -> should move right
        action = best_step_toward((0, 0), (0, 5), env)
        assert action == ACTION_RIGHT

    def test_move_down(self):
        env = make_env(survivor_positions=[[5, 0]])
        action = best_step_toward((0, 0), (5, 0), env)
        assert action == ACTION_DOWN

    def test_move_up(self):
        env = make_env(survivor_positions=[[0, 0]])
        action = best_step_toward((4, 0), (0, 0), env)
        assert action == ACTION_UP

    def test_move_left(self):
        env = make_env(survivor_positions=[[0, 0]])
        action = best_step_toward((0, 4), (0, 0), env)
        assert action == ACTION_LEFT

    def test_stay_when_at_target(self):
        env = make_env()
        action = best_step_toward((2, 2), (2, 2), env)
        assert action == ACTION_STAY

    def test_obstacle_avoidance_wall_blocks_direct_path(self):
        """Wall directly below forces a detour; must not produce ACTION_DOWN."""
        # Agent at (0,0), target at (2,0), wall at (1,0)
        env = make_env(
            survivor_positions=[[2, 0]],
            wall_positions=[[1, 0]],
        )
        action = best_step_toward((0, 0), (2, 0), env)
        # Direct path (down) is blocked; should go right or stay
        assert action != ACTION_DOWN

    def test_stay_when_all_neighbours_blocked(self):
        """All 4 neighbours are walls -> fallback to STAY."""
        env = make_env(
            width=5, height=5,
            survivor_positions=[[4, 4]],
            wall_positions=[[0, 1], [1, 0], [1, 2], [2, 1]],
        )
        action = best_step_toward((1, 1), (4, 4), env)
        # (0,1), (2,1), (1,0), (1,2) are all walls -> fallback
        assert action == ACTION_STAY

    def test_out_of_bounds_not_chosen(self):
        """Agent at top-left corner; UP and LEFT would be out-of-bounds."""
        env = make_env(survivor_positions=[[0, 6]])
        action = best_step_toward((0, 0), (0, 6), env)
        # UP (-1,0) is OOB; LEFT (0,-1) is OOB; should go RIGHT
        assert action == ACTION_RIGHT


# ─────────────────────────────────────────────────────────────────────────────
# 3. GreedyNearestPolicy
# ─────────────────────────────────────────────────────────────────────────────

class TestGreedyNearestPolicy:
    def test_name(self):
        assert GreedyNearestPolicy().name == "greedy_nearest"

    def test_callable_interface(self):
        """Policy should be callable as policy(obs_dict, env) -> dict."""
        env = make_env(survivor_positions=[[3, 3]])
        obs, _ = env.reset(seed=0)
        policy = GreedyNearestPolicy()
        actions = policy(obs, env)
        assert isinstance(actions, dict)
        assert "rescue_0" in actions

    def test_no_survivors_returns_stay(self):
        """When no survivors remain, all agents should STAY."""
        env = make_env(survivor_positions=[])
        obs, _ = env.reset(seed=0)
        policy = GreedyNearestPolicy()
        actions = policy(obs, env)
        for a in actions.values():
            assert a == ACTION_STAY

    def test_target_is_nearest_survivor(self):
        """Agent at (0,0): survivor A at (0,2), survivor B at (5,5). Should pick A."""
        env = make_env(
            survivor_positions=[[0, 2], [5, 5]],
            agent_positions=[{"id": "rescue_0", "position": [0, 0]}],
        )
        obs, _ = env.reset(seed=0)
        policy = GreedyNearestPolicy()

        # Target (0,2) is closer; first step should be RIGHT (toward col 2)
        action = policy(obs, env)["rescue_0"]
        assert action == ACTION_RIGHT

    def test_moves_toward_single_survivor(self):
        """Agent at (0,0) with survivor at (0,3): should move right repeatedly."""
        env = make_env(
            agent_positions=[{"id": "rescue_0", "position": [0, 0]}],
            survivor_positions=[[0, 3]],
        )
        obs, _ = env.reset(seed=0)
        policy = GreedyNearestPolicy()
        actions = policy(obs, env)
        assert actions["rescue_0"] == ACTION_RIGHT

    def test_tie_breaking_deterministic(self):
        """Two equidistant survivors: tie should be broken by smallest (row, col)."""
        # Agent at (2,2); survivor at (0,2) dist=2, survivor at (2,4) dist=2
        env = make_env(
            agent_positions=[{"id": "rescue_0", "position": [2, 2]}],
            survivor_positions=[[0, 2], [2, 4]],
        )
        obs, _ = env.reset(seed=0)
        policy = GreedyNearestPolicy()
        # (0,2) < (2,4) lexicographically -> (0,2) wins -> should move UP
        actions = policy(obs, env)
        assert actions["rescue_0"] == ACTION_UP

    def test_multi_agent_independent(self):
        """Two agents independently pick the nearest survivor each."""
        # rescue_0 at (0,0): nearest is (0,3)
        # rescue_1 at (6,6): nearest is (6,3)
        env = make_env(
            width=7, height=7,
            agent_positions=[
                {"id": "rescue_0", "position": [0, 0]},
                {"id": "rescue_1", "position": [6, 6]},
            ],
            survivor_positions=[[0, 3], [6, 3]],
        )
        obs, _ = env.reset(seed=0)
        policy = GreedyNearestPolicy()
        actions = policy(obs, env)
        # rescue_0 should move right (toward (0,3))
        assert actions["rescue_0"] == ACTION_RIGHT
        # rescue_1 should move left (toward (6,3))
        assert actions["rescue_1"] == ACTION_LEFT

    def test_wall_obstacle_does_not_crash(self):
        """Walls in the path should not raise any exception."""
        env = make_env(
            survivor_positions=[[4, 4]],
            wall_positions=[[2, 0], [2, 1], [2, 2], [2, 3], [2, 4]],
        )
        obs, _ = env.reset(seed=0)
        policy = GreedyNearestPolicy()
        # Should not raise
        actions = policy(obs, env)
        assert isinstance(actions, dict)

    def test_actions_in_valid_range(self):
        """All returned actions must be in [0, 4]."""
        env = make_env(survivor_positions=[[3, 3], [5, 1]])
        obs, _ = env.reset(seed=0)
        policy = GreedyNearestPolicy()
        actions = policy(obs, env)
        for a in actions.values():
            assert 0 <= a <= 4, f"Invalid action {a}"

    def test_integration_with_evaluator(self):
        """Full 5-episode evaluation smoke test via the Evaluator."""
        from evaluation import Evaluator
        ev = Evaluator("simulation/scenarios/examples/fire_small.json")
        policy = GreedyNearestPolicy()
        summary = ev.evaluate(policy, num_episodes=5, base_seed=0)
        assert summary.num_episodes == 5
        for ep in summary.episode_results:
            assert ep.episode_length > 0
            assert 0 <= ep.survivors_rescued <= ep.survivors_initial
            assert ep.survivors_rescued + ep.survivors_remaining + ep.casualties == ep.survivors_initial


# ─────────────────────────────────────────────────────────────────────────────
# 4. GreedyLargestZonePolicy
# ─────────────────────────────────────────────────────────────────────────────

class TestGreedyLargestZonePolicy:
    def test_name_contains_zone_size(self):
        assert "3" in GreedyLargestZonePolicy(zone_size=3).name

    def test_invalid_zone_size_raises(self):
        with pytest.raises(ValueError):
            GreedyLargestZonePolicy(zone_size=0)

    def test_callable_interface(self):
        env = make_env(survivor_positions=[[3, 3]])
        obs, _ = env.reset(seed=0)
        policy = GreedyLargestZonePolicy()
        actions = policy(obs, env)
        assert isinstance(actions, dict)
        assert "rescue_0" in actions

    def test_no_survivors_returns_stay(self):
        env = make_env(survivor_positions=[])
        obs, _ = env.reset(seed=0)
        policy = GreedyLargestZonePolicy()
        actions = policy(obs, env)
        for a in actions.values():
            assert a == ACTION_STAY

    def test_zone_building_correct(self):
        """Zone map should group cells into correct 3x3 buckets."""
        policy = GreedyLargestZonePolicy(zone_size=3)
        survivor_cells = np.array([[0, 0], [0, 1], [0, 2], [3, 0], [3, 1]])
        zone_map = policy._build_zone_map(survivor_cells)
        assert (0, 0) in zone_map
        assert len(zone_map[(0, 0)]) == 3   # (0,0), (0,1), (0,2) all in zone (0,0)
        assert (1, 0) in zone_map
        assert len(zone_map[(1, 0)]) == 2   # (3,0), (3,1) in zone (1,0)

    def test_zone_centre_clamped_to_bounds(self):
        """Zone centre must never exceed grid boundaries."""
        env = make_env(width=5, height=5)
        policy = GreedyLargestZonePolicy(zone_size=3)
        # Zone (1, 1) centre = (3+1, 3+1) = (4, 4) -> clamped to (4, 4)
        centre = policy._zone_centre((1, 1), env)
        assert 0 <= centre[0] <= 4
        assert 0 <= centre[1] <= 4

    def test_picks_zone_with_most_survivors(self):
        """Zone (0,0) has 3 survivors, zone (1,1) has 1. Should target (0,0)."""
        env = make_env(
            width=9, height=9,
            agent_positions=[{"id": "rescue_0", "position": [5, 5]}],
            survivor_positions=[[0, 0], [0, 1], [1, 0], [6, 6]],
        )
        obs, _ = env.reset(seed=0)
        policy = GreedyLargestZonePolicy(zone_size=3)
        # Zone (0,0): cells (0,0),(0,1),(1,0) -> 3 survivors
        # Zone (2,2): cell (6,6) -> 1 survivor
        # centre of zone (0,0) = (1,1) -> agent at (5,5) should go UP or LEFT
        actions = policy(obs, env)
        assert actions["rescue_0"] in (ACTION_UP, ACTION_LEFT)

    def test_tie_broken_by_closest_zone(self):
        """Two zones with equal survivors; agent picks the closer one."""
        # Agent at (0,0).  Zone A centre ~(1,1), Zone B centre ~(7,7)
        # Both have 1 survivor. Closer zone is A.
        env = make_env(
            width=10, height=10,
            agent_positions=[{"id": "rescue_0", "position": [0, 0]}],
            survivor_positions=[[1, 1], [7, 7]],
        )
        obs, _ = env.reset(seed=0)
        policy = GreedyLargestZonePolicy(zone_size=3)
        # Both zones have 1 survivor; (0,0) zone centre at (1,1) is closer
        # -> agent should move toward (1,1) -> ACTION_DOWN or ACTION_RIGHT
        actions = policy(obs, env)
        assert actions["rescue_0"] in (ACTION_DOWN, ACTION_RIGHT)

    def test_zone_size_one_behaves_like_greedy_nearest(self):
        """zone_size=1 means each cell is its own zone -> largest is the lone survivor."""
        env = make_env(
            agent_positions=[{"id": "rescue_0", "position": [0, 0]}],
            survivor_positions=[[0, 4]],
        )
        obs, _ = env.reset(seed=0)
        policy = GreedyLargestZonePolicy(zone_size=1)
        actions = policy(obs, env)
        assert actions["rescue_0"] == ACTION_RIGHT

    def test_multi_agent_can_pick_different_zones(self):
        """Each agent independently evaluates which zone is closest when tied."""
        env = make_env(
            width=10, height=10,
            agent_positions=[
                {"id": "rescue_0", "position": [0, 0]},
                {"id": "rescue_1", "position": [9, 9]},
            ],
            survivor_positions=[[1, 1], [8, 8]],
        )
        obs, _ = env.reset(seed=0)
        policy = GreedyLargestZonePolicy(zone_size=3)
        actions = policy(obs, env)
        # rescue_0 closer to zone(0,0): should head DOWN or RIGHT
        assert actions["rescue_0"] in (ACTION_DOWN, ACTION_RIGHT)
        # rescue_1 closer to zone(2,2): should head UP or LEFT
        assert actions["rescue_1"] in (ACTION_UP, ACTION_LEFT)

    def test_wall_does_not_crash(self):
        env = make_env(
            survivor_positions=[[4, 4]],
            wall_positions=[[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]],
        )
        obs, _ = env.reset(seed=0)
        policy = GreedyLargestZonePolicy()
        actions = policy(obs, env)
        assert isinstance(actions, dict)

    def test_actions_in_valid_range(self):
        env = make_env(survivor_positions=[[2, 2], [5, 5]])
        obs, _ = env.reset(seed=0)
        policy = GreedyLargestZonePolicy()
        actions = policy(obs, env)
        for a in actions.values():
            assert 0 <= a <= 4

    def test_integration_with_evaluator(self):
        from evaluation import Evaluator
        ev = Evaluator("simulation/scenarios/examples/fire_small.json")
        policy = GreedyLargestZonePolicy()
        summary = ev.evaluate(policy, num_episodes=5, base_seed=0)
        assert summary.num_episodes == 5
        for ep in summary.episode_results:
            assert ep.episode_length > 0
            assert ep.survivors_rescued + ep.survivors_remaining + ep.casualties == ep.survivors_initial


# ─────────────────────────────────────────────────────────────────────────────
# 5. BaselinePolicy ABC enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselinePolicyABC:
    def test_cannot_instantiate_abstract(self):
        from baselines.rule_based import BaselinePolicy
        with pytest.raises(TypeError):
            BaselinePolicy()  # type: ignore

    def test_concrete_subclass_registers(self):
        from baselines.rule_based import BaselinePolicy
        assert issubclass(GreedyNearestPolicy, BaselinePolicy)
        assert issubclass(GreedyLargestZonePolicy, BaselinePolicy)
