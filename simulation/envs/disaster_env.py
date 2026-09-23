"""PettingZoo ParallelEnv environment for multi-agent disaster response."""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv

from simulation.agents.rescue_agent import RescueAgent
from simulation.engine.cellular_automata import HazardSimulator
from simulation.engine.grid import (
    AGENT,
    DisasterGrid,
    EMPTY,
    EXIT,
    FIRE,
    SMOKE,
    SURVIVOR,
    WALL,
)
from simulation.scenarios.loader import load_scenario


class DisasterEnv(ParallelEnv):
    """
    PettingZoo-compatible multi-agent environment for disaster rescue simulation.
    Agents act simultaneously on a discrete occupancy grid under evolving hazards.
    """

    metadata = {
        "render_modes": ["ansi", "human"],
        "name": "disaster_env_v1",
    }

    def __init__(
        self,
        scenario: Union[str, Dict[str, Any]],
        render_mode: Optional[str] = None,
        reward_config: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the multi-agent disaster environment from a scenario.

        Args:
            scenario: Scenario configuration dictionary or file path.
            render_mode: Optional render mode ('ansi' or 'human').
            reward_config: Optional dictionary overriding default reward weights.
        """
        super().__init__()
        self.raw_scenario = load_scenario(scenario)
        self.render_mode = render_mode

        # Grid configuration
        self.width = self.raw_scenario["width"]
        self.height = self.raw_scenario["height"]
        self.grid = DisasterGrid(width=self.width, height=self.height)

        # Agent configuration
        self.possible_agents = [agent_data["id"] for agent_data in self.raw_scenario["agents"]]
        self.agents = copy.copy(self.possible_agents)

        self._agent_instances: Dict[str, RescueAgent] = {
            agent_data["id"]: RescueAgent(
                agent_id=agent_data["id"],
                initial_position=tuple(agent_data["position"]),
            )
            for agent_data in self.raw_scenario["agents"]
        }

        # Hazard configuration
        hazard_cfg = self.raw_scenario.get("hazard", {})
        self.spread_probability = hazard_cfg.get("spread_probability", 0.30)
        self.max_steps = hazard_cfg.get("max_steps", 100)
        self.hazard_simulator = HazardSimulator(
            grid=self.grid,
            spread_probability=self.spread_probability,
        )

        # Reward weights (Emergency Medicine Golden-Hour inspired)
        self.rewards_cfg = {
            "rescue_reward": 10.0,
            "step_penalty": -0.05,
            "fire_penalty": -5.0,
            "wall_collision_penalty": -0.1,
            "time_decay_factor": 0.005,  # Decays rescue reward over time
        }
        if reward_config:
            self.rewards_cfg.update(reward_config)

        # Action Space: 0: Stay, 1: Up, 2: Down, 3: Left, 4: Right
        self.action_spaces: Dict[str, spaces.Space] = {
            agent: spaces.Discrete(5) for agent in self.possible_agents
        }

        # Local Egocentric Observation Space: 5x5 grid with 4 binary channels
        # Channel 0: Fire, 1: Survivor, 2: Wall/OOB, 3: Exit
        self.obs_shape = (5, 5, 4)
        self.observation_spaces: Dict[str, spaces.Space] = {
            agent: spaces.Box(
                low=0.0,
                high=1.0,
                shape=self.obs_shape,
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        # State tracking
        self.current_step = 0
        self.survivors_rescued_total = 0
        self.casualties_total = 0
        self.initial_survivor_count = len(self.raw_scenario["survivors"])

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """
        Reset the environment to initial scenario configuration.

        Args:
            seed: Random seed for hazard stochasticity.
            options: Optional runtime arguments.

        Returns:
            Tuple of (observations_dict, infos_dict).
        """
        self.grid.reset()
        self.hazard_simulator.set_seed(seed)

        # Place static structures
        for wall in self.raw_scenario["walls"]:
            self.grid.add_wall(tuple(wall))

        for exit_pos in self.raw_scenario["exits"]:
            self.grid.add_exit(tuple(exit_pos))

        # Place dynamic entities
        for survivor in self.raw_scenario["survivors"]:
            self.grid.add_survivor(tuple(survivor))

        for fire in self.raw_scenario["fire"]:
            self.grid.add_fire(tuple(fire))

        # Reset agent states
        self.agents = copy.copy(self.possible_agents)
        for agent_id, agent in self._agent_instances.items():
            agent.reset()

        self.current_step = 0
        self.survivors_rescued_total = 0
        self.casualties_total = 0

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, infos

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """
        Execute simultaneous actions for all active agents.

        Args:
            actions: Dictionary mapping agent IDs to chosen discrete actions.

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos).
        """
        if not self.agents:
            return {}, {}, {}, {}, {}

        self.current_step += 1
        rewards: Dict[str, float] = {agent: self.rewards_cfg["step_penalty"] for agent in self.agents}

        # 1. Agent movement & survivor rescues
        for agent_id, action in actions.items():
            if agent_id not in self.agents:
                continue

            agent = self._agent_instances[agent_id]
            r, c = agent.position
            moves = {
                0: (0, 0),    # Stay
                1: (-1, 0),   # Up
                2: (1, 0),    # Down
                3: (0, -1),   # Left
                4: (0, 1),    # Right
            }
            dr, dc = moves.get(action, (0, 0))
            new_pos = (r + dr, c + dc)

            if self.grid.is_walkable(new_pos):
                agent.move_to(new_pos)
            else:
                # Wall or boundary collision penalty
                rewards[agent_id] += self.rewards_cfg["wall_collision_penalty"]

            # Check if agent rescued a survivor
            if self.grid.grid[agent.position[0], agent.position[1]] == SURVIVOR:
                self.grid.remove_survivor(agent.position)
                agent.record_rescue()
                self.survivors_rescued_total += 1

                # Golden-hour time decaying rescue reward
                time_weight = np.exp(-self.rewards_cfg["time_decay_factor"] * self.current_step)
                rewards[agent_id] += self.rewards_cfg["rescue_reward"] * float(time_weight)

        # 2. Hazard progression (Cellular Automata)
        new_fires = self.hazard_simulator.step()

        # Check if fire consumed survivors (casualties)
        for pos in new_fires:
            # If the cell had a survivor, that survivor is now lost
            # Note: HazardSimulator set cell to FIRE, so any SURVIVOR was overwritten
            pass

        # Calculate current remaining survivors
        current_survivor_cells = self.grid.get_survivor_cells()
        survivors_remaining = len(current_survivor_cells)
        self.casualties_total = self.initial_survivor_count - self.survivors_rescued_total - survivors_remaining

        # 3. Agent hazard collision penalty
        for agent_id in self.agents:
            agent = self._agent_instances[agent_id]
            if self.grid.grid[agent.position[0], agent.position[1]] == FIRE:
                rewards[agent_id] += self.rewards_cfg["fire_penalty"]

        # 4. Episode termination & truncation conditions
        is_terminated = survivors_remaining == 0
        is_truncated = self.current_step >= self.max_steps

        terminations = {agent: is_terminated for agent in self.agents}
        truncations = {agent: is_truncated for agent in self.agents}

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        if is_terminated or is_truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _get_observation(self, agent_id: str) -> np.ndarray:
        """
        Generate 5x5 egocentric local observation for an agent with 4 binary channels:
        Channel 0: Fire
        Channel 1: Survivor
        Channel 2: Wall / Out-of-bounds
        Channel 3: Exit
        """
        observation = np.zeros(self.obs_shape, dtype=np.float32)
        agent = self._agent_instances[agent_id]
        agent_r, agent_c = agent.position

        for i in range(5):
            for j in range(5):
                grid_r = agent_r + i - 2
                grid_c = agent_c + j - 2

                if not self.grid.is_valid((grid_r, grid_c)):
                    # Out of bounds treated as Wall / Obstacle
                    observation[i, j, 2] = 1.0
                    continue

                cell_val = self.grid.grid[grid_r, grid_c]
                if cell_val == FIRE:
                    observation[i, j, 0] = 1.0
                elif cell_val == SURVIVOR:
                    observation[i, j, 1] = 1.0
                elif cell_val == WALL:
                    observation[i, j, 2] = 1.0
                elif cell_val == EXIT:
                    observation[i, j, 3] = 1.0

        return observation

    def _get_info(self, agent_id: str) -> Dict[str, Any]:
        """Compile diagnostic metrics and state info for an agent."""
        agent = self._agent_instances[agent_id]
        survivors_left = len(self.grid.get_survivor_cells())
        return {
            "step": self.current_step,
            "agent_id": agent_id,
            "agent_position": list(agent.position),
            "rescued_by_agent": agent.rescued_count,
            "survivors_remaining": survivors_left,
            "survivors_rescued_total": self.survivors_rescued_total,
            "casualties_total": self.casualties_total,
            "fire_cells_count": len(self.grid.get_fire_cells()),
        }

    def state(self) -> np.ndarray:
        """
        Return centralized global state for Centralized Training (CTDE Critic).
        Shape: (height, width, 5) representing [WALL, FIRE, SURVIVOR, EXIT, AGENTS].
        """
        global_state = np.zeros((self.height, self.width, 5), dtype=np.float32)

        # Channel 0: Wall
        global_state[:, :, 0] = (self.grid.grid == WALL).astype(np.float32)
        # Channel 1: Fire
        global_state[:, :, 1] = (self.grid.grid == FIRE).astype(np.float32)
        # Channel 2: Survivor
        global_state[:, :, 2] = (self.grid.grid == SURVIVOR).astype(np.float32)
        # Channel 3: Exit
        global_state[:, :, 3] = (self.grid.grid == EXIT).astype(np.float32)

        # Channel 4: Agents
        for agent in self._agent_instances.values():
            if self.grid.is_valid(agent.position):
                global_state[agent.position[0], agent.position[1], 4] = 1.0

        return global_state

    def render(self, colored: bool = False) -> Optional[str]:
        """Render the environment in ASCII/colored format."""
        agent_positions = {
            agent_id: agent.position for agent_id, agent in self._agent_instances.items()
        }
        ascii_grid = self.grid.render_ascii(agent_positions=agent_positions, colored=colored)
        if self.render_mode == "human":
            print(f"\n--- Step {self.current_step} ---")
            print(ascii_grid)
            return None
        return ascii_grid

    def close(self) -> None:
        """Clean up environment resources."""
        pass
