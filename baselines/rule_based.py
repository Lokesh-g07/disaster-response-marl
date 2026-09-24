"""
Rule-based multi-agent baseline policies for CrisisRL.

Each baseline is a plain Python callable matching the PolicyFn interface:
    policy(obs_dict: Dict[str, np.ndarray], env: DisasterEnv) -> Dict[str, int]

This means every baseline can be passed directly to Evaluator.evaluate()
and compared against MAPPO on exactly the same episodes, scenarios, and metrics.

The two baselines implemented here are:

  GreedyNearestPolicy
  --------------------
  Each agent independently selects the *closest available survivor* (minimum
  Manhattan distance from the agent's current cell) and takes one step toward
  it.  When multiple survivors are equidistant the one with the smallest
  (row, col) index is chosen for determinism.

  Distance metric: Manhattan distance  |Δrow| + |Δcol|.
  This is consistent with the 4-connected movement model already used by the
  environment (actions: stay/up/down/left/right).  Euclidean distance is NOT
  used because diagonal movement is not available, making Manhattan the correct
  lower-bound estimate.

  GreedyLargestZonePolicy
  ------------------------
  Survivors are clustered into 3×3 grid zones (configurable).  The zone
  containing the *most remaining survivors* is chosen as the target zone.
  Each agent then moves toward the centre cell of that zone.  Ties in zone
  population are broken by choosing the zone closest (Manhattan) to the agent.

Both baselines share a common BaselinePolicy ABC so future rule-based or
algorithmic baselines (PPO-rollout, DQN-rollout, A* pathfinding) can easily
plug into the same Evaluator framework.
"""

from __future__ import annotations

import abc
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulation.envs.disaster_env import DisasterEnv


# ---------------------------------------------------------------------------
# Action constants – mirror the environment definition exactly
# ---------------------------------------------------------------------------
ACTION_STAY  = 0
ACTION_UP    = 1  # row - 1
ACTION_DOWN  = 2  # row + 1
ACTION_LEFT  = 3  # col - 1
ACTION_RIGHT = 4  # col + 1

_DELTA_TO_ACTION: Dict[Tuple[int, int], int] = {
    (0,  0): ACTION_STAY,
    (-1, 0): ACTION_UP,
    (1,  0): ACTION_DOWN,
    (0, -1): ACTION_LEFT,
    (0,  1): ACTION_RIGHT,
}

_CANDIDATE_MOVES: List[Tuple[int, int]] = [
    (-1, 0),   # up
    (1,  0),   # down
    (0, -1),   # left
    (0,  1),   # right
    (0,  0),   # stay (fallback)
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance between two (row, col) grid cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def best_step_toward(
    agent_pos: Tuple[int, int],
    target_pos: Tuple[int, int],
    env: DisasterEnv,
) -> int:
    """
    Return the action integer that moves the agent one step closer to target.

    Strategy: among the four cardinal neighbours, pick the walkable cell that
    minimises Manhattan distance to target.  If no cardinal move is walkable,
    fall back to ACTION_STAY.

    Args:
        agent_pos: Current (row, col) of the agent.
        target_pos: Desired destination (row, col).
        env: Live environment providing grid walkability checks.

    Returns:
        Integer action in [0, 4].
    """
    best_action = ACTION_STAY
    best_dist = math.inf

    for dr, dc in _CANDIDATE_MOVES:
        candidate = (agent_pos[0] + dr, agent_pos[1] + dc)
        if not env.grid.is_walkable(candidate):
            continue
        d = manhattan(candidate, target_pos)
        if d < best_dist:
            best_dist = d
            best_action = _DELTA_TO_ACTION[(dr, dc)]

    return best_action


# ---------------------------------------------------------------------------
# Abstract BaselinePolicy interface
# ---------------------------------------------------------------------------

class BaselinePolicy(abc.ABC):
    """
    Abstract base class for all rule-based baseline policies.

    Subclasses implement `select_actions` and are callable, producing
    the same Dict[str, int] signature as PolicyFn in evaluator.py.

    Keeping this ABC ensures future baselines (A*, BFS, DQN offline rollout)
    maintain a consistent interface for the Evaluator.
    """

    @abc.abstractmethod
    def select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        env: DisasterEnv,
    ) -> Dict[str, int]:
        """
        Compute an action for each active agent given current observations and env state.

        Args:
            obs_dict: Current observations (agent_id -> np.ndarray of shape (5,5,4)).
            env: The live DisasterEnv instance (used for grid/agent state access).

        Returns:
            Dict mapping agent_id -> discrete action integer.
        """

    def __call__(
        self,
        obs_dict: Dict[str, np.ndarray],
        env: DisasterEnv,
    ) -> Dict[str, int]:
        """Make the policy directly callable as a PolicyFn."""
        return self.select_actions(obs_dict, env)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable policy identifier."""


# ---------------------------------------------------------------------------
# Baseline 1: Greedy-Nearest
# ---------------------------------------------------------------------------

class GreedyNearestPolicy(BaselinePolicy):
    """
    Each agent greedily moves toward the *closest remaining survivor*.

    Distance metric
    ---------------
    Manhattan distance: |Δrow| + |Δcol|

    This is the natural metric for 4-connected discrete grid movement.
    Euclidean distance is deliberately avoided because the agents cannot
    move diagonally, so Manhattan gives the true lower-bound path length
    in obstacle-free terrain.

    Target selection
    ----------------
    1. Retrieve all remaining survivor cells from env.grid.get_survivor_cells().
    2. For each agent, compute Manhattan distance from agent position to every survivor.
    3. Select the survivor with the minimum distance.
    4. Ties are broken by smallest (row, col) tuple for full determinism.

    No survivor
    -----------
    If no survivors remain (all rescued or burned), the agent stays in place.

    Multi-agent coordination
    ------------------------
    Agents act independently; there is NO explicit de-confliction (two agents
    may head toward the same survivor).  This is intentional — the baseline is
    purely greedy and serves as a simple rule-based lower-bound.
    """

    @property
    def name(self) -> str:
        return "greedy_nearest"

    def _select_target(
        self,
        agent_pos: Tuple[int, int],
        survivor_cells: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """
        Select the nearest survivor cell using Manhattan distance.

        Args:
            agent_pos: The (row, col) position of the querying agent.
            survivor_cells: Array of shape (N, 2) from get_survivor_cells().

        Returns:
            (row, col) of nearest survivor, or None if no survivors remain.
        """
        if len(survivor_cells) == 0:
            return None

        best_target: Optional[Tuple[int, int]] = None
        best_dist = math.inf

        for cell in survivor_cells:
            pos = (int(cell[0]), int(cell[1]))
            d = manhattan(agent_pos, pos)
            # Tie-break: prefer smaller (row, col) for determinism
            if d < best_dist or (d == best_dist and pos < best_target):  # type: ignore[operator]
                best_dist = d
                best_target = pos

        return best_target

    def select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        env: DisasterEnv,
    ) -> Dict[str, int]:
        survivor_cells = env.grid.get_survivor_cells()
        actions: Dict[str, int] = {}

        for agent_id in env.agents:
            agent_inst = env._agent_instances[agent_id]
            agent_pos = agent_inst.position

            target = self._select_target(agent_pos, survivor_cells)

            if target is None:
                actions[agent_id] = ACTION_STAY
            else:
                actions[agent_id] = best_step_toward(agent_pos, target, env)

        return actions


# ---------------------------------------------------------------------------
# Baseline 2: Greedy-Largest Zone
# ---------------------------------------------------------------------------

class GreedyLargestZonePolicy(BaselinePolicy):
    """
    Agents move toward the centre of the grid zone with the most survivors.

    Zone definition
    ---------------
    The grid is partitioned into non-overlapping rectangular zones of size
    `zone_size × zone_size` (default 3×3).  Boundary zones may be smaller
    if the grid dimensions are not divisible by zone_size.

    Target selection
    ----------------
    1. Assign each surviving cell to its zone (floor(row/zone_size), floor(col/zone_size)).
    2. Count survivors per zone.
    3. Select the zone with the maximum survivor count.
    4. Ties in count are broken by the zone whose centre is closest (Manhattan)
       to the requesting agent, ensuring each agent may select a different zone.
    5. The target position is the centre cell of the winning zone, clamped to
       valid grid bounds.

    No survivor
    -----------
    If no survivors remain, the agent stays in place.

    Args:
        zone_size: Size of each square zone partition (default 3).
    """

    def __init__(self, zone_size: int = 3) -> None:
        if zone_size < 1:
            raise ValueError(f"zone_size must be >= 1, got {zone_size}")
        self.zone_size = zone_size

    @property
    def name(self) -> str:
        return f"greedy_largest_zone_{self.zone_size}"

    def _build_zone_map(
        self,
        survivor_cells: np.ndarray,
    ) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Partition survivor cells into zones.

        Returns:
            Dict mapping zone_key (zone_row, zone_col) -> list of survivor (row, col).
        """
        zone_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        for cell in survivor_cells:
            r, c = int(cell[0]), int(cell[1])
            zone_key = (r // self.zone_size, c // self.zone_size)
            zone_map[zone_key].append((r, c))
        return zone_map

    def _zone_centre(
        self,
        zone_key: Tuple[int, int],
        env: DisasterEnv,
    ) -> Tuple[int, int]:
        """
        Compute the centre cell of a zone, clamped to valid grid coordinates.

        Args:
            zone_key: (zone_row, zone_col) integer indices.
            env: Used to clamp to (height-1, width-1).

        Returns:
            (row, col) of the zone centre cell.
        """
        zr, zc = zone_key
        centre_r = zr * self.zone_size + self.zone_size // 2
        centre_c = zc * self.zone_size + self.zone_size // 2
        centre_r = min(centre_r, env.height - 1)
        centre_c = min(centre_c, env.width - 1)
        return (centre_r, centre_c)

    def _select_target_zone(
        self,
        agent_pos: Tuple[int, int],
        zone_map: Dict[Tuple[int, int], List[Tuple[int, int]]],
        env: DisasterEnv,
    ) -> Optional[Tuple[int, int]]:
        """
        Select the zone with most survivors; break ties by closest to agent.

        Returns:
            Centre (row, col) of the selected zone, or None if no zones.
        """
        if not zone_map:
            return None

        max_count = max(len(v) for v in zone_map.values())
        # Candidate zones: all zones tied for maximum survivor count
        candidates = [zk for zk, survivors in zone_map.items()
                      if len(survivors) == max_count]

        # Among candidates, pick the one whose centre is closest to this agent
        best_zone = min(
            candidates,
            key=lambda zk: (manhattan(agent_pos, self._zone_centre(zk, env)), zk),
        )
        return self._zone_centre(best_zone, env)

    def select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        env: DisasterEnv,
    ) -> Dict[str, int]:
        survivor_cells = env.grid.get_survivor_cells()
        zone_map = self._build_zone_map(survivor_cells)
        actions: Dict[str, int] = {}

        for agent_id in env.agents:
            agent_inst = env._agent_instances[agent_id]
            agent_pos = agent_inst.position

            target = self._select_target_zone(agent_pos, zone_map, env)

            if target is None:
                actions[agent_id] = ACTION_STAY
            else:
                actions[agent_id] = best_step_toward(agent_pos, target, env)

        return actions
