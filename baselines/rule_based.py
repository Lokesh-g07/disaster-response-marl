"""
Rule-based multi-agent baseline policies for CrisisRL.

Each baseline is a plain Python callable matching the PolicyFn interface:
    policy(obs_dict: Dict[str, np.ndarray], env: DisasterEnv) -> Dict[str, int]

This means every baseline can be passed directly to Evaluator.evaluate()
and compared against MAPPO on exactly the same episodes, scenarios, and metrics.

=====================================================================
PRIVILEGED STATE WARNING
=====================================================================
All rule-based baselines in this file access env.grid.get_survivor_cells()
and env._agent_instances directly, giving them the full ground-truth positions
of every remaining survivor on the grid.

MAPPO does NOT have this information at inference time: it only observes a
local 5x5x4 egocentric window.

Therefore the comparison between rule-based baselines and MAPPO is
INFORMATION-ASYMMETRIC:
  - Rule-based baseline : privileged-state heuristic (oracle-style)
  - MAPPO               : learned decentralized policy, local observations only

This asymmetry is intentional -- these baselines serve as upper-bound
reference points for what is achievable with global knowledge, not as
direct apples-to-apples competitors to MAPPO.
Any evaluation report MUST note this distinction.
=====================================================================

Task / Zone Allocation Architecture
-------------------------------------
Both baselines now implement an explicit two-layer decision pipeline:

    Agent --> assigned task/zone --> survivor target --> movement --> rescue
                                                                        |
                                                              task released on arrival

Layer 1: Task Allocation
    TaskAllocator  (used by GreedyNearestPolicy)
        assign_task(agent_id, survivor_cells) -> AgentTask | None
        release_task(agent_id)
        Maintains a Dict[agent_id -> AgentTask] so each agent has exactly
        one assigned survivor target at a time.  The assignment is refreshed
        each step if the target was rescued/burned.

    ZoneAllocator  (used by GreedyLargestZonePolicy)
        assign_zone(agent_id, zone_map, env) -> (AgentZoneTask, target) | (None, None)
        release_zone(agent_id)
        Maintains a Dict[agent_id -> AgentZoneTask] so each agent is
        assigned to a zone.  Within the zone the nearest surviving cell
        is used as the concrete movement target.  When the zone is empty
        the assignment is refreshed.

Layer 2: Movement
    best_step_toward(agent_pos, target_pos, env) -> int
        Unchanged from the previous implementation.  Picks the walkable
        4-connected step that minimises Manhattan distance to target.

Both baselines share the BaselinePolicy ABC so future baselines
(A*, BFS, DQN offline rollout) can plug into the Evaluator framework
without touching the evaluation code.
"""

from __future__ import annotations

import abc
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulation.envs.disaster_env import DisasterEnv


# ---------------------------------------------------------------------------
# Action constants - mirror the environment definition exactly
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
# Task / Zone data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentTask:
    """
    Represents a single survivor-rescue task assigned to one agent.

    An AgentTask is created when an agent is assigned a survivor target and
    released when the survivor is rescued (no longer present on the grid) or
    burned by hazard.

    Fields
    ------
    agent_id    : The agent that owns this task.
    target      : (row, col) of the assigned survivor cell.
    assigned_at : The env step at which this task was assigned (informational).
    """
    agent_id: str
    target: Tuple[int, int]
    assigned_at: int = 0


@dataclass
class AgentZoneTask:
    """
    Represents a zone-level assignment for one agent.

    An AgentZoneTask is created when an agent is assigned to a zone (identified
    by its zone_key) and released when the zone contains no remaining survivors.

    Fields
    ------
    agent_id    : The agent that owns this zone assignment.
    zone_key    : (zone_row, zone_col) identifier of the assigned zone.
    zone_centre : Pre-computed centre (row, col) of the zone for navigation.
    assigned_at : The env step at which this task was assigned.
    """
    agent_id: str
    zone_key: Tuple[int, int]
    zone_centre: Tuple[int, int]
    assigned_at: int = 0


# ---------------------------------------------------------------------------
# Layer 1a: TaskAllocator (for GreedyNearestPolicy)
# ---------------------------------------------------------------------------

class TaskAllocator:
    """
    Manages per-agent survivor task assignments for GreedyNearestPolicy.

    Allocation pipeline (called once per step per agent):
      1. If the agent has an existing assignment, verify the target is still a
         survivor on the grid. Release the task if the cell was rescued/burned.
      2. If the agent has no assignment (new or just released), select the
         nearest available survivor using Manhattan distance and assign it.
      3. Return the assigned target, or None if no survivors remain.

    NOTE: This allocator does NOT enforce exclusive task-to-agent assignments.
    Multiple agents may be assigned the same survivor simultaneously. This is
    intentional for a greedy heuristic -- the first to arrive rescues it, and
    the other's assignment is then released on the next step.
    """

    def __init__(self) -> None:
        # Maps agent_id -> current AgentTask (or None)
        self._assignments: Dict[str, Optional[AgentTask]] = {}

    def reset(self) -> None:
        """Clear all assignments (call at the start of each episode)."""
        self._assignments.clear()

    def _is_task_valid(self, task: AgentTask, survivor_set: set) -> bool:
        """Return True if the assigned target is still a live survivor."""
        return task.target in survivor_set

    def assign_task(
        self,
        agent_id: str,
        agent_pos: Tuple[int, int],
        survivor_cells: np.ndarray,
        current_step: int = 0,
    ) -> Optional[AgentTask]:
        """
        Ensure the agent has a valid task assignment and return it.

        Steps:
        1. Build a fast lookup set of current survivor positions.
        2. If existing assignment is still valid, return it unchanged.
        3. Otherwise release the old assignment and select a new nearest target.
        4. Return None if no survivors remain.

        Args:
            agent_id: The requesting agent.
            agent_pos: Current (row, col) of the agent.
            survivor_cells: Live survivor positions from get_survivor_cells().
            current_step: Current env timestep (stored for diagnostics).

        Returns:
            The active AgentTask, or None.
        """
        survivor_set = {(int(c[0]), int(c[1])) for c in survivor_cells}

        existing = self._assignments.get(agent_id)
        if existing is not None and self._is_task_valid(existing, survivor_set):
            return existing  # Still valid -- keep assignment

        # Release stale task
        self._assignments[agent_id] = None

        if not survivor_set:
            return None

        # Select nearest survivor (Manhattan); tie-break by smallest (row, col)
        best_target: Optional[Tuple[int, int]] = None
        best_dist = math.inf
        for pos in survivor_set:
            d = manhattan(agent_pos, pos)
            if d < best_dist or (d == best_dist and best_target is not None and pos < best_target):
                best_dist = d
                best_target = pos

        if best_target is None:
            return None

        task = AgentTask(agent_id=agent_id, target=best_target, assigned_at=current_step)
        self._assignments[agent_id] = task
        return task

    def release_task(self, agent_id: str) -> None:
        """Explicitly release the assignment for an agent."""
        self._assignments[agent_id] = None

    @property
    def assignments(self) -> Dict[str, Optional[AgentTask]]:
        """Read-only view of current assignments."""
        return dict(self._assignments)


# ---------------------------------------------------------------------------
# Layer 1b: ZoneAllocator (for GreedyLargestZonePolicy)
# ---------------------------------------------------------------------------

class ZoneAllocator:
    """
    Manages per-agent zone assignments for GreedyLargestZonePolicy.

    Allocation pipeline (called once per step per agent):
      1. If the agent has an existing zone assignment, check whether survivors
         still exist in that zone. Release if the zone is empty.
      2. If the agent has no zone assignment, select the zone with the most
         remaining survivors; break ties by distance to zone centre.
      3. Within the assigned zone, pick the nearest survivor as the concrete
         movement target.
      4. Return (zone_task, concrete_target) or (None, None).

    Args:
        zone_size: Square zone partition size (default 3).
    """

    def __init__(self, zone_size: int = 3) -> None:
        if zone_size < 1:
            raise ValueError(f"zone_size must be >= 1, got {zone_size}")
        self.zone_size = zone_size
        self._assignments: Dict[str, Optional[AgentZoneTask]] = {}

    def reset(self) -> None:
        """Clear all assignments (call at the start of each episode)."""
        self._assignments.clear()

    def _build_zone_map(
        self,
        survivor_cells: np.ndarray,
    ) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Partition survivor cells into zones keyed by (zone_row, zone_col).

        PRIVILEGED STATE: uses ground-truth survivor positions from the grid.
        """
        zone_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        for cell in survivor_cells:
            r, c = int(cell[0]), int(cell[1])
            zone_map[(r // self.zone_size, c // self.zone_size)].append((r, c))
        return zone_map

    def _zone_centre(
        self,
        zone_key: Tuple[int, int],
        env: DisasterEnv,
    ) -> Tuple[int, int]:
        """Return the centre cell of a zone, clamped to grid bounds."""
        zr, zc = zone_key
        cr = min(zr * self.zone_size + self.zone_size // 2, env.height - 1)
        cc = min(zc * self.zone_size + self.zone_size // 2, env.width - 1)
        return (cr, cc)

    def _select_best_zone(
        self,
        agent_pos: Tuple[int, int],
        zone_map: Dict[Tuple[int, int], List[Tuple[int, int]]],
        env: DisasterEnv,
    ) -> Optional[Tuple[int, int]]:
        """Select the zone key with the most survivors; break ties by closest centre."""
        if not zone_map:
            return None
        max_count = max(len(v) for v in zone_map.values())
        candidates = [zk for zk, sv in zone_map.items() if len(sv) == max_count]
        return min(
            candidates,
            key=lambda zk: (manhattan(agent_pos, self._zone_centre(zk, env)), zk),
        )

    def assign_zone(
        self,
        agent_id: str,
        agent_pos: Tuple[int, int],
        survivor_cells: np.ndarray,
        env: DisasterEnv,
        current_step: int = 0,
    ) -> Tuple[Optional[AgentZoneTask], Optional[Tuple[int, int]]]:
        """
        Ensure the agent has a valid zone assignment; return task + concrete target.

        Returns:
            (AgentZoneTask, concrete_target_pos) or (None, None).
        """
        zone_map = self._build_zone_map(survivor_cells)

        existing = self._assignments.get(agent_id)
        if existing is not None:
            if existing.zone_key in zone_map:
                # Zone still has survivors -- keep assignment
                target = self._nearest_in_zone(agent_pos, zone_map[existing.zone_key])
                return existing, target
            # Zone exhausted -- release
            self._assignments[agent_id] = None

        best_zone_key = self._select_best_zone(agent_pos, zone_map, env)
        if best_zone_key is None:
            return None, None

        centre = self._zone_centre(best_zone_key, env)
        zone_task = AgentZoneTask(
            agent_id=agent_id,
            zone_key=best_zone_key,
            zone_centre=centre,
            assigned_at=current_step,
        )
        self._assignments[agent_id] = zone_task
        target = self._nearest_in_zone(agent_pos, zone_map[best_zone_key])
        return zone_task, target

    def release_zone(self, agent_id: str) -> None:
        """Explicitly release zone assignment for an agent."""
        self._assignments[agent_id] = None

    def _nearest_in_zone(
        self,
        agent_pos: Tuple[int, int],
        zone_survivors: List[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """Select the nearest survivor within the zone by Manhattan distance."""
        if not zone_survivors:
            return None
        return min(
            zone_survivors,
            key=lambda pos: (manhattan(agent_pos, pos), pos),
        )

    @property
    def assignments(self) -> Dict[str, Optional[AgentZoneTask]]:
        """Read-only view of current zone assignments."""
        return dict(self._assignments)


# ---------------------------------------------------------------------------
# Abstract BaselinePolicy interface
# ---------------------------------------------------------------------------

class BaselinePolicy(abc.ABC):
    """
    Abstract base class for all rule-based baseline policies.

    Subclasses implement `select_actions` and are callable, producing
    the same Dict[str, int] signature as PolicyFn in evaluator.py.
    """

    @abc.abstractmethod
    def select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        env: DisasterEnv,
    ) -> Dict[str, int]:
        """
        Compute an action for each active agent.

        Args:
            obs_dict: Current observations (agent_id -> np.ndarray (5,5,4)).
            env: Live DisasterEnv. PRIVILEGED ACCESS for rule-based subclasses.

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

    def reset(self) -> None:
        """
        Optional hook called at episode start.
        Subclasses with stateful allocators should override this.
        """


# ---------------------------------------------------------------------------
# Baseline 1: Greedy-Nearest  (with explicit TaskAllocator)
# ---------------------------------------------------------------------------

class GreedyNearestPolicy(BaselinePolicy):
    """
    Each agent greedily moves toward the *closest remaining survivor*.

    PRIVILEGED STATE: reads env.grid.get_survivor_cells() for exact survivor
    coordinates.  MAPPO cannot do this at inference time.

    Two-layer decision pipeline
    ---------------------------
    Layer 1 - Task allocation (TaskAllocator):
        1. Select an available survivor as the task target using Manhattan distance.
           The task is assigned and persists across steps until the survivor is
           rescued or burned.
        2. If the target disappears (rescued by another agent or burned by hazard),
           the task is released and a new one is selected immediately.

    Layer 2 - Movement (best_step_toward):
        3. Move one step toward the assigned survivor using the 4-connected
           greedy step heuristic that respects walkability and grid bounds.
        4. When the agent arrives and the env removes the survivor, the task
           is released at the start of the next step.

    Distance metric
    ---------------
    Manhattan distance: |delta_row| + |delta_col|
    Consistent with the 4-connected action space; Euclidean is not used
    because diagonal movement is unavailable.

    Multi-agent coordination
    ------------------------
    Two agents may hold overlapping task assignments (same survivor target).
    This is intentional: the first to arrive rescues, the other's task is
    released on the next step.  No explicit de-confliction is applied.
    """

    def __init__(self) -> None:
        self._allocator = TaskAllocator()

    @property
    def name(self) -> str:
        return "greedy_nearest"

    def reset(self) -> None:
        """Release all task assignments (call at episode start if reusing policy)."""
        self._allocator.reset()

    def select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        env: DisasterEnv,
    ) -> Dict[str, int]:
        """
        For each active agent:
          1. Get/refresh task assignment via TaskAllocator.
          2. Move toward the assigned survivor target.
        """
        # PRIVILEGED STATE ACCESS -- see module docstring
        survivor_cells = env.grid.get_survivor_cells()
        actions: Dict[str, int] = {}

        for agent_id in env.agents:
            agent_inst = env._agent_instances[agent_id]
            agent_pos = agent_inst.position

            task = self._allocator.assign_task(
                agent_id=agent_id,
                agent_pos=agent_pos,
                survivor_cells=survivor_cells,
                current_step=env.current_step,
            )

            if task is None:
                actions[agent_id] = ACTION_STAY
            else:
                actions[agent_id] = best_step_toward(agent_pos, task.target, env)

        return actions

    @property
    def task_assignments(self) -> Dict[str, Optional[AgentTask]]:
        """Expose current task assignments (used in tests and diagnostics)."""
        return self._allocator.assignments


# ---------------------------------------------------------------------------
# Baseline 2: Greedy-Largest Zone  (with explicit ZoneAllocator)
# ---------------------------------------------------------------------------

class GreedyLargestZonePolicy(BaselinePolicy):
    """
    Agents are assigned to the grid zone with the most remaining survivors,
    then move toward the nearest survivor within that zone.

    PRIVILEGED STATE: reads env.grid.get_survivor_cells() for exact survivor
    coordinates.  MAPPO cannot do this at inference time.

    Zone definition
    ---------------
    The grid is partitioned into non-overlapping zone_size x zone_size tiles.
    Boundary zones are smaller when grid dimensions are not divisible by zone_size.

    Two-layer decision pipeline
    ---------------------------
    Layer 1 - Zone allocation (ZoneAllocator):
        1. Build a zone map: group all remaining survivor cells by their zone.
        2. Assign the agent to the zone with the most survivors.
           Ties in count are broken by distance to zone centre.
        3. The zone assignment persists across steps until the zone is empty.
        4. When the zone is exhausted, immediately re-allocate to the next
           best zone.

    Layer 2 - Movement (best_step_toward):
        5. Within the assigned zone, select the nearest surviving cell as the
           concrete movement target.
        6. Move one step toward that target using the greedy walkability heuristic.

    Args:
        zone_size: Size of each square zone partition (default 3).
    """

    def __init__(self, zone_size: int = 3) -> None:
        self._allocator = ZoneAllocator(zone_size=zone_size)
        self.zone_size = zone_size

    @property
    def name(self) -> str:
        return f"greedy_largest_zone_{self.zone_size}"

    def reset(self) -> None:
        """Release all zone assignments (call at episode start if reusing policy)."""
        self._allocator.reset()

    def select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        env: DisasterEnv,
    ) -> Dict[str, int]:
        """
        For each active agent:
          1. Get/refresh zone assignment via ZoneAllocator.
          2. Within the zone, pick the nearest survivor as concrete target.
          3. Move toward that target.
        """
        # PRIVILEGED STATE ACCESS -- see module docstring
        survivor_cells = env.grid.get_survivor_cells()
        actions: Dict[str, int] = {}

        for agent_id in env.agents:
            agent_inst = env._agent_instances[agent_id]
            agent_pos = agent_inst.position

            zone_task, concrete_target = self._allocator.assign_zone(
                agent_id=agent_id,
                agent_pos=agent_pos,
                survivor_cells=survivor_cells,
                env=env,
                current_step=env.current_step,
            )

            if concrete_target is None:
                actions[agent_id] = ACTION_STAY
            else:
                actions[agent_id] = best_step_toward(agent_pos, concrete_target, env)

        return actions

    @property
    def zone_assignments(self) -> Dict[str, Optional[AgentZoneTask]]:
        """Expose current zone assignments (used in tests and diagnostics)."""
        return self._allocator.assignments

    # ---- Kept for backward compatibility with existing tests ----
    def _build_zone_map(
        self,
        survivor_cells: np.ndarray,
    ) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Delegate to ZoneAllocator (preserves existing test compatibility)."""
        return self._allocator._build_zone_map(survivor_cells)

    def _zone_centre(
        self,
        zone_key: Tuple[int, int],
        env: DisasterEnv,
    ) -> Tuple[int, int]:
        """Delegate to ZoneAllocator (preserves existing test compatibility)."""
        return self._allocator._zone_centre(zone_key, env)
