"""Baselines package for CrisisRL."""

from .rule_based import (
    # ABC
    BaselinePolicy,
    # Policies
    GreedyNearestPolicy,
    GreedyLargestZonePolicy,
    # Allocators (exported for testing and diagnostics)
    TaskAllocator,
    ZoneAllocator,
    # Task data classes
    AgentTask,
    AgentZoneTask,
    # Movement helpers
    manhattan,
    best_step_toward,
    # Action constants
    ACTION_STAY,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
)

__all__ = [
    "BaselinePolicy",
    "GreedyNearestPolicy",
    "GreedyLargestZonePolicy",
    "TaskAllocator",
    "ZoneAllocator",
    "AgentTask",
    "AgentZoneTask",
    "manhattan",
    "best_step_toward",
    "ACTION_STAY",
    "ACTION_UP",
    "ACTION_DOWN",
    "ACTION_LEFT",
    "ACTION_RIGHT",
]
