"""Rescue agent representation within the disaster environment."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RescueAgent:
    """
    Tracks state and metrics for an individual emergency response unit.
    """

    agent_id: str
    initial_position: Tuple[int, int]
    position: Tuple[int, int] = field(init=False)
    rescued_count: int = 0
    steps_taken: int = 0
    is_active: bool = True

    def __post_init__(self) -> None:
        self.position = self.initial_position

    def reset(self) -> None:
        """Reset agent state to initial spawn conditions."""
        self.position = self.initial_position
        self.rescued_count = 0
        self.steps_taken = 0
        self.is_active = True

    def move_to(self, new_position: Tuple[int, int]) -> None:
        """Update agent position and step count."""
        self.position = new_position
        self.steps_taken += 1

    def record_rescue(self) -> None:
        """Increment count of successfully rescued survivors."""
        self.rescued_count += 1
