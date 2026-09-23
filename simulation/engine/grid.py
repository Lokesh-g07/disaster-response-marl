"""Discrete grid representation for CrisisRL disaster scenarios."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

EMPTY = 0
WALL = 1
FIRE = 2
SURVIVOR = 3
EXIT = 4
AGENT = 5
SMOKE = 6

CELL_SYMBOLS = {
    EMPTY: " . ",
    WALL: " # ",
    FIRE: " F ",
    SURVIVOR: " S ",
    EXIT: " E ",
    AGENT: " A ",
    SMOKE: " ~ ",
}

COLOR_SYMBOLS = {
    EMPTY: " . ",
    WALL: "\033[90m # \033[0m",
    FIRE: "\033[91m F \033[0m",
    SURVIVOR: "\033[93m S \033[0m",
    EXIT: "\033[92m E \033[0m",
    AGENT: "\033[96m A \033[0m",
    SMOKE: "\033[37m ~ \033[0m",
}


class DisasterGrid:
    """
    Maintains the 2D discrete environment grid representation,
    tracking static structures (walls, exits) and dynamic entities (fire, survivors).
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int8)

    def reset(self) -> None:
        """Reset all cells in the grid to EMPTY."""
        self.grid.fill(EMPTY)

    def is_valid(self, position: Union[Tuple[int, int], List[int]]) -> bool:
        """Check if a coordinate is within grid bounds."""
        row, col = position
        return 0 <= row < self.height and 0 <= col < self.width

    def is_walkable(self, position: Union[Tuple[int, int], List[int]]) -> bool:
        """Check if a position is valid and not blocked by a wall."""
        if not self.is_valid(position):
            return False
        row, col = position
        return bool(self.grid[row, col] != WALL)

    def add_wall(self, position: Union[Tuple[int, int], List[int]]) -> None:
        """Mark a grid cell as a wall obstacle."""
        if self.is_valid(position):
            self.grid[position[0], position[1]] = WALL

    def add_fire(self, position: Union[Tuple[int, int], List[int]]) -> None:
        """Mark a grid cell as on fire."""
        if self.is_valid(position):
            self.grid[position[0], position[1]] = FIRE

    def add_exit(self, position: Union[Tuple[int, int], List[int]]) -> None:
        """Mark a grid cell as an exit zone."""
        if self.is_valid(position):
            self.grid[position[0], position[1]] = EXIT

    def add_survivor(self, position: Union[Tuple[int, int], List[int]]) -> None:
        """Mark a grid cell as containing a survivor."""
        if self.is_valid(position):
            self.grid[position[0], position[1]] = SURVIVOR

    def remove_survivor(self, position: Union[Tuple[int, int], List[int]]) -> bool:
        """Remove a survivor at the given position if present."""
        if self.is_valid(position):
            row, col = position
            if self.grid[row, col] == SURVIVOR:
                self.grid[row, col] = EMPTY
                return True
        return False

    def get_fire_cells(self) -> np.ndarray:
        """Return array of (row, col) coordinates currently on fire."""
        return np.argwhere(self.grid == FIRE)

    def get_survivor_cells(self) -> np.ndarray:
        """Return array of (row, col) coordinates with survivors."""
        return np.argwhere(self.grid == SURVIVOR)

    def get_exit_cells(self) -> np.ndarray:
        """Return array of (row, col) coordinates for exits."""
        return np.argwhere(self.grid == EXIT)

    def get_wall_cells(self) -> np.ndarray:
        """Return array of (row, col) coordinates for walls."""
        return np.argwhere(self.grid == WALL)

    def render_ascii(
        self,
        agent_positions: Optional[Dict[str, Tuple[int, int]]] = None,
        colored: bool = False,
    ) -> str:
        """
        Generate a human-readable ASCII/colored rendering of the grid state.
        """
        sym_map = COLOR_SYMBOLS if colored else CELL_SYMBOLS
        display_grid = np.array(
            [[sym_map[self.grid[r, c]] for c in range(self.width)] for r in range(self.height)],
            dtype=object,
        )

        if agent_positions:
            for agent_id, pos in agent_positions.items():
                r, c = pos
                if 0 <= r < self.height and 0 <= c < self.width:
                    tag = f"A{agent_id[-1] if agent_id[-1].isdigit() else ''}"
                    display_grid[r, c] = f"\033[1;96m {tag} \033[0m" if colored else f" {tag} "

        horizontal_border = "+---" * self.width + "+"
        lines = [horizontal_border]
        for r in range(self.height):
            row_str = "|" + "|".join(display_grid[r]) + "|"
            lines.append(row_str)
            lines.append(horizontal_border)

        return "\n".join(lines)
