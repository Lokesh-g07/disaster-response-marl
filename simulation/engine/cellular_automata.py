"""Stochastic Cellular Automata engine for dynamic hazard propagation."""

import random
from typing import List, Optional, Tuple, Union
import numpy as np

from .grid import DisasterGrid, FIRE, WALL


class HazardSimulator:
    """
    Simulates non-deterministic hazard (fire/smoke/flood) diffusion
    across a discrete DisasterGrid using stochastic Cellular Automata rules.
    """

    def __init__(
        self,
        grid: DisasterGrid,
        spread_probability: float = 0.30,
        rng: Optional[random.Random] = None,
    ):
        self.grid = grid
        self.spread_probability = float(spread_probability)
        self.rng = rng if rng is not None else random.Random()

    def set_seed(self, seed: Optional[int] = None) -> None:
        """Set seed for reproducible hazard diffusion."""
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()

    def step(self) -> List[Tuple[int, int]]:
        """
        Advance hazard simulation by one discrete timestep.

        Iterates over all active hazard cells and attempts stochastic spread
        to adjacent 4-connected non-wall neighbors.

        Returns:
            List of newly ignited (row, col) coordinates.
        """
        current_fire = self.grid.get_fire_cells()
        if len(current_fire) == 0:
            return []

        new_fire_set = set()

        for row, col in current_fire:
            neighbors = [
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ]

            for nr, nc in neighbors:
                if not self.grid.is_valid((nr, nc)):
                    continue

                cell_val = self.grid.grid[nr, nc]
                if cell_val == WALL or cell_val == FIRE:
                    continue

                if (nr, nc) not in new_fire_set:
                    if self.rng.random() < self.spread_probability:
                        new_fire_set.add((nr, nc))

        new_fire_list = list(new_fire_set)
        for position in new_fire_list:
            self.grid.add_fire(position)

        return new_fire_list
