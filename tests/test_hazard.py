"""Unit tests for cellular automata hazard spread simulation."""

import numpy as np
from simulation.engine.cellular_automata import HazardSimulator
from simulation.engine.grid import DisasterGrid, FIRE, WALL


def test_hazard_spread_prob_one():
    """Verify deterministic spread to all 4-connected neighbors when spread_probability=1.0."""
    grid = DisasterGrid(width=5, height=5)
    grid.add_fire((2, 2))

    sim = HazardSimulator(grid=grid, spread_probability=1.0)
    new_fires = sim.step()

    # (2,2) should spread to (1,2), (3,2), (2,1), (2,3)
    expected_neighbors = {(1, 2), (3, 2), (2, 1), (2, 3)}
    assert set(new_fires) == expected_neighbors
    assert len(grid.get_fire_cells()) == 5


def test_hazard_blocked_by_walls():
    """Verify that fire cannot spread through wall obstacles."""
    grid = DisasterGrid(width=5, height=5)
    grid.add_fire((2, 2))
    grid.add_wall((2, 3))  # Wall blocks right neighbor
    grid.add_wall((1, 2))  # Wall blocks top neighbor

    sim = HazardSimulator(grid=grid, spread_probability=1.0)
    new_fires = sim.step()

    # Only (3,2) and (2,1) should catch fire
    expected_neighbors = {(3, 2), (2, 1)}
    assert set(new_fires) == expected_neighbors
    assert grid.grid[2, 3] == WALL
    assert grid.grid[1, 2] == WALL


def test_hazard_no_spread_prob_zero():
    """Verify no spread occurs when spread_probability=0.0."""
    grid = DisasterGrid(width=5, height=5)
    grid.add_fire((2, 2))

    sim = HazardSimulator(grid=grid, spread_probability=0.0)
    new_fires = sim.step()

    assert len(new_fires) == 0
    assert len(grid.get_fire_cells()) == 1


def test_hazard_reproducibility_with_seed():
    """Verify that setting the seed produces identical stochastic spread trajectories."""
    grid1 = DisasterGrid(width=10, height=10)
    grid1.add_fire((5, 5))
    sim1 = HazardSimulator(grid=grid1, spread_probability=0.5)
    sim1.set_seed(42)
    step1_fires = sim1.step()

    grid2 = DisasterGrid(width=10, height=10)
    grid2.add_fire((5, 5))
    sim2 = HazardSimulator(grid=grid2, spread_probability=0.5)
    sim2.set_seed(42)
    step2_fires = sim2.step()

    assert step1_fires == step2_fires
