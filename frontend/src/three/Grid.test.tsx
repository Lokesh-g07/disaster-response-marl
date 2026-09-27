import { vi, describe, it, expect } from 'vitest';
import { Grid } from './Grid';
import { SimulationState } from '../types';
import React from 'react';

// Mock React hooks to avoid Invalid Hook Call when rendering as a pure function
vi.mock('react', async () => {
  const actual = await vi.importActual('react');
  return {
    ...actual,
    useRef: () => ({ current: null }),
    useMemo: (cb: any) => cb(),
    useEffect: () => {}
  };
});

// Mock @react-three/fiber hooks
vi.mock('@react-three/fiber', () => ({
  useFrame: () => {}
}));

const mockState: SimulationState = {
  simulation_id: 'test_123',
  scenario: 'test',
  tick: 0,
  grid: {
    width: 10,
    height: 10,
    walls: [[5, 5]], // row 5, col 5
    exits: [[0, 9]], // row 0, col 9
    hazard_cells: [[2, 2]] // row 2, col 2
  },
  agents: [
    { id: 'rescue_0', position: [1, 1], status: 'MOVING', actionHistory: [] }
  ],
  survivors: [
    { id: 'surv_0', position: [8, 8], rescued: false }
  ],
  hazard_type: 'FIRE',
  metrics: {
    total_reward: 0,
    survivors_rescued: 0,
    casualties: 0,
    evacuation_rate: 0,
    survivors_remaining: 1
  }
};

// Helper to find groups by a specific key prefix in the React node tree
function findChildrenByKeyPrefix(node: any, prefix: string): any[] {
  let results: any[] = [];
  if (!node) return results;
  
  if (Array.isArray(node)) {
    for (const child of node) {
      results = results.concat(findChildrenByKeyPrefix(child, prefix));
    }
  } else {
    if (node.key && typeof node.key === 'string' && node.key.startsWith(prefix)) {
      results.push(node);
    }
    if (node.props && node.props.children) {
      results = results.concat(findChildrenByKeyPrefix(node.props.children, prefix));
    }
  }
  return results;
}

describe('Grid Coordinate Mapping Tests', () => {
  it('correctly maps grid dimensions to centering offsets', () => {
    const gridNode = Grid({ state: mockState }) as React.ReactElement;
    
    // Width 10, Height 10
    // offsetX = -(10 - 1) / 2 = -4.5
    // offsetZ = -(10 - 1) / 2 = -4.5
    expect(gridNode.props.position).toEqual([-4.5, 0, -4.5]);
  });

  it('correctly places floor tiles for all non-wall cells', () => {
    const gridNode = Grid({ state: mockState }) as React.ReactElement;
    const floorTiles = findChildrenByKeyPrefix(gridNode, 'floor-');
    
    // 10x10 grid = 100 cells. 1 wall = 99 floor tiles.
    expect(floorTiles.length).toBe(99);
    
    // Check specific tile placement (e.g. row 0, col 0)
    const tile0_0 = floorTiles.find(t => t.key === 'floor-0-0');
    // x = col, z = row
    expect(tile0_0.props.position).toEqual([0, 0, 0]);
  });

  it('correctly maps wall coordinates (row->Z, col->X)', () => {
    const gridNode = Grid({ state: mockState }) as React.ReactElement;
    const walls = findChildrenByKeyPrefix(gridNode, 'wall-');
    
    expect(walls.length).toBe(1);
    // Wall at [5, 5] -> x=5, z=5
    expect(walls[0].props.position).toEqual([5, 0.6, 5]);
  });

  it('correctly maps exit coordinates', () => {
    const gridNode = Grid({ state: mockState }) as React.ReactElement;
    const exits = findChildrenByKeyPrefix(gridNode, 'exit-');
    
    expect(exits.length).toBe(1);
    // Exit at [0, 9] (row 0, col 9) -> x=9, z=0
    expect(exits[0].props.position).toEqual([9, 0, 0]);
  });

  it('correctly maps survivor coordinates', () => {
    const gridNode = Grid({ state: mockState }) as React.ReactElement;
    const survivors = findChildrenByKeyPrefix(gridNode, 'survivor-');
    
    expect(survivors.length).toBe(1);
    // Survivor at [8, 8] -> x=8, z=8
    expect(survivors[0].props.position).toEqual([8, 0.45, 8]);
  });

  it('correctly maps agent coordinates via AgentDrone props', () => {
    const gridNode = Grid({ state: mockState }) as React.ReactElement;
    const agents = findChildrenByKeyPrefix(gridNode, 'agent-');
    
    expect(agents.length).toBe(1);
    // Agent properties should be passed to AgentDrone correctly
    expect(agents[0].props.agent.position).toEqual([1, 1]);
  });

  it('correctly maps hazard coordinates', () => {
    const gridNode = Grid({ state: mockState }) as React.ReactElement;
    const fireHazards = findChildrenByKeyPrefix(gridNode, 'fire-');
    
    expect(fireHazards.length).toBe(1);
    // Fire at [2, 2] -> x=2, z=2
    expect(fireHazards[0].props.position).toEqual([2, 0.4, 2]);
  });

  it('adapts to non-square grid dimensions without breaking mapping', () => {
    const rectState = { ...mockState, grid: { ...mockState.grid, width: 20, height: 5 } };
    const gridNode = Grid({ state: rectState }) as React.ReactElement;
    
    // offsetX = -(20 - 1) / 2 = -9.5
    // offsetZ = -(5 - 1) / 2 = -2.0
    expect(gridNode.props.position).toEqual([-9.5, 0, -2.0]);
  });
});
