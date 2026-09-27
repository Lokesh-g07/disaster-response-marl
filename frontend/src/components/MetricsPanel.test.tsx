import { render, screen } from '@testing-library/react';
import { MetricsPanel } from './MetricsPanel';
import { SimulationState } from '../types';
import { describe, it, expect } from 'vitest';

const mockState: SimulationState = {
  simulation_id: '123',
  scenario: 'fire_small',
  policy: 'MAPPO',
  seed: 42,
  hazard_type: 'FIRE',
  timestep: 42,
  status: 'RUNNING',
  grid: { width: 10, height: 10, walls: [], exits: [], hazard_cells: [] },
  agents: [
    { id: 'rescue_0', position: [1, 1], status: 'MOVING', actionHistory: [] },
    { id: 'rescue_1', position: [2, 2], status: 'BLOCKED', actionHistory: [] }
  ],
  survivors: [],
  metrics: {
    total_reward: 10.5,
    survivors_rescued: 2,
    evacuation_rate: 0.40,
    survivors_remaining: 2,
    casualties: 1
  }
};

describe('MetricsPanel', () => {
  it('renders nothing when state is null', () => {
    const { container } = render(<MetricsPanel state={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders top-level simulation information correctly', () => {
    render(<MetricsPanel state={mockState} />);
    
    // Scenario name
    expect(screen.getByText('fire_small')).toBeInTheDocument();
    
    // Disaster type
    expect(screen.getByText('FIRE')).toBeInTheDocument();
    
    // Policy
    expect(screen.getByText('MAPPO')).toBeInTheDocument();
    
    // Step counter
    expect(screen.getByText('STEP 42')).toBeInTheDocument();
  });

  it('renders survivor metrics correctly', () => {
    render(<MetricsPanel state={mockState} />);
    
    expect(screen.getByText('2', { selector: '.text-blue-400' })).toBeInTheDocument(); // Rescued
    expect(screen.getByText('2', { selector: '.text-orange-400' })).toBeInTheDocument(); // Remaining
    expect(screen.getByText('1', { selector: '.text-red-500' })).toBeInTheDocument(); // Casualties
    expect(screen.getByText('40%')).toBeInTheDocument(); // Evacuation rate
  });

  it('renders agent IDs and statuses correctly', () => {
    render(<MetricsPanel state={mockState} />);
    
    expect(screen.getByText('rescue 0')).toBeInTheDocument();
    expect(screen.getByText('MOVING')).toBeInTheDocument();
    
    expect(screen.getByText('rescue 1')).toBeInTheDocument();
    expect(screen.getByText('BLOCKED')).toBeInTheDocument();
  });

  it('renders completed state clearly', () => {
    const completedState = { ...mockState, status: 'COMPLETED' };
    render(<MetricsPanel state={completedState} />);
    
    expect(screen.getByText('SIMULATION COMPLETE')).toBeInTheDocument();
  });

  it('renders error state clearly', () => {
    const errorState = { ...mockState, status: 'ERROR' };
    render(<MetricsPanel state={errorState} />);
    
    expect(screen.getByText('SIMULATION ERROR')).toBeInTheDocument();
  });
});
