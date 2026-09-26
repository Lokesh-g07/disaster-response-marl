import React from 'react';
import { render, screen } from '@testing-library/react';
import { MetricsPanel } from './MetricsPanel';
import { SimulationState } from '../types';
import { describe, it, expect } from 'vitest';

describe('MetricsPanel', () => {
  it('renders nothing when state is null', () => {
    const { container } = render(<MetricsPanel state={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders metrics when state is provided', () => {
    const mockState: SimulationState = {
      simulation_id: '123',
      timestep: 5,
      status: 'RUNNING',
      grid: { width: 10, height: 10, walls: [], exits: [], fire_cells: [] },
      agents: [],
      survivors: [],
      metrics: {
        total_reward: 10.5,
        survivors_rescued: 2,
        evacuation_rate: 0.5,
        survivors_remaining: 2,
        casualties: 0
      }
    };

    render(<MetricsPanel state={mockState} />);

    expect(screen.getByText('Live Metrics')).toBeInTheDocument();
    expect(screen.getByText('RUNNING')).toBeInTheDocument();
    expect(screen.getByText('10.5')).toBeInTheDocument();
    expect(screen.getByText('50%')).toBeInTheDocument();
  });
});
