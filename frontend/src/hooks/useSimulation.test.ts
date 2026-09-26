import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, Mock } from 'vitest';
import { useSimulation } from './useSimulation';
import { createSimulation } from '../api/simulations';

vi.mock('../api/simulations', () => ({
  createSimulation: vi.fn(),
}));

class MockWebSocket {
  static OPEN = 1;
  static CLOSED = 3;
  url: string;
  onopen: (() => void) | null = null;
  onmessage: ((event: any) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  readyState: number = 1; // OPEN

  static instances: MockWebSocket[] = [];
  sentMessages: any[] = [];

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sentMessages.push(JSON.parse(data));
  }

  close() {
    this.readyState = 3; // CLOSED
    if (this.onclose) this.onclose();
  }
}

describe('useSimulation WebSocket Hook', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    MockWebSocket.instances = [];
    (globalThis as any).WebSocket = MockWebSocket;
  });

  it('starts a new simulation and connects to WebSocket', async () => {
    (createSimulation as Mock).mockResolvedValue({ simulation_id: 'test-123' });

    const { result } = renderHook(() => useSimulation());

    await act(async () => {
      await result.current.startNewSimulation('scenario', 'policy', 42);
    });

    expect(createSimulation).toHaveBeenCalledWith('scenario', 'policy', 42);
    expect(result.current.connectionStatus).toBe('Connected');
    expect(MockWebSocket.instances.length).toBe(1);
    expect(MockWebSocket.instances[0].url).toContain('/ws/simulations/test-123');
  });

  it('sends control messages correctly', async () => {
    (createSimulation as Mock).mockResolvedValue({ simulation_id: 'test-123' });
    const { result } = renderHook(() => useSimulation());

    await act(async () => {
      await result.current.startNewSimulation('scenario', 'policy', 42);
    });

    const ws = MockWebSocket.instances[0];
    
    // Clear initial step message sent on connection
    ws.sentMessages = [];

    act(() => {
      result.current.resumeSimulation();
    });
    expect(ws.sentMessages[0].type).toBe('start');
    expect(result.current.connectionStatus).toBe('Running');
    expect(result.current.isRunning).toBe(true);

    act(() => {
      result.current.pauseSimulation();
    });
    expect(ws.sentMessages[1].type).toBe('pause');
    expect(result.current.connectionStatus).toBe('Paused');

    act(() => {
      result.current.step();
    });
    expect(ws.sentMessages[2].type).toBe('step');

    act(() => {
      result.current.stopSimulation();
    });
    expect(ws.sentMessages[3].type).toBe('stop');
    expect(result.current.connectionStatus).toBe('Disconnected');
  });

  it('handles state and metric updates from WebSocket', async () => {
    (createSimulation as Mock).mockResolvedValue({ simulation_id: 'test-123' });
    const { result } = renderHook(() => useSimulation());

    await act(async () => {
      await result.current.startNewSimulation('scenario', 'policy', 42);
    });

    const ws = MockWebSocket.instances[0];

    act(() => {
      if (ws.onmessage) {
        ws.onmessage({
          data: JSON.stringify({
            type: 'state',
            step: 5,
            state: {
              width: 10,
              height: 10,
              fires: [[1, 1]],
              survivors: [[2, 2]],
              agents: {
                agent_1: { position: [0, 0], active: true }
              }
            },
            metrics: { total_reward: 100 }
          })
        });
      }
    });

    expect(result.current.simulationState).not.toBeNull();
    expect(result.current.simulationState?.timestep).toBe(5);
    expect(result.current.simulationState?.metrics.total_reward).toBe(100);
    expect(result.current.simulationState?.grid.fire_cells.length).toBe(1);
    expect(result.current.simulationState?.agents.length).toBe(1);
  });

  it('handles completed messages', async () => {
    (createSimulation as Mock).mockResolvedValue({ simulation_id: 'test-123' });
    const { result } = renderHook(() => useSimulation());

    await act(async () => {
      await result.current.startNewSimulation('scenario', 'policy', 42);
    });

    const ws = MockWebSocket.instances[0];

    act(() => {
      if (ws.onmessage) {
        ws.onmessage({
          data: JSON.stringify({
            type: 'completed',
            step: 50,
            state: { width: 10, height: 10 },
            metrics: {}
          })
        });
      }
    });

    expect(result.current.connectionStatus).toBe('Completed');
    expect(result.current.simulationState?.status).toBe('COMPLETED');
  });

  it('handles error messages from backend', async () => {
    (createSimulation as Mock).mockResolvedValue({ simulation_id: 'test-123' });
    const { result } = renderHook(() => useSimulation());

    await act(async () => {
      await result.current.startNewSimulation('scenario', 'policy', 42);
    });

    const ws = MockWebSocket.instances[0];

    act(() => {
      if (ws.onmessage) {
        ws.onmessage({
          data: JSON.stringify({
            type: 'error',
            message: 'Invalid policy'
          })
        });
      }
    });

    expect(result.current.error).toBe('Invalid policy');
    expect(result.current.connectionStatus).toBe('Error');
  });

  it('cleans up WebSocket on unmount', async () => {
    (createSimulation as Mock).mockResolvedValue({ simulation_id: 'test-123' });
    const { result, unmount } = renderHook(() => useSimulation());

    await act(async () => {
      await result.current.startNewSimulation('scenario', 'policy', 42);
    });

    const ws = MockWebSocket.instances[0];
    expect(ws.readyState).toBe(1);

    unmount();
    expect(ws.readyState).toBe(3);
  });
});
