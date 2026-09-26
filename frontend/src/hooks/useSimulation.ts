import { useState, useCallback, useRef, useEffect } from 'react';
import type { SimulationState, GridState, AgentState, SurvivorState } from '../types';
import { createSimulation } from '../api/simulations';
import { SimulationWebSocket } from '../api/websocket';

const mapBackendResponse = (simId: string, data: any): SimulationState => {
  const backendState = data.state;
  
  const grid: GridState = {
    width: backendState.width,
    height: backendState.height,
    walls: backendState.walls || [],
    exits: backendState.exits || [],
    fire_cells: backendState.fires || [],
  };

  const agents: AgentState[] = Object.entries(backendState.agents || {}).map(([id, info]: [string, any]) => ({
    id,
    position: info.position,
    status: info.active ? 'active' : 'idle',
  }));

  const survivors: SurvivorState[] = (backendState.survivors || []).map((pos: [number, number], index: number) => ({
    id: `survivor_${index}`,
    position: pos,
    rescued: false, // The backend only returns active survivors in the array
  }));

  let status = 'RUNNING';
  if (data.type === 'completed') status = 'COMPLETED';
  if (data.type === 'error') status = 'FAILED';

  return {
    simulation_id: simId,
    timestep: data.step,
    status,
    grid,
    agents,
    survivors,
    metrics: data.metrics || {},
  };
};

export type ConnectionStatus = 'Disconnected' | 'Connecting' | 'Connected' | 'Running' | 'Paused' | 'Completed' | 'Error';

export const useSimulation = () => {
  const [simulationState, setSimulationState] = useState<SimulationState | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('Disconnected');
  const [error, setError] = useState<string | null>(null);
  
  const simulationIdRef = useRef<string | null>(null);
  const wsRef = useRef<SimulationWebSocket | null>(null);

  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.disconnect();
      wsRef.current = null;
    }
  }, []);

  const handleMessage = useCallback((data: any) => {
    if (data.type === 'error') {
      setError(data.message);
      setConnectionStatus('Error');
      return;
    }

    if (data.type === 'state' || data.type === 'completed') {
      if (simulationIdRef.current) {
        setSimulationState(mapBackendResponse(simulationIdRef.current, data));
      }
      if (data.type === 'completed') {
        setConnectionStatus('Completed');
      }
    }
  }, []);

  const startNewSimulation = async (scenario: string, policy: string, seed: number) => {
    try {
      setError(null);
      setConnectionStatus('Connecting');
      cleanup(); // Disconnect existing
      setSimulationState(null);
      
      const { simulation_id } = await createSimulation(scenario, policy, seed);
      simulationIdRef.current = simulation_id;
      
      wsRef.current = new SimulationWebSocket(
        simulation_id,
        handleMessage,
        (err) => {
          setError(err);
          setConnectionStatus('Error');
        },
        () => {
          if (connectionStatus !== 'Completed' && connectionStatus !== 'Error') {
             setConnectionStatus('Disconnected');
          }
        }
      );
      
      wsRef.current.connect();
      setConnectionStatus('Connected');
      
      // Step once to get initial state immediately after connecting
      setTimeout(() => {
          if (wsRef.current) wsRef.current.send('step');
      }, 100);
      
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to start simulation');
      setConnectionStatus('Error');
    }
  };

  const step = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.send('step');
    }
  }, []);

  const pauseSimulation = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.send('pause');
      setConnectionStatus('Paused');
    }
  }, []);

  const resumeSimulation = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.send('start');
      setConnectionStatus('Running');
    }
  }, []);

  const stopSimulation = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.send('stop');
      cleanup();
      setConnectionStatus('Disconnected');
    }
  }, [cleanup]);

  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  return {
    simulationState,
    isRunning: connectionStatus === 'Running',
    connectionStatus,
    error,
    startNewSimulation,
    pauseSimulation,
    resumeSimulation,
    step,
    stopSimulation,
  };
};
