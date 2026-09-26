import { useState, useCallback, useRef, useEffect } from 'react';
import type { SimulationState, GridState, AgentState, SurvivorState } from '../types';
import { createSimulation, stepSimulation, terminateSimulation } from '../api/simulations';

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
  if (data.terminated) status = 'COMPLETED';
  if (data.truncated) status = 'FAILED';

  return {
    simulation_id: simId,
    timestep: data.step,
    status,
    grid,
    agents,
    survivors,
    metrics: data.metrics,
  };
};

export const useSimulation = () => {
  const [simulationState, setSimulationState] = useState<SimulationState | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const simulationIdRef = useRef<string | null>(null);
  const intervalRef = useRef<number | null>(null);

  const startNewSimulation = async (scenario: string, policy: string, seed: number) => {
    try {
      setError(null);
      stopSimulation(); // Stop current if running
      
      const { simulation_id } = await createSimulation(scenario, policy, seed);
      simulationIdRef.current = simulation_id;
      
      // Get initial state
      const initialResponse = await stepSimulation(simulation_id);
      setSimulationState(mapBackendResponse(simulation_id, initialResponse));
      
      setIsRunning(true);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to start simulation');
      setIsRunning(false);
    }
  };

  const step = useCallback(async () => {
    if (!simulationIdRef.current) return;
    
    try {
      const response = await stepSimulation(simulationIdRef.current);
      const state = mapBackendResponse(simulationIdRef.current, response);
      setSimulationState(state);
      
      if (state.status === 'COMPLETED' || state.status === 'FAILED') {
        setIsRunning(false);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to step simulation');
      setIsRunning(false);
    }
  }, []);

  const pauseSimulation = () => {
    setIsRunning(false);
  };

  const resumeSimulation = () => {
    if (simulationIdRef.current && simulationState?.status === 'RUNNING') {
      setIsRunning(true);
    }
  };

  const stopSimulation = async () => {
    setIsRunning(false);
    if (simulationIdRef.current) {
      try {
        await terminateSimulation(simulationIdRef.current);
      } catch (e) {
        // Ignore errors on termination
      }
      simulationIdRef.current = null;
    }
  };

  useEffect(() => {
    if (isRunning) {
      intervalRef.current = window.setInterval(() => {
        step();
      }, 500); // Step every 500ms
    } else if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isRunning, step]);

  return {
    simulationState,
    isRunning,
    error,
    startNewSimulation,
    pauseSimulation,
    resumeSimulation,
    step,
    stopSimulation,
  };
};
