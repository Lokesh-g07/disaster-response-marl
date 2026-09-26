import { useState, useCallback, useRef, useEffect } from 'react';
import { SimulationState } from '../types';
import { createSimulation, stepSimulation, terminateSimulation } from '../api/simulations';

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
      const initialState = await stepSimulation(simulation_id);
      setSimulationState(initialState);
      
      setIsRunning(true);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to start simulation');
      setIsRunning(false);
    }
  };

  const step = useCallback(async () => {
    if (!simulationIdRef.current) return;
    
    try {
      const state = await stepSimulation(simulationIdRef.current);
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
