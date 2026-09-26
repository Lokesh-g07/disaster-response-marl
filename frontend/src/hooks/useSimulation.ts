import { useState, useCallback, useRef, useEffect } from 'react';
import type { SimulationState, GridState, AgentState, SurvivorState } from '../types';
import { createSimulation } from '../api/simulations';
import { SimulationWebSocket } from '../api/websocket';

const mapBackendResponse = (
  simId: string, 
  data: any, 
  scenario: string,
  policy: string,
  seed: number,
  prevAgents: AgentState[] = []
): SimulationState => {
  const backendState = data.state;
  
  const grid: GridState = {
    width: backendState.width,
    height: backendState.height,
    walls: backendState.walls || [],
    exits: backendState.exits || [],
    hazard_cells: backendState.fires || [],
  };

  const agents: AgentState[] = Object.entries(backendState.agents || {}).map(([id, info]: [string, any]) => {
    const prevAgent = prevAgents.find(a => a.id === id);
    const action = data.actions ? data.actions[id] : undefined;
    
    let status = 'UNKNOWN';
    if (!info.active) {
      status = 'INACTIVE';
    } else if (action !== undefined) {
      if (action === 0) {
        status = 'STAYING';
      } else {
        // If agent tried to move (1=UP, 2=DOWN, 3=LEFT, 4=RIGHT)
        if (prevAgent) {
          const moved = prevAgent.position[0] !== info.position[0] || prevAgent.position[1] !== info.position[1];
          status = moved ? 'MOVING' : 'BLOCKED';
        } else {
          status = 'MOVING'; // Can't determine block status without prev pos
        }
      }
    } else {
      status = 'IDLE'; // No action taken yet (e.g., initial state)
    }

    const actionHistory = prevAgent ? [...prevAgent.actionHistory] : [];
    if (action !== undefined) {
      actionHistory.push(action);
      if (actionHistory.length > 5) {
        actionHistory.shift();
      }
    }

    return {
      id,
      position: info.position,
      status,
      lastAction: action,
      actionHistory,
    };
  });

  const survivors: SurvivorState[] = (backendState.survivors || []).map((pos: [number, number], index: number) => ({
    id: `survivor_${index}`,
    position: pos,
    rescued: false, // Backend only returns active survivors
  }));

  let status = 'RUNNING';
  if (data.type === 'completed') status = 'COMPLETED';
  if (data.type === 'error') status = 'FAILED';
  
  const hazard_type = scenario.includes('flood') ? 'FLOOD' : 'FIRE';

  return {
    simulation_id: simId,
    scenario,
    policy,
    seed,
    hazard_type,
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
  const scenarioRef = useRef<string | null>(null);
  const policyRef = useRef<string | null>(null);
  const seedRef = useRef<number | null>(null);
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
      if (simulationIdRef.current && scenarioRef.current && policyRef.current && seedRef.current !== null) {
        setSimulationState((prev) => 
          mapBackendResponse(
            simulationIdRef.current!, 
            data, 
            scenarioRef.current!,
            policyRef.current!,
            seedRef.current!,
            prev ? prev.agents : []
          )
        );
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
      scenarioRef.current = scenario;
      policyRef.current = policy;
      seedRef.current = seed;
      
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
