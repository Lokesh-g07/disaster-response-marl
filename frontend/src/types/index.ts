export interface AgentState {
  id: string;
  position: [number, number]; // [x, y]
  status: string; // active, idle, etc.
}

export interface SurvivorState {
  id: string;
  position: [number, number];
  rescued: boolean;
}

export interface GridState {
  width: number;
  height: number;
  walls: [number, number][];
  exits: [number, number][];
  fire_cells: [number, number][];
}

export interface SimulationState {
  simulation_id: string;
  timestep: number;
  status: string; // CREATED, RUNNING, COMPLETED, FAILED, TERMINATED
  grid: GridState;
  agents: AgentState[];
  survivors: SurvivorState[];
  metrics: {
    total_reward: number;
    survivors_rescued: number;
    evacuation_rate: number;
    survivors_remaining: number;
    casualties: number;
  };
}

export interface ScenarioInfo {
  id: string;
  name: string;
  description: string;
}

export interface PolicyInfo {
  id: string;
  name: string;
  description: string;
}
