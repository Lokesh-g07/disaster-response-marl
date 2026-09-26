import { apiClient } from './client';
import { SimulationState } from '../types';

export const createSimulation = async (scenario: string, policy: string, seed: number): Promise<{ simulation_id: string }> => {
  const response = await apiClient.post('/simulations', {
    scenario,
    policy,
    seed
  });
  return response.data;
};

export const stepSimulation = async (simulationId: string): Promise<SimulationState> => {
  const response = await apiClient.post(`/simulations/${simulationId}/step`);
  return response.data;
};

export const getSimulationState = async (simulationId: string): Promise<SimulationState> => {
  const response = await apiClient.get(`/simulations/${simulationId}`);
  return response.data;
};

export const terminateSimulation = async (simulationId: string): Promise<void> => {
  await apiClient.delete(`/simulations/${simulationId}`);
};
