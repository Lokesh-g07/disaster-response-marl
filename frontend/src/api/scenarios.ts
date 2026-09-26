import { apiClient } from './client';
import { ScenarioInfo } from '../types';

export const listScenarios = async (): Promise<ScenarioInfo[]> => {
  const response = await apiClient.get('/scenarios');
  return response.data;
};
