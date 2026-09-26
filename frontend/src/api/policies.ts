import { apiClient } from './client';
import { PolicyInfo } from '../types';

export const listPolicies = async (): Promise<PolicyInfo[]> => {
  const response = await apiClient.get('/policies');
  return response.data;
};
