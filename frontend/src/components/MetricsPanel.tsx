import React from 'react';
import { SimulationState } from '../types';

interface MetricsPanelProps {
  state: SimulationState | null;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ state }) => {
  if (!state) return null;
  
  const { metrics, timestep } = state;
  
  return (
    <div className="absolute top-4 right-4 bg-gray-800/90 p-4 rounded-lg shadow-lg text-white w-64 z-10 backdrop-blur-sm border border-gray-700">
      <h3 className="text-lg font-bold border-b border-gray-700 pb-2 mb-3">Live Metrics</h3>
      
      <div className="grid grid-cols-2 gap-y-3 gap-x-2 text-sm">
        <div className="text-gray-400">Status</div>
        <div className={`font-mono font-bold ${state.status === 'RUNNING' ? 'text-green-400' : state.status === 'COMPLETED' ? 'text-blue-400' : 'text-yellow-400'}`}>
          {state.status}
        </div>
        
        <div className="text-gray-400">Timestep</div>
        <div className="font-mono">{timestep}</div>
        
        <div className="text-gray-400">Reward</div>
        <div className="font-mono text-green-400">{metrics.total_reward.toFixed(1)}</div>
        
        <div className="text-gray-400">Rescued</div>
        <div className="font-mono text-blue-400">{metrics.survivors_rescued}</div>
        
        <div className="text-gray-400">Remaining</div>
        <div className="font-mono text-orange-400">{metrics.survivors_remaining}</div>
        
        <div className="text-gray-400">Evac Rate</div>
        <div className="font-mono">{(metrics.evacuation_rate * 100).toFixed(0)}%</div>
        
        <div className="text-gray-400">Casualties</div>
        <div className="font-mono text-red-500">{metrics.casualties}</div>
      </div>
    </div>
  );
};
