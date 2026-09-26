import React from 'react';
import { SimulationState } from '../types';

interface MetricsPanelProps {
  state: SimulationState | null;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ state }) => {
  if (!state) return null;
  
  const { metrics, timestep } = state;
  
  return (
    <div className="absolute top-4 right-4 bg-gray-800/90 backdrop-blur-md p-5 rounded-xl shadow-2xl text-white w-72 z-10 border border-gray-700/50 transition-all">
      <h3 className="text-xl font-black border-b border-gray-700/50 pb-3 mb-4 tracking-tight">Live Metrics</h3>
      
      <div className="grid grid-cols-2 gap-y-4 gap-x-2 text-sm font-medium">
        <div className="text-gray-400">Status</div>
        <div className={`font-mono font-bold tracking-wider ${state.status === 'RUNNING' ? 'text-green-400' : state.status === 'COMPLETED' ? 'text-blue-400' : 'text-yellow-400'}`}>
          {state.status}
        </div>
        
        <div className="text-gray-400">Timestep</div>
        <div className="font-mono font-bold">{timestep}</div>
        
        <div className="text-gray-400">Reward</div>
        <div className="font-mono font-bold text-green-400">{metrics.total_reward.toFixed(1)}</div>
        
        <div className="text-gray-400">Rescued</div>
        <div className="font-mono font-bold text-blue-400">{metrics.survivors_rescued}</div>
        
        <div className="text-gray-400">Remaining</div>
        <div className="font-mono font-bold text-orange-400">{metrics.survivors_remaining}</div>
        
        <div className="text-gray-400">Evac Rate</div>
        <div className="font-mono font-bold">{(metrics.evacuation_rate * 100).toFixed(0)}%</div>
        
        <div className="text-gray-400">Casualties</div>
        <div className="font-mono font-bold text-red-500">{metrics.casualties}</div>
      </div>
    </div>
  );
};
