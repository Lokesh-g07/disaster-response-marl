import React from 'react';
import { SimulationState, AgentState } from '../types';

interface MetricsPanelProps {
  state: SimulationState | null;
}

const ACTION_LABELS: Record<number, string> = {
  0: 'STAY',
  1: 'UP',
  2: 'DOWN',
  3: 'LEFT',
  4: 'RIGHT',
};

const ACTION_ARROWS: Record<number, string> = {
  0: '•', // STAY
  1: '↑', // UP
  2: '↓', // DOWN
  3: '←', // LEFT
  4: '→', // RIGHT
};

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ state }) => {
  if (!state) return null;
  
  const { metrics, timestep, agents } = state;
  
  return (
    <div className="absolute top-4 right-4 flex flex-col gap-4 z-10 w-80 max-h-[90vh] overflow-y-auto custom-scrollbar">
      {/* Context Panel */}
      <div className="bg-gray-800/90 backdrop-blur-md p-4 rounded-xl shadow-2xl border border-gray-700/50">
        <h3 className="text-sm font-black border-b border-gray-700/50 pb-2 mb-3 text-gray-400 tracking-widest uppercase">Context</h3>
        <div className="grid grid-cols-2 gap-y-2 text-xs">
          <div className="text-gray-500">POLICY</div>
          <div className="font-mono font-bold text-white">{state.policy}</div>
          
          <div className="text-gray-500">SCENARIO</div>
          <div className="font-mono font-bold text-white">{state.scenario}</div>
          
          <div className="text-gray-500">SEED</div>
          <div className="font-mono font-bold text-white">{state.seed}</div>
          
          <div className="text-gray-500">DISASTER</div>
          <div className="font-mono font-bold text-red-400">{state.hazard_type}</div>
        </div>
      </div>

      {/* Live Metrics Panel */}
      <div className="bg-gray-800/90 backdrop-blur-md p-4 rounded-xl shadow-2xl border border-gray-700/50">
        <h3 className="text-sm font-black border-b border-gray-700/50 pb-2 mb-3 text-gray-400 tracking-widest uppercase">Live Metrics</h3>
        
        <div className="grid grid-cols-2 gap-y-3 gap-x-2 text-sm font-medium">
          <div className="text-gray-400">Status</div>
          <div className={`font-mono font-bold tracking-wider ${state.status === 'RUNNING' ? 'text-green-400' : state.status === 'COMPLETED' ? 'text-blue-400' : 'text-yellow-400'}`}>
            {state.status}
          </div>
          
          <div className="text-gray-400">Timestep</div>
          <div className="font-mono font-bold text-white">STEP {timestep}</div>
          
          <div className="text-gray-400">Reward</div>
          <div className="font-mono font-bold text-green-400">{metrics.total_reward.toFixed(1)}</div>
          
          <div className="text-gray-400">Rescued</div>
          <div className="font-mono font-bold text-blue-400">{metrics.survivors_rescued}</div>
          
          <div className="text-gray-400">Remaining</div>
          <div className="font-mono font-bold text-orange-400">{metrics.survivors_remaining}</div>
          
          <div className="text-gray-400">Evac Rate</div>
          <div className="font-mono font-bold text-white">{(metrics.evacuation_rate * 100).toFixed(0)}%</div>
          
          <div className="text-gray-400">Casualties</div>
          <div className="font-mono font-bold text-red-500">{metrics.casualties}</div>
        </div>
      </div>

      {/* Agent Decisions Panel */}
      <div className="bg-gray-800/90 backdrop-blur-md p-4 rounded-xl shadow-2xl border border-gray-700/50">
        <h3 className="text-sm font-black border-b border-gray-700/50 pb-2 mb-3 text-gray-400 tracking-widest uppercase">Agent Decisions</h3>
        
        <div className="flex flex-col gap-4">
          {agents.map((agent: AgentState) => (
            <div key={agent.id} className="bg-gray-900/50 p-3 rounded-lg border border-gray-700/30">
              <div className="font-bold text-blue-400 mb-2 uppercase tracking-wider text-xs">{agent.id.replace('_', ' ')}</div>
              
              <div className="grid grid-cols-3 gap-2 text-xs mb-2">
                <div className="text-gray-500">Position</div>
                <div className="col-span-2 font-mono text-white">({agent.position[0]}, {agent.position[1]})</div>
                
                <div className="text-gray-500">Action</div>
                <div className="col-span-2 font-mono font-bold text-yellow-400">
                  {agent.lastAction !== undefined ? ACTION_LABELS[agent.lastAction] : 'NONE'}
                </div>
                
                <div className="text-gray-500">Status</div>
                <div className={`col-span-2 font-mono font-bold ${
                  agent.status === 'BLOCKED' ? 'text-red-400' : 
                  agent.status === 'MOVING' ? 'text-green-400' : 
                  'text-gray-400'
                }`}>
                  {agent.status}
                </div>

                <div className="text-gray-500">Target</div>
                <div className="col-span-2 font-mono text-gray-500">Not explicitly assigned</div>
              </div>
              
              {agent.actionHistory && agent.actionHistory.length > 0 && (
                <div className="mt-3 pt-2 border-t border-gray-800">
                  <div className="text-[10px] text-gray-500 mb-1 uppercase">Recent Actions</div>
                  <div className="flex gap-2">
                    {agent.actionHistory.map((act, idx) => (
                      <span key={idx} className="font-mono text-sm text-gray-300" title={ACTION_LABELS[act]}>
                        {ACTION_ARROWS[act]}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
