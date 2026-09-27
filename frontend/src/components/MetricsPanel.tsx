import React from 'react';
import { SimulationState, AgentState } from '../types';
import { Flame, Waves, Users, Activity, ShieldAlert } from 'lucide-react';

interface MetricsPanelProps {
  state: SimulationState | null;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ state }) => {
  if (!state) return null;
  
  const { metrics, timestep, agents } = state;
  const isCompleted = state.status === 'COMPLETED';
  const isError = state.status === 'ERROR' || state.status === 'FAILED';
  
  return (
    <div className="absolute top-4 right-4 flex flex-col gap-4 z-10 w-80 max-h-[90vh] overflow-y-auto custom-scrollbar font-sans">
      
      {/* HUD Header & Simulation Info */}
      <div className="bg-slate-900/80 backdrop-blur-md rounded-lg shadow-2xl border border-slate-700/50 overflow-hidden">
        <div className="bg-slate-800/80 p-3 border-b border-slate-700/50 flex justify-between items-center">
          <h2 className="text-xs font-black text-slate-300 tracking-widest uppercase">
            Disaster Response
          </h2>
          <div className="flex items-center gap-2">
            <span className={`flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded ${
              isCompleted ? 'bg-blue-500/20 text-blue-400' :
              isError ? 'bg-red-500/20 text-red-400' :
              state.status === 'RUNNING' ? 'bg-green-500/20 text-green-400' :
              'bg-yellow-500/20 text-yellow-400'
            }`}>
              {state.status === 'RUNNING' && <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />}
              {isCompleted ? 'SIMULATION COMPLETE' : isError ? 'SIMULATION ERROR' : state.status}
            </span>
          </div>
        </div>
        
        <div className="p-4 grid grid-cols-2 gap-3 text-sm">
          <div className="col-span-2 flex items-center justify-between bg-slate-800/50 p-2 rounded border border-slate-700/30">
            <div className="flex items-center gap-2">
              {state.hazard_type === 'FLOOD' ? <Waves size={16} className="text-blue-400" /> : <Flame size={16} className="text-orange-500" />}
              <span className={`font-black tracking-widest ${state.hazard_type === 'FLOOD' ? 'text-blue-400' : 'text-orange-500'}`}>
                {state.hazard_type}
              </span>
            </div>
            <div className="text-slate-300 font-mono font-bold text-xs bg-slate-900 px-2 py-1 rounded">
              STEP {timestep}
            </div>
          </div>

          <div>
            <div className="text-[10px] text-slate-500 uppercase font-bold">Scenario</div>
            <div className="font-mono text-slate-200 text-xs mt-0.5">{state.scenario}</div>
          </div>
          <div>
            <div className="text-[10px] text-slate-500 uppercase font-bold">Policy</div>
            <div className="font-mono text-slate-200 text-xs mt-0.5">{state.policy}</div>
          </div>
          <div>
            <div className="text-[10px] text-slate-500 uppercase font-bold">Seed</div>
            <div className="font-mono text-slate-200 text-xs mt-0.5">{state.seed}</div>
          </div>
        </div>
      </div>

      {/* Survivor Metrics */}
      <div className="bg-slate-900/80 backdrop-blur-md rounded-lg shadow-2xl border border-slate-700/50 p-4">
        <div className="flex items-center gap-2 border-b border-slate-700/50 pb-2 mb-3 text-slate-300">
          <Users size={14} />
          <h3 className="text-xs font-black tracking-widest uppercase">Survivors</h3>
        </div>
        
        <div className="grid grid-cols-3 gap-2 text-center mb-4">
          <div className="bg-slate-800/50 rounded p-2">
            <div className="text-[10px] text-slate-400 uppercase font-bold mb-1">Rescued</div>
            <div className="font-mono text-lg font-bold text-blue-400">{metrics.survivors_rescued}</div>
          </div>
          <div className="bg-slate-800/50 rounded p-2">
            <div className="text-[10px] text-slate-400 uppercase font-bold mb-1">Remaining</div>
            <div className="font-mono text-lg font-bold text-orange-400">{metrics.survivors_remaining}</div>
          </div>
          <div className="bg-slate-800/50 rounded p-2 border border-red-900/30">
            <div className="text-[10px] text-slate-400 uppercase font-bold mb-1">Casualties</div>
            <div className="font-mono text-lg font-bold text-red-500">{metrics.casualties}</div>
          </div>
        </div>
        
        <div className="flex items-center justify-between bg-slate-800/80 p-3 rounded border border-slate-700/50">
          <div className="flex items-center gap-2">
            <Activity size={14} className="text-emerald-400" />
            <span className="text-[10px] font-black text-slate-300 tracking-widest uppercase">Evacuation Rate</span>
          </div>
          <span className="font-mono text-lg font-bold text-emerald-400">
            {(metrics.evacuation_rate * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Agent Status Panel */}
      <div className="bg-slate-900/80 backdrop-blur-md rounded-lg shadow-2xl border border-slate-700/50 p-4">
        <div className="flex items-center gap-2 border-b border-slate-700/50 pb-2 mb-3 text-slate-300">
          <ShieldAlert size={14} />
          <h3 className="text-xs font-black tracking-widest uppercase">Agents</h3>
        </div>
        
        <div className="flex flex-col gap-2">
          {agents.map((agent: AgentState) => {
            const isBlocked = agent.status === 'BLOCKED';
            const isMoving = agent.status === 'MOVING';
            const isStaying = agent.status === 'STAYING';
            
            return (
              <div key={agent.id} className="flex items-center justify-between bg-slate-800/50 p-2.5 rounded border border-slate-700/30">
                <span className="font-bold text-slate-200 text-xs uppercase tracking-wide">
                  {agent.id.replace('_', ' ')}
                </span>
                
                <div className="flex items-center gap-2">
                  <span className={`text-[10px] font-bold uppercase tracking-wider ${
                    isBlocked ? 'text-red-400' :
                    isMoving ? 'text-blue-400' :
                    isStaying ? 'text-slate-400' :
                    'text-slate-500'
                  }`}>
                    {agent.status}
                  </span>
                  <div className={`w-2 h-2 rounded-full ${
                    isBlocked ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]' :
                    isMoving ? 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)]' :
                    isStaying ? 'bg-slate-400' :
                    'bg-slate-600'
                  }`} />
                </div>
              </div>
            );
          })}
          {agents.length === 0 && (
            <div className="text-center text-slate-500 text-xs italic py-2">
              No active agents
            </div>
          )}
        </div>
      </div>
      
    </div>
  );
};
