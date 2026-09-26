import React from 'react';
import { ScenarioInfo, PolicyInfo } from '../types';
import { Play, Pause, Square, SkipForward } from 'lucide-react';

interface SimulationControlsProps {
  scenarios: ScenarioInfo[];
  policies: PolicyInfo[];
  selectedScenario: string;
  selectedPolicy: string;
  seed: number;
  onScenarioChange: (val: string) => void;
  onPolicyChange: (val: string) => void;
  onSeedChange: (val: number) => void;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStep: () => void;
  onStop: () => void;
  isRunning: boolean;
  hasActiveSimulation: boolean;
}

export const SimulationControls: React.FC<SimulationControlsProps> = ({
  scenarios,
  policies,
  selectedScenario,
  selectedPolicy,
  seed,
  onScenarioChange,
  onPolicyChange,
  onSeedChange,
  onStart,
  onPause,
  onResume,
  onStep,
  onStop,
  isRunning,
  hasActiveSimulation,
}) => {
  return (
    <div className="bg-gray-800 p-4 rounded-lg shadow-lg flex flex-col gap-4 text-white w-64 flex-shrink-0 z-10 relative">
      <h2 className="text-xl font-bold border-b border-gray-700 pb-2">Controls</h2>
      
      <div className="flex flex-col gap-2">
        <label className="text-sm font-semibold text-gray-400">Scenario</label>
        <select 
          className="bg-gray-700 p-2 rounded border border-gray-600 focus:outline-none focus:border-blue-500"
          value={selectedScenario} 
          onChange={(e) => onScenarioChange(e.target.value)}
          disabled={hasActiveSimulation}
        >
          {scenarios.map(s => (
            <option key={s.id} value={s.id}>{s.name}</option>
          ))}
        </select>
      </div>

      <div className="flex flex-col gap-2">
        <label className="text-sm font-semibold text-gray-400">Policy</label>
        <select 
          className="bg-gray-700 p-2 rounded border border-gray-600 focus:outline-none focus:border-blue-500"
          value={selectedPolicy} 
          onChange={(e) => onPolicyChange(e.target.value)}
          disabled={hasActiveSimulation}
        >
          {policies.map(p => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
      </div>

      <div className="flex flex-col gap-2">
        <label className="text-sm font-semibold text-gray-400">Seed</label>
        <input 
          type="number" 
          className="bg-gray-700 p-2 rounded border border-gray-600 focus:outline-none focus:border-blue-500"
          value={seed} 
          onChange={(e) => onSeedChange(Number(e.target.value))}
          disabled={hasActiveSimulation}
        />
      </div>

      <div className="grid grid-cols-2 gap-2 mt-4">
        {!hasActiveSimulation ? (
          <button 
            className="col-span-2 bg-blue-600 hover:bg-blue-500 flex items-center justify-center gap-2 p-2 rounded transition-colors font-medium"
            onClick={onStart}
          >
            <Play size={18} /> Start New
          </button>
        ) : (
          <>
            {isRunning ? (
              <button 
                className="bg-yellow-600 hover:bg-yellow-500 flex items-center justify-center p-2 rounded transition-colors"
                onClick={onPause}
                title="Pause"
              >
                <Pause size={18} />
              </button>
            ) : (
              <button 
                className="bg-green-600 hover:bg-green-500 flex items-center justify-center p-2 rounded transition-colors"
                onClick={onResume}
                title="Resume"
              >
                <Play size={18} />
              </button>
            )}
            
            <button 
              className="bg-gray-600 hover:bg-gray-500 flex items-center justify-center p-2 rounded transition-colors disabled:opacity-50"
              onClick={onStep}
              disabled={isRunning}
              title="Step Forward"
            >
              <SkipForward size={18} />
            </button>
            
            <button 
              className="col-span-2 bg-red-600 hover:bg-red-500 flex items-center justify-center gap-2 p-2 rounded transition-colors mt-2 font-medium"
              onClick={onStop}
            >
              <Square size={18} /> Stop
            </button>
          </>
        )}
      </div>
    </div>
  );
};
