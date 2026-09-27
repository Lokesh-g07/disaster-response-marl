import { useEffect, useState } from 'react';
import { SimulationControls } from './components/SimulationControls';
import { MetricsPanel } from './components/MetricsPanel';
import { DisasterScene } from './three/DisasterScene';
import { useSimulation } from './hooks/useSimulation';
import { listScenarios } from './api/scenarios';
import { listPolicies } from './api/policies';
import { ScenarioInfo, PolicyInfo } from './types';

function App() {
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [policies, setPolicies] = useState<PolicyInfo[]>([]);
  
  const [selectedScenario, setSelectedScenario] = useState<string>('');
  const [selectedPolicy, setSelectedPolicy] = useState<string>('');
  const [seed, setSeed] = useState<number>(42);

  const {
    simulationState,
    isRunning,
    connectionStatus,
    error,
    startNewSimulation,
    pauseSimulation,
    resumeSimulation,
    step,
    stopSimulation
  } = useSimulation();

  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        const [scens, pols] = await Promise.all([
          listScenarios(),
          listPolicies()
        ]);
        setScenarios(scens);
        setPolicies(pols);
        
        if (scens.length > 0) setSelectedScenario(scens[0].id);
        if (pols.length > 0) setSelectedPolicy(pols[0].id);
      } catch (err) {
        console.error("Failed to fetch scenarios/policies", err);
      }
    };
    
    fetchMetadata();
  }, []);

  const handleStart = () => {
    if (selectedScenario && selectedPolicy) {
      startNewSimulation(selectedScenario, selectedPolicy, seed);
    }
  };

  return (
    <div className="w-full h-screen overflow-hidden flex relative bg-[#0f172a] font-sans">
      <DisasterScene state={simulationState} />
      
      <div className="absolute top-4 left-4 z-10 flex flex-col gap-4">
        <div className="bg-gray-800/90 backdrop-blur-md p-4 rounded-xl shadow-2xl border border-gray-700 w-64 transition-all">
          <div className="flex justify-between items-center mb-1">
            <h1 className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-teal-300 tracking-tight">
              CrisisRL
            </h1>
            <div className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider flex items-center gap-1 ${
              connectionStatus === 'Connected' || connectionStatus === 'Running' ? 'bg-green-500/20 text-green-400 border border-green-500/30' :
              connectionStatus === 'Disconnected' ? 'bg-gray-500/20 text-gray-400 border border-gray-500/30' :
              connectionStatus === 'Error' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
              'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
            }`}>
              {connectionStatus === 'Running' && <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"></div>}
              {connectionStatus}
            </div>
          </div>
          <p className="text-gray-400 text-xs font-medium uppercase tracking-widest opacity-80">Multi-Agent Responder</p>
        </div>
        
        <SimulationControls 
          scenarios={scenarios}
          policies={policies}
          selectedScenario={selectedScenario}
          selectedPolicy={selectedPolicy}
          seed={seed}
          onScenarioChange={setSelectedScenario}
          onPolicyChange={setSelectedPolicy}
          onSeedChange={setSeed}
          onStart={handleStart}
          onPause={pauseSimulation}
          onResume={resumeSimulation}
          onStep={step}
          onStop={stopSimulation}
          isRunning={isRunning}
          hasActiveSimulation={!!simulationState && !['COMPLETED', 'FAILED', 'TERMINATED'].includes(simulationState.status)}
        />

        {error && (
          <div className="bg-red-900/90 border border-red-700 text-white p-3 rounded-lg text-sm w-64 backdrop-blur-sm shadow-lg shadow-red-900/20">
            <span className="font-bold block mb-1">Error</span>
            {error}
          </div>
        )}
      </div>

      <MetricsPanel state={simulationState} />
      
      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-gray-800/90 backdrop-blur-md px-4 py-3 rounded-xl border border-gray-700/50 text-xs text-gray-300 flex flex-wrap gap-5 shadow-2xl items-center font-medium tracking-wide">
        <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-sm bg-[#3182ce] shadow-[0_0_8px_rgba(49,130,206,0.6)]"></div> Rescue Agent</div>
        <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#ed8936] shadow-[0_0_8px_rgba(237,137,54,0.6)]"></div> Survivor</div>
        <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-sm bg-[#38a169] shadow-[0_0_8px_rgba(56,161,105,0.6)]"></div> Evac Zone</div>
        {(!simulationState || simulationState.hazard_type === 'FIRE') && (
          <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-sm bg-[#e53e3e] shadow-[0_0_8px_rgba(229,62,62,0.6)]"></div> Fire Hazard</div>
        )}
        {simulationState?.hazard_type === 'FLOOD' && (
          <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-sm bg-[#00b5d8] shadow-[0_0_8px_rgba(0,181,216,0.6)]"></div> Flood Hazard</div>
        )}
        <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-sm bg-[#1a202c] border border-gray-600"></div> Impassable Structure</div>
      </div>
    </div>
  );
}

export default App;
