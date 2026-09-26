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
    <div className="w-full h-screen overflow-hidden flex relative bg-gray-900 font-sans">
      <DisasterScene state={simulationState} />
      
      <div className="absolute top-4 left-4 z-10 flex flex-col gap-4">
        <div className="bg-gray-800/90 backdrop-blur-sm p-4 rounded-lg shadow-lg border border-gray-700 w-64">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
              CrisisRL
            </h1>
            <div className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider flex items-center gap-1 ${
              connectionStatus === 'Connected' || connectionStatus === 'Running' ? 'bg-green-900/50 text-green-400 border border-green-700' :
              connectionStatus === 'Disconnected' ? 'bg-gray-900/50 text-gray-400 border border-gray-700' :
              connectionStatus === 'Error' ? 'bg-red-900/50 text-red-400 border border-red-700' :
              'bg-yellow-900/50 text-yellow-400 border border-yellow-700'
            }`}>
              {connectionStatus === 'Running' && <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"></div>}
              {connectionStatus}
            </div>
          </div>
          <p className="text-gray-400 text-xs mt-1">Multi-Agent Disaster Response</p>
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
      <div className="absolute bottom-4 left-4 bg-gray-800/90 backdrop-blur-sm p-3 rounded-lg border border-gray-700 text-xs text-gray-300 flex gap-4">
        <div className="flex items-center gap-2"><div className="w-3 h-3 rounded bg-[#4299e1]"></div> Agent</div>
        <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#ecc94b]"></div> Survivor</div>
        <div className="flex items-center gap-2"><div className="w-3 h-3 bg-[#38a169]"></div> Exit</div>
        <div className="flex items-center gap-2"><div className="w-3 h-3 bg-[#e53e3e]"></div> Fire</div>
        <div className="flex items-center gap-2"><div className="w-3 h-3 bg-[#4a5568]"></div> Wall</div>
      </div>
    </div>
  );
}

export default App;
