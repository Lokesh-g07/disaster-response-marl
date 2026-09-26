import React, { useMemo } from 'react';
import { SimulationState } from '../types';

interface GridProps {
  state: SimulationState;
}

export const Grid: React.FC<GridProps> = ({ state }) => {
  const { width, height, walls, exits, fire_cells } = state.grid;

  // Center the grid around origin
  const offsetX = -width / 2;
  const offsetZ = -height / 2;

  // Ground plane
  const groundMesh = useMemo(() => (
    <mesh position={[0, -0.5, 0]} receiveShadow>
      <boxGeometry args={[width, 1, height]} />
      <meshStandardMaterial color="#2d3748" />
    </mesh>
  ), [width, height]);

  return (
    <group position={[offsetX + 0.5, 0, offsetZ + 0.5]}>
      {groundMesh}
      
      {/* Exits */}
      {exits.map((pos, i) => (
        <mesh key={`exit-${i}`} position={[pos[0], 0.1, pos[1]]} receiveShadow>
          <boxGeometry args={[1, 0.2, 1]} />
          <meshStandardMaterial color="#38a169" />
          <pointLight position={[pos[0], 1, pos[1]]} color="#68d391" distance={3} intensity={2} />
        </mesh>
      ))}

      {/* Walls */}
      {walls.map((pos, i) => (
        <mesh key={`wall-${i}`} position={[pos[0], 0.5, pos[1]]} castShadow receiveShadow>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color="#4a5568" />
        </mesh>
      ))}

      {/* Fire */}
      {fire_cells.map((pos, i) => (
        <group key={`fire-${i}`} position={[pos[0], 0.2, pos[1]]}>
          <mesh castShadow>
            <coneGeometry args={[0.4, 0.8, 4]} />
            <meshStandardMaterial color="#e53e3e" emissive="#c53030" emissiveIntensity={0.5} />
          </mesh>
          <pointLight color="#fc8181" distance={2} intensity={1} />
        </group>
      ))}

      {/* Survivors */}
      {state.survivors.map((survivor) => !survivor.rescued && (
        <mesh key={`survivor-${survivor.id}`} position={[survivor.position[0], 0.3, survivor.position[1]]} castShadow>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshStandardMaterial color="#ecc94b" />
        </mesh>
      ))}

      {/* Agents */}
      {state.agents.map((agent) => (
        <group key={`agent-${agent.id}`} position={[agent.position[0], 0.4, agent.position[1]]}>
          <mesh castShadow>
            <cylinderGeometry args={[0.3, 0.3, 0.8, 8]} />
            <meshStandardMaterial color="#4299e1" />
          </mesh>
          {/* Agent Indicator Light */}
          <pointLight color="#63b3ed" distance={2} intensity={0.5} />
        </group>
      ))}
    </group>
  );
};
