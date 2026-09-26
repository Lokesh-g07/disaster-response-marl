import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { SimulationState } from '../types';

interface GridProps {
  state: SimulationState;
}

export const Grid: React.FC<GridProps> = ({ state }) => {
  const { width, height, walls, exits, hazard_cells } = state.grid;
  const isFlood = state.hazard_type === 'FLOOD';

  // Center the grid around origin
  // X-axis maps to columns (width), Z-axis maps to rows (height)
  const offsetX = -(width - 1) / 2;
  const offsetZ = -(height - 1) / 2;

  // Animation Refs
  const timeRef = useRef(0);
  const waterMaterialRef = useRef<THREE.MeshPhysicalMaterial>(null);
  const fireGroupRef = useRef<THREE.Group>(null);
  const survivorGroupRef = useRef<THREE.Group>(null);

  useFrame((_, delta) => {
    timeRef.current += delta;
    
    // Animate water
    if (waterMaterialRef.current) {
      // Subtle pulse/wave effect via opacity or displacement (simulated simply)
      waterMaterialRef.current.opacity = 0.7 + Math.sin(timeRef.current * 2) * 0.1;
    }

    // Animate fire scaling/flicker
    if (fireGroupRef.current) {
      fireGroupRef.current.children.forEach((child, i) => {
        const scale = 1 + Math.sin(timeRef.current * 5 + i) * 0.1;
        child.scale.set(scale, scale, scale);
      });
    }

    // Animate survivors (subtle hover/pulse)
    if (survivorGroupRef.current) {
      survivorGroupRef.current.children.forEach((child, i) => {
        child.position.y = 0.4 + Math.sin(timeRef.current * 3 + i) * 0.05;
      });
    }
  });

  // Ground plane with grid lines
  const groundMesh = useMemo(() => {
    // The ground mesh needs to be centered relative to the local origin
    // The local origin (0,0,0) corresponds to grid cell (row=0, col=0)
    // The center of the grid is at (col=(width-1)/2, row=(height-1)/2)
    const centerX = (width - 1) / 2;
    const centerZ = (height - 1) / 2;
    
    return (
      <group position={[centerX, -0.5, centerZ]}>
        <mesh receiveShadow position={[0, 0.49, 0]}>
          <boxGeometry args={[width, 0.1, height]} />
          <meshStandardMaterial color="#2d3748" />
        </mesh>
        {/* Draw grid lines */}
        <gridHelper 
          args={[Math.max(width, height), Math.max(width, height), '#4a5568', '#4a5568']} 
          position={[0, 0.55, 0]} 
        />
      </group>
    );
  }, [width, height]);

  return (
    <group position={[offsetX, 0, offsetZ]}>
      {groundMesh}
      
      {/* Exits */}
      {exits.map((pos, i) => (
        // X = col (pos[1]), Z = row (pos[0])
        <group key={`exit-${i}`} position={[pos[1], 0.05, pos[0]]}>
          <mesh receiveShadow>
            <boxGeometry args={[1, 0.1, 1]} />
            <meshStandardMaterial color="#38a169" emissive="#276749" emissiveIntensity={0.2} />
          </mesh>
          <mesh position={[0, 0.06, 0]} rotation={[-Math.PI/2, 0, 0]}>
            <planeGeometry args={[0.6, 0.6]} />
            <meshBasicMaterial color="#68d391" transparent opacity={0.5} />
          </mesh>
          <pointLight position={[0, 0.5, 0]} color="#68d391" distance={3} intensity={1} />
        </group>
      ))}

      {/* Walls / Buildings */}
      {walls.map((pos, i) => (
        <mesh key={`wall-${i}`} position={[pos[1], 0.75, pos[0]]} castShadow receiveShadow>
          <boxGeometry args={[1, 1.5, 1]} />
          <meshStandardMaterial color="#1a202c" roughness={0.9} />
        </mesh>
      ))}

      {/* Hazards (Fire or Flood) */}
      <group ref={isFlood ? null : fireGroupRef}>
        {hazard_cells.map((pos, i) => {
          if (isFlood) {
            return (
              <group key={`flood-${i}`} position={[pos[1], 0.15, pos[0]]}>
                <mesh receiveShadow castShadow>
                  <boxGeometry args={[1, 0.3, 1]} />
                  <meshPhysicalMaterial 
                    ref={i === 0 ? waterMaterialRef : null}
                    color="#00b5d8" 
                    transparent 
                    opacity={0.8} 
                    roughness={0.1}
                    transmission={0.5}
                    thickness={0.5}
                  />
                </mesh>
              </group>
            );
          } else {
            return (
              <group key={`fire-${i}`} position={[pos[1], 0.4, pos[0]]}>
                {/* Outer Flame */}
                <mesh castShadow>
                  <coneGeometry args={[0.35, 0.8, 5]} />
                  <meshStandardMaterial color="#e53e3e" emissive="#c53030" emissiveIntensity={0.6} transparent opacity={0.9} />
                </mesh>
                {/* Inner Flame */}
                <mesh position={[0, -0.1, 0]}>
                  <coneGeometry args={[0.2, 0.5, 4]} />
                  <meshStandardMaterial color="#ecc94b" emissive="#d69e2e" emissiveIntensity={1} />
                </mesh>
                <pointLight color="#fc8181" distance={3} intensity={1.5} />
              </group>
            );
          }
        })}
      </group>

      {/* Survivors */}
      <group ref={survivorGroupRef}>
        {state.survivors.map((survivor) => !survivor.rescued && (
          <group key={`survivor-${survivor.id}`} position={[survivor.position[1], 0.4, survivor.position[0]]}>
            <mesh castShadow>
              <sphereGeometry args={[0.2, 16, 16]} />
              <meshStandardMaterial color="#ed8936" roughness={0.5} />
            </mesh>
            <mesh position={[0, 0.3, 0]}>
              <sphereGeometry args={[0.1, 8, 8]} />
              <meshStandardMaterial color="#fbd38d" />
            </mesh>
            <pointLight color="#f6ad55" distance={1.5} intensity={0.5} />
          </group>
        ))}
      </group>

      {/* Rescue Agents (Drones) */}
      {state.agents.map((agent) => (
        <group key={`agent-${agent.id}`} position={[agent.position[1], 0.6, agent.position[0]]}>
          {/* Drone Body */}
          <mesh castShadow>
            <boxGeometry args={[0.5, 0.2, 0.5]} />
            <meshStandardMaterial color="#3182ce" metalness={0.5} roughness={0.2} />
          </mesh>
          {/* Top light / Sensor */}
          <mesh position={[0, 0.15, 0]}>
            <sphereGeometry args={[0.08, 8, 8]} />
            <meshStandardMaterial color="#63b3ed" emissive="#63b3ed" emissiveIntensity={1} />
          </mesh>
          {/* Drone Arms */}
          <mesh position={[0, 0, 0]} rotation={[0, Math.PI/4, 0]}>
            <boxGeometry args={[0.7, 0.05, 0.05]} />
            <meshStandardMaterial color="#2b6cb0" />
          </mesh>
          <mesh position={[0, 0, 0]} rotation={[0, -Math.PI/4, 0]}>
            <boxGeometry args={[0.7, 0.05, 0.05]} />
            <meshStandardMaterial color="#2b6cb0" />
          </mesh>
          <pointLight color="#90cdf4" distance={3} intensity={1} />
        </group>
      ))}
    </group>
  );
};

