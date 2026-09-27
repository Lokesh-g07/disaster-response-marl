import React, { useMemo, useRef, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as THREE from 'three';
import { SimulationState, AgentState } from '../types';

interface AgentDroneProps {
  agent: AgentState;
}

const AgentDrone: React.FC<AgentDroneProps> = ({ agent }) => {
  const groupRef = useRef<THREE.Group>(null);
  const isBlocked = agent.status === 'BLOCKED';
  const isMoving = agent.status === 'MOVING';
  const isInactive = agent.status === 'INACTIVE';
  const bodyColor = isInactive ? "#718096" : "#3182ce";
  const emissiveColor = isInactive ? "#000000" : (isBlocked ? "#fc8181" : (isMoving ? "#63b3ed" : "#4299e1"));

  // Target grid position (col, height, row) -> (x, y, z)
  const targetPos = useMemo(() => new THREE.Vector3(agent.position[1], 0.6, agent.position[0]), [agent.position]);
  const currentPos = useRef(targetPos.clone());
  
  // Track previous grid position to calculate rotation direction
  const prevGridPos = useRef([...agent.position]);
  const targetRot = useRef(0);
  const currentRot = useRef(0);

  useEffect(() => {
    const dx = agent.position[1] - prevGridPos.current[1]; // col diff
    const dz = agent.position[0] - prevGridPos.current[0]; // row diff

    // Update target rotation if moving
    if (dx !== 0 || dz !== 0) {
      targetRot.current = Math.atan2(dx, dz);
      
      // Ensure shortest rotation path
      let diff = targetRot.current - currentRot.current;
      while (diff < -Math.PI) diff += Math.PI * 2;
      while (diff > Math.PI) diff -= Math.PI * 2;
      targetRot.current = currentRot.current + diff;
    }
    
    prevGridPos.current = [...agent.position];
  }, [agent.position]);

  useFrame((state, delta) => {
    if (!groupRef.current) return;

    // Translation interpolation (damped) - reached ~95% in 0.2s
    if (isBlocked) {
      currentPos.current.copy(targetPos);
    } else {
      const dampFactor = 1 - Math.exp(-15 * delta);
      currentPos.current.lerp(targetPos, dampFactor);
    }

    // Rotation interpolation
    const rotDamp = 1 - Math.exp(-12 * delta);
    currentRot.current += (targetRot.current - currentRot.current) * rotDamp;

    // Subtle hover/bobbing based on time
    const hoverAmplitude = isMoving ? 0.05 : 0.02;
    const hoverFrequency = isMoving ? 10 : 3;
    const hoverOffset = isInactive ? 0 : Math.sin(state.clock.elapsedTime * hoverFrequency) * hoverAmplitude;

    groupRef.current.position.set(
      currentPos.current.x,
      currentPos.current.y + hoverOffset,
      currentPos.current.z
    );
    groupRef.current.rotation.y = currentRot.current;
  });

  return (
    <group ref={groupRef}>
      {/* Drone Chassis */}
      <mesh castShadow>
        <cylinderGeometry args={[0.3, 0.3, 0.15, 8]} />
        <meshStandardMaterial color={bodyColor} metalness={0.6} roughness={0.2} />
      </mesh>
      {/* Sensor Dome */}
      <mesh position={[0, 0.12, 0]} castShadow>
        <sphereGeometry args={[0.12, 8, 8]} />
        <meshStandardMaterial color="#2d3748" roughness={0.1} metalness={0.8} />
      </mesh>
      {/* Status Beacon */}
      <mesh position={[0, 0.2, 0]}>
        <sphereGeometry args={[0.05, 8, 8]} />
        <meshStandardMaterial color={emissiveColor} emissive={emissiveColor} emissiveIntensity={isMoving ? 2 : 1} />
      </mesh>
      
      <Html position={[0, 0.6, 0]} center zIndexRange={[100, 0]}>
        <div className={`text-[10px] font-bold px-1.5 py-0.5 rounded shadow-sm whitespace-nowrap bg-gray-900/80 backdrop-blur-sm ${isBlocked ? 'text-red-400 border border-red-500/50' : isMoving ? 'text-blue-300 border border-blue-500/30' : 'text-gray-300 border border-gray-600/50'}`}>
          RESCUE UNIT
          {isBlocked && ' [BLOCKED]'}
        </div>
      </Html>

      {/* Rotors/Arms */}
      {[0, 1, 2, 3].map((armIdx) => (
        <group key={armIdx} rotation={[0, (Math.PI / 2) * armIdx + Math.PI / 4, 0]}>
          <mesh position={[0.3, 0, 0]} castShadow>
            <boxGeometry args={[0.2, 0.05, 0.05]} />
            <meshStandardMaterial color={isInactive ? "#4a5568" : "#2b6cb0"} />
          </mesh>
          <mesh position={[0.4, 0.05, 0]}>
            <cylinderGeometry args={[0.1, 0.1, 0.02, 8]} />
            <meshStandardMaterial color="#1a202c" />
          </mesh>
        </group>
      ))}
      {!isInactive && <pointLight color={emissiveColor} distance={2.5} intensity={isMoving ? 1.5 : 1} />}
    </group>
  );
};

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
        child.position.y = 0.45 + Math.sin(timeRef.current * 3 + i) * 0.05;
      });
    }
  });

  // Ground plane - Immersive floor
  const groundMesh = useMemo(() => {
    const centerX = (width - 1) / 2;
    const centerZ = (height - 1) / 2;
    
    // Build array of floor tiles for non-wall cells
    const floorTiles = [];
    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        const isWall = walls.some(w => w[0] === r && w[1] === c);
        if (!isWall) {
          const isDarker = (r + c) % 2 === 0;
          floorTiles.push(
            <mesh key={`floor-${r}-${c}`} position={[c, 0, r]} receiveShadow>
              <boxGeometry args={[0.98, 0.1, 0.98]} />
              <meshStandardMaterial color={isDarker ? "#cbd5e0" : "#e2e8f0"} roughness={0.7} />
            </mesh>
          );
        }
      }
    }

    return (
      <group position={[0, -0.05, 0]}>
        {floorTiles}
        <mesh receiveShadow position={[centerX, -0.1, centerZ]}>
          <boxGeometry args={[width + 1, 0.1, height + 1]} />
          <meshStandardMaterial color="#4a5568" roughness={0.9} />
        </mesh>
      </group>
    );
  }, [width, height, walls]);

  return (
    <group position={[offsetX, 0, offsetZ]}>
      {groundMesh}
      
      {/* Exits as Doorways/Safe Zones */}
      {exits.map((pos, i) => (
        <group key={`exit-${i}`} position={[pos[1], 0, pos[0]]}>
          <mesh position={[0, 0.06, 0]} receiveShadow>
            <boxGeometry args={[0.9, 0.12, 0.9]} />
            <meshStandardMaterial color="#38a169" roughness={0.3} />
          </mesh>
          <mesh position={[0, 0.13, 0]} rotation={[-Math.PI/2, 0, 0]}>
            <ringGeometry args={[0.3, 0.4, 16]} />
            <meshBasicMaterial color="#68d391" side={THREE.DoubleSide} transparent opacity={0.8} />
          </mesh>
          <Html position={[0, 1.2, 0]} center zIndexRange={[100, 0]}>
            <div className="text-[10px] font-bold text-green-400 bg-gray-900/80 px-1.5 py-0.5 rounded shadow-sm border border-green-500/50 backdrop-blur-sm">
              SAFE EXIT
            </div>
          </Html>
          <pointLight position={[0, 0.5, 0]} color="#68d391" distance={2.5} intensity={1} />
        </group>
      ))}

      {/* Walls as Architectural Segments */}
      {walls.map((pos, i) => (
        <group key={`wall-${i}`} position={[pos[1], 0.6, pos[0]]}>
          <mesh castShadow receiveShadow>
            <boxGeometry args={[1, 1.2, 1]} />
            <meshStandardMaterial color="#a0aec0" roughness={0.8} />
          </mesh>
          <mesh position={[0, 0.61, 0]} castShadow receiveShadow>
            <boxGeometry args={[1.02, 0.05, 1.02]} />
            <meshStandardMaterial color="#718096" roughness={0.9} />
          </mesh>
        </group>
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
                <mesh castShadow>
                  <coneGeometry args={[0.35, 0.8, 5]} />
                  <meshStandardMaterial color="#e53e3e" emissive="#c53030" emissiveIntensity={0.6} transparent opacity={0.9} />
                </mesh>
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

      {/* Survivors (Humanoid silhouettes) */}
      <group ref={survivorGroupRef}>
        {state.survivors.map((survivor) => !survivor.rescued && (
          <group key={`survivor-${survivor.id}`} position={[survivor.position[1], 0.45, survivor.position[0]]}>
            <mesh castShadow position={[0, -0.15, 0]}>
              <capsuleGeometry args={[0.12, 0.3, 4, 8]} />
              <meshStandardMaterial color="#ed8936" roughness={0.7} />
            </mesh>
            <mesh castShadow position={[0, 0.2, 0]}>
              <sphereGeometry args={[0.12, 16, 16]} />
              <meshStandardMaterial color="#fbd38d" roughness={0.5} />
            </mesh>
            <pointLight color="#f6ad55" distance={1.5} intensity={0.5} />
            <Html position={[0, 0.6, 0]} center zIndexRange={[100, 0]}>
              <div className="text-[9px] font-bold text-orange-300 px-1 rounded-sm bg-gray-900/60 backdrop-blur-sm shadow-sm border border-orange-500/30">
                SURVIVOR
              </div>
            </Html>
          </group>
        ))}
      </group>

      {/* Rescue Agents (Drones) */}
      {state.agents.map((agent) => (
        <AgentDrone key={`agent-${agent.id}`} agent={agent} />
      ))}
    </group>
  );
};

