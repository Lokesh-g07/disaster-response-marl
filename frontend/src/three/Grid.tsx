import React, { useMemo, useRef, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as THREE from 'three';
import { SimulationState, AgentState, SurvivorState } from '../types';

/* =========================================================================
   AGENT DRONE
   ========================================================================= */
const AgentDrone: React.FC<{ agent: AgentState }> = ({ agent }) => {
  const groupRef = useRef<THREE.Group>(null);
  const rotorsRef = useRef<THREE.Group>(null);
  
  const isBlocked = agent.status === 'BLOCKED';
  const isMoving = agent.status === 'MOVING';
  const isInactive = agent.status === 'INACTIVE';
  
  const bodyColor = isInactive ? "#718096" : "#3182ce";
  const emissiveColor = isInactive ? "#000000" : (isBlocked ? "#fc8181" : (isMoving ? "#63b3ed" : "#4299e1"));

  const targetPos = useMemo(() => new THREE.Vector3(agent.position[1], 0.6, agent.position[0]), [agent.position]);
  const currentPos = useRef(targetPos.clone());
  const prevGridPos = useRef([...agent.position]);
  const targetRot = useRef(0);
  const currentRot = useRef(0);

  useEffect(() => {
    const dx = agent.position[1] - prevGridPos.current[1];
    const dz = agent.position[0] - prevGridPos.current[0];
    if (dx !== 0 || dz !== 0) {
      targetRot.current = Math.atan2(dx, dz);
      let diff = targetRot.current - currentRot.current;
      while (diff < -Math.PI) diff += Math.PI * 2;
      while (diff > Math.PI) diff -= Math.PI * 2;
      targetRot.current = currentRot.current + diff;
    }
    prevGridPos.current = [...agent.position];
  }, [agent.position]);

  useFrame((state, delta) => {
    if (!groupRef.current) return;

    if (isBlocked) {
      currentPos.current.copy(targetPos);
    } else {
      currentPos.current.lerp(targetPos, 1 - Math.exp(-15 * delta));
    }

    currentRot.current += (targetRot.current - currentRot.current) * (1 - Math.exp(-12 * delta));

    const hoverOffset = isInactive ? 0 : Math.sin(state.clock.elapsedTime * (isMoving ? 10 : 3)) * (isMoving ? 0.05 : 0.02);
    groupRef.current.position.set(currentPos.current.x, currentPos.current.y + hoverOffset, currentPos.current.z);
    groupRef.current.rotation.y = currentRot.current;

    // Spin rotors
    if (rotorsRef.current && !isInactive) {
      rotorsRef.current.rotation.y += delta * (isMoving ? 25 : 8);
    }
  });

  return (
    <group ref={groupRef}>
      <mesh castShadow>
        <cylinderGeometry args={[0.3, 0.3, 0.15, 8]} />
        <meshStandardMaterial color={bodyColor} metalness={0.6} roughness={0.2} />
      </mesh>
      <mesh position={[0, 0.12, 0]} castShadow>
        <sphereGeometry args={[0.12, 8, 8]} />
        <meshStandardMaterial color="#2d3748" roughness={0.1} metalness={0.8} />
      </mesh>
      <mesh position={[0, 0.2, 0]}>
        <sphereGeometry args={[0.05, 8, 8]} />
        <meshStandardMaterial color={emissiveColor} emissive={emissiveColor} emissiveIntensity={isMoving ? 2 : 1} />
      </mesh>
      
      <Html position={[0, 0.6, 0]} center zIndexRange={[100, 0]}>
        <div className={`text-[10px] font-bold px-1.5 py-0.5 rounded shadow-sm whitespace-nowrap bg-gray-900/80 backdrop-blur-sm ${isBlocked ? 'text-red-400 border border-red-500/50' : isMoving ? 'text-blue-300 border border-blue-500/30' : 'text-gray-300 border border-gray-600/50'}`}>
          {agent.id.replace('_', ' ').toUpperCase()}
          {isBlocked && ' [BLOCKED]'}
        </div>
      </Html>

      <group ref={rotorsRef}>
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
      </group>
      {!isInactive && <pointLight color={emissiveColor} distance={2.5} intensity={isMoving ? 1.5 : 0.8} />}
    </group>
  );
};

/* =========================================================================
   SURVIVOR
   ========================================================================= */
const SurvivorEntity: React.FC<{ survivor: SurvivorState }> = ({ survivor }) => {
  const beaconRef = useRef<THREE.Mesh>(null);
  const hoverRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (hoverRef.current) {
      hoverRef.current.position.y = 0.45 + Math.sin(state.clock.elapsedTime * 3 + survivor.position[0]) * 0.05;
    }
    if (beaconRef.current) {
      const scale = 1 + (state.clock.elapsedTime * 2 % 1);
      beaconRef.current.scale.set(scale, scale, scale);
      (beaconRef.current.material as THREE.MeshBasicMaterial).opacity = Math.max(0, 0.8 - scale * 0.4);
    }
  });

  return (
    <group position={[survivor.position[1], 0, survivor.position[0]]}>
      {/* Floor beacon pulse */}
      <mesh ref={beaconRef} position={[0, 0.02, 0]} rotation={[-Math.PI/2, 0, 0]}>
        <ringGeometry args={[0.3, 0.4, 16]} />
        <meshBasicMaterial color="#f6ad55" side={THREE.DoubleSide} transparent opacity={0.6} />
      </mesh>

      <group ref={hoverRef}>
        <mesh castShadow position={[0, -0.15, 0]}>
          <capsuleGeometry args={[0.12, 0.3, 4, 8]} />
          <meshStandardMaterial color="#ed8936" roughness={0.7} />
        </mesh>
        <mesh castShadow position={[0, 0.2, 0]}>
          <sphereGeometry args={[0.12, 16, 16]} />
          <meshStandardMaterial color="#fbd38d" roughness={0.5} emissive="#dd6b20" emissiveIntensity={0.2} />
        </mesh>
        <Html position={[0, 0.6, 0]} center zIndexRange={[100, 0]}>
          <div className="text-[9px] font-bold text-orange-200 px-1 rounded-sm bg-orange-900/80 backdrop-blur-sm shadow-sm border border-orange-500/50">
            SURVIVOR
          </div>
        </Html>
      </group>
    </group>
  );
};

/* =========================================================================
   HAZARDS
   ========================================================================= */
const FireHazard: React.FC<{ pos: [number, number] }> = ({ pos }) => {
  const embersRef = useRef<THREE.Group>(null);
  const flameRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (flameRef.current) {
      const scale = 1 + Math.sin(state.clock.elapsedTime * 8 + pos[0] + pos[1]) * 0.1;
      flameRef.current.scale.set(scale, scale, scale);
    }
    if (embersRef.current) {
      embersRef.current.children.forEach((child, i) => {
        child.position.y = (state.clock.elapsedTime * (1 + i * 0.5) + i) % 1.2;
        child.position.x = Math.sin(state.clock.elapsedTime * 2 + i) * 0.2;
      });
    }
  });

  return (
    <group position={[pos[1], 0.4, pos[0]]}>
      <group ref={flameRef}>
        <mesh castShadow>
          <coneGeometry args={[0.35, 0.8, 5]} />
          <meshStandardMaterial color="#e53e3e" emissive="#c53030" emissiveIntensity={0.6} transparent opacity={0.8} />
        </mesh>
        <mesh position={[0, -0.1, 0]}>
          <coneGeometry args={[0.2, 0.5, 4]} />
          <meshStandardMaterial color="#ecc94b" emissive="#d69e2e" emissiveIntensity={1} />
        </mesh>
      </group>
      {/* Floating Embers */}
      <group ref={embersRef} position={[0, -0.3, 0]}>
        {[0, 1, 2].map(i => (
          <mesh key={i} position={[0, 0, 0]}>
            <sphereGeometry args={[0.03, 4, 4]} />
            <meshBasicMaterial color="#fbd38d" />
          </mesh>
        ))}
      </group>
      {/* Local floor scorch mark */}
      <mesh position={[0, -0.38, 0]} rotation={[-Math.PI/2, 0, 0]}>
        <planeGeometry args={[0.9, 0.9]} />
        <meshBasicMaterial color="#4a1c1c" transparent opacity={0.5} />
      </mesh>
    </group>
  );
};

const FloodHazard: React.FC<{ pos: [number, number] }> = ({ pos }) => {
  const matRef = useRef<THREE.MeshPhysicalMaterial>(null);
  
  useFrame((state) => {
    if (matRef.current) {
      matRef.current.opacity = 0.65 + Math.sin(state.clock.elapsedTime * 2.5 + pos[0]) * 0.15;
    }
  });

  return (
    <group position={[pos[1], 0.15, pos[0]]}>
      <mesh receiveShadow castShadow>
        <boxGeometry args={[1, 0.3, 1]} />
        <meshPhysicalMaterial 
          ref={matRef}
          color="#00b5d8" 
          transparent 
          opacity={0.8} 
          roughness={0.05}
          transmission={0.6}
          thickness={0.8}
        />
      </mesh>
    </group>
  );
};

/* =========================================================================
   ARCHITECTURAL ELEMENTS
   ========================================================================= */
const SafeExit: React.FC<{ pos: [number, number] }> = ({ pos }) => (
  <group position={[pos[1], 0, pos[0]]}>
    {/* Floor hazard striping leading to exit */}
    <mesh position={[0, 0.02, 0]} rotation={[-Math.PI/2, 0, 0]}>
      <planeGeometry args={[1, 1]} />
      <meshStandardMaterial color="#eab308" roughness={0.9} transparent opacity={0.3} />
    </mesh>
    <mesh position={[0, 0.06, 0]} receiveShadow>
      <boxGeometry args={[0.9, 0.12, 0.9]} />
      <meshStandardMaterial color="#38a169" roughness={0.3} />
    </mesh>
    <mesh position={[0, 0.13, 0]} rotation={[-Math.PI/2, 0, 0]}>
      <ringGeometry args={[0.3, 0.4, 16]} />
      <meshBasicMaterial color="#68d391" side={THREE.DoubleSide} transparent opacity={0.8} />
    </mesh>
    <Html position={[0, 1.2, 0]} center zIndexRange={[100, 0]}>
      <div className="text-[10px] font-bold text-green-300 bg-green-900/90 px-1.5 py-0.5 rounded shadow-lg border border-green-500/50 backdrop-blur-sm">
        EVACUATION POINT
      </div>
    </Html>
    <pointLight position={[0, 0.8, 0]} color="#68d391" distance={3} intensity={1.2} />
  </group>
);

const WallSegment: React.FC<{ pos: [number, number] }> = ({ pos }) => (
  <group position={[pos[1], 0.6, pos[0]]}>
    <mesh castShadow receiveShadow>
      <boxGeometry args={[1, 1.2, 1]} />
      <meshStandardMaterial color="#a0aec0" roughness={0.8} />
    </mesh>
    {/* Structural support column inside wall */}
    <mesh castShadow receiveShadow position={[0, 0, 0]}>
      <cylinderGeometry args={[0.55, 0.55, 1.25, 4]} />
      <meshStandardMaterial color="#cbd5e0" roughness={0.9} />
    </mesh>
    {/* Top trim */}
    <mesh position={[0, 0.63, 0]} castShadow receiveShadow>
      <boxGeometry args={[1.05, 0.06, 1.05]} />
      <meshStandardMaterial color="#4a5568" roughness={0.9} />
    </mesh>
  </group>
);

/* =========================================================================
   MAIN GRID
   ========================================================================= */
export const Grid: React.FC<{ state: SimulationState }> = ({ state }) => {
  const { width, height, walls, exits, hazard_cells } = state.grid;
  const isFlood = state.hazard_type === 'FLOOD';

  const offsetX = -(width - 1) / 2;
  const offsetZ = -(height - 1) / 2;

  // Generate immersive floor with zones and labels
  const groundMesh = useMemo(() => {
    const centerX = (width - 1) / 2;
    const centerZ = (height - 1) / 2;
    
    const floorTiles: React.ReactNode[] = [];
    const labels: React.ReactNode[] = [];
    
    // Add sector labels strictly inside bounds (NW, NE, SW, SE)
    const sectors = [
      { name: "SECTOR A", pos: [width * 0.25, height * 0.25] },
      { name: "SECTOR B", pos: [width * 0.75, height * 0.25] },
      { name: "SECTOR C", pos: [width * 0.25, height * 0.75] },
      { name: "SECTOR D", pos: [width * 0.75, height * 0.75] }
    ];

    sectors.forEach((sec, i) => {
      labels.push(
        <Html key={`label-${i}`} position={[sec.pos[0], 0.02, sec.pos[1]]} rotation={[-Math.PI/2, 0, 0]} transform>
          <div className="text-gray-500/30 text-3xl font-black font-sans pointer-events-none select-none tracking-widest">
            {sec.name}
          </div>
        </Html>
      );
    });

    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        const isWall = walls.some(w => w[0] === r && w[1] === c);
        if (!isWall) {
          // Different tones for different quadrants to simulate rooms
          const isDarker = (r + c) % 2 === 0;
          const isNorth = r < height / 2;
          const isWest = c < width / 2;
          
          let baseColor = "#e2e8f0";
          if (isNorth && isWest) baseColor = isDarker ? "#cbd5e0" : "#e2e8f0";
          else if (isNorth && !isWest) baseColor = isDarker ? "#e2e8f0" : "#edf2f7";
          else if (!isNorth && isWest) baseColor = isDarker ? "#a0aec0" : "#cbd5e0";
          else baseColor = isDarker ? "#edf2f7" : "#f7fafc";

          floorTiles.push(
            <mesh key={`floor-${r}-${c}`} position={[c, 0, r]} receiveShadow>
              <boxGeometry args={[0.98, 0.1, 0.98]} />
              <meshStandardMaterial color={baseColor} roughness={0.8} />
            </mesh>
          );
        }
      }
    }

    return (
      <group position={[0, -0.05, 0]}>
        {floorTiles}
        {labels}
        <mesh receiveShadow position={[centerX, -0.1, centerZ]}>
          <boxGeometry args={[width + 1.5, 0.1, height + 1.5]} />
          <meshStandardMaterial color="#2d3748" roughness={0.9} />
        </mesh>
        {/* Subtle baseboard/wall boundary line */}
        <mesh receiveShadow position={[centerX, 0, centerZ]}>
          <boxGeometry args={[width + 0.2, 0.11, height + 0.2]} />
          <meshBasicMaterial color="#1a202c" />
        </mesh>
      </group>
    );
  }, [width, height, walls]);

  return (
    <group position={[offsetX, 0, offsetZ]}>
      {groundMesh}
      
      {exits.map((pos, i) => <SafeExit key={`exit-${i}`} pos={pos} />)}
      
      {walls.map((pos, i) => <WallSegment key={`wall-${i}`} pos={pos} />)}

      <group>
        {hazard_cells.map((pos, i) => (
          isFlood 
            ? <FloodHazard key={`flood-${i}`} pos={pos} /> 
            : <FireHazard key={`fire-${i}`} pos={pos} />
        ))}
      </group>

      <group>
        {state.survivors.map((survivor) => !survivor.rescued && (
          <SurvivorEntity key={`survivor-${survivor.id}`} survivor={survivor} />
        ))}
      </group>

      {state.agents.map((agent) => (
        <AgentDrone key={`agent-${agent.id}`} agent={agent} />
      ))}
    </group>
  );
};

