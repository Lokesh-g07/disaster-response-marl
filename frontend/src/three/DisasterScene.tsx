import React, { useEffect } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { SimulationState } from '../types';
import { Grid } from './Grid';

interface DisasterSceneProps {
  state: SimulationState | null;
}

const CameraSetup: React.FC<{ state: SimulationState }> = ({ state }) => {
  const { camera, controls } = useThree();
  const { width, height } = state.grid;

  useEffect(() => {
    if (!controls) return;
    
    // Dynamic framing based on grid dimensions
    const maxDim = Math.max(width, height);
    const distance = maxDim * 1.5;
    
    // Isometric/three-quarter elevated view
    camera.position.set(-distance * 0.7, distance * 0.8, distance * 0.7);
    camera.lookAt(0, 0, 0);
    
    if ((controls as any).target) {
      (controls as any).target.set(0, 0, 0);
      (controls as any).update();
    }
  }, [width, height, camera, controls]);

  return null;
};

export const DisasterScene: React.FC<DisasterSceneProps> = ({ state }) => {
  return (
    <div className="w-full h-full bg-gray-900 absolute inset-0">
      <Canvas shadows camera={{ fov: 45 }}>
        <color attach="background" args={['#1a202c']} />
        
        {state && <CameraSetup state={state} />}
        
        {/* Soft environment lighting */}
        <ambientLight intensity={0.6} />
        <directionalLight 
          position={[10, 20, 5]} 
          castShadow 
          intensity={1.2} 
          shadow-mapSize-width={2048} 
          shadow-mapSize-height={2048} 
          shadow-camera-far={50}
          shadow-camera-left={-15}
          shadow-camera-right={15}
          shadow-camera-top={15}
          shadow-camera-bottom={-15}
        />
        <hemisphereLight args={['#2b6cb0', '#2d3748', 0.4]} />
        
        {state && <Grid state={state} />}
        
        <OrbitControls 
          makeDefault 
          maxPolarAngle={Math.PI / 2 - 0.1}
          minPolarAngle={0.1}
          minDistance={5}
          maxDistance={60}
        />
      </Canvas>
    </div>
  );
};
