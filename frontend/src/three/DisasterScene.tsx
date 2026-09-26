import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { SimulationState } from '../types';
import { Grid } from './Grid';

interface DisasterSceneProps {
  state: SimulationState | null;
}

export const DisasterScene: React.FC<DisasterSceneProps> = ({ state }) => {
  return (
    <div className="w-full h-full bg-gray-900 absolute inset-0">
      <Canvas shadows camera={{ position: [-10, 15, 10], fov: 45 }}>
        <color attach="background" args={['#1a202c']} />
        
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
          maxPolarAngle={Math.PI / 2 - 0.1} // Prevent camera from going completely below ground
          minPolarAngle={0.1} // Prevent going completely top-down to preserve 3D feel
          minDistance={5}
          maxDistance={60}
          target={[0, 0, 0]}
        />
      </Canvas>
    </div>
  );
};
