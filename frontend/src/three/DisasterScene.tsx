import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sky } from '@react-three/drei';
import { SimulationState } from '../types';
import { Grid } from './Grid';

interface DisasterSceneProps {
  state: SimulationState | null;
}

export const DisasterScene: React.FC<DisasterSceneProps> = ({ state }) => {
  return (
    <div className="w-full h-full bg-gray-900 absolute inset-0">
      <Canvas shadows camera={{ position: [0, 15, 15], fov: 45 }}>
        <color attach="background" args={['#1a202c']} />
        <Sky distance={450000} sunPosition={[0, 1, 0]} inclination={0} azimuth={0.25} />
        
        <ambientLight intensity={0.4} />
        <directionalLight 
          position={[10, 20, 10]} 
          castShadow 
          intensity={1} 
          shadow-mapSize-width={2048} 
          shadow-mapSize-height={2048} 
        />
        
        {state && <Grid state={state} />}
        
        <OrbitControls 
          makeDefault 
          maxPolarAngle={Math.PI / 2 - 0.05} // Prevent camera from going below ground
          minDistance={5}
          maxDistance={50}
        />
      </Canvas>
    </div>
  );
};
