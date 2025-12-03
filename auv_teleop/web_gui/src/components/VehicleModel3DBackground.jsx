import React, { Suspense, useRef } from 'react';
import { Canvas, useLoader, useFrame } from '@react-three/fiber';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';
import * as THREE from 'three';

function STLModel({ url, color = '#00D9FF' }) {
  const meshRef = useRef();
  const geometry = useLoader(STLLoader, url);

  // Center and compute bounding box for auto-scaling
  React.useEffect(() => {
    if (geometry) {
      geometry.center();
      geometry.computeBoundingBox();
      const box = geometry.boundingBox;
      const size = new THREE.Vector3();
      box.getSize(size);
      console.log('STL Model loaded, size:', size.x, size.y, size.z);
    }
  }, [geometry, url]);

  // Auto-rotate animation
  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.z += 0.002;
    }
  });

  return (
    <mesh ref={meshRef} geometry={geometry} rotation={[Math.PI / -2, 0, 0]} scale={3}>
      <meshStandardMaterial 
        color={color} 
        metalness={0.6} 
        roughness={0.3}
        side={THREE.DoubleSide}
        emissive={color}
        emissiveIntensity={0.15}
      />
    </mesh>
  );
}

function VehicleModel3DBackground({ modelPath = '/body.stl', color = '#00D9FF' }) {
  React.useEffect(() => {
    console.log('VehicleModel3DBackground mounted, loading:', modelPath);
  }, [modelPath]);

  return (
    <div 
      style={{ 
        position: 'absolute',
        top: '-10%',
        left: '-3%',
        width: '50%', 
        height: '50%',
        zIndex: 0,
        pointerEvents: 'none',
      }}
    >
      <Canvas 
        style={{ width: '100%', height: '100%' }}
        gl={{ alpha: true, antialias: true }}
        camera={{ position: [0, 2, 5], fov: 45 }}
        onCreated={({ gl }) => {
          gl.setClearColor(0x000000, 0);
          console.log('Canvas created successfully');
        }}
      >
        
        {/* Lighting */}
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} color={color} />
        <directionalLight position={[-10, -10, -5]} intensity={0.5} color="#ffffff" />
        <pointLight position={[0, 5, 0]} intensity={0.6} color={color} />
        
        {/* STL Model */}
        <Suspense fallback={null}>
          <STLModel url={modelPath} color={color} />
        </Suspense>
      </Canvas>
    </div>
  );
}

export default VehicleModel3DBackground;
